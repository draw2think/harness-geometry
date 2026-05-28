"""Baseline Eval — Single-turn direct QA (no GeoGebra tools, no multi-turn).

Each problem is one request: system prompt + image + question + choices → ANSWER.

Runs online by default (sequential, or --workers N for parallel). A Gemini
Batch API path (~50% cost) is also supported:
  (default) — online requests, scored immediately
  --collect — poll/retrieve a previously submitted batch job and score results

Pipeline
--------
  1. Load problems (geometry3k / pgps9k) — same loaders as main eval
  2. Build one request per problem (system + image + question)
  3. Submit batch job to Gemini API
  4. Poll until completion, then parse answers and compute accuracy

Output layout
-------------
  eval/
    geometry3k/                 (shared with main eval)
      2101/
        <slug>_baseline_result.json
      summary_<slug>_baseline_<ts>.json

Usage
-----
  # Submit a batch (10 problems, medium thinking):
  python eval/eval_baseline.py --dataset geometry3k \\
      --data_dir /data/geometry3k/val --sample 10 --thinking medium

  # Submit PGPS9K batch:
  python eval/eval_baseline.py --dataset pgps9k \\
      --data_dir /data/PGPS9K --sample 20 --thinking low

  # Collect results from a previous batch:
  python eval/eval_baseline.py --collect <batch_job_name>

  # Submit + wait (poll until done, then score):
  python eval/eval_baseline.py --dataset geometry3k \\
      --data_dir /data/geometry3k/val --sample 10 --wait

  # Online mode (no batch, sequential requests — for debugging):
  python eval/eval_baseline.py --dataset geometry3k \\
      --data_dir /data/geometry3k/val --sample 3 --online
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from symbolic.utils.model_registry import get_model, make_client
import eval_config as cfg

EVAL_ROOT = Path(__file__).parent

# Shared loaders + helpers
from eval_common import (
    parse_answer,
    validate,
    format_choices,
    load_image,
    image_content_openai,
    image_content_anthropic,
)
from loaders import DATASET_LOADERS


# ── Baseline system prompt ───────────────────────────────────────────────────
# Loaded dynamically per dataset from eval/prompts/<dataset>.json["baseline"].

BASELINE_SYSTEM = ""

def _load_prompts(dataset: str):
    global BASELINE_SYSTEM
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{dataset}.json"
    if not path.exists():
        path = prompts_dir / "geometry3k.json"
    _prompts = json.load(path.open())
    BASELINE_SYSTEM = _prompts.get("baseline", _prompts.get("direct", ""))


# ── Build batch request ──────────────────────────────────────────────────────

def _user_text(prob: dict) -> str:
    """Format question + choices for baseline prompt."""
    choices = prob.get("choices", [])
    if choices:
        return (f"{prob['question']}\n\n"
                f"Choices: {format_choices(choices)}\n"
                f"Answer with the letter (A, B, C, or D) of the correct choice.")
    else:
        return (f"{prob['question']}\n\n"
                f"This is an open-ended question. "
                f"Answer with a numeric value or expression (e.g. 12, 3.5, 2√3, 5π/6). "
                f"Do NOT answer with a letter even if the image shows options.")


def _build_request(prob: dict, model: str, thinking_level: str,
                   temperature: float):
    """Build a single Gemini Batch API InlinedRequest for one problem."""
    from google.genai import types

    user_text = _user_text(prob)

    parts = []
    for img in prob.get("images", [prob["image"]]):
        img_bytes, mime = load_image(img)
        parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)))
    parts.append(types.Part(text=user_text))

    gem_level = "NONE" if thinking_level.lower() == "off" else thinking_level.upper()
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=BASELINE_SYSTEM,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(
                thinking_level=gem_level,
            ),
        ),
    )


def _build_request_for_online_gemini(prob: dict, thinking_level: str,
                                     temperature: float):
    """Build google.genai types for online (non-batch) calls."""
    from google.genai import types

    user_text = _user_text(prob)

    # Support multi-image (SolidGeo choice images)
    parts = []
    img_list = prob.get("images", [prob["image"]])
    for img in img_list:
        img_bytes, mime = load_image(img)
        parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)))
    parts.append(types.Part(text=user_text))

    contents = [types.Content(role="user", parts=parts)]
    gem_level = "NONE" if thinking_level.lower() == "off" else thinking_level.upper()
    config = types.GenerateContentConfig(
        system_instruction=BASELINE_SYSTEM,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(
            thinking_level=gem_level,
        ),
    )
    return contents, config


def _build_messages_for_anthropic(prob: dict):
    """Build Anthropic-compatible messages for baseline call.
    Returns (messages, system_prompt)."""
    user_text = _user_text(prob)
    content = []
    for img in prob.get("images", [prob["image"]]):
        content.append(image_content_anthropic(img))
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    return messages, BASELINE_SYSTEM


def _build_messages_for_openai(prob: dict, temperature: float):
    """Build OpenAI-compatible messages for baseline call."""
    user_text = _user_text(prob)
    content = []
    img_list = prob.get("images", [prob["image"]] if prob.get("image") else [])
    for img in img_list:
        if img and Path(img).exists() and Path(img).is_file():
            content.append(image_content_openai(img))
    content.append({"type": "text", "text": user_text})
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": content},
    ]
    return messages


# ── Submit batch ─────────────────────────────────────────────────────────────

def submit_batch(client, problems: list[dict], model: str,
                 thinking_level: str, temperature: float) -> str:
    """Submit a batch job. Returns the batch job name."""
    requests = []
    for prob in problems:
        req = _build_request(prob, model, thinking_level, temperature)
        requests.append(req)

    print(f"  Submitting batch with {len(requests)} requests...")
    batch_job = client.batches.create(
        model=model,
        src=requests,
    )
    print(f"  Batch job created: {batch_job.name}")
    print(f"  State: {batch_job.state}")
    return batch_job.name


# ── Poll / collect results ───────────────────────────────────────────────────

_DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}


def poll_batch(client, job_name: str, interval: int = 30,
               timeout: int = 7200) -> object:
    """Poll until batch job completes. Returns the final job object."""
    start = time.time()
    while True:
        job = client.batches.get(name=job_name)
        state = str(job.state) if hasattr(job, "state") else str(job)
        # Normalize: state might be enum or string
        state_str = state.split(".")[-1] if "." in state else state
        print(f"  [{time.strftime('%H:%M:%S')}] {job_name} → {state_str}")
        if state_str in _DONE_STATES:
            return job
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"  [TIMEOUT] Batch not done after {timeout}s")
            return job
        time.sleep(interval)


def collect_results(job) -> list[str]:
    """Extract response texts from a completed batch job."""
    texts = []
    if hasattr(job, "dest") and job.dest:
        # Inline results are in job.dest
        dest = job.dest
        if isinstance(dest, list):
            for item in dest:
                # Each item should have response.candidates[0].content.parts
                text = _extract_text_from_response(item)
                texts.append(text)
        elif hasattr(dest, 'responses'):
            for resp in dest.responses:
                text = _extract_text_from_response(resp)
                texts.append(text)
    # Fallback: try to iterate the job object itself
    if not texts and hasattr(job, '__iter__'):
        for item in job:
            text = _extract_text_from_response(item)
            texts.append(text)
    return texts


def _extract_text_from_response(resp) -> str:
    """Extract text from various response object shapes."""
    # google.genai response object
    if hasattr(resp, 'candidates'):
        for cand in resp.candidates:
            parts = cand.content.parts if hasattr(cand, 'content') else []
            text_parts = []
            for p in parts:
                if hasattr(p, 'thought') and p.thought:
                    continue  # skip thinking parts
                if hasattr(p, 'text') and p.text:
                    text_parts.append(p.text)
            if text_parts:
                return "\n".join(text_parts)
    # Dict-style response
    if isinstance(resp, dict):
        # Try response.candidates path
        cands = resp.get("response", resp).get("candidates", [])
        for cand in cands:
            parts = cand.get("content", {}).get("parts", [])
            text_parts = [p["text"] for p in parts
                         if "text" in p and not p.get("thought")]
            if text_parts:
                return "\n".join(text_parts)
    return str(resp)


# ── Log writing ──────────────────────────────────────────────────────────────

def _write_log(prob_dir: Path, slug: str, prob: dict, result: dict,
               model: str, thinking_level: str, response_text: str):
    """Write a human-readable log for one problem."""
    log_path = prob_dir / f"{slug}_baseline_log.txt"
    lines = [
        "=" * 64,
        f"  Problem  : {prob['id']}",
        f"  Dataset  : {prob.get('dataset', '?')}",
        f"  Model    : {model}",
        f"  Mode     : baseline ({result.get('submission', '?')})",
        f"  Thinking : {thinking_level}",
        f"  Hint     : {prob.get('hint_mode', 'none')}",
        "=" * 64,
        "",
        "── Question ──",
        prob["question"],
        "",
        "Choices: " + "  ".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(prob.get("choices", []))
        ),
        f"Answer key: {prob['answer_label']}  (expected: {prob.get('expected')})",
        "",
        "── Response ──",
        response_text or "(no response)",
        "",
        "── Result ──",
        f"  Passed       : {result['passed']}",
        f"  Detail       : {result['detail']}",
        f"  Answer raw   : {result.get('answer_raw')}",
        "",
        "── Token Usage ──",
        f"  Input tokens   : {result.get('input_tokens', 0):,}",
        f"  Output tokens  : {result.get('output_tokens', 0):,}",
        f"  Thought tokens : {result.get('thought_tokens', 0):,}",
        f"  Total tokens   : {result.get('total_tokens', 0):,}",
        f"  Time (sec)     : {result.get('t_last_sec', 0):.1f}",
        "",
        "── Cost Estimate ──",
        f"  Input  : ${result.get('input_tokens', 0) * 0.5 / 1e6:.6f}",
        f"  Output : ${(result.get('output_tokens', 0) + result.get('thought_tokens', 0)) * 3.0 / 1e6:.6f}",
        f"  Total  : ${(result.get('input_tokens', 0) * 0.5 + (result.get('output_tokens', 0) + result.get('thought_tokens', 0)) * 3.0) / 1e6:.6f}",
        "=" * 64,
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


# ── Online mode (sequential or parallel) ─────────────────────────────────────

_print_lock = threading.Lock()


def _run_one_online(client, prob: dict, model: str,
                    thinking_level: str, temperature: float,
                    dataset_dir: Path, slug: str, ttft_timeout: int,
                    sdk_type: str = "google-genai", provider: str = "",
                    idx: int = 0, total: int = 0) -> dict:
    """Run a single problem via online API. Thread-safe. Supports google-genai and openai SDKs."""

    t0 = time.time()
    resp = None
    timed_out = False
    full_text = ""
    input_tokens = output_tokens = thought_tokens = 0

    try:
        if sdk_type == "google-genai":
            contents, config = _build_request_for_online_gemini(
                prob, thinking_level, temperature)
            config.http_options = {"timeout": ttft_timeout * 1000}  # ms (SDK divides by 1000)
            resp = client.models.generate_content(
                model=model, contents=contents, config=config)
        elif sdk_type == "anthropic":
            anthropic_msgs, anthropic_sys = _build_messages_for_anthropic(prob)
            anthropic_kwargs = dict(
                model=model, messages=anthropic_msgs,
                system=anthropic_sys, max_tokens=65536,
                timeout=float(ttft_timeout),
            )
            _effort = thinking_level.lower() if thinking_level else "high"
            if _effort != "off":
                eff = {"minimal": "low"}.get(_effort, _effort)
                # 4.5 models: manual thinking with budget_tokens
                # 4.6 models: adaptive thinking + output_config.effort
                if "4-5" in model:
                    anthropic_kwargs["thinking"] = {
                        "type": "enabled", "budget_tokens": 10000}
                else:
                    anthropic_kwargs["thinking"] = {"type": "adaptive"}
                # effort via output_config (Opus 4.6, Sonnet 4.6, Opus 4.5)
                if "sonnet-4-5" not in model and \
                        eff in ("low", "medium", "high", "max"):
                    anthropic_kwargs["output_config"] = {"effort": eff}
            with client.messages.stream(**anthropic_kwargs) as _stream:
                resp = _stream.get_final_message()
        else:
            messages = _build_messages_for_openai(prob, temperature)
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature,
                          timeout=float(ttft_timeout))
            if provider == "dashscope":
                enable = thinking_level.lower() != "off"
                kwargs["extra_body"] = {"enable_thinking": enable}
            elif provider.startswith("openai"):
                _is_reasoning = thinking_level.lower() not in ("off", "") and \
                    any(k in model for k in ("o1", "o3", "o4", "gpt-5", "5.1", "5.2", "5.3", "5.4"))
                if _is_reasoning:
                    effort = thinking_level or cfg.REASONING_EFFORT
                    # BL mode has no tools so reasoning_effort is safe for mini/nano
                    if effort and effort.lower() != "none":
                        kwargs["reasoning_effort"] = effort
                        kwargs.pop("temperature", None)
                    kwargs["max_completion_tokens"] = cfg.MAX_COMPLETION_TOKENS
            resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        exc_name = type(exc).__name__
        if "timeout" in exc_name.lower() or "timed" in str(exc).lower() \
                or "timeout" in str(exc).lower():
            timed_out = True
        else:
            with _print_lock:
                print(f"  [{idx}/{total}] {prob['id']}  [ERROR] {exc_name}: {exc}")

    elapsed = time.time() - t0

    # Extract text + tokens — SDK-specific
    if sdk_type == "google-genai":
        if resp and resp.candidates:
            text_parts = []
            for p in resp.candidates[0].content.parts:
                if getattr(p, "thought", False):
                    continue
                if p.text:
                    text_parts.append(p.text)
            full_text = "\n".join(text_parts)
            um = resp.usage_metadata
            if um:
                input_tokens = getattr(um, "prompt_token_count", 0) or 0
                output_tokens = getattr(um, "candidates_token_count", 0) or 0
                thought_tokens = getattr(um, "thoughts_token_count", 0) or 0
    elif sdk_type == "anthropic":
        # Anthropic response: resp.content = list of blocks (text / thinking)
        if resp and resp.content:
            text_parts = []
            for block in resp.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "thinking":
                    thought_tokens += len(getattr(block, "thinking", "") or "") // 4
            full_text = "\n".join(text_parts)
            if hasattr(resp, "usage") and resp.usage:
                input_tokens = getattr(resp.usage, "input_tokens", 0) or 0
                output_tokens = getattr(resp.usage, "output_tokens", 0) or 0
    else:
        # OpenAI-compatible response
        if resp and resp.choices:
            msg = resp.choices[0].message
            # Filter out thinking content (reasoning_content for some providers)
            full_text = getattr(msg, "content", "") or ""
            # Some providers put thinking in reasoning_content
            think_text = getattr(msg, "reasoning_content", "") or ""
            if hasattr(resp, "usage") and resp.usage:
                input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
                # Some providers report thinking tokens separately
                thought_tokens = getattr(resp.usage, "completion_tokens_details", None)
                if thought_tokens and hasattr(thought_tokens, "reasoning_tokens"):
                    thought_tokens = thought_tokens.reasoning_tokens or 0
                else:
                    thought_tokens = 0

    answer = parse_answer(full_text) if full_text else None

    # MathCanvas → GPT-4.1 judge; other datasets → validate
    if prob.get("dataset") == "mathcanvas" and answer is not None:
        from eval_common import judge_mathcanvas_answer
        pred_str = (json.dumps(answer, ensure_ascii=False)
                    if isinstance(answer, (list, dict)) else str(answer))
        judge = judge_mathcanvas_answer(
            question=prob.get("question", ""),
            gt_answer=prob.get("expected_raw", ""),
            pred_answer=pred_str,
        )
        passed = bool(judge.get("complete_score", 0))
        detail = (f"judge weighted={judge.get('weighted_score',0):.2f} "
                  f"complete={judge.get('complete_score',0)}")
    elif prob.get("dataset") == "mathcanvas":
        passed, detail, judge = False, "no answer emitted", {}
    else:
        passed, detail = validate(answer, prob)
        judge = None

    if timed_out:
        detail = f"TIMEOUT ({ttft_timeout}s)"

    tag = "[+]" if passed else "[-]"
    with _print_lock:
        print(f"  [{idx}/{total}] {prob['id']:10s}  {tag}  {detail}  "
              f"(in={input_tokens:,} out={output_tokens:,} "
              f"think={thought_tokens:,} t={elapsed:.1f}s)")

    result = _make_result(prob, answer, input_tokens, output_tokens,
                          thought_tokens, elapsed, "online",
                          model_id=model, thinking_level=thinking_level)
    result["passed"] = passed
    result["detail"] = detail
    if judge:
        result["judge"] = judge
    result["response_text"] = full_text
    result["timed_out"] = timed_out

    # Save per-problem result + log
    prob_dir = dataset_dir / prob["id"]
    prob_dir.mkdir(parents=True, exist_ok=True)
    (prob_dir / f"{slug}_baseline_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    _write_log(prob_dir, slug, prob, result,
               model, thinking_level, full_text)

    return result


def run_online(client, problems: list[dict], model: str,
               thinking_level: str, temperature: float,
               dataset_dir: Path, slug: str, ttft_timeout: int,
               workers: int = 1,
               sdk_type: str = "google-genai",
               provider: str = "") -> list[dict]:
    """Run problems via online API — sequential or parallel."""
    total = len(problems)
    common = dict(sdk_type=sdk_type, provider=provider)

    if workers <= 1:
        return [
            _run_one_online(client, prob, model, thinking_level, temperature,
                            dataset_dir, slug, ttft_timeout,
                            idx=i + 1, total=total, **common)
            for i, prob in enumerate(problems)
        ]

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one_online, client, prob, model,
                thinking_level, temperature,
                dataset_dir, slug, ttft_timeout,
                idx=i + 1, total=total, **common,
            ): prob
            for i, prob in enumerate(problems)
        }
        for fut in as_completed(futures):
            prob = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                with _print_lock:
                    print(f"  [FATAL] {prob['id']}: {e}")
    return results


# ── Result construction ──────────────────────────────────────────────────────

def _make_result(prob: dict, answer: dict | None,
                 input_tokens: int, output_tokens: int,
                 thought_tokens: int, elapsed: float,
                 submission_mode: str,
                 model_id: str = "", thinking_level: str = "") -> dict:
    """Build a result dict compatible with main eval format."""
    _EXTRA_FIELDS = ["knowledge_type", "book", "page", "expression",
                     "parsing_stru_seqs", "parsing_sem_seqs"]
    extra = {f: prob[f] for f in _EXTRA_FIELDS if f in prob and prob[f]}

    return {
        "id":           prob["id"],
        "question":     prob["question"],
        "choices":      prob.get("choices", []),
        "types":        (prob.get("problem_type_graph", [])
                         + prob.get("problem_type_goal", [])),
        "answer_label": prob["answer_label"],
        "expected":     prob["expected"],
        "answer_raw":   answer,
        "passed":       False,
        "detail":       "",
        "hint_mode":    prob.get("hint_mode", "none"),
        "mode":         "baseline",
        "model":        model_id,
        "thinking_level": thinking_level,
        "submission":   submission_mode,   # "batch" | "online"
        **extra,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens":  input_tokens + output_tokens + thought_tokens,
        "ggb_used":     False,
        "t_last_sec":   elapsed,
    }


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_batch_results(problems: list[dict], response_texts: list[str],
                        dataset_dir: Path, slug: str,
                        model: str = "", thinking_level: str = "") -> list[dict]:
    """Parse and score batch results. Save per-problem result + log files."""
    results = []
    for prob, text in zip(problems, response_texts):
        answer = parse_answer(text)

        # MathCanvas → GPT-4.1 judge; other datasets → validate
        if prob.get("dataset") == "mathcanvas" and answer is not None:
            from eval_common import judge_mathcanvas_answer
            pred_str = (json.dumps(answer, ensure_ascii=False)
                        if isinstance(answer, (list, dict)) else str(answer))
            judge = judge_mathcanvas_answer(
                question=prob.get("question", ""),
                gt_answer=prob.get("expected_raw", ""),
                pred_answer=pred_str,
            )
            passed = bool(judge.get("complete_score", 0))
            detail = (f"judge weighted={judge.get('weighted_score',0):.2f} "
                      f"complete={judge.get('complete_score',0)}")
        elif prob.get("dataset") == "mathcanvas":
            passed, detail, judge = False, "no answer emitted", {}
        else:
            passed, detail = validate(answer, prob)
            judge = None

        result = _make_result(prob, answer, 0, 0, 0, 0.0, "batch",
                              model_id=model, thinking_level=thinking_level)
        result["passed"] = passed
        result["detail"] = detail
        if judge:
            result["judge"] = judge
        result["response_text"] = text
        results.append(result)

        # Save per-problem result + log
        prob_dir = dataset_dir / prob["id"]
        prob_dir.mkdir(parents=True, exist_ok=True)
        (prob_dir / f"{slug}_baseline_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        _write_log(prob_dir, slug, prob, result, model, thinking_level, text)

    return results


def print_summary(results: list[dict], dataset: str, dataset_dir: Path,
                  slug: str):
    """Print and save summary."""
    n_pass = sum(1 for r in results if r.get("passed"))
    pct = n_pass / len(results) * 100 if results else 0

    print(f"\n{'='*64}")
    print(f"  BASELINE SUMMARY  {dataset}  ({len(results)} problems)")
    print(f"{'='*64}")
    for r in results:
        tag = "[+]" if r["passed"] else "[-]"
        typs = "+".join(r.get("types", []))
        print(f"  {tag} {r['id']:10s}  [{typs:28s}]  {r['detail']}")

    print(f"\n  Passed: {n_pass}/{len(results)} ({pct:.1f}%)")

    # Token stats
    total_in = sum(r.get("input_tokens", 0) for r in results)
    total_out = sum(r.get("output_tokens", 0) for r in results)
    total_think = sum(r.get("thought_tokens", 0) for r in results)
    print(f"  Tokens: in={total_in:,}  out={total_out:,}  think={total_think:,}")

    # Cost estimate (Gemini 3 Flash Preview pricing)
    cost_in = total_in * 0.5 / 1_000_000
    cost_out = (total_out + total_think) * 3.0 / 1_000_000
    batch_mult = 0.5  # batch = 50% discount
    cost_online = cost_in + cost_out
    cost_batch = cost_online * batch_mult
    print(f"  Est. cost: ${cost_online:.4f} (online) / ${cost_batch:.4f} (batch)")

    # Save summary
    ts = time.strftime("%Y%m%d_%H%M")
    summary_path = dataset_dir / f"summary_{slug}_baseline_{ts}.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Summary → {summary_path}")
    print("=" * 64 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline eval — single-turn QA (batch or online)",
    )
    parser.add_argument("--dataset", default=cfg.DEFAULT_DATASET,
                        choices=list(DATASET_LOADERS))
    parser.add_argument("--data_dir", default=cfg.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--sample", type=int, default=cfg.DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--id", default=None)
    parser.add_argument("--hint", default=cfg.DEFAULT_HINT,
                        choices=["none", "points", "logic_form",
                                 "parsing_sem", "parsing_stru"])
    parser.add_argument("--model", default=cfg.DEFAULT_MODEL)
    parser.add_argument("--thinking", default=cfg.THINKING_LEVEL,
                        choices=["off", "minimal", "low", "medium", "high"],
                        help=f"Thinking level; 'off' disables thinking for DashScope (default: {cfg.THINKING_LEVEL})")
    parser.add_argument("--temperature", type=float, default=cfg.TEMPERATURE)
    parser.add_argument("--ttft-timeout", type=int, default=cfg.TTFT_TIMEOUT,
                        help=f"Per-request timeout in seconds (default: {cfg.TTFT_TIMEOUT})")

    # PGPS9K
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--exclude-book", default=None)
    parser.add_argument("--subject", default=None,
                        help="MathVerse subject / SolidGeo problem_type filter")
    parser.add_argument("--level", default=None,
                        help="SolidGeo complexity level (e.g. 'Level 1')")
    parser.add_argument("--split", default=None,
                        help="OlympiadBench split (en_comp/zh_comp/zh_cee/all)")
    parser.add_argument("--skip-done", action="store_true")

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--online", action="store_true", default=True,
                            help="Run via online API (default; use --batch for batch mode)")
    mode_group.add_argument("--collect", default=None, metavar="JOB_NAME",
                            help="Collect results from a finished batch job")

    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for online mode (default: 1)")
    parser.add_argument("--wait", action="store_true",
                        help="After submitting batch, poll until done and score")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between batch status polls (default: 30)")
    parser.add_argument("--poll-timeout", type=int, default=7200,
                        help="Max seconds to wait for batch (default: 7200)")

    args = parser.parse_args()

    # Resolve model
    try:
        card = get_model(args.model)
    except KeyError as e:
        print(f"[ERROR] {e}"); return

    model_id = args.model
    model_name = card.model_name
    slug = model_id.replace("/", "-")

    # thinking_level priority: --thinking CLI (when non-default) > card.thinking_level > cfg default
    if args.thinking == cfg.THINKING_LEVEL and card.thinking_level:
        args.thinking = card.thinking_level

    # Build client
    try:
        client, _ = make_client(model_id)
    except Exception as exc:
        print(f"[ERROR] {exc}"); return

    # Output directory
    dataset_dir = EVAL_ROOT / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect mode ──
    if args.collect:
        # Load the problems manifest saved alongside the batch
        manifest_path = dataset_dir / f"{slug}_baseline_manifest.json"
        if not manifest_path.exists():
            print(f"[ERROR] Manifest not found: {manifest_path}")
            print("  The manifest is saved when you submit a batch.")
            return
        problems = json.loads(manifest_path.read_text())
        print(f"  Loaded {len(problems)} problems from manifest")

        job = poll_batch(client, args.collect, interval=args.poll_interval,
                         timeout=0)  # don't wait, just get current state
        results = collect_results(job)
        if len(results) != len(problems):
            print(f"  [WARN] Got {len(results)} results for {len(problems)} problems")

        scored = score_batch_results(problems, results, dataset_dir, slug,
                                     model=model_name,
                                     thinking_level=args.thinking)
        print_summary(scored, args.dataset, dataset_dir, slug)
        return

    # ── Load prompts for this dataset ──
    _load_prompts(args.dataset)

    # ── Load problems ──
    loader = DATASET_LOADERS[args.dataset]
    kwargs = {"data_dir": args.data_dir, "sample": args.sample,
              "seed": args.seed, "hint": args.hint}
    if args.id:
        id_val = args.id
        if id_val.endswith(".json"):
            id_path = Path(id_val) if Path(id_val).is_absolute() else Path(args.data_dir).parent / id_val
            if not id_path.exists():
                id_path = Path(__file__).parent / args.dataset / id_val
            with open(id_path) as f:
                id_list = json.load(f)
            id_val = ",".join(id_list)
        kwargs["problem_id"] = id_val
        # When IDs are explicitly specified, don't apply sample limit
        kwargs["sample"] = None
    if args.dataset == "pgps9k":
        if args.image_dir:
            kwargs["image_dir"] = args.image_dir
        if args.exclude_book:
            kwargs["exclude_book"] = args.exclude_book
    if args.dataset == "geolaux":
        kwargs["problem_type"] = "calculation"
    if args.dataset == "mathverse" and hasattr(args, 'subject') and args.subject:
        kwargs["subject"] = args.subject
    if args.dataset == "solidgeo":
        if hasattr(args, 'level') and args.level:
            kwargs["level"] = args.level
        if hasattr(args, 'subject') and args.subject:
            kwargs["problem_type"] = args.subject
    if args.dataset == "olympiadbench":
        kwargs["split"] = getattr(args, 'split', None) or "en_comp"
    problems = loader(**kwargs)

    # Skip done
    if args.skip_done:
        before = len(problems)
        problems = [p for p in problems
                    if not (dataset_dir / p["id"]
                            / f"{slug}_baseline_result.json").exists()]
        print(f"  Skip-done: {before - len(problems)} done, "
              f"{len(problems)} to run")

    if not problems:
        print("  No problems to run."); return

    print(f"\n{'='*64}")
    print(f"  Baseline Eval — {'Online' if args.online else 'Batch'}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Model    : {model_id} ({model_name})")
    print(f"  Problems : {len(problems)}")
    print(f"  Thinking : {args.thinking}")
    print(f"  Hint     : {args.hint}")
    if args.online and args.workers > 1:
        print(f"  Workers  : {args.workers}")
    print(f"{'='*64}")

    # ── Online mode ──
    if args.online:
        results = run_online(client, problems, model_name,
                             args.thinking, args.temperature,
                             dataset_dir, slug, args.ttft_timeout,
                             workers=args.workers,
                             sdk_type=card.sdk,
                             provider=card.provider)
        print_summary(results, args.dataset, dataset_dir, slug)
        return

    # ── Batch mode ──
    # Save manifest (problem metadata) so we can match results later
    manifest_path = dataset_dir / f"{slug}_baseline_manifest.json"
    manifest_path.write_text(json.dumps(problems, indent=2, ensure_ascii=False))
    print(f"  Manifest → {manifest_path}")

    job_name = submit_batch(client, problems, model_name,
                            args.thinking, args.temperature)

    # Save job name for later collection
    job_info = {
        "job_name": job_name,
        "model": model_name,
        "model_id": model_id,
        "dataset": args.dataset,
        "n_problems": len(problems),
        "thinking": args.thinking,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    job_path = dataset_dir / f"{slug}_baseline_batch_job.json"
    job_path.write_text(json.dumps(job_info, indent=2))
    print(f"  Job info → {job_path}")

    if not args.wait:
        print(f"\n  Batch submitted. To collect results later:")
        print(f"    python eval/eval_baseline.py "
              f"--dataset {args.dataset} --collect {job_name}")
        return

    # Wait for completion
    print(f"\n  Waiting for batch to complete (poll every {args.poll_interval}s)...")
    job = poll_batch(client, job_name,
                     interval=args.poll_interval,
                     timeout=args.poll_timeout)

    state_str = str(job.state).split(".")[-1]
    if state_str != "JOB_STATE_SUCCEEDED":
        print(f"  [ERROR] Batch ended with state: {state_str}")
        return

    response_texts = collect_results(job)
    if len(response_texts) != len(problems):
        print(f"  [WARN] {len(response_texts)} results for {len(problems)} problems")

    scored = score_batch_results(problems, response_texts, dataset_dir, slug,
                                 model=model_name,
                                 thinking_level=args.thinking)
    print_summary(scored, args.dataset, dataset_dir, slug)


if __name__ == "__main__":
    main()
