"""GeoGoal-SGVR runner — Engine Faithfulness empirical validation.

Pipeline per problem
--------------------
  1. CONSTRUCT : LLM realizes NL construction via GGB tools (Gemini)
  2. ANSWER    : LLM emits ordered T_i list via `ANSWER:` JSON line
  3. EXTRACT   : enumerate all named points on canvas, collect (x, y) coords
  4. VERIFY    : newclid-based predicate check against solution_FL,
                 producing premise/numcheck/derived SR + IR

Output layout
-------------
  eval/geogoal/
    {problem_id}/
      {slug}_construct.json    tool-call trace + tokens + wall time
      {slug}_canvas.json       {point_name: (x, y)} dict from final canvas
      {slug}_verify.json       full predicate-level verifier output
      {slug}_result.json       summary (SR, IR, FA match, tokens)
      {slug}_log.txt           console log
  summary_{slug}_{timestamp}.json

Usage
-----
  # Smoke 5 problems
  python eval/geogoal/eval_geogoal.py --sample 5

  # Specific problem(s)
  python eval/geogoal/eval_geogoal.py --id geogal_00000,geogal_00032

  # Specific model
  python eval/geogoal/eval_geogoal.py --model gemini-3-flash-preview@medium --sample 10

  # Incremental resume
  python eval/geogoal/eval_geogoal.py --skip-done --workers 2
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parent
REPO_ROOT = EVAL_ROOT.parent
for p in (THIS_DIR, EVAL_ROOT, REPO_ROOT):
    sys.path.insert(0, str(p))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import base64
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import (
    build_gemini_tools,
    CanvasTracker,
)
from symbolic.utils.model_registry import get_model, make_client
import eval_config as cfg

from loaders import load_geogoal_sgvr
from eval_common import parse_answer
import geogoal_verifier
import geogoal_tiparser

OUTPUT_ROOT = EVAL_ROOT / "geogoal"
MAX_TURNS = getattr(cfg, "MAX_TURNS", 30)

# ── Thread-local TeeLog ────────────────────────────────────────────────────
_thread_local = threading.local()
_real_stdout = sys.stdout


class _ThreadDispatchStdout:
    def __init__(self, orig):
        self._orig = orig
        self._lock = threading.Lock()

    def write(self, data):
        target = getattr(_thread_local, "target", None)
        if target is not None:
            target.write_both(data)
        else:
            with self._lock:
                self._orig.write(data)

    def flush(self):
        target = getattr(_thread_local, "target", None)
        if target is not None:
            target.flush_both()
        else:
            self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


class TeeLog:
    def __init__(self, path, mode="w"):
        self.path = Path(path)
        self._mode = mode
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(self._mode, buffering=1, encoding="utf-8")
        _thread_local.target = self
        return self

    def write_both(self, data):
        _real_stdout.write(data)
        self._fh.write(data)

    def flush_both(self):
        _real_stdout.flush()
        self._fh.flush()

    def __exit__(self, *_):
        _thread_local.target = None
        if self._fh:
            self._fh.close()


# ── Config ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str | None = None
MODEL_ID: str | None = None   # registry ID (keeps @level suffix) — for file naming + make_client
MODEL: str | None = None       # card.model_name — for API call bodies


def _load_prompt():
    global SYSTEM_PROMPT
    path = EVAL_ROOT / "prompts" / "geogoal_sgvr.json"
    SYSTEM_PROMPT = json.load(path.open())["construct"]


def _model_slug() -> str:
    """Disk-friendly slug from registry ID (preserves @level)."""
    return MODEL_ID.replace("/", "_").replace(":", "_")


# ── Lazy GeoGebra ──────────────────────────────────────────────────────────
class _LazyGGB:
    def __init__(self):
        self._ctx = self._ggb = None
        self._was_used = False

    def get(self):
        if self._ggb is None:
            print("  [GGB] Starting Selenium (2D)...", flush=True)
            self._ctx = GeoGebraAPI(mode="selenium", headless=True, enable_3d=False)
            self._ggb = self._ctx.__enter__()
            self._ggb.reset()
            self._was_used = True
        return self._ggb

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = self._ggb = None

    @property
    def used(self):
        return self._was_used


# ── Tool result printing ───────────────────────────────────────────────────
def _print_tool_result(fn_name: str, log_entry: dict):
    ok = log_entry["ok"]
    tag = "OK  " if ok else "FAIL"
    cmd = log_entry.get("cmd", "")
    res = log_entry.get("result", "")
    preview = f" → {str(res)[:60]}" if res else ""
    print(f"    [{tag}] {fn_name}({cmd[:40]}){preview}", flush=True)


# ── Canvas export (reuse test_construct conventions) ───────────────────────
# Import the stateless helpers from test_construct so rendering matches the
# main planar-geometry-solving runner.
from test_agentic_geo_constructer import (  # noqa: E402
    _prepare_canvas_export,
    _fix_underscore_labels,
)


def _save_turn_canvas(lazy, prob_dir, turn_num):
    """Save a canvas PNG after a turn completes. No-op if flag off."""
    if not getattr(cfg, "SAVE_PER_TURN", False):
        return
    if lazy._ggb is None or prob_dir is None:
        return
    try:
        ggb = lazy._ggb
        png_path = prob_dir / f"{_model_slug()}_canvas_turn{turn_num}.png"
        if _prepare_canvas_export(ggb):
            try:
                _fix_underscore_labels(ggb)
            except Exception:
                pass
            ggb.export_png(png_path)
            print(f"  [GGB] Turn {turn_num} canvas -> {png_path.name}", flush=True)
    except Exception as exc:
        print(f"  [GGB] Turn {turn_num} screenshot failed: {exc}", flush=True)


def _finalize_ggb(lazy, prob_dir):
    """Export final canvas PNG and close GeoGebra. Call in finally block."""
    if lazy._ggb is not None and prob_dir is not None:
        try:
            png_path = prob_dir / f"{_model_slug()}_canvas.png"
            if _prepare_canvas_export(lazy._ggb):
                try:
                    _fix_underscore_labels(lazy._ggb)
                except Exception:
                    pass
                lazy._ggb.export_png(png_path)
                print(f"  [GGB] Canvas saved -> {png_path}", flush=True)
            else:
                print("  [GGB] No geometric objects -- canvas skipped", flush=True)
        except Exception as exc:
            print(f"  [GGB] Screenshot failed: {exc}", flush=True)
    try:
        lazy.close()
    except Exception:
        pass


# ── Canvas coords extraction ───────────────────────────────────────────────
def extract_canvas_coords(ggb) -> dict[str, tuple[float, float]]:
    """Enumerate all named points on the canvas, return {name: (x, y)}."""
    st = ggb.get_construction_state()
    coords: dict[str, tuple[float, float]] = {}
    for info in st.get("objects", {}).values():
        if info.get("type") == "point" and "x" in info and "y" in info:
            name = info.get("name", "")
            if name:
                coords[name] = (float(info["x"]), float(info["y"]))
    return coords


# ── Gemini construct loop ──────────────────────────────────────────────────
def run_construct_gemini(client, prob, prob_dir, slug) -> dict:
    """Single-phase construct (no render, no judge). Break on ANSWER line."""
    from google.genai import types

    def _gemini_config(turn: int):
        fc_mode = types.FunctionCallingConfig(
            mode="ANY") if turn == 0 else types.FunctionCallingConfig(mode="AUTO")
        thinking = None
        if cfg.THINKING_LEVEL:
            lvl = "NONE" if cfg.THINKING_LEVEL.lower() == "off" \
                else cfg.THINKING_LEVEL.upper()
            thinking = types.ThinkingConfig(thinking_level=lvl)
        return types.GenerateContentConfig(
            tools=build_gemini_tools(include_render=False),
            tool_config=types.ToolConfig(function_calling_config=fc_mode),
            system_instruction=SYSTEM_PROMPT,
            temperature=cfg.TEMPERATURE,
            thinking_config=thinking,
        )

    # User turn: strip <image> placeholder (image is delivered as a separate Part)
    question_text = prob["question"].replace("<image>\n", "", 1).lstrip()
    img_bytes = Path(prob["image"]).read_bytes()
    print(f"  [INPUT] question {len(question_text)} chars, image {len(img_bytes)//1024} KB",
          flush=True)
    contents = [types.Content(role="user", parts=[
        types.Part(text=question_text),
        types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
    ])]

    lazy = _LazyGGB()
    tracker = CanvasTracker()
    tok_i = tok_o = tok_think = tok_t = 0
    t0 = time.perf_counter()
    process_log: list[dict] = []
    answer_text: str | None = None
    no_tool_streak = 0
    ttft = None

    try:
        for turn in range(MAX_TURNS):
            print(f"  [LLM] turn {turn + 1} — ", end="", flush=True)
            turn_entry = {"turn": turn + 1, "tool_calls": []}

            _ttft_limit = cfg.TTFT_TIMEOUT
            if _ttft_limit:
                pool = ThreadPoolExecutor(max_workers=1)
                fut = pool.submit(client.models.generate_content,
                                  model=MODEL, contents=contents,
                                  config=_gemini_config(turn))
                try:
                    resp = fut.result(timeout=_ttft_limit)
                except FuturesTimeout:
                    pool.shutdown(wait=False, cancel_futures=True)
                    print(f"\n  [TIMEOUT] turn {turn + 1} TTFT>{_ttft_limit}s")
                    raise TimeoutError(f"TTFT>{_ttft_limit}s")
                finally:
                    pool.shutdown(wait=False)
            else:
                resp = client.models.generate_content(
                    model=MODEL, contents=contents,
                    config=_gemini_config(turn))
            if ttft is None:
                ttft = time.perf_counter() - t0

            parts = (resp.candidates[0].content.parts or []) if resp.candidates else []
            for p in parts:
                if getattr(p, "text", None) and not getattr(p, "thought", False):
                    print(p.text[:120].strip(), end="", flush=True)
            print(flush=True)

            if u := getattr(resp, "usage_metadata", None):
                tok_i += int(getattr(u, "prompt_token_count", None) or 0)
                tok_o += int(getattr(u, "candidates_token_count", None) or 0)
                tok_think += int(getattr(u, "thoughts_token_count", None) or 0)
                tok_t += int(getattr(u, "total_token_count", None) or 0)

            history_parts = [p for p in parts
                             if not getattr(p, "thought", False)
                             or getattr(p, "function_call", None)]
            contents.append(types.Content(role="model", parts=history_parts or parts))

            text = " ".join(p.text for p in parts
                            if getattr(p, "text", None)
                            and not getattr(p, "thought", False))

            # CONSTRUCTION_DONE signal OR legacy ANSWER line — both terminate
            if "CONSTRUCTION_DONE" in text.upper() or "ANSWER:" in text:
                print("  [SIGNAL] construction complete")
                answer_text = text  # may still contain LLM-emitted answer for reference
                process_log.append(turn_entry)
                _save_turn_canvas(lazy, prob_dir, turn + 1)
                break

            fcs = [p.function_call for p in parts
                   if getattr(p, "function_call", None)]
            if not fcs:
                no_tool_streak += 1
                if no_tool_streak >= 2:
                    print(f"  [AUTO-STOP] {no_tool_streak} turns without tool calls")
                    process_log.append(turn_entry)
                    break
                process_log.append(turn_entry)
                continue
            else:
                no_tool_streak = 0

            tool_parts = []
            for fc in fcs:
                args = dict(fc.args)
                ggb = lazy.get()
                result, log_entry = tracker.execute(ggb, fc.name, args)
                _print_tool_result(fc.name, log_entry)
                turn_entry["tool_calls"].append(log_entry)
                tool_parts.append(types.Part(function_response=types.FunctionResponse(
                    name=fc.name, response=result)))
            contents.append(types.Content(role="user", parts=tool_parts))
            process_log.append(turn_entry)
            _save_turn_canvas(lazy, prob_dir, turn + 1)

    except TimeoutError as te:
        print(f"  [SKIP] {te}", flush=True)

    # Extract canvas coords before closing GGB
    canvas_coords: dict[str, tuple[float, float]] = {}
    if lazy.used:
        try:
            canvas_coords = extract_canvas_coords(lazy.get())
        except Exception as e:
            print(f"  [WARN] canvas extraction failed: {e}")

    # Final canvas PNG + GGB cleanup (one call)
    _finalize_ggb(lazy, prob_dir)
    tend = time.perf_counter()

    return {
        "answer_text": answer_text,
        "canvas_coords": canvas_coords,
        "tools_ok": tracker.ok_n, "tools_fail": tracker.fail_n,
        "tools_total": tracker.total_n,
        "process": {"total_turns": len(process_log), "turns": process_log},
        "metrics": {
            "input_tokens": tok_i, "output_tokens": tok_o,
            "thought_tokens": tok_think, "total_tokens": tok_t,
            "ggb_used": lazy.used,
            "t_ttft_sec": round(ttft, 3) if ttft else None,
            "t_total_sec": round(tend - t0, 3),
        },
    }


# ── Final answer list parsing & equivalence ────────────────────────────────
def _parse_value(s: str):
    """Fraction-aware numeric parser. Returns float or None."""
    from fractions import Fraction
    if s is None:
        return None
    s = str(s).strip()
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        try:
            return float(s)
        except ValueError:
            return None


def compare_answer_list(pred: list, gt: list, tol: float = 0.01) -> dict:
    """Element-wise numeric comparison, with length-mismatch tolerance.
    Returns {'match': int, 'total': int, 'per_idx': [bool, ...]}."""
    n = min(len(pred), len(gt))
    per_idx: list[bool] = []
    match = 0
    for i in range(n):
        pv = _parse_value(pred[i])
        gv = _parse_value(gt[i])
        if pv is None or gv is None:
            per_idx.append(False)
            continue
        ok = abs(pv - gv) < tol
        per_idx.append(ok)
        if ok:
            match += 1
    # Pad length mismatches with False
    while len(per_idx) < len(gt):
        per_idx.append(False)
    return {"match": match, "total": len(gt), "per_idx": per_idx,
            "pred_len": len(pred), "gt_len": len(gt)}


# ── Per-problem orchestration ──────────────────────────────────────────────
def run_problem(client, prob, prob_dir: Path, slug: str) -> dict:
    """Execute construct → verify for a single problem."""
    prob_dir.mkdir(parents=True, exist_ok=True)

    # 1. Construct phase
    c_result = run_construct_gemini(client, prob, prob_dir, slug)
    (prob_dir / f"{slug}_construct.json").write_text(
        json.dumps(c_result, indent=2, ensure_ascii=False, default=str))

    # 2. Canvas snapshot
    (prob_dir / f"{slug}_canvas.json").write_text(
        json.dumps({"coords": c_result["canvas_coords"]},
                   indent=2, ensure_ascii=False))

    gt_list = prob.get("expected") or []

    # 3a. LOCAL T_i evaluation (primary — Engine Faithfulness)
    fa_local: dict | None = None
    ti_preds: list[dict] = []
    if c_result["canvas_coords"]:
        try:
            ti_preds = geogoal_tiparser.evaluate_all(
                prob["question"], c_result["canvas_coords"])
            fa_local = geogoal_tiparser.compare_to_gt(ti_preds, gt_list)
        except Exception as e:
            fa_local = {"error": f"{type(e).__name__}: {e}",
                        "match": 0, "total": len(gt_list)}
    (prob_dir / f"{slug}_ti.json").write_text(
        json.dumps({"predictions": ti_preds, "compare": fa_local},
                   indent=2, ensure_ascii=False, default=str))

    # 3b. Legacy: parse LLM-emitted ANSWER if present (diagnostic, not scored)
    ans_parse = {"raw": c_result.get("answer_text") or "", "value": None,
                 "type": None}
    pred_list_llm: list = []
    if c_result["answer_text"]:
        ap = parse_answer(c_result["answer_text"])
        if ap:
            ans_parse.update(ap)
            v = ap.get("value")
            if isinstance(v, list):
                pred_list_llm = [str(x) for x in v]
    fa_llm = (compare_answer_list(pred_list_llm, gt_list)
              if pred_list_llm and gt_list else None)

    # 4. Predicate verification (engine-exact)
    verify_out: dict = {}
    if c_result["canvas_coords"]:
        try:
            verify_out = geogoal_verifier.verify(
                solution_fl=prob["solution_FL"],
                canvas_coords=c_result["canvas_coords"],
                include_details=True,
            )
        except Exception as e:
            verify_out = {"error": f"{type(e).__name__}: {e}"}
    else:
        verify_out = {"error": "no canvas coords"}
    (prob_dir / f"{slug}_verify.json").write_text(
        json.dumps(verify_out, indent=2, ensure_ascii=False, default=str))

    # 5. Summary — all scores consolidated into result.json
    summary = {
        "id": prob["id"],
        "model": MODEL,
        "FA_local": fa_local,                 # our T_i parser vs GT
        "FA_llm": fa_llm,                     # LLM-emitted list vs GT (diagnostic)
        "llm_answer": ans_parse,              # raw LLM emit (if any)
        "SR": verify_out.get("SR"),           # per-tier skeleton rates
        "IR": verify_out.get("IR"),           # integrity rate (bool)
        "total": verify_out.get("total"),
        "passed": verify_out.get("passed"),
        "missing_point": verify_out.get("missing_point"),
        "unsupported": verify_out.get("unsupported"),
        "canvas_points": sorted(c_result["canvas_coords"].keys()),
        "metrics": c_result["metrics"],
        "tools": {
            "total": c_result["tools_total"],
            "ok": c_result["tools_ok"],
            "fail": c_result["tools_fail"],
        },
    }
    (prob_dir / f"{slug}_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # Console summary
    ir_mark = "✓" if verify_out.get("IR") else "✗"
    sr = verify_out.get("SR", {})
    fa_str = f"FA_local {fa_local['match']}/{fa_local['total']}" \
             if fa_local and "match" in fa_local else "FA_local n/a"
    print(f"  [DONE] {prob['id']}  {fa_str}  "
          f"SR pre={sr.get('premise', 0):.2f} num={sr.get('numcheck', 0):.2f} "
          f"der={sr.get('derived', 0):.2f}  IR {ir_mark}", flush=True)
    return summary


# ── Driver ─────────────────────────────────────────────────────────────────
def _run_one(prob, skip_done: bool):
    """Worker entry for parallel execution."""
    slug = _model_slug()
    prob_dir = OUTPUT_ROOT / prob["id"]
    result_path = prob_dir / f"{slug}_result.json"
    log_path = prob_dir / f"{slug}_log.txt"

    if skip_done and result_path.exists():
        try:
            return json.loads(result_path.read_text())
        except Exception:
            pass

    with TeeLog(log_path):
        print(f"\n━━━━ {prob['id']} ━━━━", flush=True)
        t0 = time.perf_counter()
        try:
            client, _card = make_client(MODEL_ID)
            summary = run_problem(client, prob, prob_dir, slug)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            summary = {"id": prob["id"], "error": f"{type(e).__name__}: {e}"}
        summary["_wall_sec"] = round(time.perf_counter() - t0, 2)
        return summary


def main():
    global MODEL, MODEL_ID

    parser = argparse.ArgumentParser(
        description="GeoGoal-SGVR construct-then-verify runner")
    parser.add_argument("--data", default="/data/geogoal_sgvr", type=Path,
                        help="Dataset root (contains data/ and images/)")
    parser.add_argument("--sample", type=int, default=cfg.DEFAULT_SAMPLE,
                        help="Random sample size; None = all 256 test problems")
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--id", default=None,
                        help="Comma-separated problem IDs (e.g. geogal_00000,geogal_00032)")
    parser.add_argument("--model", default=cfg.DEFAULT_MODEL,
                        help="Model registry ID (see symbolic.utils.model_registry)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip problems with existing result.json")
    parser.add_argument("--ttft-timeout", type=int, default=None,
                        help="Override cfg.TTFT_TIMEOUT (seconds)")
    parser.add_argument("--thinking", default=None,
                        help="Override cfg.THINKING_LEVEL (off/minimal/low/medium/high)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (each spawns own GGB Selenium)")
    parser.add_argument("--save-screenshot-per-turn", action="store_true",
                        help="Save a canvas PNG after every turn")
    args = parser.parse_args()

    cfg.SAVE_PER_TURN = args.save_screenshot_per_turn

    if args.list_models:
        from symbolic.utils.model_registry import list_models
        list_models()
        return

    if args.ttft_timeout is not None:
        cfg.TTFT_TIMEOUT = args.ttft_timeout
    if args.thinking is not None:
        cfg.THINKING_LEVEL = args.thinking

    # Resolve model card; may override cfg.THINKING_LEVEL / cfg.TEMPERATURE
    card = get_model(args.model)
    MODEL_ID = args.model
    MODEL = card.model_name
    if getattr(card, "thinking_level", None):
        cfg.THINKING_LEVEL = card.thinking_level
    if getattr(card, "fixed_temperature", None) is not None:
        cfg.TEMPERATURE = card.fixed_temperature

    _load_prompt()

    # Thread-local stdout dispatch for per-problem TeeLog
    sys.stdout = _ThreadDispatchStdout(_real_stdout)

    problems = load_geogoal_sgvr(
        data_dir=Path(args.data), sample=args.sample,
        seed=args.seed, problem_id=args.id,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Startup banner (mirrors test_agentic_geo_constructer.py convention)
    print("\n" + "=" * 64)
    print(f"  GeoGoal-SGVR  —  Construct-then-Verify")
    print(f"  Dataset   : {args.data}  (test split, N={len(problems)})")
    print(f"  Model     : {MODEL_ID}  ({card.provider} / {card.sdk})")
    print(f"  API model : {MODEL}")
    print(f"  Thinking  : {cfg.THINKING_LEVEL}")
    print(f"  TTFT      : {cfg.TTFT_TIMEOUT}s")
    print(f"  Prompt    : {len(SYSTEM_PROMPT)} chars")
    # Tools disclosed: all non-render (LLM sees the full planar toolset)
    from symbolic.tools.geogebra_tools import build_gemini_tools as _bgt
    _ntools = len(_bgt(include_render=False)[0].function_declarations)
    print(f"  Tools     : {_ntools} (all non-render disclosed)")
    print(f"  Output    : {OUTPUT_ROOT}/")
    if args.skip_done:
        slug = _model_slug()
        done = sum(1 for p in problems
                   if (OUTPUT_ROOT / p["id"] / f"{slug}_result.json").exists())
        print(f"  Skip-done : {done} done, {len(problems) - done} to run")
    if args.workers > 1:
        print(f"  Workers   : {args.workers}")
    print("=" * 64, flush=True)

    results: list[dict] = []
    if args.workers <= 1:
        for p in problems:
            results.append(_run_one(p, args.skip_done))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_run_one, p, args.skip_done) for p in problems]
            for f in futs:
                results.append(f.result())

    # Write aggregate summary
    slug = _model_slug()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_ROOT / f"summary_{slug}_{ts}.json"

    # Aggregate SR / IR / FA_local across results (skip errored)
    ok_results = [r for r in results if "error" not in r]
    tot = {"premise": 0, "numcheck": 0, "derived": 0}
    pas = {"premise": 0, "numcheck": 0, "derived": 0}
    ir_count = 0
    fa_match = fa_total = 0
    for r in ok_results:
        t = r.get("total") or {}
        p = r.get("passed") or {}
        for k in tot:
            tot[k] += int(t.get(k, 0) or 0)
            pas[k] += int(p.get(k, 0) or 0)
        if r.get("IR"):
            ir_count += 1
        fa = r.get("FA_local") or {}
        fa_match += int(fa.get("match", 0) or 0)
        fa_total += int(fa.get("total", 0) or 0)

    agg = {
        "model": MODEL,
        "N_total": len(results),
        "N_ok": len(ok_results),
        "SR_aggregate": {
            k: (pas[k] / tot[k]) if tot[k] else None for k in tot
        } | {
            "overall": (sum(pas.values()) / sum(tot.values())) if sum(tot.values()) else None,
        },
        "IR_rate": ir_count / len(ok_results) if ok_results else None,
        "FA_micro": fa_match / fa_total if fa_total else None,
        "totals": tot,
        "passed": pas,
        "per_problem": [
            {"id": r.get("id"),
             "SR_overall": (r.get("SR") or {}).get("overall"),
             "IR": r.get("IR"),
             "FA_match": (r.get("FA_local") or {}).get("match"),
             "FA_total": (r.get("FA_local") or {}).get("total"),
             "tokens": (r.get("metrics") or {}).get("total_tokens"),
             "wall_sec": r.get("_wall_sec")}
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False, default=str))
    print(f"\n[geogoal] summary -> {summary_path}")
    print(f"  SR: {agg['SR_aggregate']}")
    print(f"  IR rate: {agg['IR_rate']}")
    print(f"  FA micro: {agg['FA_micro']}")


if __name__ == "__main__":
    main()
