"""LLM Image+Text Geometry QA — Benchmark Evaluation Script.

Tests the LLM on real geometry benchmark problems (image + text).

Pipeline
--------
  1. PERCEIVE  — LLM systematically reads every visual marking in the image
                 (point labels, tick marks, angle arcs, parallel arrows, etc.)
  2. CONSTRUCT — LLM rebuilds the figure in GeoGebra (makes reasoning
                 transparent and human-verifiable)
  3. QUERY     — LLM measures required quantity via query_* tools
  4. ANSWER    — ANSWER: {"value": <v>, "type": "numerical"}

Output layout
-------------
  eval/
    geometry3k/
      2101/
        log.txt        full conversation transcript
        result.json    pass/fail + metrics
      2102/ ...
      summary.json     cumulative results (all completed problems)

Usage
-----
  # List problems:
  python eval/test_agentic_geo_constructer.py \\
      --dataset geometry3k --data_dir /data/geometry3k/val --list 15

  # Run a random sample:
  python eval/test_agentic_geo_constructer.py \\
      --dataset geometry3k --data_dir /data/geometry3k/val --sample 10

  # Single problem:
  python eval/test_agentic_geo_constructer.py \\
      --dataset geometry3k --data_dir /data/geometry3k/val --id 2101

  # With logic-form hints:
  python eval/test_agentic_geo_constructer.py \\
      --dataset geometry3k --data_dir /data/geometry3k/val --sample 10 \\
      --hint logic_form

  # PGPS9K — list problems:
  python eval/test_agentic_geo_constructer.py \\
      --dataset pgps9k --data_dir /data/PGPS9K --list 10

  # PGPS9K — exclude Geometry3K overlap:
  python eval/test_agentic_geo_constructer.py \\
      --dataset pgps9k --data_dir /data/PGPS9K --exclude-book Geometry3K --list 10

  # PGPS9K — single problem (supports "prob_13" or "13"):
  python eval/test_agentic_geo_constructer.py \\
      --dataset pgps9k --data_dir /data/PGPS9K --id 13

  # PGPS9K — with structural parsing hints:
  python eval/test_agentic_geo_constructer.py \\
      --dataset pgps9k --data_dir /data/PGPS9K --sample 10 --hint parsing_stru

Geometry3K per-problem files:
  data.json              problem_text, choices, answer, precise_value
  logic_form.json        diagram_logic_form, point_positions (pixel coords)
  img_diagram.png        clean diagram
  img_diagram_point.png  diagram with labeled points  <- used by default
  symbols/               PASCAL VOC bounding-box annotations (not used here)
"""
import sys, os
# Make project root and eval/ importable when run from any directory
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from pathlib import Path

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import (
    build_gemini_tools,
    build_openai_tools,
    build_anthropic_tools,
    CanvasTracker,
)
from symbolic.tools.geogebra_tools_solid import (
    SOLID_GEOGEBRA_TOOLS, SOLID_QUERY_TOOLS, SOLID_RENDER_TOOLS,
    exec_with_solid_routing,
    build_gemini_solid_tools,
    build_openai_solid_tools,
    build_anthropic_solid_tools,
)
from symbolic.utils.model_registry import get_model, make_client, list_models
import eval_config as cfg


def _is_solid(prob: dict) -> bool:
    """Return True if this problem requires 3D GeoGebra.

    Checks multiple fields for compatibility across datasets:
    - MathVerse: metadata.subject = "Solid Geometry"
    - SolidGeo: metadata.subject = "Solid Geometry" (set by loader)
    - MathCanvas-Bench: knowledge = "Solid Geometry"
    """
    meta_subj = str(prob.get("metadata", {}).get("subject", ""))
    knowledge = str(prob.get("knowledge", ""))
    return "Solid" in meta_subj or "Solid" in knowledge

from eval_common import (
    format_choices,
    image_part_gemini,
    image_content_openai,
    image_content_anthropic,
    parse_answer,
    validate,
)
from loaders import DATASET_LOADERS

EVAL_ROOT = Path(__file__).parent   # eval/


# ── Thread-safe TeeLog ─────────────────────────────────────────────────────────
# Each thread stores its own TeeLog in _thread_local.  A single
# _ThreadDispatchStdout sits on sys.stdout and routes write/flush to the
# calling thread's TeeLog (if any) or to the real stdout.

_thread_local = threading.local()
_real_stdout  = sys.stdout          # captured before any replacement


class _ThreadDispatchStdout:
    """Route stdout writes to the calling thread's TeeLog or to the real stdout."""

    def __init__(self, orig):
        self._orig = orig
        self._lock = threading.Lock()

    def write(self, data):
        tee = getattr(_thread_local, "tee", None)
        with self._lock:
            if tee:
                tee.write_both(data)
            else:
                self._orig.write(data)

    def flush(self):
        tee = getattr(_thread_local, "tee", None)
        with self._lock:
            if tee:
                tee.flush_both()
            else:
                self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# Install once at import time — all existing print() calls work unchanged.
sys.stdout = _ThreadDispatchStdout(_real_stdout)


class TeeLog:
    """Context manager: tee stdout to a log file for the current thread."""

    def __init__(self, path):
        self._path = Path(path)
        self._file = None

    def __enter__(self):
        self._file = self._path.open("w", encoding="utf-8")
        _thread_local.tee = self
        return self

    def write_both(self, data):
        _real_stdout.write(data)
        self._file.write(data)

    def flush_both(self):
        _real_stdout.flush()
        self._file.flush()

    def __exit__(self, *_):
        _thread_local.tee = None
        self._file.close()
        _real_stdout.write(f"  [LOG] {self._path}\n")


# ── Model / config ────────────────────────────────────────────────────────────
# Defaults loaded from eval/eval_config.py — edit that file for persistent changes.
# CLI args override at runtime.  All globals below are set by main().

MODEL_ID  = cfg.DEFAULT_MODEL           # registry ID (for file naming)
MODEL     = cfg.DEFAULT_MODEL           # card.model_name (for API calls)
PROVIDER  = ""                          # card.provider (display + conditional logic)
SDK_TYPE  = "google-genai"              # card.sdk (query loop routing)
MAX_TURNS = cfg.MAX_TURNS


_HINT_SUFFIX = ""   # set by main() when --hint is non-default

def _model_slug() -> str:
    """Filename-safe model identifier (uses registry ID)."""
    base = MODEL_ID.replace("/", "-")
    return f"{base}_{_HINT_SUFFIX}" if _HINT_SUFFIX else base


def _prepare_canvas_export(ggb) -> bool:
    """Set labels visible for geometric objects before PNG export.

    Returns True if there are renderable geometric elements, False if canvas
    would be blank (only lists, numerics, functions, etc.).

    - points:              show NAME        (e.g. "A")
    - segments:            show NAME_VALUE  (e.g. "a = 12")
    - angles:              show NAME_VALUE  (e.g. "alpha = 45 deg")
    - polygons, circles:   hide label (shape itself still renders)
    - functions, lists:    hide (auxiliary)
    """
    # label style constants: 0=NAME, 1=NAME_VALUE, 2=VALUE, 3=CAPTION
    SHOW_NAME       = {"point"}
    SHOW_NAME_VALUE = {"segment", "angle"}
    GEOMETRIC       = {"point", "segment", "angle", "line", "ray", "vector",
                       "triangle", "quadrilateral", "polygon",
                       "circle", "conic", "arc"}
    ANALYTIC        = {"function", "inequality"}  # need axes
    has_geometry = False
    has_analytic = False
    for name in ggb.get_all_object_names():
        otype = (ggb.get_object_type(name) or "").lower()
        if otype in GEOMETRIC:
            has_geometry = True
        if otype in ANALYTIC:
            has_analytic = True
        if otype in SHOW_NAME:
            if getattr(cfg, "HIDE_POINT_LABELS", False):
                ggb.set_label_visible(name, False)    # keep dot, drop label
            else:
                ggb.set_label_visible(name, True)
                ggb.set_label_style(name, 0)          # NAME
        elif otype in SHOW_NAME_VALUE:
            ggb.set_label_visible(name, True)
            ggb.set_label_style(name, 1)          # NAME_VALUE
        else:
            ggb.set_label_visible(name, False)
    has_content = has_geometry or has_analytic
    if has_content:
        ggb.fit_view(padding=2.5)
        if has_analytic:
            # Functions/inequalities need axes and grid for readability
            ggb.set_axes_visible(True, True)
            ggb.set_grid_visible(True)
        else:
            # Pure geometry: clean canvas without axes
            ggb.set_grid_visible(False)
            ggb.set_axes_visible(False, False)
    return has_content

# Prompts live in eval/prompts/<dataset>.json — edit there, not here.
# _load_prompts() is called from main() after --dataset is known.
_PROMPTS: dict = {}
SYSTEM_INSTRUCTION           = ""
SYSTEM_INSTRUCTION_CONSTRUCT = ""

def _load_prompts(dataset: str):
    """Load prompt file for *dataset*, fallback to geometry3k.json."""
    global _PROMPTS, SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION_CONSTRUCT
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{dataset}.json"
    if not path.exists():
        path = prompts_dir / "geometry3k.json"
    _PROMPTS = json.load(path.open())
    SYSTEM_INSTRUCTION           = _PROMPTS.get("direct", _PROMPTS.get("construct", ""))
    SYSTEM_INSTRUCTION_CONSTRUCT = _PROMPTS.get("construct", "")


# ── Lazy GeoGebra ─────────────────────────────────────────────────────────────

class _LazyGGB:
    def __init__(self, enable_3d: bool = False):
        self._ctx = self._ggb = None
        self._was_used = False
        self._enable_3d = enable_3d

    def get(self):
        if self._ggb is None:
            tag = "3D" if self._enable_3d else "2D"
            print(f"  [GGB] Starting Selenium ({tag})...", flush=True)
            self._ctx = GeoGebraAPI(mode="selenium", headless=True,
                                    enable_3d=self._enable_3d)
            self._ggb = self._ctx.__enter__()
            self._ggb.reset()
            self._was_used = True
        return self._ggb

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = self._ggb = None
            # keep _was_used=True so _build_result can read it after close

    @property
    def used(self): return self._was_used


# ── Tool result printing ─────────────────────────────────────────────────────

def _print_tool_result(fn_name: str, log_entry: dict):
    """Print a tool execution result to stdout (captured by TeeLog)."""
    cmd = log_entry["cmd"]
    ok  = log_entry["ok"]
    tag = "OK  " if ok else "FAIL"
    if fn_name.startswith("query_"):
        val = log_entry.get("value")
        suffix = f"  -> {log_entry['error']}" if not ok else ""
        print(f"    [{tag}] {cmd}  value={val}{suffix}", flush=True)
        if ok:
            print(f"           canvas: {json.dumps(log_entry['canvas'])}")
    else:
        suffix = f"  -> {log_entry['error']}" if not ok else ""
        print(f"    [{tag}] {cmd}{suffix}", flush=True)
        if ok:
            if "new_objects" in log_entry:
                removed = log_entry.get("removed_objects", [])
                if removed:
                    print(f"           removed: {removed}")
                print(f"           new: {json.dumps(log_entry['new_objects'])}")
                print(f"           canvas: {json.dumps(log_entry['canvas'])}")
            elif "applied" in log_entry:
                # display/set_value tools — no new objects, just confirmation
                pass


# ── Question formatting ──────────────────────────────────────────────────────

def _build_question(prob: dict, mode: str) -> str:
    """Append choices and mode-specific instructions to the question text."""
    import re as _re
    question = prob["question"]
    if prob.get("choices"):
        opts = format_choices(prob["choices"])
        # Skip duplicate "Choices:" prefix when the question already lists
        # options inline (e.g. "A. 6\nB. 7\nC. 8\nD. 9" or "(A) yes (B) no").
        already_inline = bool(_re.search(
            r'(?:^|\n)\s*[A-E][.\s]+\S|\([A-E]\)\s*\S', question))
        if mode == "construct":
            instr = (" Construct the figure, measure the required quantity,"
                     " then identify which choice matches."
                     " Answer with the choice letter (A/B/C/D) or its numeric value.")
            question += (instr if already_inline
                         else f"\nChoices: {opts}\n{instr.lstrip()}")
        else:
            question += (" Answer with the numeric value." if already_inline
                         else f"\nChoices: {opts}\nAnswer with the numeric value.")
    return question


# ── Per-turn canvas snapshot (opt-in via --save-screenshot-per-turn) ─────────────────────

def _fix_underscore_labels(ggb):
    """Fix GeoGebra subscript rendering for names with underscores.

    GeoGebra only subscripts the single char after '_', so 'A_ref'
    renders as 'A_r ef'.  This wraps the suffix in braces via caption:
    'A_ref' → caption 'A_{ref}' so it renders as 'A_{ref}' properly.
    Auxiliary objects (names with _) get smaller gray points but stay visible.
    """
    import re
    for name in (ggb.get_all_object_names() or []):
        if "_" not in name:
            continue
        otype = (ggb.get_object_type(name) or "").lower()
        # Angles/numerics: keep VALUE mode (shows "105°"), skip caption override
        if otype in ("angle", "numeric"):
            continue
        # Other objects with _: fix subscript via caption
        parts = name.split("_")
        if len(parts) >= 2:
            caption = parts[0]
            for p in parts[1:]:
                caption += "_{" + p + "}"
            try:
                ggb._execute_js(
                    f'ggbApplet.setCaption("{name}","{caption}")')
                ggb.set_label_style(name, 3)  # CAPTION mode
            except Exception:
                pass
        # Gray out auxiliary points (keep visible)
        if otype == "point":
            try:
                ggb.set_color(name, 140, 140, 140)
                ggb._execute_js(f'ggbApplet.setPointSize("{name}", 3)')
            except Exception:
                pass


def _save_turn_canvas(lazy, prob_dir, turn_num):
    """Save a canvas PNG after a turn completes.  No-op if flag is off."""
    if not getattr(cfg, "SAVE_PER_TURN", False):
        return
    if lazy._ggb is None or prob_dir is None:
        return
    try:
        ggb = lazy._ggb
        png_path = prob_dir / f"{_model_slug()}_canvas_turn{turn_num}.png"
        if _prepare_canvas_export(ggb):
            _fix_underscore_labels(ggb)
            ggb.export_png(png_path)
            print(f"  [GGB] Turn {turn_num} canvas -> {png_path.name}",
                  flush=True)
    except Exception as exc:
        print(f"  [GGB] Turn {turn_num} screenshot failed: {exc}",
              flush=True)


# ── Canvas export + cleanup (shared finally block) ────────────────────────────

def _finalize_ggb(lazy, prob_dir):
    """Export canvas PNG and close GeoGebra.  Call in finally block."""
    if lazy._ggb is not None and prob_dir is not None:
        try:
            png_path = prob_dir / f"{_model_slug()}_canvas.png"
            if _prepare_canvas_export(lazy._ggb):
                try:
                    _fix_underscore_labels(lazy._ggb)
                except Exception:
                    pass  # label fix is cosmetic, don't let it break export
                lazy._ggb.export_png(png_path)
                print(f"  [GGB] Canvas saved -> {png_path}", flush=True)
            else:
                print("  [GGB] No geometric objects -- canvas skipped", flush=True)
        except Exception as exc:
            print(f"  [GGB] Screenshot failed: {exc}", flush=True)
    try:
        lazy.close()
    except Exception:
        pass  # ensure Chrome is killed even if close() fails


# ── Build result dict (shared across all query loops) ────────────────────────

def _build_result(answer_raw, prob, tracker, process_log, lazy,
                  tok_i, tok_o, tok_think, tok_t, t0, t1, t_final):
    tend = time.perf_counter()

    # MathCanvas: skip validate — judge runs separately after _build_result
    if prob.get("dataset") == "mathcanvas":
        ok_flag, detail = None, "pending judge"
    else:
        ok_flag, detail = validate(answer_raw, prob)

    return {
        "answer": answer_raw, "passed": ok_flag, "detail": detail,
        "tools_ok": tracker.ok_n, "tools_fail": tracker.fail_n,
        "tools_total": tracker.total_n,
        "process": {"total_turns": len(process_log), "turns": process_log},
        "metrics": {
            "input_tokens": tok_i, "output_tokens": tok_o,
            "thought_tokens": tok_think, "total_tokens": tok_t,
            "ggb_used": lazy.used,
            "ttft_sec":    round(t1     - t0, 3) if t1     else None,
            "t_final_sec": round(t_final - t0, 3) if t_final else None,
            "t_last_sec":  round(tend    - t0, 3),
        },
    }


# ── OpenAI Responses API adapter (for mini models: reasoning + tools) ────────

# Global flag set from card.use_responses_api in main()
USE_RESPONSES_API = False

def _use_responses_api() -> bool:
    """Return True if this model needs the Responses API for reasoning + tools."""
    return USE_RESPONSES_API


def _convert_tools_for_responses(tools_def: list) -> list:
    """Chat Completions tool format → Responses API tool format."""
    out = []
    for t in tools_def:
        if t.get("type") == "function":
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
    return out


def _messages_to_responses_input(messages: list) -> list:
    """Convert Chat Completions messages list to Responses API input items."""
    items = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            items.append({"role": "developer", "content": msg["content"]})
        elif role == "user":
            content = msg["content"]
            if isinstance(content, list):
                parts = []
                for c in content:
                    if c.get("type") == "text":
                        parts.append({"type": "input_text", "text": c["text"]})
                    elif c.get("type") == "image_url":
                        parts.append({"type": "input_image",
                                      "image_url": c["image_url"]["url"]})
                items.append({"role": "user", "content": parts})
            else:
                items.append({"role": "user", "content": content})
        elif role == "assistant":
            # Re-emit function_call items from the assistant turn
            tcs = msg.get("tool_calls", [])
            for tc in tcs:
                items.append({
                    "type": "function_call",
                    "id": tc["id"],
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
            if msg.get("content"):
                items.append({"role": "assistant", "content": msg["content"]})
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg["content"],
            })
    return items


class _FakeChoice:
    """Mimic Chat Completions Choice for Responses API output."""
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason

class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls

class _FakeTC:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = type('F', (), {'name': name, 'arguments': arguments})()

class _FakeUsage:
    def __init__(self, usage):
        self.prompt_tokens = getattr(usage, 'input_tokens', 0)
        self.completion_tokens = getattr(usage, 'output_tokens', 0)
        self.total_tokens = getattr(usage, 'total_tokens', 0)
        rt = 0
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            rt = getattr(usage.output_tokens_details, 'reasoning_tokens', 0) or 0
        self.completion_tokens_details = type('D', (), {'reasoning_tokens': rt})()

class _FakeResponse:
    def __init__(self, resp):
        text_parts = []
        tool_calls = []
        for item in resp.output:
            if getattr(item, 'type', '') == 'function_call':
                tool_calls.append(_FakeTC(
                    id=getattr(item, 'call_id', getattr(item, 'id', '')),
                    name=item.name,
                    arguments=item.arguments,
                ))
            elif getattr(item, 'type', '') == 'message':
                for c in getattr(item, 'content', []):
                    if getattr(c, 'type', '') == 'output_text':
                        text_parts.append(c.text)
        content = "\n".join(text_parts) if text_parts else None
        fr = "tool_calls" if tool_calls else "stop"
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls or None), fr)]
        self.usage = _FakeUsage(resp.usage) if resp.usage else None


_last_response_id = None  # tracks previous_response_id across turns

def _call_responses_api(client, model, messages, tools_def, effort,
                        tool_results=None, **extra):
    """Call OpenAI Responses API and return a Chat-Completions-compatible object.

    Uses previous_response_id for multi-turn to avoid resending full history,
    saving significant input tokens.
    """
    global _last_response_id
    tools_resp = _convert_tools_for_responses(tools_def)

    kwargs = dict(model=model, tools=tools_resp)
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    if _last_response_id and tool_results:
        # Multi-turn: reference previous response + send only new tool outputs
        kwargs["previous_response_id"] = _last_response_id
        kwargs["input"] = tool_results  # only the new function_call_output items
    else:
        # First turn: send full input
        kwargs["input"] = _messages_to_responses_input(messages)

    if extra.get("timeout"):
        kwargs["timeout"] = extra["timeout"]
    resp = client.responses.create(**kwargs)
    _last_response_id = resp.id  # save for next turn
    return _FakeResponse(resp)


def _reset_responses_state():
    """Reset multi-turn state between problems."""
    global _last_response_id
    _last_response_id = None


# ── OpenAI-compatible query loop ─────────────────────────────────────────────

def run_query_openai(client, prob: dict, mode: str = "direct",
                     prob_dir: Path | None = None) -> dict:
    """
    Multi-turn agentic loop via OpenAI Chat Completions API.
    Works with DashScope Qwen3-VL (vision) and OpenAI o-series models.

    Thinking content (reasoning_content) is stripped from history — same
    principle as Gemini thought-parts filtering.
    Uses Responses API (v1/responses) when card.use_responses_api=True,
    with previous_response_id for multi-turn token savings.
    """
    import json as _json

    is_3d = _is_solid(prob)
    question = _build_question(prob, mode)
    sys_instr = SYSTEM_INSTRUCTION_CONSTRUCT if mode == "construct" else SYSTEM_INSTRUCTION
    tools_def = build_openai_tools()
    _resp_tool_results = None  # Responses API: tool results for next turn
    if _use_responses_api():
        _reset_responses_state()
    if is_3d:
        tools_def = tools_def + build_openai_solid_tools()

    # Build user content — support 0~N images (pure text OK)
    user_content = []
    img_list = prob.get("images", [prob["image"]] if prob.get("image") else [])
    for img in img_list:
        if img and Path(img).exists() and Path(img).is_file():
            user_content.append(image_content_openai(img))
    user_content.append({"type": "text", "text": question})
    messages = [
        {"role": "system", "content": sys_instr},
        {"role": "user",   "content": user_content},
    ]

    lazy        = _LazyGGB(enable_3d=is_3d)
    answer_raw  = None
    tracker     = CanvasTracker()
    tok_i = tok_o = tok_think = tok_t = 0
    t0 = time.perf_counter(); t1 = t_final = None
    process_log: list[dict] = []

    try:
        for turn in range(MAX_TURNS):
            print(f"  [LLM] turn {turn+1} — ", end="", flush=True)
            turn_entry: dict = {"turn": turn + 1, "tool_calls": [],
                                "answer_skipped": False, "answer_raw": None}

            # construct mode turn 0: force tool use (some models only support auto/none)
            tc_mode = ("required" if mode == "construct" and turn == 0
                       and PROVIDER not in ("moonshot",) else "auto")
            kwargs = dict(
                model=MODEL,
                messages=messages,
                tools=tools_def,
                tool_choice=tc_mode,
                temperature=cfg.TEMPERATURE,
            )

            # Zhipu GLM-5: thinking {"type": "enabled"/"disabled"}
            if PROVIDER == "zhipu":
                enable = cfg.THINKING_LEVEL.lower() != "off" if cfg.THINKING_LEVEL else True
                kwargs["extra_body"] = {"thinking": {"type": "enabled" if enable else "disabled"}}
            # DashScope Qwen: thinking_level "off" -> disable thinking
            elif PROVIDER == "dashscope":
                enable = cfg.THINKING_LEVEL.lower() != "off" if cfg.THINKING_LEVEL else True
                kwargs["extra_body"] = {"enable_thinking": enable}
            # OpenAI reasoning models: reasoning_effort + max_completion_tokens
            elif PROVIDER.startswith("openai"):
                effort = cfg.THINKING_LEVEL or cfg.REASONING_EFFORT
                if effort and effort.lower() in ("none", "off", ""):
                    effort = None
                if _use_responses_api() and effort:
                    # mini/nano: use Responses API for reasoning + tools
                    pass  # handled below
                else:
                    if effort:
                        kwargs["reasoning_effort"] = effort
                        kwargs.pop("temperature", None)
                    kwargs["max_completion_tokens"] = cfg.MAX_COMPLETION_TOKENS

            if _use_responses_api() and effort:
                _ttft = cfg.TTFT_TIMEOUT or 120
                resp = _call_responses_api(client, MODEL, messages, tools_def, effort,
                                           tool_results=_resp_tool_results,
                                           timeout=float(_ttft))
                _resp_tool_results = None  # consumed
            else:
                resp = client.chat.completions.create(**kwargs)
            if t1 is None: t1 = time.perf_counter()

            if resp.usage:
                tok_i     += resp.usage.prompt_tokens     or 0
                tok_o     += resp.usage.completion_tokens or 0
                tok_think += getattr(getattr(resp.usage, "completion_tokens_details", None),
                                    "reasoning_tokens", 0) or 0
                tok_t     += resp.usage.total_tokens      or 0

            choice = resp.choices[0]
            msg = choice.message
            text = msg.content or ""
            tool_calls = msg.tool_calls or []
            print(text[:120].strip(), flush=True)

            # Strip reasoning_content before appending to history
            asst_msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                asst_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            messages.append(asst_msg)

            # ── Parse answer ────────────────────────────────────────────────
            if ans := parse_answer(text):
                # In construct mode with no prior tool usage, only skip the
                # answer if there are also tool calls pending (model is about
                # to construct).  If no tool calls, the model chose a direct
                # answer — accept it as a simple-problem fallback.
                if mode == "construct" and tracker.total_n == 0 and tool_calls:
                    print("  [SKIP ANSWER] construct mode — tool calls pending, deferring answer")
                    turn_entry["answer_skipped"] = True
                else:
                    answer_raw = ans; t_final = time.perf_counter()
                    turn_entry["answer_raw"] = ans
                    print(f"  [ANSWER] {json.dumps(ans)}")
                    process_log.append(turn_entry); break

            # ── Execute tools ───────────────────────────────────────────────
            if not tool_calls:
                t_final = time.perf_counter()
                process_log.append(turn_entry); break

            tool_results = []
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = _json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                ggb = lazy.get()

                if is_3d:
                    result, log_entry = exec_with_solid_routing(
                        tracker, ggb, fn_name, args)
                else:
                    result, log_entry = tracker.execute(ggb, fn_name, args)
                _print_tool_result(fn_name, log_entry)
                turn_entry["tool_calls"].append(log_entry)

                tool_results.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(result, ensure_ascii=False)})
            messages.extend(tool_results)
            # Prepare Responses API multi-turn: convert tool results to function_call_output
            if _use_responses_api():
                _resp_tool_results = [
                    {"type": "function_call_output",
                     "call_id": tr["tool_call_id"],
                     "output": tr["content"]}
                    for tr in tool_results
                ]
            process_log.append(turn_entry)
            _save_turn_canvas(lazy, prob_dir, turn + 1)

    finally:
        _finalize_ggb(lazy, prob_dir)

    return _build_result(answer_raw, prob, tracker, process_log, lazy,
                         tok_i, tok_o, tok_think, tok_t, t0, t1, t_final)


# ── Anthropic query loop ─────────────────────────────────────────────────────

def run_query_anthropic(client, prob: dict, mode: str = "direct",
                        prob_dir: Path | None = None) -> dict:
    """
    Multi-turn agentic loop via Anthropic Messages API.
    Supports adaptive thinking + tool use.
    """
    is_3d = _is_solid(prob)
    question = _build_question(prob, mode)
    sys_instr = SYSTEM_INSTRUCTION_CONSTRUCT if mode == "construct" else SYSTEM_INSTRUCTION
    tools_def = build_anthropic_tools()
    if is_3d:
        tools_def = tools_def + build_anthropic_solid_tools()

    # Build user content — support 0~N images (pure text OK)
    user_content = []
    img_list = prob.get("images", [prob["image"]] if prob.get("image") else [])
    for img in img_list:
        if img and Path(img).exists() and Path(img).is_file():
            user_content.append(image_content_anthropic(img))
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]

    lazy        = _LazyGGB(enable_3d=is_3d)
    answer_raw  = None
    tracker     = CanvasTracker()
    tok_i = tok_o = tok_think = tok_t = 0
    t0 = time.perf_counter(); t1 = t_final = None
    process_log: list[dict] = []

    try:
        for turn in range(MAX_TURNS):
            print(f"  [LLM] turn {turn+1} — ", end="", flush=True)
            turn_entry: dict = {"turn": turn + 1, "tool_calls": [],
                                "answer_skipped": False, "answer_raw": None}

            # Build API kwargs
            kwargs = dict(
                model=MODEL,
                system=sys_instr,
                messages=messages,
                tools=tools_def,
                max_tokens=65536,
                cache_control={"type": "ephemeral"},
            )
            # Thinking: 4.5 → enabled+budget; 4.6+ → adaptive + output_config.effort
            _tl = cfg.THINKING_LEVEL.lower() if cfg.THINKING_LEVEL else "high"
            if _tl != "off":
                eff = {"minimal": "low"}.get(_tl, _tl)
                if "4-5" in MODEL:
                    kwargs["thinking"] = {
                        "type": "enabled", "budget_tokens": 10000}
                else:
                    kwargs["thinking"] = {"type": "adaptive"}
                # effort via output_config (Opus 4.6, Sonnet 4.6, Opus 4.5)
                if "sonnet-4-5" not in MODEL and \
                        eff in ("low", "medium", "high", "max"):
                    kwargs["output_config"] = {"effort": eff}

            # construct mode turn 0: force tool use (only when thinking is off;
            # Anthropic forbids tool_choice=any with thinking enabled)
            if mode == "construct" and turn == 0 and _tl == "off":
                kwargs["tool_choice"] = {"type": "any"}
            else:
                kwargs["tool_choice"] = {"type": "auto"}

            with client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
            if t1 is None: t1 = time.perf_counter()

            # Token accounting
            if resp.usage:
                tok_i += getattr(resp.usage, "input_tokens", 0) or 0
                tok_o += getattr(resp.usage, "output_tokens", 0) or 0
                tok_t += tok_i + tok_o

            # Parse response blocks
            text = ""
            tool_uses = []
            for block in resp.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text += block.text
                elif btype == "thinking":
                    tok_think += len(getattr(block, "thinking", "") or "") // 4  # rough estimate
                elif btype == "tool_use":
                    tool_uses.append(block)

            if text:
                print(text[:120].strip(), flush=True)

            # ── Parse answer ──────────────────────────────────────────────
            if ans := parse_answer(text):
                # In construct mode with no prior tool usage, only skip the
                # answer if there are also tool uses pending (model is about
                # to construct).  If no tool uses, the model chose a direct
                # answer — accept it as a simple-problem fallback.
                if mode == "construct" and tracker.total_n == 0 and tool_uses:
                    print("  [SKIP ANSWER] construct mode — tool calls pending, deferring answer")
                    turn_entry["answer_skipped"] = True
                else:
                    answer_raw = ans; t_final = time.perf_counter()
                    turn_entry["answer_raw"] = ans
                    print(f"  [ANSWER] {json.dumps(ans)}")
                    process_log.append(turn_entry); break

            # ── Execute tools ─────────────────────────────────────────────
            if not tool_uses:
                t_final = time.perf_counter()
                process_log.append(turn_entry); break

            tool_results = []
            for tu in tool_uses:
                fn_name = tu.name
                args = tu.input if isinstance(tu.input, dict) else {}
                ggb = lazy.get()

                if is_3d:
                    result, log_entry = exec_with_solid_routing(
                        tracker, ggb, fn_name, args)
                else:
                    result, log_entry = tracker.execute(ggb, fn_name, args)
                _print_tool_result(fn_name, log_entry)
                turn_entry["tool_calls"].append(log_entry)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Append assistant turn (full content) + tool results as user turn
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_results})
            process_log.append(turn_entry)
            _save_turn_canvas(lazy, prob_dir, turn + 1)

    finally:
        _finalize_ggb(lazy, prob_dir)

    return _build_result(answer_raw, prob, tracker, process_log, lazy,
                         tok_i, tok_o, tok_think, tok_t, t0, t1, t_final)


# ── Gemini query loop ────────────────────────────────────────────────────────

def run_query_gemini(client, prob: dict, mode: str = "direct",
                     prob_dir: Path | None = None) -> dict:
    """
    mode="direct"    — show MC choices; allow direct reasoning (default)
    mode="construct" — hide choices; require GeoGebra construction
    """
    from google.genai import types

    if not Path(prob["image"]).exists():
        return _empty("image missing: " + prob["image"])

    question = _build_question(prob, mode)
    sys_instr = (SYSTEM_INSTRUCTION_CONSTRUCT if mode == "construct"
                 else SYSTEM_INSTRUCTION)
    is_3d = _is_solid(prob)

    # construct mode turn 0: force tool use, then auto
    def _gemini_config(turn: int):
        if mode == "construct" and turn == 0:
            fc_mode = types.FunctionCallingConfig(mode="ANY")
        else:
            fc_mode = types.FunctionCallingConfig(mode="AUTO")
        # Gemini 3+ thinking level control; "off" -> NONE
        thinking = None
        if cfg.THINKING_LEVEL:
            gem_level = "NONE" if cfg.THINKING_LEVEL.lower() == "off" else cfg.THINKING_LEVEL.upper()
            thinking = types.ThinkingConfig(thinking_level=gem_level)
        # Merge 2D + 3D tools for solid problems
        tools_2d = build_gemini_tools()
        if is_3d:
            tools_3d = build_gemini_solid_tools()
            all_decls = (tools_2d[0].function_declarations
                         + tools_3d[0].function_declarations)
            merged = [types.Tool(function_declarations=all_decls)]
        else:
            merged = tools_2d
        return types.GenerateContentConfig(
            tools=merged,
            tool_config=types.ToolConfig(function_calling_config=fc_mode),
            system_instruction=sys_instr,
            temperature=cfg.TEMPERATURE,
            thinking_config=thinking,
        )

    # Build user content — support multi-image (SolidGeo choice images)
    user_parts = []
    img_list = prob.get("images", [prob["image"]])
    for img in img_list:
        if Path(img).exists():
            user_parts.append(image_part_gemini(img))
    user_parts.append(types.Part(text=question))
    contents = [types.Content(role="user", parts=user_parts)]

    lazy        = _LazyGGB(enable_3d=is_3d)
    answer_raw  = None
    tracker     = CanvasTracker()
    tok_i = tok_o = tok_think = tok_t = 0
    t0 = time.perf_counter(); t1 = t_final = None
    process_log: list[dict] = []

    try:
        for turn in range(MAX_TURNS):
            print(f"  [LLM] turn {turn+1} — ", end="", flush=True)
            turn_entry: dict = {"turn": turn + 1, "tool_calls": [],
                                "answer_skipped": False, "answer_raw": None}
            # ── TTFT timeout guard ──
            _ttft_limit = cfg.TTFT_TIMEOUT
            if _ttft_limit:
                _pool = ThreadPoolExecutor(max_workers=1)
                _fut = _pool.submit(client.models.generate_content,
                                    model=MODEL, contents=contents,
                                    config=_gemini_config(turn))
                try:
                    resp = _fut.result(timeout=_ttft_limit)
                except FuturesTimeout:
                    _pool.shutdown(wait=False, cancel_futures=True)
                    print(f"\n  [TIMEOUT] LLM turn {turn+1} exceeded {_ttft_limit}s — skipping problem")
                    raise TimeoutError(f"TTFT>{_ttft_limit}s")
                finally:
                    _pool.shutdown(wait=False)
            else:
                resp = client.models.generate_content(
                    model=MODEL, contents=contents, config=_gemini_config(turn))
            if t1 is None: t1 = time.perf_counter()
            parts = (resp.candidates[0].content.parts or []) if resp.candidates else []
            for p in parts:
                if getattr(p, "text", None) and not getattr(p, "thought", False):
                    print(p.text, end="", flush=True)
            print(flush=True)

            if u := getattr(resp, "usage_metadata", None):
                tok_i     += int(getattr(u, "prompt_token_count",     None) or 0)
                tok_o     += int(getattr(u, "candidates_token_count", None) or 0)
                tok_think += int(getattr(u, "thoughts_token_count",   None) or 0)
                tok_t     += int(getattr(u, "total_token_count",      None) or 0)

            # Gemini thinking models: strip thought TEXT from history,
            # but keep function calls even if they carry thoughtSignature.
            history_parts = [p for p in parts
                             if not getattr(p, "thought", False)
                             or getattr(p, "function_call", None)]
            contents.append(types.Content(role="model", parts=history_parts or parts))

            text = " ".join(p.text for p in parts if getattr(p, "text", None)
                            and not getattr(p, "thought", False))
            if ans := parse_answer(text):
                answer_raw = ans; t_final = time.perf_counter()
                turn_entry["answer_raw"] = ans
                print(f"  [ANSWER] {json.dumps(ans)}")
                process_log.append(turn_entry); break

            fcs = [p.function_call for p in parts
                   if getattr(p, "function_call", None)]
            if not fcs:
                print(f"  [LLM] {text[:200].strip()}"); t_final = time.perf_counter()
                process_log.append(turn_entry); break

            tool_parts = []
            for fc in fcs:
                args = dict(fc.args)
                ggb  = lazy.get()

                if is_3d:
                    result, log_entry = exec_with_solid_routing(
                        tracker, ggb, fc.name, args)
                else:
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
        # Export canvas BEFORE closing — construction may be partially complete
        _finalize_ggb(lazy, prob_dir)
        tend = time.perf_counter()
        return {
            "answer": None, "passed": False,
            "detail": str(te),
            "tools_ok": tracker.ok_n, "tools_fail": tracker.fail_n,
            "tools_total": tracker.total_n,
            "process": {"total_turns": len(process_log), "turns": process_log},
            "metrics": {
                "input_tokens": tok_i, "output_tokens": tok_o,
                "thought_tokens": tok_think, "total_tokens": tok_t,
                "ggb_used": lazy.used,
                "ttft_sec": None, "t_final_sec": None,
                "t_last_sec": round(tend - t0, 3),
            },
        }
    finally:
        _finalize_ggb(lazy, prob_dir)

    return _build_result(answer_raw, prob, tracker, process_log, lazy,
                         tok_i, tok_o, tok_think, tok_t, t0, t1, t_final)


def _empty(detail: str) -> dict:
    return {"answer": None, "passed": False, "detail": detail,
            "tools_ok": 0, "tools_fail": 0, "tools_total": 0,
            "process": {"total_turns": 0, "turns": []},
            "metrics": {"input_tokens": 0, "output_tokens": 0,
                        "thought_tokens": 0, "total_tokens": 0,
                        "ggb_used": False, "ttft_sec": None,
                        "t_final_sec": None, "t_last_sec": 0}}


# ── Problem runner ────────────────────────────────────────────────────────────

def run_problem(client, prob: dict, prob_dir: Path, mode: str = "direct") -> dict:
    types_str = " + ".join(prob.get("problem_type_graph", []) +
                           prob.get("problem_type_goal",  []))
    print(f"\n{'='*64}")
    # Show knowledge/subknowledge if available (MathCanvas, SolidGeo)
    k_str = prob.get("knowledge", "")
    sk_str = prob.get("subknowledge", "")
    k_display = f"  {k_str}/{sk_str}" if k_str else ""
    is_3d_tag = " [3D]" if _is_solid(prob) else ""
    print(f"  [{prob['dataset']}:{prob['split']}]  id={prob['id']}  [{types_str}]"
          f"  mode={mode}  model={MODEL_ID}{is_3d_tag}{k_display}")
    print(f"  {prob['question'][:120]}{'...' if len(prob['question'])>120 else ''}")
    print(f"  Choices : {prob['choices']}")
    exp_display = prob['answer_label'] if prob.get('answer_label') else prob['expected']
    raw = prob.get('expected_raw', '')
    if raw and raw != str(exp_display):
        exp_display = f"{exp_display} ({raw})"
    print(f"  Expected: {exp_display}")
    print(f"  Image   : {prob['image']}")
    print(f"{'='*64}")

    if SDK_TYPE == "google-genai":
        result = run_query_gemini(client, prob, mode=mode, prob_dir=prob_dir)
    elif SDK_TYPE == "openai":
        result = run_query_openai(client, prob, mode=mode, prob_dir=prob_dir)
    elif SDK_TYPE == "anthropic":
        result = run_query_anthropic(client, prob, mode=mode, prob_dir=prob_dir)
    else:
        raise NotImplementedError(f"Eval loop for SDK '{SDK_TYPE}' not yet implemented")

    m = result["metrics"]
    print(f"\n  Approach: {'GGB' if m['ggb_used'] else 'direct'}  |  "
          f"Tools: {result['tools_ok']} OK / {result['tools_fail']} fail of {result['tools_total']}")
    print(f"  Tokens: in={m['input_tokens']} out={m['output_tokens']} "
          f"think={m['thought_tokens']} total={m['total_tokens']}  "
          f"TTFT={m['ttft_sec']}s  t_final={m['t_final_sec']}s")

    # MathCanvas → GPT-4.1 judge (replaces validate)
    if prob.get("dataset") == "mathcanvas" and result["answer"] is not None:
        from eval_common import judge_mathcanvas_answer
        pred_str = (json.dumps(result["answer"], ensure_ascii=False)
                    if isinstance(result["answer"], (list, dict))
                    else str(result["answer"]))
        print(f"\n  [JUDGE] calling GPT-4.1 ...", end="", flush=True)
        judge = judge_mathcanvas_answer(
            question=prob.get("question", ""),
            gt_answer=prob.get("expected_raw", ""),
            pred_answer=pred_str,
        )
        result["passed"] = bool(judge.get("complete_score", 0))
        result["detail"] = (f"judge weighted={judge.get('weighted_score',0):.2f} "
                            f"complete={judge.get('complete_score',0)}")
        result["judge"] = judge
        tag = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"\r  [JUDGE] {tag} weighted={judge.get('weighted_score',0):.2f}  "
              f"complete={judge.get('complete_score',0)}")
        corr = judge.get("correctness", [])
        if corr:
            print(f"  [JUDGE] sub-answers: {corr}")
    elif prob.get("dataset") == "mathcanvas":
        result["passed"] = False
        result["detail"] = "no answer emitted"
        print(f"\n  [FAIL] no answer emitted")
    else:
        passed = result["passed"]
        print(f"\n  {'[PASS]' if passed else '[FAIL]'} {result['detail']}")

    # Save per-problem result.json
    prob_dir.mkdir(parents=True, exist_ok=True)
    _EXTRA_FIELDS = ["knowledge_type", "book", "page", "expression",
                     "parsing_stru_seqs", "parsing_sem_seqs"]
    extra = {f: prob[f] for f in _EXTRA_FIELDS if f in prob and prob[f]}
    (prob_dir / f"{_model_slug()}_result.json").write_text(json.dumps({
        "id":           prob["id"],
        "question":     prob["question"],
        "choices":      prob.get("choices", []),
        "types":        prob.get("problem_type_graph",[]) + prob.get("problem_type_goal",[]),
        "answer_label": prob["answer_label"],
        "expected":     prob["expected"],
        "answer_raw":   result["answer"],
        "passed":       result["passed"],
        "detail":       result["detail"],
        "hint_mode":    prob.get("hint_mode", "none"),
        "mode":         mode,
        "model":        MODEL,
        **extra,
        **result["metrics"],
        "process":      result["process"],
        **({"judge": result["judge"]} if "judge" in result else {}),
    }, indent=2, ensure_ascii=False))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def _rebuild_summary(dataset_dir: Path, model_slug: str) -> list[dict]:
    """Scan all <model>_result.json files for this model and rebuild summary."""
    rows = []
    for result_file in sorted(dataset_dir.glob(f"*/{model_slug}_result.json")):
        rows.append(json.loads(result_file.read_text()))
    return rows


def main():
    global MODEL_ID, MODEL, PROVIDER, SDK_TYPE

    parser = argparse.ArgumentParser(
        description="LLM image+text geometry QA eval",
        epilog="Models: python -m symbolic.utils.model_registry --vision --tool-calling --thinking",
    )
    parser.add_argument("--dataset",  default=cfg.DEFAULT_DATASET,
                        choices=list(DATASET_LOADERS))
    parser.add_argument("--data_dir", default=cfg.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--sample",   type=int, default=cfg.DEFAULT_SAMPLE)
    parser.add_argument("--seed",     type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--id",       default=None)
    parser.add_argument("--list",     type=int, default=0, metavar="N")
    parser.add_argument("--hint",     default=cfg.DEFAULT_HINT,
                        choices=["none", "points", "logic_form",
                                 "parsing_sem", "parsing_stru", "auxiliary"])
    parser.add_argument("--mode",     default=cfg.DEFAULT_MODE,
                        choices=["direct", "construct"],
                        help="direct: show MC choices, allow direct reasoning  "
                             "construct: hide choices, require GeoGebra construction")
    parser.add_argument("--model",    default=cfg.DEFAULT_MODEL,
                        help="Model registry ID (see: --list-models)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    # ── PGPS9K-specific options ──
    parser.add_argument("--image-dir", default=None,
                        help="Image sub-dir override (e.g. Diagram, Diagram_Visual)")
    parser.add_argument("--exclude-book", default=None,
                        help="Exclude problems from a specific book (e.g. Geometry3K)")
    parser.add_argument("--subject", default=None,
                        help="MathVerse subject / SolidGeo problem_type filter")
    parser.add_argument("--level", default=None,
                        help="SolidGeo complexity level filter (e.g. 'Level 1', 'Level 2', 'Level 3')")
    parser.add_argument("--split", default=None,
                        help="OlympiadBench split (en_comp/zh_comp/zh_cee/all)")
    # ── Incremental runs ──
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip problems that already have <model>_result.json")
    parser.add_argument("--ttft-timeout", type=int, default=None,
                        help="Per-turn TTFT timeout in seconds (overrides eval_config)")
    parser.add_argument("--thinking", default=None,
                        choices=["off", "minimal", "low", "medium", "high"],
                        help="Thinking level; 'off' disables thinking (overrides eval_config)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--save-screenshot-per-turn", action="store_true",
                        help="Save a canvas PNG after each turn (for figure generation)")
    parser.add_argument("--hide-point-labels", action="store_true",
                        help="Hide point NAME labels in exported canvas (keeps "
                             "the point dots, axes, and grid). Useful for clean "
                             "figure/letter renderings.")
    parser.add_argument("--out-dir", default=None, type=Path,
                        help="Override output root (default: eval/{dataset}/); "
                             "when set, artifacts land in <out-dir>/<prob_id>/ instead.")
    args = parser.parse_args()

    # CLI overrides for eval_config
    if args.ttft_timeout is not None:
        cfg.TTFT_TIMEOUT = args.ttft_timeout
    if args.thinking is not None:
        cfg.THINKING_LEVEL = args.thinking
    cfg.SAVE_PER_TURN = args.save_screenshot_per_turn
    cfg.HIDE_POINT_LABELS = args.hide_point_labels

    # --list-models: show registry and exit
    if args.list_models:
        from symbolic.utils.model_registry import print_registry
        print_registry(list_models(vision=True, tool_calling=True))
        return

    # Resolve model from registry
    try:
        card = get_model(args.model)
    except KeyError as e:
        print(f"[ERROR] {e}"); return

    MODEL_ID = args.model
    MODEL    = card.model_name
    PROVIDER = card.provider
    SDK_TYPE = card.sdk

    global USE_RESPONSES_API
    USE_RESPONSES_API = getattr(card, 'use_responses_api', False)
    if USE_RESPONSES_API:
        print(f"  [CONFIG] Using OpenAI Responses API (v1/responses) for {MODEL_ID}")

    _load_prompts(args.dataset)

    # thinking_level priority: --thinking CLI > card.thinking_level > cfg default
    if args.thinking is None and card.thinking_level:
        cfg.THINKING_LEVEL = card.thinking_level
    # fixed_temperature: override cfg.TEMPERATURE if model requires it (e.g. Kimi K2.5 = 1.0)
    if getattr(card, 'fixed_temperature', None) is not None:
        cfg.TEMPERATURE = card.fixed_temperature

    loader   = DATASET_LOADERS[args.dataset]
    kwargs   = {"data_dir": args.data_dir, "sample": args.sample,
                "seed": args.seed, "hint": args.hint}
    if args.id:
        id_val = args.id
        if id_val.endswith(".json"):
            id_path = Path(id_val) if Path(id_val).is_absolute() else Path(args.data_dir).parent / id_val
            if not id_path.exists():
                id_path = EVAL_ROOT / args.dataset / id_val
            with open(id_path) as f:
                id_list = json.load(f)
            id_val = ",".join(id_list)
        kwargs["problem_id"] = id_val
        kwargs["sample"] = None  # don't truncate when IDs explicitly specified
    if args.dataset == "geometry3k" and args.image_dir:
        kwargs["image_style"] = args.image_dir  # "point" or "plain"
    if args.dataset == "pgps9k":
        if args.image_dir:
            kwargs["image_dir"] = args.image_dir
        if args.exclude_book:
            kwargs["exclude_book"] = args.exclude_book
    if args.dataset == "geolaux":
        kwargs["problem_type"] = "calculation"
    if args.dataset == "mathverse" and args.subject:
        kwargs["subject"] = args.subject
    if args.dataset == "solidgeo":
        if args.level:
            kwargs["level"] = args.level
        if args.subject:  # reuse --subject for problem_type filter
            kwargs["problem_type"] = args.subject
    if args.dataset == "mathcanvas" and args.subject:
        kwargs["knowledge"] = args.subject
    if args.dataset == "olympiadbench":
        kwargs["split"] = getattr(args, 'split', None) or "en_comp"
    problems = loader(**kwargs)

    if args.list:
        print(f"\n{args.dataset}/{args.data_dir.name} — first {args.list} problems:")
        for p in problems[:args.list]:
            typs = "+".join(p.get("problem_type_graph",[]) + p.get("problem_type_goal",[]))
            print(f"  {p['id']:6s}  [{typs:30s}]  {p['question'][:60]}")
        return

    # Output dir: eval/{dataset}/ by default, or --out-dir override
    dataset_dir = args.out_dir if args.out_dir is not None else EVAL_ROOT / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Set hint suffix for result filenames
    global _HINT_SUFFIX
    if args.hint and args.hint != "none":
        _HINT_SUFFIX = f"hint_{args.hint}"

    print("\n" + "="*64)
    print(f"  LLM Image+Text Geometry QA")
    print(f"  Dataset   : {args.dataset}/{args.data_dir.name}")
    print(f"  Model     : {MODEL_ID}  ({card.provider} / {card.sdk})")
    print(f"  API model : {MODEL}")
    print(f"  Problems  : {len(problems)}")
    print(f"  Hint mode : {args.hint}")
    print(f"  Mode      : {args.mode}")
    print(f"  Output    : {dataset_dir}/")
    print("="*64)

    try:
        client, _ = make_client(MODEL_ID)
    except Exception as exc:
        print(f"[ERROR] {exc}"); return

    slug     = _model_slug()
    run_ids  = set()

    # Filter out already-done problems upfront
    todo = []
    for prob in problems:
        prob_dir = dataset_dir / prob["id"]
        if args.skip_done and (prob_dir / f"{slug}_result.json").exists():
            continue
        todo.append(prob)
    skipped = len(problems) - len(todo)

    if args.skip_done:
        print(f"  Skip-done : {skipped} already completed, {len(todo)} to run")
    if args.workers > 1:
        print(f"  Workers   : {args.workers}")
    print("="*64)

    def _run_one(prob):
        """Execute a single problem (safe to call from a worker thread)."""
        prob_dir = dataset_dir / prob["id"]
        prob_dir.mkdir(parents=True, exist_ok=True)
        log_path = prob_dir / f"{slug}_log.txt"
        with TeeLog(log_path):
            result = run_problem(client, prob, prob_dir, mode=args.mode)
        return prob["id"], result

    if args.workers <= 1:
        # ── Sequential (original behaviour) ──
        consecutive_errors = 0
        for prob in todo:
            try:
                pid, _ = _run_one(prob)
                run_ids.add(pid)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                wait = min(consecutive_errors * 10, 60)
                print(f"  [ERROR] {prob['id']}: {e}  (backoff {wait}s)")
                time.sleep(wait)
    else:
        # ── Parallel via subprocesses (full isolation: own Chrome, API client) ──
        import subprocess, math
        n_workers = min(args.workers, len(todo))
        # Round-robin split into N chunks
        chunks: list[list[str]] = [[] for _ in range(n_workers)]
        for i, prob in enumerate(todo):
            chunks[i % n_workers].append(prob["id"])

        # Build base command (inherit all CLI args except --workers and --id)
        base_cmd = [
            sys.executable, __file__,
            "--dataset", args.dataset,
            "--data_dir", str(args.data_dir),
            "--sample", "0",
            "--mode", args.mode,
            "--model", args.model,
            "--hint", args.hint,
            "--skip-done",
            "--workers", "1",          # each subprocess runs sequentially
        ]
        if args.ttft_timeout is not None:
            base_cmd += ["--ttft-timeout", str(args.ttft_timeout)]
        if args.thinking is not None:
            base_cmd += ["--thinking", args.thinking]
        if args.image_dir:
            base_cmd += ["--image-dir", args.image_dir]
        if args.exclude_book:
            base_cmd += ["--exclude-book", args.exclude_book]

        procs = []
        for i, chunk_ids in enumerate(chunks):
            if not chunk_ids:
                continue
            cmd = base_cmd + ["--id", ",".join(chunk_ids)]
            print(f"  [subprocess {i}] {len(chunk_ids)} problems: {chunk_ids[0]}..{chunk_ids[-1]}")
            procs.append(subprocess.Popen(cmd))

        # Wait for all subprocesses
        for p in procs:
            p.wait()
        failed = sum(1 for p in procs if p.returncode != 0)
        if failed:
            print(f"  [WARN] {failed}/{len(procs)} subprocess(es) exited with error")

        # Collect all IDs that now have results
        for prob in todo:
            res_path = dataset_dir / prob["id"] / f"{slug}_result.json"
            if res_path.exists():
                run_ids.add(prob["id"])

    if skipped:
        print(f"\n  [skip-done] Skipped {skipped} already-completed problems")

    # Rebuild cumulative summary for this model only
    summary = _rebuild_summary(dataset_dir, slug)
    ts = time.strftime("%Y%m%d_%H%M")
    summary_path = dataset_dir / f"summary_{slug}_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Console summary
    n_pass = sum(1 for r in summary if r.get("passed"))
    n_ggb  = sum(1 for r in summary if r.get("ggb_used"))
    pct    = n_pass / len(summary) * 100 if summary else 0

    print(f"\n{'='*64}")
    print(f"  SUMMARY  {args.dataset}  ({len(summary)} problems evaluated so far)")
    print(f"{'='*64}")
    # Show only the problems actually run this session
    for r in summary:
        if r["id"] not in run_ids: continue
        tag  = "[+]" if r["passed"] else "[-]"
        app  = "GGB" if r.get("ggb_used") else "dir"
        typs = "+".join(r.get("types", []))
        print(f"  {tag} [{app}] {r['id']:6s}  [{typs:28s}]  {r['detail']}")
    print(f"\n  This run : {sum(1 for r in summary if r['id'] in run_ids and r['passed'])}"
          f"/{len(run_ids)} passed")
    print(f"  All-time : {n_pass}/{len(summary)} passed ({pct:.0f}%)  "
          f"| GeoGebra used: {n_ggb}/{len(summary)}")
    print(f"  Results  -> {summary_path}")
    print("="*64 + "\n")


if __name__ == "__main__":
    main()
