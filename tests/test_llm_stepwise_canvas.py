"""
LLM -> GeoGebra render pipeline — stepwise canvas feedback.

Variant of test_llm_to_draw that appends a canvas snapshot (name->type)
to every successful tool-call response, so the model always knows the
current state of the construction.

Run: python tests/test_llm_stepwise_canvas.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import build_gemini_tools, execute_geogebra_tool
from symbolic.utils import get_api_key

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_llm_stepwise_canvas
(OUTDIR / "fig").mkdir(parents=True, exist_ok=True)
(OUTDIR / "log").mkdir(parents=True, exist_ok=True)


class TeeLog:
    """Mirror stdout to a file while keeping terminal output intact."""
    def __init__(self, path):
        self._path = Path(path)
        self._file = None
        self._orig = None

    def __enter__(self):
        self._orig = sys.stdout
        self._file = self._path.open("w", encoding="utf-8")
        sys.stdout = self
        return self

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def __exit__(self, *_):
        sys.stdout = self._orig
        self._file.close()
        self._orig.write(f"  [LOG] {self._path}\n")


MODEL     = "gemini-3-flash-preview"
MAX_TURNS = 30

PROBLEMS = [
    {
        "id":   "sc_01_isosceles",
        "name": "Isosceles triangle + perpendicular bisector",
        "nl": (
            "Construct isosceles triangle ABC where AB = AC = 5 and BC = 6. "
            "Draw the perpendicular bisector of BC and mark its intersection "
            "with BC as M. Show that AM is perpendicular to BC by marking the right angle."
        ),
    },
    {
        "id":   "sc_02_inscribed_angle",
        "name": "Inscribed angle theorem",
        "nl": (
            "Draw a circle with centre O and radius 3. "
            "Place three points A, B, C on the circle. "
            "Draw the inscribed angle ABC and the central angle AOC "
            "subtending the same arc AC. Label both angles."
        ),
    },
    {
        "id":   "sc_03_tangent",
        "name": "Tangent-radius perpendicularity",
        "nl": (
            "Draw a circle with centre O and radius 3. "
            "Place point P outside the circle at (7, 0). "
            "Draw both tangent lines from P to the circle. "
            "Mark the tangent contact points T1 and T2. "
            "Draw segment OT1 and mark the 90-degree angle at T1."
        ),
    },
    {
        "id":   "sc_04_midsegment",
        "name": "Triangle midsegment theorem",
        "nl": (
            "Draw triangle ABC. "
            "Mark D as the midpoint of AB and E as the midpoint of AC. "
            "Draw segment DE. "
            "Label DE and BC lengths to show DE = BC / 2."
        ),
    },
    {
        "id":   "sc_05_pythagorean",
        "name": "Pythagorean theorem visual proof",
        "nl": (
            "Draw right triangle ABC with the right angle at C, AC = 3, BC = 4. "
            "Construct squares on all three sides. "
            "Label the area of each square."
        ),
    },
]

SYSTEM_INSTRUCTION = """\
You are a GeoGebra construction assistant.

Your job is to construct the requested geometry figure by tool-calling only.
Each tool call maps to one GeoGebra command and is executed immediately.
After each call, you receive {success, error, canvas} feedback.
  - canvas: current {name: type} snapshot of every object on the canvas.

Execution protocol (strict):
1) Work in dependency order: anchors -> constraints -> derived objects -> measurements.
2) Make at most 2 tool calls per turn, then wait for feedback.
3) If any call fails, fix that dependency first before expanding construction.
4) Before creating an object, check canvas to avoid duplicating existing names.
5) Keep coordinates in [-10, 10] x [-10, 10].
6) Do not call styling/zoom/visibility-management tools unless explicitly requested.

Completion policy:
- Return exactly "DONE" only when all required objects are present.
- Do not output explanations, markdown, or extra text.
- If construction is incomplete, continue calling tools and do not emit DONE.
"""


# ── Tool-calling loop ──────────────────────────────────────────────────────────

def run_tool_calling(ggb, client, nl: str):
    from google.genai import types

    tools  = build_gemini_tools()
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Construct this geometry figure:\n\n{nl}")]
        )
    ]

    succeeded = failed = total = 0
    token_in = token_out = token_total = 0
    t_submit = time.perf_counter()
    t_first_response = t_final = None

    for turn in range(MAX_TURNS):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config,
        )
        if t_first_response is None:
            t_first_response = time.perf_counter()

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            in_tok  = getattr(usage, "prompt_token_count",     None) or getattr(usage, "input_token_count",  None) or 0
            out_tok = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None) or 0
            tot_tok = getattr(usage, "total_token_count", None) or (in_tok + out_tok)
            token_in    += int(in_tok)
            token_out   += int(out_tok)
            token_total += int(tot_tok)

        model_parts = response.candidates[0].content.parts
        contents.append(types.Content(role="model", parts=model_parts))

        fn_calls = [p.function_call for p in model_parts if getattr(p, "function_call", None)]

        if not fn_calls:
            txt = response.text or ""
            print(f"  [LLM] {txt[:120].strip()}")
            t_final = time.perf_counter()
            break

        tool_responses = []
        for fc in fn_calls:
            total += 1
            args = dict(fc.args)
            cmd, ok, err = execute_geogebra_tool(ggb, fc.name, args)
            status = "OK  " if ok else "FAIL"
            print(f"    [{status}] {cmd}" + (f"  -> {err}" if not ok else ""))
            if ok:
                succeeded += 1
            else:
                failed += 1

            resp = {"command": cmd, "success": ok, "error": err}

            if ok:
                # Snapshot canvas after every successful command
                objs = ggb.get_construction_state().get("objects", {})
                canvas = {name: info.get("type", "?") for name, info in objs.items()}
                resp["canvas"] = canvas
                print(f"           canvas: {json.dumps(canvas)}")

            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=resp,
                    )
                )
            )

        contents.append(types.Content(role="user", parts=tool_responses))

    t_end = time.perf_counter()
    metrics = {
        "input_tokens":        token_in,
        "output_tokens":       token_out,
        "total_tokens":        token_total,
        "ttft_sec":            round(t_first_response - t_submit, 3) if t_first_response else None,
        "submit_to_final_sec": round(t_final - t_submit, 3)          if t_final         else None,
        "submit_to_last_sec":  round(t_end   - t_submit, 3),
    }
    return succeeded, total, failed, metrics


# ── Problem runner ─────────────────────────────────────────────────────────────

def run_problem(ggb, client, prob: dict):
    from geogebra_render_common import apply_global_style, fit_view_square

    print(f"\n{'='*62}")
    print(f"  {prob['name']}")
    print(f"{'='*62}")
    print(f"  NL: {prob['nl'][:100]}...")

    ggb.reset()
    print()
    succeeded, total, failed, metrics = run_tool_calling(ggb, client, prob["nl"])

    print(f"\n  Tool calls: {succeeded} OK, {failed} failed (of {total} total)")
    print(f"  Tokens in/out/total: {metrics['input_tokens']}/{metrics['output_tokens']}/{metrics['total_tokens']}")
    print(f"  Timing TTFT={metrics['ttft_sec']}s | submit->final={metrics['submit_to_final_sec']}s")

    apply_global_style(ggb)
    fit_view_square(ggb, padding=1.2)

    out = OUTDIR / "fig" / f"{prob['id']}.png"
    ok  = ggb.export_png(out)
    print(f"  {'[SAVED]' if ok else '[FAIL export]'} {out}")

    return succeeded, total, failed, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  LLM -> GeoGebra  (stepwise canvas feedback)")
    print(f"  Model     : {MODEL}")
    print(f"  Max turns : {MAX_TURNS}")
    print("="*62)

    try:
        from symbolic.utils.env_loader import make_genai_client
        client = make_genai_client()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    summary = []
    for prob in PROBLEMS:
        with TeeLog(OUTDIR / "log" / f"{prob['id']}.txt"):
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                ok, total, fail, metrics = run_problem(ggb, client, prob)
            summary.append({
                "name":                prob["name"],
                "ok":                  ok,
                "total":               total,
                "fail":                fail,
                "total_tokens":        metrics["total_tokens"],
                "ttft_sec":            metrics["ttft_sec"],
                "submit_to_final_sec": metrics["submit_to_final_sec"],
            })

    print(f"\n{'='*62}")
    print("  Summary")
    print(f"{'='*62}")
    for s in summary:
        pct = s["ok"] / s["total"] * 100 if s["total"] else 0
        tag = "[OK]" if s["fail"] == 0 else "[!!]"
        print(
            f"  {tag} {s['name'][:30]:30s} "
            f"{s['ok']}/{s['total']} ({pct:.0f}%) | "
            f"tok={s['total_tokens']} | ttft={s['ttft_sec']}s | "
            f"t_final={s['submit_to_final_sec']}s"
        )

    print(f"\n  Figures -> {OUTDIR / 'fig'}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
