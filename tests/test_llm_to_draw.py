"""
LLM -> GeoGebra render pipeline (v3 — tool calling).

The model is given the NL problem and a set of GeoGebra construction tools.
It calls the tools one by one; each call is executed in GeoGebra immediately
and the result (success / error) is fed back so the model can self-correct.

After construction:
  - apply_global_style  : consistent JS API styling by object type
  - fit_view_square     : square viewport from point bounding box
  - export PNG to temp/

Run: python tests/test_llm_to_draw.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import build_gemini_tools, execute_geogebra_tool
from symbolic.utils import get_api_key

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_llm_to_draw
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

MODEL      = "gemini-3.1-pro-preview-customtools"
MAX_TURNS  = 30   # max tool-call rounds per problem

PROBLEMS = [
    {
        "id": "llm_01_isosceles",
        "name": "Isosceles triangle + perpendicular bisector",
        "nl": (
            "Construct isosceles triangle ABC where AB = AC = 5 and BC = 6. "
            "Draw the perpendicular bisector of BC and mark its intersection "
            "with BC as M. Show that AM is perpendicular to BC by marking the right angle."
        ),
    },
    {
        "id": "llm_02_inscribed_angle",
        "name": "Inscribed angle theorem",
        "nl": (
            "Draw a circle with centre O and radius 3. "
            "Place three points A, B, C on the circle. "
            "Draw the inscribed angle ABC and the central angle AOC "
            "subtending the same arc AC. Label both angles."
        ),
    },
    {
        "id": "llm_03_tangent",
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
        "id": "llm_04_midsegment",
        "name": "Triangle midsegment theorem",
        "nl": (
            "Draw triangle ABC. "
            "Mark D as the midpoint of AB and E as the midpoint of AC. "
            "Draw segment DE. "
            "Label DE and BC lengths to show DE = BC / 2."
        ),
    },
    {
        "id": "llm_05_pythagorean",
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
After each call, you receive {success, error} feedback.

Execution protocol (strict):
1) Work in dependency order: anchors -> constraints -> derived objects -> measurements.
2) Make at most 2 tool calls per turn, then wait for feedback.
3) If any call fails, fix that dependency first before expanding construction.
4) Prefer constraint-based construction over ad-hoc coordinates.
5) Keep coordinates in [-10, 10] x [-10, 10].
6) Do not call styling/zoom/visibility-management tools unless explicitly requested.

Completion policy:
- Return exactly "DONE" only when all required objects are present.
- Do not output explanations, markdown, or extra text.
- If construction is incomplete, continue calling tools and do not emit DONE.
"""

# Tool definitions and execution are now centralized in
# symbolic.tools.geogebra_tools for cross-provider reuse.


# ── GeoGebra JS API helpers ────────────────────────────────────────────────────

def js(ggb, script):
    return ggb._driver.execute_script(script)


def set_color(ggb, n, r, g, b):
    js(ggb, f'ggbApplet.setColor("{n}", {r}, {g}, {b})')


def set_thickness(ggb, n, t):
    js(ggb, f'ggbApplet.setLineThickness("{n}", {t})')


def set_filling(ggb, n, a):
    js(ggb, f'ggbApplet.setFilling("{n}", {a})')


def apply_global_style(ggb):
    """Post-construction styling via JS API, by object type."""
    state = ggb.get_construction_state()
    for name, info in state["objects"].items():
        t = info.get("type", "")
        if t == "polygon":
            set_color(ggb, name, 173, 216, 230)
            set_filling(ggb, name, 0.25)
            set_thickness(ggb, name, 2)
        elif t == "line":
            set_color(ggb, name, 160, 160, 160)
            set_thickness(ggb, name, 1)
        elif t == "segment":
            set_thickness(ggb, name, 2)
        elif t in ("circle", "conic"):
            set_thickness(ggb, name, 2)
        elif t == "angle":
            set_color(ggb, name, 220, 120, 0)


def fit_view_square(ggb, padding=1.5):
    """Square viewport centred on point bounding box."""
    state = ggb.get_construction_state()
    pts = [
        (info["x"], info["y"])
        for info in state["objects"].values()
        if info.get("type") == "point" and "x" in info and "y" in info
    ]
    if not pts:
        js(ggb, "ggbApplet.setCoordSystem(-7, 11, -7, 11)")
        return
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding
    # Force square
    xspan, yspan = xmax - xmin, ymax - ymin
    if xspan > yspan:
        d = (xspan - yspan) / 2
        ymin -= d; ymax += d
    else:
        d = (yspan - xspan) / 2
        xmin -= d; xmax += d
    js(ggb, f"ggbApplet.setCoordSystem({xmin:.2f}, {xmax:.2f}, {ymin:.2f}, {ymax:.2f})")


# ── Tool-calling loop ──────────────────────────────────────────────────────────

def run_tool_calling(ggb, client, nl: str):
    """
    Single-model tool-calling pipeline.
    Returns (succeeded, total_calls, failed_calls).
    """
    from google.genai import types

    tools  = build_gemini_tools()
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
    )

    # Conversation history
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Construct this geometry figure:\n\n{nl}")]
        )
    ]

    succeeded = 0
    failed = 0
    total = 0
    token_in = 0
    token_out = 0
    token_total = 0
    t_submit = time.perf_counter()
    t_first_response = None
    t_final = None

    for turn in range(MAX_TURNS):
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        if t_first_response is None:
            t_first_response = time.perf_counter()

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            in_tok = (
                getattr(usage, "prompt_token_count", None)
                or getattr(usage, "input_token_count", None)
                or 0
            )
            out_tok = (
                getattr(usage, "candidates_token_count", None)
                or getattr(usage, "output_token_count", None)
                or 0
            )
            tot_tok = getattr(usage, "total_token_count", None)
            if tot_tok is None:
                tot_tok = in_tok + out_tok
            token_in += int(in_tok)
            token_out += int(out_tok)
            token_total += int(tot_tok)

        # Collect model output
        model_parts = response.candidates[0].content.parts
        contents.append(types.Content(role="model", parts=model_parts))

        # Gather all function calls in this turn
        fn_calls = [p.function_call for p in model_parts if getattr(p, "function_call", None)]

        if not fn_calls:
            # Model finished (text response)
            txt = response.text or ""
            print(f"  [LLM] {txt[:120].strip()}")
            t_final = time.perf_counter()
            break

        # Execute each tool call and collect responses
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

            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={
                            "command": cmd,
                            "success": ok,
                            "error":   err,
                        },
                    )
                )
            )

        contents.append(types.Content(role="user", parts=tool_responses))

    t_end = time.perf_counter()
    metrics = {
        "input_tokens": token_in,
        "output_tokens": token_out,
        "total_tokens": token_total,
        "ttft_sec": round((t_first_response - t_submit), 3) if t_first_response is not None else None,
        "submit_to_final_sec": round((t_final - t_submit), 3) if t_final is not None else None,
        "submit_to_last_sec": round((t_end - t_submit), 3),
    }
    return succeeded, total, failed, metrics


# ── Problem runner ─────────────────────────────────────────────────────────────

def run_problem(ggb, client, prob: dict):
    print(f"\n{'='*60}")
    print(f"  {prob['name']}")
    print(f"{'='*60}")
    print(f"  NL: {prob['nl'][:100]}...")

    ggb.reset()
    print()
    succeeded, total, failed, metrics = run_tool_calling(ggb, client, prob["nl"])
    print(f"\n  Tool calls: {succeeded} OK, {failed} failed (of {total} total)")
    print(
        f"  Tokens in/out/total: {metrics['input_tokens']}/"
        f"{metrics['output_tokens']}/{metrics['total_tokens']}"
    )
    print(
        f"  Timing TTFT={metrics['ttft_sec']}s | "
        f"submit->final={metrics['submit_to_final_sec']}s | "
        f"submit->last={metrics['submit_to_last_sec']}s"
    )

    apply_global_style(ggb)
    fit_view_square(ggb, padding=1.2)

    out = OUTDIR / "fig" / f"{prob['id']}.png"
    ok  = ggb.export_png(out)
    print(f"  {'[SAVED]' if ok else '[FAIL export]'} {out}")

    return succeeded, total, failed, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  LLM -> GeoGebra Render Pipeline  (tool-calling, v3)")
    print(f"  Model     : {MODEL}")
    print(f"  Max turns : {MAX_TURNS}")
    print("="*60)

    try:
        from google import genai
        client = genai.Client(api_key=get_api_key("google"))
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    summary = []
    for prob in PROBLEMS:
        with TeeLog(OUTDIR / "log" / f"{prob['id']}.txt"):
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                ok, total, fail, metrics = run_problem(ggb, client, prob)
            summary.append(
                {
                    "name": prob["name"],
                    "ok": ok,
                    "total": total,
                    "fail": fail,
                    "input_tokens": metrics["input_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    "total_tokens": metrics["total_tokens"],
                    "ttft_sec": metrics["ttft_sec"],
                    "submit_to_final_sec": metrics["submit_to_final_sec"],
                }
            )

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for s in summary:
        pct = s["ok"] / s["total"] * 100 if s["total"] else 0
        tag = "[OK]" if s["fail"] == 0 else "[!!]"
        print(
            f"  {tag} {s['name'][:26]:26s} "
            f"{s['ok']}/{s['total']} ({pct:.0f}%) | "
            f"tok={s['total_tokens']} | ttft={s['ttft_sec']}s | "
            f"t_final={s['submit_to_final_sec']}s"
        )

    print(f"\n  Figures saved to {OUTDIR / 'fig'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()