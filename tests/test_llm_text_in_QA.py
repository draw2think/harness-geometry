"""
LLM Geometry Query Pipeline.

Tests the LLM's ability to answer geometry questions with verifiable answers,
not just draw figures.  The model may:

  LEVEL 0 — direct reasoning  : answer from theorems/formulas, no tools needed.
  LEVEL 1 — GeoGebra query    : construct objects, measure via query_* tools.
  LEVEL 2 — hybrid            : construct + measure + symbolic synthesis.

Answer format (model must emit exactly one line):
  ANSWER: {"value": <v>, "type": "<boolean|numerical|symbolic|coordinate>"}
    boolean    : true / false
    numerical  : a decimal  (e.g. 5.0, 120.0)
    symbolic   : a string using sqrt(), pi, fractions  (e.g. "sqrt(2)", "6*pi")
    coordinate : [x, y] JSON array

Categories tested:
  A  Boolean      — true/false, solvable by theorem recall
  B  Numerical    — decimal answer from a formula
  C  Symbolic     — exact answer with sqrt / pi / fractions
  D  Coordinate   — points, slopes, intersections
  E  Mixed        — concrete geometry; model chooses direct reasoning or tools

GeoGebra is started lazily: Selenium is only launched if the model actually
calls a construction or query tool.  Level-0 problems incur zero startup cost.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import math
import re
import time
from pathlib import Path

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import (
    build_gemini_tools,
    execute_geogebra_tool,
    execute_query_tool,
)
from symbolic.utils import get_api_key

OUTDIR = Path("temp") / Path(__file__).stem
(OUTDIR / "log").mkdir(parents=True, exist_ok=True)


# ── TeeLog ────────────────────────────────────────────────────────────────────

class TeeLog:
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


# ── Model / config ────────────────────────────────────────────────────────────

MODEL     = "gemini-3-flash-preview"
MAX_TURNS = 30

SYSTEM_INSTRUCTION = """\
You are a geometry problem solver with optional access to GeoGebra tools.

Decision policy:
  DIRECT ANSWER — preferred whenever you can derive the result confidently
    from a theorem, formula, or short algebraic computation.
    Use this for abstract questions, standard formulas, and anything where
    you are certain of the answer without needing to construct a figure.

  GEOGEBRA TOOLS — use when direct computation is error-prone or you want
    to verify a non-trivial measurement by actually building the figure.
    draw_* tools create geometric objects; query_* tools measure them and
    return {"value": <number>} in the response.
    After every successful call you also receive a canvas snapshot
    {name: {type, x?, y?, val?}} showing the current construction state.

Tool protocol (when using GeoGebra):
  - At most 2 tool calls per turn, then wait for feedback.
  - Dependency order: anchor points → derived objects → measurement queries.
  - Keep coordinates in [−10, 10] × [−10, 10].
  - Do not call styling or visibility tools.
  - When a query_* response contains {"value": <number>}, that number IS the
    final measurement. Emit ANSWER immediately — no further tool calls needed.

Answer protocol (always):
  Emit EXACTLY one line the moment you have the answer:
    ANSWER: {"value": <v>, "type": "<boolean|numerical|symbolic|coordinate>"}
  Formats:
    boolean    → JSON true or false
    numerical  → decimal number          e.g. 5.0
    symbolic   → string with sqrt()/pi   e.g. "sqrt(3)", "6*pi", "12*sqrt(2)/7"
    coordinate → [x, y] JSON array       e.g. [2.0, 1.5]
  Output ANSWER as the first and only line. No preamble, no explanation.
"""


# ── Problem bank ──────────────────────────────────────────────────────────────

PROBLEMS = [

    # ── A: Boolean  (LEVEL 0 expected — pure theorem recall) ─────────────────

    # {
    #     "id": "bool_01_pythagorean_check",
    #     "category": "A_boolean",
    #     "nl": "Is a triangle with sides 3, 4, and 5 a right triangle?",
    #     "expected": True,
    #     "expected_type": "boolean",
    # },
    # {
    #     "id": "bool_02_square_diagonals_perp",
    #     "category": "A_boolean",
    #     "nl": "Are the diagonals of a square perpendicular to each other?",
    #     "expected": True,
    #     "expected_type": "boolean",
    # },
    # {
    #     "id": "bool_03_exterior_angle_theorem",
    #     "category": "A_boolean",
    #     "nl": (
    #         "The exterior angle of a triangle at vertex A is 110°. "
    #         "The interior angles at B and C are 60° and 50° respectively. "
    #         "Does the exterior angle equal the sum of the two non-adjacent interior angles?"
    #     ),
    #     "expected": True,
    #     "expected_type": "boolean",
    # },
    # {
    #     "id": "bool_04_inscribed_angle_half",
    #     "category": "A_boolean",
    #     "nl": (
    #         "A central angle subtends an arc of 80°. "
    #         "An inscribed angle subtending the same arc measures 40°. "
    #         "Is this consistent with the inscribed angle theorem?"
    #     ),
    #     "expected": True,
    #     "expected_type": "boolean",
    # },
    # {
    #     "id": "bool_05_midsegment_parallel",
    #     "category": "A_boolean",
    #     "nl": (
    #         "In triangle ABC, D is the midpoint of AB and E is the midpoint of AC. "
    #         "Is segment DE parallel to BC and equal to half the length of BC?"
    #     ),
    #     "expected": True,
    #     "expected_type": "boolean",
    # },

    # # ── B: Numerical  (LEVEL 0 — formula application) ────────────────────────

    # {
    #     "id": "num_01_rectangle_diagonal",
    #     "category": "B_numerical",
    #     "nl": "What is the length of the diagonal of a rectangle with sides 5 and 12?",
    #     "expected": 13.0,
    #     "expected_type": "numerical",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "num_02_interior_angle_pentagon",
    #     "category": "B_numerical",
    #     "nl": "What is the measure in degrees of each interior angle of a regular pentagon?",
    #     "expected": 108.0,
    #     "expected_type": "numerical",
    #     "tolerance": 0.01,
    # },
    # {
    #     "id": "num_03_circumradius_right_triangle",
    #     "category": "B_numerical",
    #     "nl": (
    #         "What is the circumradius of a right triangle with legs 3 and 4? "
    #         "Give a decimal answer."
    #     ),
    #     "expected": 2.5,
    #     "expected_type": "numerical",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "num_04_chord_center_distance",
    #     "category": "B_numerical",
    #     "nl": (
    #         "A chord of length 8 is drawn in a circle of radius 5. "
    #         "What is the perpendicular distance from the center to the chord?"
    #     ),
    #     "expected": 3.0,
    #     "expected_type": "numerical",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "num_05_apothem_hexagon",
    #     "category": "B_numerical",
    #     "nl": (
    #         "What is the apothem (perpendicular distance from center to a side) "
    #         "of a regular hexagon with side length 4? Give a decimal answer."
    #     ),
    #     # apothem = s * sqrt(3) / 2 = 4 * sqrt(3) / 2 = 2*sqrt(3)
    #     "expected": 2 * math.sqrt(3),
    #     "expected_type": "numerical",
    #     "tolerance": 0.01,
    # },

    # # ── C: Symbolic  (LEVEL 0 — exact form with sqrt / pi) ───────────────────

    # {
    #     "id": "sym_01_unit_square_diagonal",
    #     "category": "C_symbolic",
    #     "nl": "What is the length of the diagonal of a unit square? Express in exact symbolic form.",
    #     "expected": "sqrt(2)",
    #     "expected_numeric": math.sqrt(2),
    #     "expected_type": "symbolic",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "sym_02_equilateral_area_side2",
    #     "category": "C_symbolic",
    #     "nl": (
    #         "What is the area of an equilateral triangle with side length 2? "
    #         "Express in exact symbolic form."
    #     ),
    #     # Area = (sqrt(3)/4)*4 = sqrt(3)
    #     "expected": "sqrt(3)",
    #     "expected_numeric": math.sqrt(3),
    #     "expected_type": "symbolic",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "sym_03_circle_circumference",
    #     "category": "C_symbolic",
    #     "nl": "What is the circumference of a circle with radius 3? Express using pi.",
    #     "expected": "6*pi",
    #     "expected_numeric": 6 * math.pi,
    #     "expected_type": "symbolic",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "sym_04_sector_area_60deg",
    #     "category": "C_symbolic",
    #     "nl": (
    #         "A circle has radius 6 and a sector with central angle 60°. "
    #         "What is the area of the sector? Express in exact symbolic form."
    #     ),
    #     # (60/360) * pi * 36 = 6*pi
    #     "expected": "6*pi",
    #     "expected_numeric": 6 * math.pi,
    #     "expected_type": "symbolic",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "sym_05_inradius_equilateral_6",
    #     "category": "C_symbolic",
    #     "nl": (
    #         "What is the inradius of an equilateral triangle with side length 6? "
    #         "Express in exact symbolic form."
    #     ),
    #     # r = Area/s = (9*sqrt(3)) / 9 = sqrt(3)
    #     "expected": "sqrt(3)",
    #     "expected_numeric": math.sqrt(3),
    #     "expected_type": "symbolic",
    #     "tolerance": 0.001,
    # },

    # # ── D: Coordinate  (LEVEL 0 preferred; GeoGebra also valid) ──────────────

    # {
    #     "id": "coord_01_midpoint",
    #     "category": "D_coordinate",
    #     "nl": "Find the midpoint of the segment with endpoints A(−2, 3) and B(6, −1).",
    #     "expected": [2.0, 1.0],
    #     "expected_type": "coordinate",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "coord_02_slope",
    #     "category": "D_coordinate",
    #     "nl": "What is the slope of the line passing through (1, 2) and (4, 11)?",
    #     "expected": 3.0,
    #     "expected_type": "numerical",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "coord_03_line_intersection",
    #     "category": "D_coordinate",
    #     "nl": "Find the intersection point of the lines  y = 2x + 1  and  y = −x + 4.",
    #     "expected": [1.0, 3.0],
    #     "expected_type": "coordinate",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "coord_04_point_to_line_distance",
    #     "category": "D_coordinate",
    #     "nl": "What is the distance from point (3, 4) to the line  3x + 4y = 0?",
    #     # |3*3 + 4*4| / sqrt(9+16) = 25/5 = 5
    #     "expected": 5.0,
    #     "expected_type": "numerical",
    #     "tolerance": 0.001,
    # },
    # {
    #     "id": "coord_05_circumcenter_right",
    #     "category": "D_coordinate",
    #     "nl": (
    #         "Find the circumcenter of the triangle with vertices "
    #         "A(0, 0),  B(4, 0),  C(0, 3). "
    #         "Give the answer as [x, y]."
    #     ),
    #     # Right triangle → circumcenter = midpoint of hypotenuse BC = (2, 1.5)
    #     "expected": [2.0, 1.5],
    #     "expected_type": "coordinate",
    #     "tolerance": 0.001,
    # },

    # ── E: Mixed  (concrete coordinates; model chooses approach) ─────────────
    # These problems can be solved analytically or via GeoGebra tools.
    # Neither approach is hinted at — the model decides.

    {
        "id": "ggb_01_angle_at_vertex",
        "category": "E_geogebra",
        "nl": (
            "Triangle ABC has vertices A=(0,0), B=(4,0), C=(1,3). "
            "What is the measure of angle BAC in degrees?"
        ),
        # cos(BAC) = AB·AC / (|AB||AC|) = 4/(4*sqrt(10)) = 1/sqrt(10)
        "expected": math.degrees(math.acos(1 / math.sqrt(10))),
        "expected_type": "numerical",
        "tolerance": 0.1,
    },
    {
        "id": "ggb_02_median_length",
        "category": "E_geogebra",
        "nl": (
            "In triangle ABC with A=(0,0), B=(6,0), C=(2,4), "
            "what is the length of the median from vertex A to the midpoint of BC?"
        ),
        # M = (4,2), AM = sqrt(16+4) = sqrt(20)
        "expected": math.sqrt(20),
        "expected_type": "numerical",
        "tolerance": 0.01,
    },
    {
        "id": "ggb_03_midsegment_length",
        "category": "E_geogebra",
        "nl": (
            "Triangle ABC has vertices A=(0,4), B=(−3,0), C=(3,0). "
            "D and E are the midpoints of AB and AC respectively. "
            "What is the length of DE?"
        ),
        # D=(-1.5,2), E=(1.5,2), DE=3
        "expected": 3.0,
        "expected_type": "numerical",
        "tolerance": 0.001,
    },
    {
        "id": "ggb_04_perpendicular_chord",
        "category": "E_geogebra",
        "nl": (
            "A chord of the circle with center O=(0,0) and radius 5 "
            "passes through interior point P=(3,0) and is perpendicular to OP. "
            "What is the length of this chord?"
        ),
        # d(O,chord)=3; half-chord=sqrt(25−9)=4; chord=8
        "expected": 8.0,
        "expected_type": "numerical",
        "tolerance": 0.01,
    },
    {
        "id": "ggb_05_tangent_length",
        "category": "E_geogebra",
        "nl": (
            "From external point P=(7,0), a tangent is drawn to the circle "
            "centered at O=(0,0) with radius 3. "
            "What is the length of the tangent segment from P to the point of tangency?"
        ),
        # PT = sqrt(OP² − r²) = sqrt(49−9) = sqrt(40)
        "expected": math.sqrt(40),
        "expected_type": "numerical",
        "tolerance": 0.01,
    },
]


# ── Lazy GeoGebra init ────────────────────────────────────────────────────────

class _LazyGGB:
    """
    Starts GeoGebra (Selenium) only on the first call to .get().
    Zero cost for problems the model solves without any tool calls.
    """
    def __init__(self):
        self._ctx = None
        self._ggb = None
        self._was_used = False   # never reset by close()

    def get(self):
        if self._ggb is None:
            print("  [GGB] Starting Selenium…")
            self._ctx = GeoGebraAPI(mode="selenium", headless=True)
            self._ggb = self._ctx.__enter__()
            self._ggb.reset()
            self._was_used = True
        return self._ggb

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._ggb = None
        # _was_used intentionally preserved after close

    @property
    def used(self) -> bool:
        return self._was_used


# ── Canvas helper ─────────────────────────────────────────────────────────────

def build_rich_canvas(ggb) -> dict:
    """
    Return a compact canvas snapshot: {name: {type, x?, y?, val?}}.
    Points include coordinates; other objects include their value_string.
    """
    objs = ggb.get_construction_state().get("objects", {})
    canvas = {}
    for name, info in objs.items():
        t = info.get("type", "?")
        entry: dict = {"type": t}
        if t == "point":
            if "x" in info:
                entry["x"] = round(info["x"], 4)
            if "y" in info:
                entry["y"] = round(info["y"], 4)
        vs = info.get("value_string")
        if vs:
            entry["val"] = vs
        canvas[name] = entry
    return canvas


# ── Answer parsing + validation ───────────────────────────────────────────────

def parse_answer(text: str) -> dict | None:
    """Extract ANSWER: {...} from model text (anywhere in the string)."""
    m = re.search(r"ANSWER:\s*(\{.*?\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def eval_symbolic(expr: str) -> float:
    """Numerically evaluate a symbolic string.
    Supports: sqrt, pi, sin/cos/tan, asin/acos/atan/atan2, log, exp, abs.
    """
    safe = str(expr).strip().replace("^", "**")
    ns = {
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin,  "cos": math.cos,  "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "atan2": math.atan2, "log": math.log, "exp": math.exp,
        "abs": abs, "__builtins__": {},
    }
    return eval(safe, ns)  # noqa: S307


def validate(answer: dict | None, prob: dict) -> tuple[bool, str]:
    """Return (passed, detail_string)."""
    if answer is None:
        return False, "no answer emitted"

    t    = answer.get("type", "")
    v    = answer.get("value")
    exp  = prob["expected"]
    tol  = prob.get("tolerance", 0.001)

    if t == "boolean":
        if isinstance(v, bool):
            actual = v
        else:
            actual = str(v).strip().lower() in ("true", "1")
        return actual == exp, f"{actual} vs {exp}"

    # numerical and symbolic are both compared numerically —
    # the model may legitimately return either form for a numeric answer.
    if t in ("numerical", "symbolic"):
        # evaluate to float (symbolic string or plain number)
        try:
            if t == "symbolic" and isinstance(v, str):
                actual = eval_symbolic(v)
            else:
                actual = float(v)
        except Exception as e:
            return False, f"cannot evaluate {v!r}: {e}"
        # use expected_numeric if provided, otherwise fall back to expected
        exp_num = float(prob.get("expected_numeric", exp))
        ok = abs(actual - exp_num) <= tol
        tag = f"eval({v!r})" if t == "symbolic" else str(round(actual, 4))
        return ok, f"{tag}={actual:.4f} vs {exp_num:.4f}  (tol={tol})"

    if t == "coordinate":
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return False, f"expected [x,y] list, got {v!r}"
        try:
            ax, ay = float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return False, f"cannot parse coordinate {v!r}"
        ex, ey = exp[0], exp[1]
        ok = abs(ax - ex) <= tol and abs(ay - ey) <= tol
        return ok, f"({ax:.3f},{ay:.3f}) vs ({ex:.3f},{ey:.3f})"

    return False, f"unknown answer type {t!r}"


# ── Query loop ────────────────────────────────────────────────────────────────

def run_query(client, prob: dict) -> dict:
    """
    Run one problem.  GeoGebra is started lazily — only if the model calls a tool.
    Returns a result dict with answer, pass/fail, tool counts, and token metrics.
    """
    from google.genai import types

    tools  = build_gemini_tools()
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prob["nl"])],
        )
    ]

    lazy      = _LazyGGB()
    answer_raw   = None
    succeeded = failed = total = 0
    token_in = token_out = token_total = 0
    t_submit  = time.perf_counter()
    t_first   = t_final = None

    # Track canvas names seen before this turn to report only new objects
    _known_names: set[str] = set()

    try:
        for turn in range(MAX_TURNS):
            print(f"  [LLM] turn {turn + 1} — waiting for model…", flush=True)
            response = client.models.generate_content(
                model=MODEL, contents=contents, config=config,
            )
            if t_first is None:
                t_first = time.perf_counter()

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                in_tok  = getattr(usage, "prompt_token_count",     None) or 0
                out_tok = getattr(usage, "candidates_token_count", None) or 0
                tot_tok = getattr(usage, "total_token_count",      None) or (in_tok + out_tok)
                token_in    += int(in_tok)
                token_out   += int(out_tok)
                token_total += int(tot_tok)

            model_parts = response.candidates[0].content.parts
            contents.append(types.Content(role="model", parts=model_parts))

            # Check text parts for ANSWER
            text_parts = [p.text for p in model_parts if getattr(p, "text", None)]
            full_text  = " ".join(text_parts)
            parsed = parse_answer(full_text)
            if parsed:
                answer_raw = parsed
                t_final = time.perf_counter()
                print(f"  [ANSWER] {json.dumps(parsed)}")
                break

            fn_calls = [p.function_call for p in model_parts if getattr(p, "function_call", None)]

            if not fn_calls:
                print(f"  [LLM] {full_text[:160].strip()}")
                t_final = time.perf_counter()
                break

            # Execute tool calls — initialise GeoGebra on first call
            tool_responses = []
            for fc in fn_calls:
                total += 1
                args = dict(fc.args)
                ggb  = lazy.get()          # no-op after first call

                if fc.name.startswith("query_"):
                    cmd, ok, err, value = execute_query_tool(ggb, fc.name, args)
                    status = "OK  " if ok else "FAIL"
                    print(f"    [{status}] {cmd}  value={value}" + (f"  -> {err}" if not ok else ""))
                    if ok:
                        succeeded += 1
                        # Full canvas only for query results (model needs value in context)
                        canvas = build_rich_canvas(ggb)
                        _known_names.update(canvas.keys())
                        resp = {"command": cmd, "success": True, "error": "",
                                "value": value, "canvas": canvas}
                        print(f"           canvas: {json.dumps(canvas)}")
                    else:
                        failed += 1
                        resp = {"command": cmd, "success": False, "error": err}
                else:
                    cmd, ok, err = execute_geogebra_tool(ggb, fc.name, args)
                    status = "OK  " if ok else "FAIL"
                    print(f"    [{status}] {cmd}" + (f"  -> {err}" if not ok else ""))
                    if ok:
                        succeeded += 1
                        # For draw tools: report only objects added by this call
                        full_canvas = build_rich_canvas(ggb)
                        new_objs = {n: v for n, v in full_canvas.items()
                                    if n not in _known_names}
                        _known_names.update(full_canvas.keys())
                        resp = {"command": cmd, "success": True, "error": "",
                                "new_objects": new_objs}
                        print(f"           new: {json.dumps(new_objs)}")
                    else:
                        failed += 1
                        resp = {"command": cmd, "success": False, "error": err}

                tool_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(name=fc.name, response=resp)
                    )
                )

            contents.append(types.Content(role="user", parts=tool_responses))

    finally:
        lazy.close()

    t_end = time.perf_counter()
    ok_flag, detail = validate(answer_raw, prob)

    metrics = {
        "input_tokens":  token_in,
        "output_tokens": token_out,
        "total_tokens":  token_total,
        "ggb_used":      lazy.used,
        "ttft_sec":      round(t_first - t_submit, 3) if t_first else None,
        "t_final_sec":   round(t_final - t_submit, 3) if t_final else None,
        "t_last_sec":    round(t_end   - t_submit, 3),
    }
    return {
        "answer":      answer_raw,
        "passed":      ok_flag,
        "detail":      detail,
        "tools_ok":    succeeded,
        "tools_fail":  failed,
        "tools_total": total,
        "metrics":     metrics,
    }


# ── Problem runner ────────────────────────────────────────────────────────────

def run_problem(client, prob: dict) -> dict:
    cat = prob["category"]
    print(f"\n{'='*64}")
    print(f"  [{cat}]  {prob['id']}")
    print(f"  {prob['nl'][:100]}{'...' if len(prob['nl']) > 100 else ''}")
    print(f"{'='*64}")

    result = run_query(client, prob)

    passed = result["passed"]
    tag    = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {tag} {result['detail']}")
    m = result["metrics"]
    ggb_tag = "GGB" if m["ggb_used"] else "direct"
    print(f"  Approach: {ggb_tag}  |  "
          f"Tools: {result['tools_ok']} OK / {result['tools_fail']} fail of {result['tools_total']}")
    print(f"  Tokens: {m['input_tokens']}/{m['output_tokens']}/{m['total_tokens']}  "
          f"TTFT={m['ttft_sec']}s  t_final={m['t_final_sec']}s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

CATEGORY_LABELS = {
    "A_boolean":   "A  Boolean      (direct reasoning)",
    "B_numerical": "B  Numerical    (formula/computation)",
    "C_symbolic":  "C  Symbolic     (sqrt / pi / fractions)",
    "D_coordinate":"D  Coordinate   (points / slopes)",
    "E_geogebra":  "E  Mixed        (concrete geometry, any approach)",
}


def main():
    print("\n" + "=" * 64)
    print("  LLM Geometry Query Pipeline")
    print(f"  Model     : {MODEL}")
    print(f"  Problems  : {len(PROBLEMS)}")
    print("=" * 64)

    try:
        from symbolic.utils.env_loader import make_genai_client
        client = make_genai_client()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    summary: list[dict] = []
    for prob in PROBLEMS:
        log_path = OUTDIR / "log" / f"{prob['id']}.txt"
        with TeeLog(log_path):
            result = run_problem(client, prob)
        summary.append({
            "id":       prob["id"],
            "category": prob["category"],
            "passed":   result["passed"],
            "detail":   result["detail"],
            "ggb_used": result["metrics"]["ggb_used"],
            "tokens":   result["metrics"]["total_tokens"],
            "t_final":  result["metrics"]["t_final_sec"],
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")

    by_cat: dict[str, list] = {}
    for s in summary:
        by_cat.setdefault(s["category"], []).append(s)

    total_pass = total_all = 0
    for cat, label in CATEGORY_LABELS.items():
        items = by_cat.get(cat, [])
        if not items:
            continue
        n_pass = sum(1 for s in items if s["passed"])
        total_pass += n_pass
        total_all  += len(items)
        print(f"\n  {label}  ({n_pass}/{len(items)})")
        for s in items:
            tag     = "[+]" if s["passed"] else "[-]"
            approach = "GGB" if s["ggb_used"] else "dir"
            print(f"    {tag} [{approach}] {s['id']:33s}  {s['detail']}")

    print(f"\n{'='*64}")
    pct = total_pass / total_all * 100 if total_all else 0
    print(f"  Total: {total_pass}/{total_all} passed ({pct:.0f}%)")
    print(f"  Logs → {OUTDIR / 'log'}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
