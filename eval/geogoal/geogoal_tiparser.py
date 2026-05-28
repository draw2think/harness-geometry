"""GeoGoal-SGVR T_i expression parser + evaluator.

Purpose
-------
Extract the T_0, T_1, ..., T_N expressions from a SGVR question and
evaluate each of them on a canvas dict {point_name: (x, y)}, returning
numerical values that can be compared against the ground-truth answer list.

This replaces "LLM emits ANSWER line" with deterministic local computation,
aligning with the Engine Faithfulness claim: we do not trust the LLM's
arithmetic — only its construction fidelity.

Atoms handled
-------------
  |AB|                     length
  AB  (bare, in arithmetic) length (fallback when no |...|)
  angle(AB, CD)            directed angle between lines AB and CD (deg)
  \\angle(AB, CD)           same, LaTeX form
  area_triangle(A, B, C)   area of triangle ABC
  area(△A B C)              same, Unicode triangle symbol

Arithmetic: +, -, *, /, parentheses, integer/decimal literals.
Modifier:   "(in degrees, mod 180)" — final value reduced to [0, 180).
"""

from __future__ import annotations

import math
import re

# ── Regex ──────────────────────────────────────────────────────────────────
_TI_SPLIT_RE = re.compile(r"\bT?(\d+)\s*=")
# Accept all these mod180 qualifier variants:
#   "(in degrees, mod 180)"     "(degrees mod 180)"
#   "in degrees (mod 180)"      "(in degrees) mod 180"
#   "in degrees modulo 180"     "degrees, modulo 180"
#   etc. — crucially allow arbitrary whitespace, commas, and/or parens between
#   'degrees' and 'mod', since SGVR question generator uses many variants.
_MOD180_RE = re.compile(
    r"\(?\s*(?:in\s+)?degrees\s*[,\s\)\(]*\(?\s*(?:mod|modulo)\s*180\s*\)?",
    re.IGNORECASE,
)
_MOD180_SHORT_RE = re.compile(r"\(?\s*(?:mod|modulo)\s*180\s*\)?", re.IGNORECASE)
# Standalone "in degrees" (without mod) — strip but don't treat as mod180
_DEGREES_ONLY_RE = re.compile(r"\(?\s*(?:in\s+)?degrees\s*\)?", re.IGNORECASE)
_AREA_UNI_RE = re.compile(
    r"area\s*\(\s*△\s*([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])\s*\)", re.UNICODE
)
_AREA_LATEX_RE = re.compile(
    r"area\s*\(\s*\\triangle\s*([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])\s*\)",
    re.IGNORECASE,
)
_AREA_TRI_RE = re.compile(
    r"area(?:_triangle)?\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)"
)
# Natural-language variant: "area of triangle A B C" (with/without commas/spaces/brackets)
_AREA_NL_RE = re.compile(
    r"area\s+of\s+triangle\s+([A-Za-z])\s*[,\s]*([A-Za-z])\s*[,\s]*([A-Za-z])",
    re.IGNORECASE,
)
# Line-pair angle: angle(AB, CD)
_ANGLE_RE = re.compile(
    r"(?:∠|\\?angle)\s*\(\s*([A-Za-z])([A-Za-z])\s*,\s*([A-Za-z])([A-Za-z])\s*\)",
    re.UNICODE,
)
# Three-point angle: ∠ABC  →  angle at vertex B between rays BA and BC
_ANGLE_3PT_RE = re.compile(
    r"(?:∠|\\?angle)\s*([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])(?![A-Za-z_])",
    re.UNICODE,
)
_LEN_EXPLICIT_RE = re.compile(r"\|\s*([A-Za-z])\s*([A-Za-z])\s*\|")
_LEN_BARE_RE = re.compile(r"(?<![A-Za-z_0-9])([A-Za-z])([A-Za-z])(?![A-Za-z_0-9])")


# ── Coord lookup helper (case-insensitive) ─────────────────────────────────
def _lookup(name: str, coords: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    if name in coords:
        return coords[name]
    if name.lower() in coords:
        return coords[name.lower()]
    if name.upper() in coords:
        return coords[name.upper()]
    return None


def _length(a: str, b: str, coords: dict) -> float | None:
    pa, pb = _lookup(a, coords), _lookup(b, coords)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _angle_lines(a: str, b: str, c: str, d: str, coords: dict) -> float | None:
    """Directed angle between line AB and line CD, reduced to [0, 180) degrees."""
    pa, pb = _lookup(a, coords), _lookup(b, coords)
    pc, pd = _lookup(c, coords), _lookup(d, coords)
    if None in (pa, pb, pc, pd):
        return None
    v1 = (pb[0] - pa[0], pb[1] - pa[1])
    v2 = (pd[0] - pc[0], pd[1] - pc[1])
    if (v1[0] ** 2 + v1[1] ** 2) < 1e-12 or (v2[0] ** 2 + v2[1] ** 2) < 1e-12:
        return None
    ang1 = math.degrees(math.atan2(v1[1], v1[0]))
    ang2 = math.degrees(math.atan2(v2[1], v2[0]))
    return (ang2 - ang1) % 180


def _angle_three_point(a: str, b: str, c: str, coords: dict) -> float | None:
    """Angle at vertex B formed by rays BA and BC, in degrees [0, 180]."""
    pa, pb, pc = _lookup(a, coords), _lookup(b, coords), _lookup(c, coords)
    if None in (pa, pb, pc):
        return None
    v1 = (pa[0] - pb[0], pa[1] - pb[1])
    v2 = (pc[0] - pb[0], pc[1] - pb[1])
    if (v1[0] ** 2 + v1[1] ** 2) < 1e-12 or (v2[0] ** 2 + v2[1] ** 2) < 1e-12:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(abs(cross), dot))


def _area_triangle(a: str, b: str, c: str, coords: dict) -> float | None:
    pa, pb, pc = _lookup(a, coords), _lookup(b, coords), _lookup(c, coords)
    if None in (pa, pb, pc):
        return None
    return abs((pb[0] - pa[0]) * (pc[1] - pa[1])
               - (pc[0] - pa[0]) * (pb[1] - pa[1])) / 2.0


# ── T_i list extraction from question ──────────────────────────────────────
def parse_ti_list(question: str) -> list[tuple[int, str]]:
    """Return [(index, expr_string), ...] ordered by T_i index.

    Locates the `T0 = ... T1 = ...` block in the question, tolerates
    comma-inline and line-split formats, and trims trailing answer-list
    boilerplate ("Provide your answers...", "Compute and return...").
    """
    start = question.find("T0")
    if start < 0:
        return []
    end = len(question)
    # Any closing phrase after T_i list. Match newline OR sentence-end before verb.
    tail_re = re.compile(
        r"(?:\n|\.\s+)(Provide|Compute|Find|Explain|Give|Return|Output|Express|Determine)\b"
    )
    m = tail_re.search(question, start)
    if m:
        end = m.start()
    section = question[start:end]

    matches = list(_TI_SPLIT_RE.finditer(section))
    defs: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        expr_start = m.end()
        expr_end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        expr = section[expr_start:expr_end].strip()
        # trim trailing "," or "." or ";"
        expr = expr.rstrip(",.; \t\n")
        # Strip redundant leading "T = " that appears in some source questions
        # (e.g. "T0 = T = |ac|/|bc|").
        expr = re.sub(r"^T\s*=\s*", "", expr)
        defs.append((idx, expr))
    defs.sort(key=lambda x: x[0])
    return defs


# ── Single T_i evaluator ───────────────────────────────────────────────────
def evaluate_ti(expr: str, coords: dict[str, tuple[float, float]]) -> float | None:
    """Evaluate one T_i expression.  Returns float or None on failure."""
    # Detect modifier before stripping
    mod180 = bool(_MOD180_RE.search(expr) or _MOD180_SHORT_RE.search(expr))
    expr = _MOD180_RE.sub("", expr)
    expr = _MOD180_SHORT_RE.sub("", expr)
    # Strip standalone "in degrees" qualifier (without mod), e.g. "angle(..) in degrees"
    expr = _DEGREES_ONLY_RE.sub("", expr)
    # Normalize Unicode minus (−, U+2212) and multiplication (×, U+00D7) so eval works
    expr = expr.replace("−", "-").replace("–", "-").replace("—", "-").replace("×", "*")

    def _sub_area_uni(m):
        v = _area_triangle(m.group(1), m.group(2), m.group(3), coords)
        return f"({v})" if v is not None else "None"

    def _sub_area_tri(m):
        v = _area_triangle(m.group(1), m.group(2), m.group(3), coords)
        return f"({v})" if v is not None else "None"

    def _sub_angle(m):
        v = _angle_lines(m.group(1), m.group(2), m.group(3), m.group(4), coords)
        return f"({v})" if v is not None else "None"

    def _sub_angle_3pt(m):
        v = _angle_three_point(m.group(1), m.group(2), m.group(3), coords)
        return f"({v})" if v is not None else "None"

    def _sub_len_explicit(m):
        v = _length(m.group(1), m.group(2), coords)
        return f"({v})" if v is not None else "None"

    def _sub_len_bare(m):
        v = _length(m.group(1), m.group(2), coords)
        return f"({v})" if v is not None else "None"

    # Strip bracket wrappers like "[area of triangle A B C]"
    expr = expr.replace("[", " ").replace("]", " ")

    # Order matters: more specific patterns first
    expr = _AREA_UNI_RE.sub(_sub_area_uni, expr)
    expr = _AREA_LATEX_RE.sub(_sub_area_tri, expr)  # \triangle form
    expr = _AREA_NL_RE.sub(_sub_area_tri, expr)     # "area of triangle A B C"
    expr = _AREA_TRI_RE.sub(_sub_area_tri, expr)    # area_triangle(A, B, C)
    expr = _ANGLE_RE.sub(_sub_angle, expr)          # line-pair: angle(AB, CD)
    expr = _ANGLE_3PT_RE.sub(_sub_angle_3pt, expr)  # three-point: ∠ABC
    expr = _LEN_EXPLICIT_RE.sub(_sub_len_explicit, expr)
    expr = _LEN_BARE_RE.sub(_sub_len_bare, expr)

    # Any remaining 'None' means a missing point: cannot evaluate
    if "None" in expr:
        return None

    # Guard: allow only arithmetic characters after substitution
    if not re.fullmatch(r"[\d\s\+\-\*/\(\)\.eE]*", expr.strip()):
        return None

    try:
        val = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None

    if val is None:
        return None
    # Note: do NOT apply `% 180` here. The mod180 qualifier is handled in
    # compare_to_gt via circular distance, so that GT values of literal 180
    # match predictions of 0/360 (lines are direction-agnostic mod 180).
    return float(val)


# ── Public: evaluate full T_i list against canvas ──────────────────────────
def evaluate_all(question: str,
                 coords: dict[str, tuple[float, float]]) -> list[dict]:
    """Evaluate every T_i in the question.  Returns ordered list of
    {index, expr, value, mod180} dicts; `value` is float or None if
    evaluation failed (missing point, unsupported syntax, ...).
    `mod180` is True iff the expression was qualified "(in degrees mod 180)"
    or equivalent — used for circular comparison in compare_to_gt."""
    out = []
    for idx, expr in parse_ti_list(question):
        is_mod180 = bool(_MOD180_RE.search(expr) or _MOD180_SHORT_RE.search(expr))
        out.append({
            "index": idx,
            "expr": expr,
            "value": evaluate_ti(expr, coords),
            "mod180": is_mod180,
        })
    return out


# ── GT compare (fraction-aware) ────────────────────────────────────────────
def compare_to_gt(predictions: list[dict],
                  gt_list: list[str],
                  tol: float = 0.01) -> dict:
    """Element-wise compare predicted T_i values against GT answer list.

    GT entries are strings; numeric parsing is Fraction-aware ('2/1', '65/23').
    Returns aggregate + per-index detail.
    """
    from fractions import Fraction

    def _parse_gt(s):
        s = str(s).strip()
        try:
            return float(Fraction(s))
        except (ValueError, ZeroDivisionError):
            try:
                return float(s)
            except ValueError:
                return None

    per_idx: list[dict] = []
    match = 0
    N = max(len(predictions), len(gt_list))
    for i in range(N):
        pred_row = predictions[i] if i < len(predictions) else None
        pv = pred_row["value"] if pred_row else None
        gv = _parse_gt(gt_list[i]) if i < len(gt_list) else None
        mod180 = bool(pred_row and pred_row.get("mod180"))
        if pv is None or gv is None:
            ok = False
        elif mod180:
            # Circular distance on the 180-cyclic number line: 0 ≡ 180.
            d = abs(pv - gv) % 180
            ok = min(d, 180 - d) < tol
        else:
            ok = abs(pv - gv) < tol
        if ok:
            match += 1
        per_idx.append({
            "index": i,
            "expr": (pred_row or {}).get("expr"),
            "pred": pv,
            "gt": gv,
            "ok": ok,
            "mod180": mod180,
        })
    return {
        "match": match,
        "total": len(gt_list),
        "pred_len": len(predictions),
        "per_idx": per_idx,
    }
