"""GeoGoal-SGVR predicate verifier.

Purpose
-------
Empirical validation of the Engine Faithfulness claim (paper §3.2): checks
whether the Draw2Think CT canvas, after the LLM completes its construction,
satisfies every ground-truth predicate specified by SGVR's `solution_FL`.

Pipeline
--------
    solution_FL (AG-style FL predicates)
        -> parse_solution_fl: tier-classified predicate list
        -> _expand_eqratio3: SGVR-shorthand -> Newclid atomic form
        -> Newclid predicate_from_construction + Predicate.check_numerical
        -> 3-tier Skeleton Rate (SR) + Integrity Rate (IR)

Newclid (3.0.1, local clone) provides the numerical check layer; this file
is the FL parser + registry builder + tier aggregator. Point the NEWCLID_SRC
environment variable at your local Newclid clone's src/ directory.
"""

from __future__ import annotations

import os
import re
import sys
import types
import typing
from pathlib import Path

# ── Newclid bootstrap (Py 3.10 compat + skip deductors/symengine) ─────────
# Point NEWCLID_SRC at your local Newclid clone's src/ directory.
_NEWCLID_SRC = Path(os.environ.get("NEWCLID_SRC", "newclid/src"))


def _bootstrap_newclid() -> None:
    """Patch typing.Self for Py<3.11, then stub out `newclid/__init__.py`
    so we import only the predicate subsystem (no deductors -> no symengine)."""
    if "newclid.predicates" in sys.modules:
        return
    if not hasattr(typing, "Self"):
        from typing_extensions import Self  # noqa
        typing.Self = Self  # type: ignore[attr-defined]
    if "newclid" not in sys.modules:
        pkg = types.ModuleType("newclid")
        pkg.__path__ = [str(_NEWCLID_SRC / "newclid")]  # type: ignore[attr-defined]
        sys.modules["newclid"] = pkg
    if str(_NEWCLID_SRC) not in sys.path:
        sys.path.insert(0, str(_NEWCLID_SRC))


_bootstrap_newclid()

from newclid.numerical.geometries import PointNum  # noqa: E402
from newclid.predicate_types import PredicateArgument  # noqa: E402
from newclid.predicates import PREDICATES, predicate_from_construction  # noqa: E402
from newclid.problem import PredicateConstruction  # noqa: E402
from newclid.symbols.points_registry import Point, PointsRegisty  # noqa: E402

_NEWCLID_TYPES: set[str] = {k.value for k in PREDICATES.keys()}


# ── Tier classification ──────────────────────────────────────────────────
# SGVR uses 30 tag labels; we aggregate to 3 tiers:
#   premise  : problem-statement constraints (perception fidelity)
#   numcheck : explicit length/angle/pythagoras value checks
#   derived  : rule-derived conclusions (reasoning fidelity)
_TIER_PREMISE = {"Premise"}
_TIER_NUMCHECK = {"Numerical Check", "Pythagoras Verification"}
_TIER_DERIVED_NAMED = {"Ratio Chasing", "Angle Chasing"}
_RULE_ID_RE = re.compile(r"^r\d+\b")


def _tier_of(label: str) -> str | None:
    label = label.strip()
    if label in _TIER_PREMISE:
        return "premise"
    if label in _TIER_NUMCHECK:
        return "numcheck"
    if label in _TIER_DERIVED_NAMED or _RULE_ID_RE.match(label):
        return "derived"
    return None


# ── FL line parser ───────────────────────────────────────────────────────
_TAG_RE = re.compile(r"\(\s*([^)]{2,80}?)\s*\)\s*=>")
_PRED_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]")
_FRAC_RE = re.compile(r"Fraction\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def _normalize_name(name: str) -> str:
    """CamelCase → snake_case.  `PythagoreanPremises` → `pythagorean_premises`."""
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


def _parse_args(args_str: str) -> list[str]:
    """Split comma-separated args, respecting parens so `Fraction(6, 1)` stays whole.
    Convert `Fraction(n, m)` literals to `n/m` (Newclid expects `str_to_fraction`-friendly)."""
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in args_str:
        if ch in "([":
            depth += 1
            cur += ch
        elif ch in ")]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [_FRAC_RE.sub(r"\1/\2", a) for a in out]


def parse_solution_fl(solution_fl: str) -> list[dict]:
    """Extract tier-classified predicates from SGVR's `solution_FL`.

    Only predicates appearing AFTER a `(TierTag)=>` token on a line are kept;
    predicates before the tag are justification references to earlier tiers.

    Returns:
        List of dicts: {tier, name, args, raw}
            tier  ∈ {'premise', 'numcheck', 'derived'}
            name  is snake_case normalized
            args  is a list of string args (Fraction literals normalized)
    """
    out: list[dict] = []
    for line in solution_fl.splitlines():
        m = _TAG_RE.search(line)
        if m is None:
            continue
        tier = _tier_of(m.group(1))
        if tier is None:
            continue
        rest = line[m.end():]
        for pm in _PRED_RE.finditer(rest):
            name = _normalize_name(pm.group(1))
            args = _parse_args(pm.group(2))
            out.append({
                "tier": tier,
                "name": name,
                "args": args,
                "raw": pm.group(0),
            })
    return out


# ── eqratio3 expansion (SGVR shorthand -> Newclid r07 Thales atomic form) ─
def _expand_eqratio3(args: list[str]) -> list[dict]:
    """SGVR format [O, O, A, B, C, D] → 1 simtri + 2 eqratio (Newclid r07).

    Per AlphaGeometry rules.txt: `para A B C D, coll O A C, coll O B D =>
    eqratio3 A B C D O O` (apex at end); SGVR reverses arg order so apex is
    doubled at the front. We emit the Newclid atomic form of Thales I."""
    if len(args) != 6 or args[0] != args[1]:
        return []
    o, _, a, b, c, d = args
    return [
        {"name": "simtri",  "args": [o, a, b, o, c, d]},
        {"name": "eqratio", "args": [o, a, c, a, o, b, b, d]},
        {"name": "eqratio", "args": [o, c, a, c, o, d, b, d]},
    ]


# ── PointsRegistry builder ───────────────────────────────────────────────
def build_registry(coords: dict[str, tuple[float, float]]) -> PointsRegisty:
    """Construct Newclid PointsRegisty from canvas point name → (x, y).

    Point names are lowercased; Newclid's canonicalization uses that form."""
    reg = PointsRegisty()
    for name, (x, y) in coords.items():
        pt = Point(name=PredicateArgument(name.lower()),
                   num=PointNum(x=float(x), y=float(y)))
        reg.add_point(pt)
    return reg


# ── Single predicate check ───────────────────────────────────────────────
def _check_one(name: str, args: list[str], reg: PointsRegisty) -> str:
    """Returns one of: 'true', 'false', 'missing-point', 'unsupported'."""
    if name not in _NEWCLID_TYPES:
        return "unsupported"
    try:
        pc = PredicateConstruction(string=f"{name} {' '.join(args)}")
    except ValueError:
        return "unsupported"
    try:
        pred = predicate_from_construction(pc, reg)
    except ValueError as exc:
        if "find point" in str(exc).lower():
            return "missing-point"
        return "unsupported"
    if pred is None:
        return "unsupported"
    try:
        return "true" if pred.check_numerical() else "false"
    except Exception:
        return "unsupported"


# ── Main entry ───────────────────────────────────────────────────────────
def verify(solution_fl: str,
           canvas_coords: dict[str, tuple[float, float]],
           include_details: bool = False) -> dict:
    """Verify all GT predicates against the given canvas.

    Args:
        solution_fl: SGVR's `solution_FL` string.
        canvas_coords: {point_name: (x, y)} queried from the CT canvas.
        include_details: if True, include per-predicate result list.

    Returns dict with keys:
        total/passed/missing_point/unsupported: per-tier counters
        SR: per-tier skeleton rate + overall
        IR: bool (all predicates across tiers pass)
        per_predicate: (optional) detailed list
    """
    reg = build_registry(canvas_coords)
    preds = parse_solution_fl(solution_fl)

    # Expand eqratio3 before dispatch
    expanded: list[dict] = []
    for p in preds:
        if p["name"] == "eqratio3":
            subs = _expand_eqratio3(p["args"])
            for sub in subs:
                expanded.append({
                    "tier": p["tier"],
                    "name": sub["name"],
                    "args": sub["args"],
                    "raw": p["raw"] + "(expanded)",
                })
        else:
            expanded.append(p)

    tiers = ("premise", "numcheck", "derived")
    total = {t: 0 for t in tiers}
    passed = {t: 0 for t in tiers}
    missing = {t: 0 for t in tiers}
    unsup = {t: 0 for t in tiers}
    details: list[dict] = []

    for p in expanded:
        result = _check_one(p["name"], p["args"], reg)
        t = p["tier"]
        total[t] += 1
        if result == "true":
            passed[t] += 1
        elif result == "missing-point":
            missing[t] += 1
        elif result == "unsupported":
            unsup[t] += 1
        if include_details:
            details.append({**p, "result": result})

    def _sr(t: str) -> float:
        return passed[t] / total[t] if total[t] else 1.0

    tot_count = sum(total.values())
    tot_pass = sum(passed.values())
    ir = all(passed[t] == total[t] for t in tiers if total[t] > 0)

    result: dict = {
        "total": total,
        "passed": passed,
        "missing_point": missing,
        "unsupported": unsup,
        "SR": {t: _sr(t) for t in tiers} | {
            "overall": tot_pass / tot_count if tot_count else 1.0
        },
        "IR": ir,
    }
    if include_details:
        result["per_predicate"] = details
    return result


def coverage_report(solution_fl: str) -> dict:
    """Diagnostic: per-predicate-name counts + Newclid support status."""
    from collections import Counter

    preds = parse_solution_fl(solution_fl)
    counts: Counter[str] = Counter()
    for p in preds:
        counts[p["name"]] += 1

    unsupported: list[tuple[str, int]] = []
    by_name: dict[str, dict] = {}
    for name, cnt in counts.items():
        is_supported = name in _NEWCLID_TYPES or name == "eqratio3"
        by_name[name] = {"count": cnt, "supported": is_supported}
        if not is_supported:
            unsupported.append((name, cnt))

    return {
        "total": sum(counts.values()),
        "by_name": by_name,
        "unsupported": unsupported,
        "newclid_types": sorted(_NEWCLID_TYPES),
    }
