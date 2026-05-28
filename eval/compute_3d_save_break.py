"""Compute paired Save/Break transitions for the two 3D rows in Table 4.

This is intentionally separate from compute_save_break_bl_think.py, whose
appendix totals are defined over the eight planar benchmarks.

Rows:
  - MathVerse Solid Geometry, Vision Only (119), Gemini-3-Flash@medium
  - SolidGeo Level 3 (177), Gemini-3-Flash@high
"""

import json
from pathlib import Path


EVAL_ROOT = Path("eval")
MODEL = "gemini-3-flash-preview"


def ids_mathverse_solid_vo() -> set[str]:
    fp = Path("/data/mathverse/testmini.json")
    raw = json.load(open(fp))
    return {
        str(d["sample_index"])
        for d in raw
        if d.get("problem_version") == "Vision Only"
        and d.get("metadata", {}).get("subject") == "Solid Geometry"
    }


def ids_solidgeo_level3() -> set[str]:
    fp = Path("/data/solidgeo/SolidGeo.json")
    raw = json.load(open(fp))
    items = raw.values() if isinstance(raw, dict) else raw
    return {
        str(d.get("qa_id", ""))
        for d in items
        if d.get("complexity_level") == "Level 3"
    }


def load_result(case_dir: Path, baseline: bool, level: str):
    suffix = "_baseline_result.json" if baseline else "_result.json"
    fp = case_dir / f"{MODEL}@{level}{suffix}"
    if not fp.exists():
        return None
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    passed = d.get("passed")
    think = d.get("thought_tokens", d.get("think_tokens"))
    return {
        "passed": bool(passed) if passed is not None else None,
        "think": int(think) if think is not None else None,
    }


def compute(name: str, dir_name: str, level: str, id_filter: set[str]):
    bb = save = brk = ff = 0
    bl_pass = ct_pass = n = unmatched = filtered = 0
    base = EVAL_ROOT / dir_name

    for case_dir in sorted(base.iterdir()):
        if not case_dir.is_dir():
            continue
        pid = case_dir.name
        if pid not in id_filter and pid.replace("prob_", "") not in id_filter:
            filtered += 1
            continue
        bl = load_result(case_dir, baseline=True, level=level)
        ct = load_result(case_dir, baseline=False, level=level)
        if bl is None or ct is None:
            unmatched += 1
            continue

        bl_p = bool(bl["passed"])
        ct_p = bool(ct["passed"])
        n += 1
        bl_pass += int(bl_p)
        ct_pass += int(ct_p)
        if bl_p and ct_p:
            bb += 1
        elif (not bl_p) and ct_p:
            save += 1
        elif bl_p and (not ct_p):
            brk += 1
        else:
            ff += 1

    win = save / brk if brk else float("inf")
    print(f"{name:<18} N={n:>3}  BL={bl_pass:>3} ({100*bl_pass/max(n,1):4.1f})  "
          f"CT={ct_pass:>3} ({100*ct_pass/max(n,1):4.1f})  "
          f"BB={bb:>3} Save={save:>2} Break={brk:>2} FF={ff:>2} "
          f"Net={save-brk:+3d} Win={win:.2f}")
    if unmatched or filtered:
        print(f"  [info] {name}: {filtered} filtered, {unmatched} unmatched")


def main():
    compute("MathVerse-solid", "mathverse", "medium", ids_mathverse_solid_vo())
    compute("SolidGeo-hard", "solidgeo", "high", ids_solidgeo_level3())


if __name__ == "__main__":
    main()
