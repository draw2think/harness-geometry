"""Compute Save/Break × BL-think profile across the 8 benchmarks reported in the paper.

For every problem with both a BL and a CT result file, classify the pair as:
  BL_PASS_CT_PASS, SAVE (BL fail -> CT pass), BREAK (BL pass -> CT fail), BL_FAIL_CT_FAIL.

Then for SAVE and BREAK groups, summarise BL thinking-token usage:
  - Count of timeouts / empty responses (BL_think == 0)
  - Median / mean BL_think tokens for the rest
  - All numbers per benchmark + grand totals

Splits and per-benchmark thinking levels match the paper's Table~tab:main-results:
  - Geo3K        : test split only (601)
  - MathVerse    : Plane Geometry, Vision Only version (510)
  - MathVista    : 'geometry problem solving' subset (208)
  - PGPS9K       : 1000 problems = 910 @medium + 90 @high (hard subset, IDs from hard_test_ids.json)
  - OlympiadBench: @high (per exp_results.md)
  - Others       : @medium

Usage:  python eval/compute_save_break_bl_think.py
"""
import json
from pathlib import Path
from statistics import median, mean

EVAL_ROOT = Path("eval")
DATA_ROOT = Path("data")

MODEL = "gemini-3-flash-preview"


# ── Build per-benchmark "valid problem IDs" set (filtered to paper splits) ──

def ids_geo3k() -> set:
    """Geometry3K test split: 601 problem IDs."""
    test_dir = Path("/data/geometry3k/test")
    if test_dir.exists():
        return {p.name for p in test_dir.iterdir() if p.is_dir()}
    # Fallback: read the eval testmini list if any
    return None  # use directory listing


def ids_mathverse_plane_vo() -> set:
    """MathVerse Plane Geometry, Vision Only: 510 problems."""
    fp = DATA_ROOT / "mathverse" / "testmini.json"
    raw = json.load(open(fp))
    return {str(d["sample_index"]) for d in raw
            if d.get("metadata", {}).get("subject") == "Plane Geometry"
            and d.get("problem_version") == "Vision Only"}


def ids_mathvista_gps() -> set:
    """MathVista 'geometry problem solving' subset: 208 problems."""
    fp = Path("/data/mathvista/testmini.json")
    if not fp.exists():
        return None
    raw = json.load(open(fp))
    keep = set()
    for entry in raw:
        meta = entry.get("metadata", {})
        if "geometry problem solving" in meta.get("task", ""):
            keep.add(str(entry.get("pid", entry.get("id", ""))))
    return keep


def ids_pgps9k_hard() -> set:
    """PGPS9K hard subset: 90 problem IDs (these use @high)."""
    fp = EVAL_ROOT / "pgps9k" / "hard_test_ids.json"
    return set(json.load(open(fp)))


# (display name, eval-dir, level-default, special)
# special: dict with optional 'hard_ids' (set, evaluated at @high) and 'id_filter' (set)
BENCHMARKS = [
    ("GeoQA/UniGeo",  "unigeo",        "medium", None),
    ("PGPS9K",        "pgps9k",        "medium", {"hard_ids": ids_pgps9k_hard(),
                                                    "hard_level": "high"}),
    ("MathVista",     "mathvista",     "medium", {"id_filter": ids_mathvista_gps()}),
    ("Geo3K",         "geometry3k",    "medium", {"id_filter": ids_geo3k()}),
    ("GeoLaux",       "geolaux",       "medium", None),
    ("MathVerse",     "mathverse",     "medium", {"id_filter": ids_mathverse_plane_vo()}),
    ("GeoSketch",     "geosketch",     "medium", None),
    ("OlympiadBench", "olympiadbench", "high",   None),
]


def load_result(dir_path: Path, baseline: bool, level: str):
    """Return dict with keys passed (bool|None), think_tokens (int|None) for the given thinking level."""
    suffix = "_baseline_result.json" if baseline else "_result.json"
    fp = dir_path / f"{MODEL}@{level}{suffix}"
    if not fp.exists():
        return None
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    passed = d.get("passed")
    think = d.get("thought_tokens")
    if think is None:
        think = d.get("think_tokens")
    return {"passed": bool(passed) if passed is not None else None,
            "think": int(think) if think is not None else None}


def classify_pair(bl, ct):
    if bl is None or ct is None:
        return None
    bl_p = bool(bl.get("passed"))
    ct_p = bool(ct.get("passed"))
    if bl_p and ct_p:           return "BB"
    if (not bl_p) and ct_p:     return "SAVE"
    if bl_p and (not ct_p):     return "BREAK"
    return "FF"


def summarise_think(values):
    if not values:
        return {"n": 0, "n_zero": 0, "median": None, "mean": None}
    nz = sum(1 for v in values if v == 0)
    nonzero = [v for v in values if v > 0]
    return {
        "n": len(values),
        "n_zero": nz,
        "median": median(nonzero) if nonzero else 0,
        "mean": int(mean(nonzero)) if nonzero else 0,
    }


def main():
    rows = []
    grand_save_think, grand_break_think = [], []
    grand = {"n": 0, "bb": 0, "save": 0, "brk": 0, "ff": 0, "bl_pass": 0, "ct_pass": 0}

    for name, dir_name, default_level, special in BENCHMARKS:
        save_think, break_think = [], []
        n_total = n_bb = n_save = n_break = n_ff = 0
        n_bl_pass = n_ct_pass = 0
        n_unmatched = 0
        n_filtered = 0

        base = EVAL_ROOT / dir_name
        if not base.exists():
            print(f"  [warn] missing dir: {base}", flush=True)
            continue

        id_filter = special.get("id_filter") if special else None
        hard_ids = special.get("hard_ids") if special else None
        hard_level = special.get("hard_level") if special else None

        for case_dir in sorted(base.iterdir()):
            if not case_dir.is_dir():
                continue
            pid = case_dir.name

            # Apply per-benchmark split filter
            if id_filter is not None:
                if pid not in id_filter and pid.replace("prob_", "") not in id_filter:
                    n_filtered += 1
                    continue

            # Pick thinking level: hard subset uses hard_level, else default.
            # If the hard-level pair is incomplete (e.g. CT missing for one case),
            # fall back to the default level so the case is not silently dropped.
            level = default_level
            if hard_ids and pid in hard_ids:
                level = hard_level

            bl = load_result(case_dir, baseline=True, level=level)
            ct = load_result(case_dir, baseline=False, level=level)
            if (bl is None or ct is None) and level != default_level:
                # fall back to default level for both BL and CT
                bl = load_result(case_dir, baseline=True, level=default_level)
                ct = load_result(case_dir, baseline=False, level=default_level)
                if bl is not None and ct is not None:
                    print(f"  [info] {name}: {pid} fell back from @{level} to @{default_level} (hard-level pair incomplete)", flush=True)
            if bl is None or ct is None:
                n_unmatched += 1
                continue
            kind = classify_pair(bl, ct)
            if kind is None:
                n_unmatched += 1
                continue
            n_total += 1
            if kind == "BB":     n_bb += 1
            elif kind == "SAVE":
                n_save += 1
                save_think.append(bl["think"] or 0)
            elif kind == "BREAK":
                n_break += 1
                break_think.append(bl["think"] or 0)
            else:                n_ff += 1

            if bl.get("passed"):  n_bl_pass += 1
            if ct.get("passed"):  n_ct_pass += 1

        rows.append({
            "name": name, "n": n_total,
            "bb": n_bb, "save": n_save, "brk": n_break, "ff": n_ff,
            "bl_pass": n_bl_pass, "ct_pass": n_ct_pass,
            "save_think": summarise_think(save_think),
            "break_think": summarise_think(break_think),
        })
        grand_save_think.extend(save_think)
        grand_break_think.extend(break_think)
        grand["n"] += n_total
        grand["bb"] += n_bb
        grand["save"] += n_save
        grand["brk"] += n_break
        grand["ff"] += n_ff
        grand["bl_pass"] += n_bl_pass
        grand["ct_pass"] += n_ct_pass

        if n_unmatched or n_filtered:
            extra = f"{n_filtered} filtered" if n_filtered else ""
            extra2 = f"{n_unmatched} unmatched/incomplete" if n_unmatched else ""
            joined = ", ".join(s for s in (extra, extra2) if s)
            print(f"  [info] {name}: {joined}", flush=True)

    gs = summarise_think(grand_save_think)
    gb = summarise_think(grand_break_think)

    # ════════ Table 1: ACCURACY (matches main-results) ════════
    print()
    print("=" * 86)
    print("Table 1 — Accuracy (BL & CT) per benchmark — for tab:main-results")
    print("=" * 86)
    print(f"{'Benchmark':<14} {'N':>5} {'BL pass':>8} {'BL %':>7} {'CT pass':>8} {'CT %':>7} {'Δ (pp)':>8}")
    print("-" * 86)
    for r in rows:
        bl = 100 * r['bl_pass'] / max(1, r['n'])
        ct = 100 * r['ct_pass'] / max(1, r['n'])
        print(f"{r['name']:<14} {r['n']:>5} {r['bl_pass']:>8} {bl:>7.1f} "
              f"{r['ct_pass']:>8} {ct:>7.1f} {ct-bl:>+8.1f}")
    print("-" * 86)
    bl_pct = 100 * grand['bl_pass'] / max(1, grand['n'])
    ct_pct = 100 * grand['ct_pass'] / max(1, grand['n'])
    print(f"{'TOTAL':<14} {grand['n']:>5} {grand['bl_pass']:>8} {bl_pct:>7.1f} "
          f"{grand['ct_pass']:>8} {ct_pct:>7.1f} {ct_pct-bl_pct:>+8.1f}")

    # ════════ Table 2: OUTCOME TRANSITION (matches tab:outcome-transition) ════════
    print()
    print("=" * 92)
    print("Table 2 — Outcome transition per benchmark — for tab:outcome-transition")
    print("=" * 92)
    print(f"{'Benchmark':<14} {'N':>5} {'BB':>5} {'Save':>5} {'Break':>6} {'FF':>5} "
          f"{'Net':>6} {'Win':>6}")
    print("-" * 92)
    for r in rows:
        net = r['save'] - r['brk']
        win = r['save'] / r['brk'] if r['brk'] else float('inf')
        win_s = f"{win:.2f}" if r['brk'] else "∞"
        print(f"{r['name']:<14} {r['n']:>5} {r['bb']:>5} {r['save']:>5} {r['brk']:>6} "
              f"{r['ff']:>5} {net:>+6} {win_s:>6}")
    print("-" * 92)
    net = grand['save'] - grand['brk']
    win = grand['save'] / grand['brk'] if grand['brk'] else float('inf')
    print(f"{'TOTAL':<14} {grand['n']:>5} {grand['bb']:>5} {grand['save']:>5} {grand['brk']:>6} "
          f"{grand['ff']:>5} {net:>+6} {win:>6.2f}")

    # ════════ Table 3: BL_THINK PROFILE (Save vs Break) ════════
    print()
    print("=" * 110)
    print("Table 3 — BL thinking-token profile of Save vs Break — for tab:save-break-profile")
    print("=" * 110)
    print(f"{'Benchmark':<14} {'Save':>5} {'Save 0-tok':>11} {'Save median':>12} {'Save mean':>10}  "
          f"{'Break':>6} {'Break 0-tok':>12} {'Break median':>13} {'Break mean':>11}")
    print("-" * 110)
    for r in rows:
        s, b = r['save_think'], r['break_think']
        print(f"{r['name']:<14} {r['save']:>5} {s['n_zero']:>11} {(s['median'] or 0):>12} {(s['mean'] or 0):>10}  "
              f"{r['brk']:>6} {b['n_zero']:>12} {(b['median'] or 0):>13} {(b['mean'] or 0):>11}")
    print("-" * 110)
    print(f"{'TOTAL':<14} {grand['save']:>5} {gs['n_zero']:>11} {(gs['median'] or 0):>12} {(gs['mean'] or 0):>10}  "
          f"{grand['brk']:>6} {gb['n_zero']:>12} {(gb['median'] or 0):>13} {(gb['mean'] or 0):>11}")

    print()
    print("=" * 110)
    print("Headline numbers")
    print("-" * 110)
    if gb['median']:
        print(f"  Median BL_think (Save) / Median BL_think (Break) = {(gs['median'] or 0)/gb['median']:.2f}x")
    if grand['save']:
        print(f"  Save 0-tok rate: {gs['n_zero']}/{grand['save']} = {100*gs['n_zero']/grand['save']:.0f}%")
    if grand['brk']:
        print(f"  Break 0-tok rate: {gb['n_zero']}/{grand['brk']} = {100*gb['n_zero']/grand['brk']:.0f}%")
    print()


if __name__ == "__main__":
    main()
