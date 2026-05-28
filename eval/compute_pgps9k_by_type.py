"""Per-knowledge-type BL/CT pass counts on PGPS9K-test (1000 problems).

Uses the same merge convention as compute_save_break_bl_think.py:
  - default level: @medium
  - hard subset (90 IDs from hard_test_ids.json): @high
  - if @high pair is incomplete, fall back to @medium for that pid

Maps each pid to its 'type' field in data/PGPS9K/PGPS9K/test.json (30 categories,
identical to the labels in knowledge_point.txt). Outputs per-type pass counts and
groups them into Group 1 (CT>BL), Group 2 (CT=BL), Group 3 (CT<BL) so the result
slots straight into Writing/sections/appendix_cross_system.tex tab:pgps9k-type.

Usage:  python eval/compute_pgps9k_by_type.py
"""
import json
from pathlib import Path

EVAL_ROOT = Path("eval")
DATA_ROOT = Path("data")
MODEL = "gemini-3-flash-preview"

PGPS_DIR = EVAL_ROOT / "pgps9k"
HARD_IDS = set(json.load(open(PGPS_DIR / "hard_test_ids.json")))


def load_pid_to_type():
    fp = DATA_ROOT / "PGPS9K" / "PGPS9K" / "test.json"
    raw = json.load(open(fp))
    return {pid: v["type"].strip() for pid, v in raw.items()}


def load_result(case_dir, baseline, level):
    suffix = "_baseline_result.json" if baseline else "_result.json"
    fp = case_dir / f"{MODEL}@{level}{suffix}"
    if not fp.exists():
        return None
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    return d.get("passed")


def main():
    pid_to_type = load_pid_to_type()

    per_type = {}  # type -> {"n": int, "bl": int, "ct": int}
    for t in set(pid_to_type.values()):
        per_type[t] = {"n": 0, "bl": 0, "ct": 0}

    n_total = n_bl = n_ct = 0
    n_dropped = 0

    for case_dir in sorted(PGPS_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        pid = case_dir.name
        if pid not in pid_to_type:
            continue

        level = "high" if pid in HARD_IDS else "medium"
        bl = load_result(case_dir, baseline=True, level=level)
        ct = load_result(case_dir, baseline=False, level=level)
        if (bl is None or ct is None) and level != "medium":
            bl = load_result(case_dir, baseline=True, level="medium")
            ct = load_result(case_dir, baseline=False, level="medium")
        if bl is None or ct is None:
            n_dropped += 1
            continue

        t = pid_to_type[pid]
        per_type[t]["n"] += 1
        if bl: per_type[t]["bl"] += 1
        if ct: per_type[t]["ct"] += 1
        n_total += 1
        if bl: n_bl += 1
        if ct: n_ct += 1

    print(f"\nTotal: N={n_total}  BL={n_bl}  CT={n_ct}  Δ={n_ct-n_bl:+d}  (dropped={n_dropped})\n")

    # Group rows
    g1, g2, g3 = [], [], []
    for t, s in per_type.items():
        row = {"type": t, "n": s["n"], "bl": s["bl"], "ct": s["ct"]}
        if s["ct"] > s["bl"]:   g1.append(row)
        elif s["ct"] == s["bl"]: g2.append(row)
        else:                    g3.append(row)

    # Sort: G1 by CT desc, G2 by N desc, G3 by BL desc (matches existing table)
    g1.sort(key=lambda r: (-r["ct"], -r["n"], r["type"]))
    g2.sort(key=lambda r: (-r["n"], r["type"]))
    g3.sort(key=lambda r: (-r["bl"], -r["n"], r["type"]))

    def emit(title, rows):
        print(f"── {title}  ({len(rows)} types) ──")
        print(f"  {'N':>4} {'BL':>4} {'CT':>4}  Type")
        sn = sb = sc = 0
        for r in rows:
            ceil_bl = "*" if r["bl"] == r["n"] else " "
            ceil_ct = "*" if r["ct"] == r["n"] else " "
            print(f"  {r['n']:>4} {r['bl']:>4}{ceil_bl}{r['ct']:>4}{ceil_ct} {r['type']}")
            sn += r["n"]; sb += r["bl"]; sc += r["ct"]
        print(f"  {sn:>4} {sb:>4} {sc:>4}  (group sum)\n")
        return sn, sb, sc

    s1 = emit("Group 1: CT > BL (positive transition; sort by CT desc)", g1)
    s2 = emit("Group 2: CT = BL (matched; sort by N desc)",                g2)
    s3 = emit("Group 3: CT < BL (negative transition; sort by BL desc)",   g3)
    print(f"Sums: N={s1[0]+s2[0]+s3[0]}  BL={s1[1]+s2[1]+s3[1]}  CT={s1[2]+s2[2]+s3[2]}")

    # Emit a LaTeX-ready 2-col table (15 rows × 2 columns = 30 entries) when groups
    # have correct sizes. We just dump groups sequentially; user can paste/adjust.
    print("\n── LaTeX rows (one per type, group order) ──")
    for r in g1 + g2 + g3:
        n, bl, ct = r["n"], r["bl"], r["ct"]
        bl_s = f"\\ceil{{{bl}}}" if bl == n else f"{bl}"
        ct_s = f"\\ceil{{{ct}}}" if ct == n else f"{ct}"
        if ct > bl:   rel = "\\rellt"
        elif ct < bl: rel = "\\relgt"
        else:         rel = "\\releq"
        # type names with & need escaping
        tname = r["type"].replace("and", "\\&") if " and " in r["type"] else r["type"]
        # match existing table's "&"-style for compound names
        tname = (r["type"]
                 .replace("Perimeter and Area", "Perimeter \\& Area")
                 .replace("Circumference and Area", "Circumference \\& Area")
                 .replace("Rhombus and Square", "Rhombus \\& Square")
                 .replace("Trapezoid and Kite", "Trapezoid \\& Kite"))
        print(f"  {tname} & {n} & {bl_s} & {rel} & {ct_s} \\\\")


if __name__ == "__main__":
    main()
