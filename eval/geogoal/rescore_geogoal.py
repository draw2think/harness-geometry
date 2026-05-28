"""Offline rescore: re-evaluate T_i from stored canvas.json + update result.json.

No LLM calls, no GeoGebra; pure Python (~30s over 256 problems).

Use after parser fixes (regex changes, new atom support) to refresh
FA_local without re-running the LLM construction.
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parent
REPO_ROOT = EVAL_ROOT.parent
for p in (THIS_DIR, EVAL_ROOT, REPO_ROOT):
    sys.path.insert(0, str(p))

import json
import pandas as pd

import geogoal_tiparser as tp


ROOT = EVAL_ROOT / "geogoal"
SLUG = "gemini-3-flash-preview@medium"
DATA = Path("/data/geogoal_sgvr/data/test-00000-of-00001.parquet")


def main():
    df = pd.read_parquet(DATA)
    q_map = dict(zip(df["id"], df["question"]))
    ans_map = {}
    import ast
    for _id, raw in zip(df["id"], df["answer"]):
        try:
            ans_map[_id] = ast.literal_eval(raw)
        except Exception:
            ans_map[_id] = []

    dirs = sorted(d for d in ROOT.iterdir() if d.name.startswith("geogal_") and d.is_dir())
    print(f"Rescoring {len(dirs)} problems...")

    new_fa_sum = old_fa_sum = total = 0
    changed = 0
    for d in dirs:
        qid = d.name
        canvas_path = d / f"{SLUG}_canvas.json"
        ti_path = d / f"{SLUG}_ti.json"
        res_path = d / f"{SLUG}_result.json"

        if not (canvas_path.exists() and res_path.exists()):
            continue

        canvas = json.loads(canvas_path.read_text()).get("coords", {})
        # coords stored as [x, y] list — convert back to tuple
        coords = {k: tuple(v) for k, v in canvas.items()}

        question = q_map.get(qid, "")
        gt = ans_map.get(qid, [])

        # Re-evaluate all T_i from current parser
        preds = tp.evaluate_all(question, coords)
        cmp_new = tp.compare_to_gt(preds, gt)

        # Load old FA for diff
        try:
            old_cmp = json.loads(res_path.read_text()).get("FA_local") or {}
            old_match = int(old_cmp.get("match", 0) or 0)
        except Exception:
            old_match = 0

        # Write updated ti.json and result.json
        ti_path.write_text(json.dumps(
            {"predictions": preds, "compare": cmp_new},
            indent=2, ensure_ascii=False, default=str,
        ))
        res = json.loads(res_path.read_text())
        res["FA_local"] = cmp_new
        res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))

        new_match = int(cmp_new.get("match", 0) or 0)
        tot = int(cmp_new.get("total", 0) or 0)
        new_fa_sum += new_match
        old_fa_sum += old_match
        total += tot
        if new_match != old_match:
            changed += 1

    print(f"\nRescore complete.")
    print(f"  Problems with FA_local changed: {changed}/{len(dirs)}")
    if total:
        print(f"  Old FA micro: {old_fa_sum}/{total} = {100*old_fa_sum/total:.2f}%")
        print(f"  New FA micro: {new_fa_sum}/{total} = {100*new_fa_sum/total:.2f}%")
        print(f"  Δ FA match  : +{new_fa_sum - old_fa_sum}")


if __name__ == "__main__":
    main()
