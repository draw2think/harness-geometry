"""Recompute every PGPS9K-specific number used in the paper under the unified
@medium + @high (hard subset) merge — same fallback convention as
compute_save_break_bl_think.py.

Outputs feed:
  • Writing/sections/04_experiments.tex tab:main-results            (BL/CT acc, think tokens)
  • Writing/sections/appendix_process_stats.tex tab:answer-source   (4 paper categories + Engine-Involved)
  • Writing/sections/appendix_process_stats.tex tab:turn-dist       (1/2/3/4/5+ + mean)
  • Writing/sections/appendix_process_stats.tex tab:token-efficiency (BL/CT think+output tokens, wall time)
  • Writing/sections/appendix_process_stats.tex tab:outcome-transition (BB/Save/Break/FF)

Usage:  python eval/compute_pgps9k_full_stats.py
"""
import json
import sys
from pathlib import Path
from statistics import mean
from collections import Counter

sys.path.insert(0, "eval")
from analyze_answer_source import classify  # noqa: E402

EVAL_DIR = Path("eval/pgps9k")
HARD_IDS = set(json.load(open(EVAL_DIR / "hard_test_ids.json")))
MODEL = "gemini-3-flash-preview"


def load(case_dir: Path, baseline: bool, level: str):
    suffix = "_baseline_result.json" if baseline else "_result.json"
    fp = case_dir / f"{MODEL}@{level}{suffix}"
    if not fp.exists():
        return None
    try:
        return json.load(open(fp))
    except Exception:
        return None


def main():
    n = bl_pass = ct_pass = 0
    bb = save = brk = ff = 0
    bl_think, ct_think = [], []
    bl_out, ct_out = [], []
    bl_time, ct_time, ct_per_turn = [], [], []
    turn_dist = Counter()
    turn_total = []
    cat_counts_correct = Counter()
    n_dropped = 0
    fallback_pids = []

    for case_dir in sorted(EVAL_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        pid = case_dir.name

        level = "high" if pid in HARD_IDS else "medium"
        bl = load(case_dir, baseline=True, level=level)
        ct = load(case_dir, baseline=False, level=level)
        if (bl is None or ct is None) and level != "medium":
            bl_m = load(case_dir, baseline=True, level="medium")
            ct_m = load(case_dir, baseline=False, level="medium")
            if bl_m is not None and ct_m is not None:
                bl, ct = bl_m, ct_m
                fallback_pids.append(pid)
        if bl is None or ct is None:
            n_dropped += 1
            continue

        n += 1
        bl_p = bool(bl.get("passed"))
        ct_p = bool(ct.get("passed"))
        bl_pass += int(bl_p)
        ct_pass += int(ct_p)

        # Outcome transition
        if bl_p and ct_p:           bb += 1
        elif (not bl_p) and ct_p:   save += 1
        elif bl_p and (not ct_p):   brk += 1
        else:                       ff += 1

        # Tokens (per-problem)
        bl_think.append(bl.get("thought_tokens") or 0)
        ct_think.append(ct.get("thought_tokens") or 0)
        bl_out.append(bl.get("output_tokens") or 0)
        ct_out.append(ct.get("output_tokens") or 0)

        # Wall time
        bl_t = bl.get("t_last_sec") or bl.get("t_final_sec") or 0
        ct_t = ct.get("t_last_sec") or ct.get("t_final_sec") or 0
        bl_time.append(bl_t)
        ct_time.append(ct_t)

        # Turn count and per-turn time. nt=0 means the first turn timed out
        # before completing, so we count it as one attempted turn (matches
        # the convention in tab:turn-dist of the @medium-only run).
        proc = ct.get("process") or {}
        nt = proc.get("total_turns") or 0
        nt_for_bucket = max(nt, 1)
        turn_total.append(nt_for_bucket)
        bucket = ("1" if nt_for_bucket == 1 else "2" if nt_for_bucket == 2 else
                  "3" if nt_for_bucket == 3 else "4" if nt_for_bucket == 4 else "5+")
        turn_dist[bucket] += 1
        ct_per_turn.append(ct_t / nt_for_bucket)

        # Answer-source classification (paper table reports over correct only)
        if ct_p:
            try:
                cat = classify(ct)
            except Exception:
                cat = "unknown"
            cat_counts_correct[cat] += 1

    print(f"\n=== PGPS9K @medium + @high (hard subset) merge ===")
    print(f"N = {n}   dropped = {n_dropped}   fell-back to @medium = {len(fallback_pids)} {fallback_pids}")
    print()

    # ─── §04 Experiments — tab:main-results ───
    print("── tab:main-results PGPS9K row ──")
    print(f"  BL pass: {bl_pass}/{n} = {100*bl_pass/n:.1f}%")
    print(f"  CT pass: {ct_pass}/{n} = {100*ct_pass/n:.1f}%")
    print(f"  Δ = {(100*ct_pass/n - 100*bl_pass/n):+.1f} pp")
    print(f"  BL think mean: {mean(bl_think):.0f}    CT think mean: {mean(ct_think):.0f}    ratio: {mean(ct_think)/mean(bl_think):.2f}")
    print()

    # ─── tab:outcome-transition row ───
    win = save / brk if brk else float("inf")
    print("── tab:outcome-transition PGPS9K row ──")
    print(f"  N={n}  BB={bb}  Save={save}  Break={brk}  FF={ff}  Net={save-brk:+d}  Win={win:.2f}")
    print()

    # ─── tab:turn-dist row ───
    total = sum(turn_dist.values())
    print("── tab:turn-dist PGPS9K row ──")
    print(f"  N={total}", end="  ")
    for k in ["1", "2", "3", "4", "5+"]:
        c = turn_dist[k]
        print(f"{k}={c} ({100*c/total:.0f}%)", end="  ")
    print(f"mean={mean(turn_total):.1f}")
    print()

    # ─── tab:token-efficiency row ───
    print("── tab:token-efficiency PGPS9K row ──")
    print(f"  Think:  BL={mean(bl_think):.0f}  CT={mean(ct_think):.0f}  ratio={mean(ct_think)/mean(bl_think):.2f}")
    print(f"  Output: BL={mean(bl_out):.0f}   CT={mean(ct_out):.0f}    ratio={mean(ct_out)/mean(bl_out):.2f}")
    print(f"  Wall:   BL={mean(bl_time):.1f}s  CT-tot={mean(ct_time):.1f}s  CT-pt={mean(ct_per_turn):.1f}s")
    print()

    # ─── tab:answer-source row (over correct) ───
    correct = sum(cat_counts_correct.values())
    paper_cats = {
        "Clean Oracle": ["clean_oracle"],
        "Hybrid":       ["hybrid"],
        "Resilient":    ["resilient_oracle", "resilient_fallback"],
        "LLM Bypass":   ["llm_bypass", "direct_reasoning", "no_answer", "unknown"],
    }
    engine_involved_keys = ["clean_oracle", "resilient_oracle", "hybrid"]

    print("── tab:answer-source PGPS9K row ──")
    print(f"  Correct CT answers: {correct}/{n}")
    sum_pct = 0.0
    for label, keys in paper_cats.items():
        c = sum(cat_counts_correct[k] for k in keys)
        pct = 100 * c / correct if correct else 0
        sum_pct += pct
        print(f"  {label:<15s} {c:>4d}   {pct:5.1f}%")
    ei = sum(cat_counts_correct[k] for k in engine_involved_keys)
    print(f"  Engine-Involved {ei:>4d}   {100*ei/correct:5.1f}%")
    print(f"  (sanity: 4-cat sum = {sum_pct:.1f}%)")
    print(f"  Raw category counts: {dict(cat_counts_correct)}")


if __name__ == "__main__":
    main()
