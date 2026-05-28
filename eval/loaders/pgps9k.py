"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── PGPS9K loader ────────────────────────────────────────────────────────────

def load_pgps9k(data_dir: Path, sample: int | None = None, seed: int = 42,
                problem_id: str | None = None, hint: str = "none",
                image_dir: str = "Diagram_Visual",
                exclude_book: str | None = None) -> list[dict]:
    """
    Load problems from PGPS9K test split.

    data_dir:      root of PGPS9K dataset (contains PGPS9K/, Diagram_Visual/, ...)
    image_dir:     "Diagram_Visual" (with labels) or "Diagram" (clean)
    exclude_book:  drop problems from a specific book (e.g. "Geometry3K")
    hint:
      "none"          image + question only
      "parsing_sem"   append semantic parsing (known relations)
      "parsing_stru"  append structural parsing (diagram structure)
    """
    test_json = data_dir / "PGPS9K" / "test.json"
    raw: dict = json.loads(test_json.read_text())

    items = sorted(raw.items(), key=lambda kv: int(kv[0].split("_")[1]))

    # Filter by book
    if exclude_book:
        items = [(k, v) for k, v in items if v.get("book") != exclude_book]

    # Filter by problem_id  ("prob_13" or "13" or "13,14,15")
    if problem_id is not None:
        id_set = set()
        for x in str(problem_id).split(","):
            x = x.strip()
            id_set.add(x if x.startswith("prob_") else f"prob_{x}")
        items = [(k, v) for k, v in items if k in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
        items.sort(key=lambda kv: int(kv[0].split("_")[1]))

    def _fmt(x: float) -> str:
        """25.0 -> '25', 33.941... -> '33.941'"""
        if x == int(x):
            return str(int(x))
        return f"{x:.3f}".rstrip("0").rstrip(".")

    problems = []
    for key, entry in items:
        # Image
        img = data_dir / image_dir / entry["diagram"]
        if not img.exists():
            continue

        choices_raw = entry["choices"]          # list[float]
        choices_str = [_fmt(c) for c in choices_raw]

        # Derive answer_label by matching float answer to closest choice
        ans_val = float(entry["answer"])
        diffs = [(abs(ans_val - c), i) for i, c in enumerate(choices_raw)]
        _, best_idx = min(diffs)
        answer_label = chr(65 + best_idx)       # A/B/C/D
        expected = choices_raw[best_idx]         # use choice value (more precise)

        # Hint text
        hint_text = ""
        if hint == "parsing_sem" and entry.get("parsing_sem_seqs"):
            rels = "; ".join(entry["parsing_sem_seqs"])
            hint_text = f"\n[Known relations: {rels}]"
        elif hint == "parsing_stru" and entry.get("parsing_stru_seqs"):
            stru = "; ".join(entry["parsing_stru_seqs"])
            hint_text = f"\n[Diagram structure: {stru}]"

        problems.append({
            "id":                  key,          # "prob_13"
            "dataset":             "pgps9k",
            "split":               "test",
            "question":            entry["text"] + hint_text,
            "choices":             choices_str,
            "answer_label":        answer_label,
            "expected":            expected,
            "image":               str(img),
            "tolerance":           0.01,
            "hint_mode":           hint,
            # PGPS9K-specific (transparent to result.json)
            "knowledge_type":      entry.get("type", "").strip(),
            "book":                entry.get("book", ""),
            "page":                entry.get("page", ""),
            "expression":          entry.get("expression", ""),
            "parsing_stru_seqs":   entry.get("parsing_stru_seqs", []),
            "parsing_sem_seqs":    entry.get("parsing_sem_seqs", []),
        })
    return problems
