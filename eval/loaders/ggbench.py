"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── GGBench loader ──────────────────────────────────────────────────────────

def load_ggbench(data_dir: Path, sample: int | None = None, seed: int = 42,
                 problem_id: str | None = None, hint: str = "none",
                 difficulty: str | None = None) -> list[dict]:
    """
    Load problems from GGBench (construction tasks).

    data_dir:    root of GGBench (contains GGBench_dataset.json, Q&A_image/, ...)
    difficulty:  filter by "Easy", "Medium", "Hard", or None for all

    Note: GGBench is a construction benchmark — no numeric answers/choices.
    task_type="construction". Eval scripts should detect this and skip
    numeric validation, using VLM-judge or image comparison instead.
    """
    json_path = data_dir / "GGBench_dataset.json"
    raw = json.loads(json_path.read_text())

    items = raw

    # Filter by difficulty
    if difficulty:
        items = [d for d in items if d.get("difficulty") == difficulty]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [d for d in items if str(d["id"]) in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
    items.sort(key=lambda d: d["id"])

    problems = []
    for entry in items:
        # question_image path: "/Q&A_image/3_1.png" → strip leading /
        q_img = data_dir / entry["question_image"].lstrip("/")
        if not q_img.exists():
            continue

        ref_img = data_dir / entry["res_image"].lstrip("/")

        problems.append({
            "id":                   str(entry["id"]),
            "dataset":              "ggbench",
            "split":                "test",
            "question":             entry["question"],
            "choices":              [],
            "answer_label":         "",
            "expected":             None,
            "image":                str(q_img),
            "tolerance":            0.01,
            "hint_mode":            hint,
            "task_type":            "construction",
            # GGBench-specific
            "difficulty":           entry.get("difficulty", ""),
            "text_answer":          entry.get("text_answer", ""),
            "reference_image":      str(ref_img) if ref_img.exists() else "",
            "complete_image":       entry.get("complete_image", ""),
            "inspection_content":   entry.get("inspection_content", ""),
            "skill_classification": entry.get("skill_classification", ""),
        })
    return problems
