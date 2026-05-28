"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── GenExam loader ────────────────────────────────────────────────────────────

def load_genexam(data_dir: Path = Path("/data/genexam"),
                 gt_image_dir: Path = Path("/data/genexam/data/images"),
                 sample: int | None = None, seed: int = 42,
                 problem_id: str | None = None,
                 difficulty: str | None = None, **_) -> list[dict]:
    """Load GenExam Mathematics problems (text-to-image construction).

    data_dir:     directory containing mathematics.json
    gt_image_dir: directory containing ground-truth images (Mathematics/*.png)
    difficulty:   filter by "easy", "medium", "hard", or None for all
    """
    json_path = data_dir / "mathematics.json"
    items = json.loads(json_path.read_text())

    # Filter by difficulty
    if difficulty:
        items = [d for d in items if d.get("difficulty", "").lower() == difficulty.lower()]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [d for d in items if d["id"] in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
    items.sort(key=lambda d: d["id"])

    problems = []
    for entry in items:
        gt_img = gt_image_dir / entry["image_path"]
        problems.append({
            "id":              entry["id"],
            "dataset":         "genexam",
            "split":           "test",
            "prompt":          entry["prompt"],
            "scoring_points":  entry["scoring_points"],
            "gt_image":        str(gt_img) if gt_img.exists() else "",
            "taxonomy":        entry.get("taxonomy", ""),
            "subject":         entry.get("subject", "Mathematics"),
            "difficulty":      entry.get("difficulty", ""),
            "img_type":        entry.get("img_type", ""),
            "task_type":       "construction",
        })
    return problems
