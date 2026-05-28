"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── UniGeo calc loader ──────────────────────────────────────────────────────

def load_unigeo(data_dir: Path, sample: int | None = None, seed: int = 42,
                problem_id: str | None = None, hint: str = "none",
                language: str = "cn") -> list[dict]:
    """
    Load problems from UniGeo calculation test split (= GeoQA with English).

    data_dir:  root of UniGeo dataset (contains calculation_test.pk, ...)
    language:  "cn" for Chinese subject, "en" for English_problem
    hint:      "none" only (no extra hint modes)
    """
    import pickle
    import numpy as np
    from PIL import Image

    pk_path = data_dir / "calculation_test.pk"
    with open(pk_path, "rb") as f:
        raw = pickle.load(f)

    # Cache directory for extracted images
    img_cache = data_dir / "images_cache"
    img_cache.mkdir(exist_ok=True)

    items = list(enumerate(raw))

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [(i, d) for i, d in items if str(i) in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
        items.sort(key=lambda x: x[0])

    problems = []
    for idx, entry in items:
        # Export image from numpy array → PNG (cached)
        img_path = img_cache / f"{idx}.png"
        if not img_path.exists():
            img_arr = entry["image"]
            Image.fromarray(img_arr).save(img_path)

        # Question text
        question = entry.get("English_problem", entry["subject"]) \
            if language == "en" else entry["subject"]

        choices = entry["choices"]               # e.g. ['40°', '60°', '120°', '140°']
        label = entry["label"]                   # int 0-3
        answer_label = chr(65 + label)
        expected = float(entry["target_number"])

        # Try to get numeric value from choice for tolerance
        choice_val = entry["choice_nums"][label] if "choice_nums" in entry else expected

        problems.append({
            "id":              str(idx),
            "dataset":         "unigeo",
            "split":           "test",
            "question":        question,
            "choices":         choices,
            "answer_label":    answer_label,
            "expected":        expected,
            "image":           str(img_path),
            "tolerance":       0.01,
            "hint_mode":       hint,
            # UniGeo-specific
            "knowledge_type":  ", ".join(sorted(entry.get("formal_point", []))),
        })
    return problems
