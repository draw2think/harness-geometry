"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── MathVerse loader ────────────────────────────────────────────────────────

def load_mathverse(data_dir: Path, sample: int | None = None, seed: int = 42,
                   problem_id: str | None = None, hint: str = "none",
                   version: str = "Vision Only",
                   subject: str = "Plane Geometry") -> list[dict]:
    """
    Load problems from MathVerse testmini.

    data_dir:  root of MathVerse (contains testmini.json, images/, ...)
    version:   problem version filter (default: "Vision Only")
    subject:   metadata.subject filter (default: "Plane Geometry")

    Supported versions: Text Dominant, Text Lite, Vision Intensive,
                        Vision Dominant, Vision Only
    """
    json_path = data_dir / "testmini.json"
    raw = json.loads(json_path.read_text())

    # Filter by version + subject
    filtered = [d for d in raw
                if d.get("problem_version") == version
                and d.get("metadata", {}).get("subject") == subject]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        filtered = [d for d in filtered if str(d["sample_index"]) in id_set]

    # Sample
    if sample and len(filtered) > sample:
        filtered = random.Random(seed).sample(filtered, sample)
    filtered.sort(key=lambda d: d["sample_index"])

    problems = []
    for entry in filtered:
        # Image: JSON stores "images_version_6/image_N.png",
        # actual location is data_dir / "images" / "images_version_6/image_N.png"
        img_path = data_dir / "images" / entry["image"]
        if not img_path.exists():
            img_path = data_dir / entry["image"]  # fallback
        if not img_path.exists():
            continue

        # Parse choices from question_for_eval (if MC):
        #   "... ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
        eval_q = entry.get("question_for_eval", "")
        choices = []
        choice_matches = re.findall(
            r'(?:^|\n)([A-E])\s*[:\.]\s*(.+?)(?=\n[A-E]\s*[:\.]\s*|\Z)', eval_q)
        choice_matches = [val for _, val in choice_matches]
        for val in choice_matches:
            choices.append(val.strip())

        raw_answer = entry["answer"].strip()

        # MC (single letter) vs open-ended (numeric value)
        if len(raw_answer) == 1 and raw_answer.upper() in "ABCDE":
            answer_label = raw_answer.upper()
            expected = None   # letter matching in validate() handles it
        else:
            # Open-ended: try to parse numeric value, keep raw as fallback
            answer_label = ""
            choices = []
            clean = _clean_gt_value(raw_answer)
            try:
                expected = float(clean)
            except ValueError:
                try:
                    val = eval_symbolic(clean)
                    expected = float(val)  # ensure scalar, not tuple/list
                except Exception:
                    expected = None  # validate() will fall back to expected_raw

        # Question text: for Vision Only use the generic query prompt
        question = entry.get("query_wo", "").strip()
        if not question:
            question = ("Solve the geometry problem shown in the image "
                        "and provide the correct option letter.")

        problems.append({
            "id":               str(entry["sample_index"]),
            "dataset":          "mathverse",
            "split":            "testmini",
            "question":         question,
            "choices":          choices,
            "answer_label":     answer_label,
            "expected":         expected,
            "expected_raw":     raw_answer,   # original GT string for fallback
            "image":            str(img_path),
            "tolerance":        0.01,
            "hint_mode":        hint,
            # MathVerse-specific
            "problem_version":  entry["problem_version"],
            "source":           entry.get("metadata", {}).get("source", ""),
            "subfield":         entry.get("metadata", {}).get("subfield", ""),
            "metadata":         entry.get("metadata", {}),
        })
    return problems
