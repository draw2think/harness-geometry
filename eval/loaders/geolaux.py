"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── GeoLaux loader ───────────────────────────────────────────────────────────

def load_geolaux(data_dir: Path, sample: int | None = None, seed: int = 42,
                 problem_id: str | None = None, hint: str = "none",
                 problem_type: str | None = None) -> list[dict]:
    """
    Load problems from GeoLaux-mini (330 problems, Chinese zhongkao geometry).

    data_dir:     root of geolaux repo (contains data/GeoLaux_minidata.json)
    problem_type: "calculation", "proving", or None for all
    hint:         "none" or "auxiliary" (append auxiliary line text hint)
    """
    json_path = data_dir / "data" / "GeoLaux_minidata.json"
    raw: dict = json.loads(json_path.read_text())

    img_dir = data_dir / "data" / "mini_original_images"
    aux_img_dir = data_dir / "data" / "mini_auxiliary_images"

    items = list(raw.items())

    # Filter by problem type
    if problem_type:
        items = [(k, v) for k, v in items if v.get("type") == problem_type]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [(k, v) for k, v in items if k in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
    items.sort(key=lambda kv: kv[0])

    problems = []
    for pid, entry in items:
        img_name = entry.get("original_image_name", "")
        img_path = img_dir / f"{img_name}.png"
        if not img_path.exists():
            continue

        # Parse answer for calculation problems
        ans_num = None
        is_proving = entry.get("type") == "proving"
        raw_answer = entry.get("number_answer")
        choices_raw = entry.get("choices", [])

        # Choices are "A.xxx", "B.xxx", ... — strip letter prefix for display,
        # derive answer_label by matching raw_answer against choice values.
        answer_label = ""
        choices = []
        for c in choices_raw:
            m = re.match(r'^([A-D])\.(.+)', c)
            if m:
                letter, val = m.group(1), m.group(2).strip()
                choices.append(val)
                # Match: strip trailing °/' for comparison
                val_cmp = re.sub(r"[°']+\s*$", '', val)
                ans_cmp = re.sub(r"[°']+\s*$", '', str(raw_answer)) if raw_answer else ""
                if val_cmp == ans_cmp:
                    answer_label = letter
            else:
                choices.append(c)

        if raw_answer and not is_proving:
            try:
                ans_num = eval_symbolic(str(raw_answer))
            except Exception:
                pass

        # Optional auxiliary line hint
        hint_text = ""
        if hint == "auxiliary" and entry.get("auxiliary_text"):
            hint_text = f"\n[Auxiliary line hint: {entry['auxiliary_text']}]"

        question = entry["problem_text"] + hint_text

        # Auxiliary image (if available)
        aux_img_name = entry.get("auxiliary_image_name")
        aux_img = ""
        if aux_img_name:
            aux_path = aux_img_dir / f"{aux_img_name}.png"
            if aux_path.exists():
                aux_img = str(aux_path)

        problems.append({
            "id":              pid,
            "dataset":         "geolaux",
            "split":           "mini",
            "question":        question,
            "choices":         choices,
            "answer_label":    answer_label,
            "expected":        ans_num,
            "image":           str(img_path),
            "tolerance":       0.01,
            "hint_mode":       hint,
            "task_type":       "proving" if is_proving else "calculation",
            # GeoLaux-specific
            "raw_answer":      raw_answer,
            "problem_type":    entry.get("type", ""),
            "step_length":     entry.get("step_length", 0),
            "auxiliary_type":  entry.get("auxiliary_type", 0),
            "auxiliary_text":  entry.get("auxiliary_text", ""),
            "auxiliary_image": aux_img,
            "solution":        entry.get("solution", ""),
        })
    return problems
