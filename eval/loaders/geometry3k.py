"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── Geometry3K loader ────────────────────────────────────────────────────────

def load_geometry3k(data_dir: Path, sample: int | None = None, seed: int = 42,
                    problem_id: str | None = None, hint: str = "none",
                    image_style: str = "plain") -> list[dict]:
    """
    Load problems from a Geometry3K split (val / test / train).

    hint:
      "none"       image + question only  (pure vision challenge)
      "points"     append pixel coords of labeled points to question
      "logic_form" append full diagram_logic_form predicates to question
    """
    dirs = sorted(
        [d for d in data_dir.iterdir()
         if d.is_dir() and (d / "data.json").exists()],
        key=lambda p: int(p.name) if p.name.isdigit() else 0,
    )
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        dirs = [d for d in dirs if d.name in id_set]
    if sample and len(dirs) > sample:
        dirs = random.Random(seed).sample(dirs, sample)
        dirs.sort(key=lambda p: int(p.name) if p.name.isdigit() else 0)

    problems = []
    for d in dirs:
        data = json.loads((d / "data.json").read_text())
        lf   = json.loads((d / "logic_form.json").read_text()) \
               if (d / "logic_form.json").exists() else {}

        # image_style: "point" (labeled vertices, default), "plain" (clean diagram)
        if image_style == "plain":
            img = d / "img_diagram.png"
            if not img.exists():
                img = d / "img_diagram_point.png"
        else:
            img = d / "img_diagram_point.png"
            if not img.exists():
                img = d / "img_diagram.png"
        if not img.exists():
            continue

        choices      = data.get("choices", [])
        answer_label = data["answer"]
        ans_idx      = ord(answer_label) - ord("A")

        # Derive expected value from the correct choice text -- not precise_value,
        # which has known annotation mismatches in geometry3k (e.g. id=2315).
        correct_choice = choices[ans_idx] if ans_idx < len(choices) else ""
        is_symbolic = False
        ans_num = None

        # 1. Try direct float parse (handles "-1", "140", "16.76", ...)
        try:
            ans_num = float(correct_choice.replace(" ", ""))
        except (ValueError, TypeError):
            pass

        # 2. Try symbolic evaluation (handles "4 \sqrt 6 + 2 \sqrt{14}", ...)
        if ans_num is None and correct_choice:
            try:
                ans_num = eval_symbolic(correct_choice)
                is_symbolic = True
            except Exception:
                pass

        # 3. Fall back to precise_value
        if ans_num is None:
            pv_list = data.get("precise_value", [])
            try:
                ans_num = float(pv_list[ans_idx]) if ans_idx < len(pv_list) else None
            except (ValueError, TypeError):
                ans_num = None

        # Optional hint appended to question
        hint_text = ""
        if hint == "points" and lf.get("point_positions"):
            pts = ", ".join(
                f"{k}=pixel({int(v[0])},{int(v[1])})"
                for k, v in lf["point_positions"].items()
            )
            hint_text = f"\n[Point pixel positions: {pts}]"
        elif hint == "logic_form":
            dlf = lf.get("diagram_logic_form", [])
            if dlf:
                hint_text = "\n[Diagram relations: " + "; ".join(dlf) + "]"

        problems.append({
            "id":                str(data["id"]),
            "dataset":           "geometry3k",
            "split":             data_dir.name,
            "question":          data["problem_text"] + hint_text,
            "choices":           choices,
            "answer_label":      answer_label,
            "expected":          ans_num,
            "image":             str(img),
            "tolerance":         0.01,
            "problem_type_graph": data.get("problem_type_graph", []),
            "problem_type_goal":  data.get("problem_type_goal",  []),
            "logic_form":        lf,
            "hint_mode":         hint,
        })
    return problems
