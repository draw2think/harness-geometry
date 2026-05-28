"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── GeoSketch loader ─────────────────────────────────────────────────────────

def load_geosketch(data_dir: Path, sample: int | None = None, seed: int = 42,
                   problem_id: str | None = None, hint: str = "none",
                   answer_type: str | None = None) -> list[dict]:
    """
    Load problems from GeoSketch benchmark (390 problems, English).

    data_dir:    root of geosketch (contains data/, images/, tasks/)
    answer_type: "numerical", "ratio", "descriptor", or None for all
    hint:        "none" or "logic_form" (append parsed geometry from tasks/)
    """
    prob_dir = data_dir / "data"
    img_dir = data_dir / "images"
    tasks_dir = data_dir / "tasks"

    files = sorted(
        [f for f in prob_dir.iterdir() if f.suffix == ".json"],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        files = [f for f in files if f.stem in id_set]

    # Sample
    if sample and len(files) > sample:
        files = random.Random(seed).sample(files, sample)
        files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 0)

    problems = []
    for f in files:
        entry = json.loads(f.read_text())
        pid = f.stem

        img_path = img_dir / f"{pid}.png"
        if not img_path.exists():
            continue

        raw_answer = entry.get("answer", "")

        # Clean LaTeX wrapper artifacts: \(...\), trailing \), leading \(
        clean_ans = raw_answer.strip()
        clean_ans = re.sub(r'^\\\(|\\\)$', '', clean_ans).strip()
        clean_ans = re.sub(r'^\(|\)$', '', clean_ans).strip() \
            if clean_ans.startswith('(') and clean_ans.endswith(')') \
            and ':' not in clean_ans else clean_ans

        # Classify answer type and parse numeric value
        ans_num = None
        a_type = "descriptor"  # default

        # Try ratio parse first (e.g., "1:1", "2√3:1", "\sqrt{2}:1")
        ratio_m = re.match(
            r'^(.+?)\s*:\s*(.+)$', clean_ans)
        if ratio_m:
            a_type = "ratio"
            try:
                num = eval_symbolic(ratio_m.group(1))
                den = eval_symbolic(ratio_m.group(2))
                if den != 0:
                    ans_num = num / den
            except Exception:
                pass
        else:
            # Try numeric parse
            try:
                ans_num = eval_symbolic(clean_ans)
                a_type = "numerical"
            except Exception:
                pass

        # Filter by answer type
        if answer_type and a_type != answer_type:
            continue

        # Optional logic form hint from tasks/
        hint_text = ""
        if hint == "logic_form":
            task_path = tasks_dir / pid / "ex.json"
            if task_path.exists():
                task = json.loads(task_path.read_text())
                forms = task.get("diagram_logic_forms", [])
                if forms:
                    hint_text = "\n[Diagram relations: " + "; ".join(forms) + "]"

        question = entry.get("problem_text", "") + hint_text

        problems.append({
            "id":            pid,
            "dataset":       "geosketch",
            "split":         "test",
            "question":      question,
            "choices":       [],
            "answer_label":  "",
            "expected":      ans_num,
            "image":         str(img_path),
            "tolerance":     0.01,
            "hint_mode":     hint,
            "task_type":     "calculation",
            # GeoSketch-specific
            "expected_raw":  clean_ans,
            "raw_answer":    raw_answer,
            "answer_type":   a_type,
            "question_zh":   entry.get("problem_text_zh", ""),
        })
    return problems
