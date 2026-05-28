"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── MathCanvas-Bench loader ────────────────────────────────────────────────

def load_mathcanvas(data_dir: Path, sample: int | None = None, seed: int = 42,
                    problem_id: str | None = None, hint: str = "none",
                    knowledge: str | None = None) -> list[dict]:
    """
    Load MathCanvas-Bench (3,079 visual reasoning problems).

    data_dir:   root containing MathCanvas_Bench.jsonl + images/
    knowledge:  filter by knowledge category (e.g. 'Solid Geometry', 'Plane Geometry')
    """
    jsonl_path = data_dir / "MathCanvas_Bench.jsonl"
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    # Filter by knowledge
    if knowledge:
        items = [d for d in items if knowledge.lower() in d.get("knowledge", "").lower()]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [d for d in items if d["id"] in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)

    img_dir = data_dir / "images"
    problems = []
    for entry in items:
        pid = entry["id"]

        # Collect question images: {id}-pro0.png, {id}-pro1.png, ...
        img_paths = []
        for i in range(10):
            p = img_dir / f"{pid}-pro{i}.png"
            if p.exists():
                img_paths.append(str(p))
            else:
                break

        # Build question text from question_interleave
        # Note: field is "content" (not "text") in MathCanvas-Bench
        q_parts = []
        img_idx = 0
        for part in entry.get("question_interleave", []):
            if part["type"] == "text":
                text = (part.get("content") or part.get("text") or "").strip()
                if text:
                    q_parts.append(text)
            elif part["type"] == "image":
                img_idx += 1
                q_parts.append(f"[Image {img_idx}]")

        question = "\n".join(q_parts).strip()
        if not question:
            question = "Solve the math problem shown in the image."

        # Parse answer — may have <1>...</1><2>...</2> sub-question tags
        raw_answer = str(entry.get("answer", ""))

        # For single answer, try numeric parse
        answer_label = ""
        if re.search(r'<\d+>', raw_answer):
            # Multi sub-question — extract first sub-answer for validation
            m = re.search(r'<1>(.*?)</1>', raw_answer)
            clean = _clean_gt_value(m.group(1)) if m else _clean_gt_value(raw_answer)
        else:
            clean = _clean_gt_value(raw_answer)

        try:
            expected = float(clean)
        except ValueError:
            try:
                val = eval_symbolic(clean)
                expected = float(val)
            except Exception:
                expected = None

        problems.append({
            "id":               pid,
            "dataset":          "mathcanvas",
            "split":            "bench",
            "question":         question,
            "choices":          [],
            "answer_label":     answer_label,
            "expected":         expected,
            "expected_raw":     raw_answer,
            "image":            img_paths[0] if img_paths else "",
            "images":           img_paths,
            "tolerance":        0.01,
            "hint_mode":        hint,
            # MathCanvas-specific
            "knowledge":        entry.get("knowledge", ""),
            "subknowledge":     entry.get("subknowledge", ""),
        })
    return problems
