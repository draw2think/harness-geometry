"""MathVista GPS loader. See eval/loaders/__init__.py."""

import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


def load_mathvista(data_dir: Path, sample: int | None = None, seed: int = 42,
                   problem_id: str | None = None, hint: str = "none",
                   task_filter: str = "geometry problem solving") -> list[dict]:
    """
    Load problems from MathVista testmini (parquet + images).

    data_dir:     root of MathVista (contains data/, images/)
    task_filter:  metadata.task filter (default: "geometry problem solving" = GPS)
                  Set to None or "" to load all 1000 problems.
    """
    import pandas as pd

    parquet = data_dir / "data" / "testmini-00000-of-00001-725687bf7a18d64b.parquet"
    df = pd.read_parquet(parquet)

    # Filter by task type (GPS by default)
    if task_filter:
        df = df[df['metadata'].apply(lambda x: x.get('task', '') == task_filter)]

    items = list(df.iterrows())

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [(i, row) for i, row in items if str(row['pid']) in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
        items.sort(key=lambda x: x[1]['pid'])

    problems = []
    for _, row in items:
        pid = row['pid']
        meta = row['metadata'] if isinstance(row['metadata'], dict) else {}

        # Image path
        img_path = data_dir / "images" / f"{pid}.jpg"
        if not img_path.exists():
            img_path = data_dir / "images" / f"{pid}.png"
        if not img_path.exists():
            continue

        # Question
        question = row.get('question', '') or ''

        # Choices
        choices_raw = row.get('choices', None)
        choices = []
        if choices_raw is not None and hasattr(choices_raw, '__len__') and len(choices_raw) > 0:
            choices = list(choices_raw)

        # Answer
        raw_answer = str(row.get('answer', '')).strip()
        answer_label = ""
        expected = None

        if row.get('question_type') == 'multi_choice' and choices:
            # MC: answer is a letter or the choice text
            if len(raw_answer) == 1 and raw_answer.upper() in "ABCDE":
                answer_label = raw_answer.upper()
                ans_idx = ord(answer_label) - ord("A")
                if ans_idx < len(choices):
                    clean = _clean_gt_value(str(choices[ans_idx]))
                    try:
                        expected = float(clean)
                    except ValueError:
                        try:
                            expected = float(eval_symbolic(clean))
                        except Exception:
                            expected = None
            else:
                # answer is the choice text — reverse-lookup to find the label
                for ci, ch in enumerate(choices):
                    if str(ch).strip() == raw_answer:
                        answer_label = chr(ord('A') + ci)
                        break
                # Also try to parse numeric value
                clean = _clean_gt_value(raw_answer)
                try:
                    expected = float(clean)
                except ValueError:
                    try:
                        expected = float(eval_symbolic(clean))
                    except Exception:
                        expected = None
        else:
            # Free-form: parse numeric
            clean = _clean_gt_value(raw_answer)
            try:
                expected = float(clean)
            except ValueError:
                try:
                    expected = float(eval_symbolic(clean))
                except Exception:
                    expected = None

        problems.append({
            "id":            str(pid),
            "dataset":       "mathvista",
            "split":         "testmini",
            "question":      question,
            "choices":       choices,
            "answer_label":  answer_label,
            "expected":      expected,
            "expected_raw":  raw_answer,
            "image":         str(img_path),
            "tolerance":     0.01,
            "hint_mode":     hint,
            # MathVista-specific
            "source":        meta.get("source", ""),
            "task":          meta.get("task", ""),
            "grade":         meta.get("grade", ""),
            "language":      meta.get("language", ""),
        })

    return problems
