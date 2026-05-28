"""Auto-extracted loader. See eval/loaders/__init__.py."""

import json
import os
import random
import re
from pathlib import Path

# Shared helpers from eval_common
from eval_common import _clean_gt_value, eval_symbolic


# ── SolidGeo loader ────────────────────────────────────────────────────────

def load_solidgeo(data_dir: Path, sample: int | None = None, seed: int = 42,
                  problem_id: str | None = None, hint: str = "none",
                  level: str | None = None,
                  problem_type: str | None = None) -> list[dict]:
    """
    Load SolidGeo benchmark (3,113 solid geometry problems).

    data_dir:     root containing SolidGeo.json + images/
    level:        filter by complexity_level ("Level 1", "Level 2", "Level 3")
    problem_type: filter by problem_type substring (e.g. "Multi-view Projection")
    """
    json_path = data_dir / "SolidGeo.json"
    raw = json.loads(json_path.read_text())

    # raw is dict {str_index: {...}}
    items = list(raw.values()) if isinstance(raw, dict) else raw

    # Filter by level
    if level:
        items = [d for d in items if d.get("complexity_level") == level]

    # Filter by problem_type
    if problem_type:
        items = [d for d in items
                 if any(problem_type.lower() in pt.lower()
                        for pt in d.get("problem_type", []))]

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [d for d in items if str(d.get("qa_id", "")) in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)

    problems = []
    for entry in items:
        # Image paths: entry["image"] = ["images\\1.jpg", ...] or string repr
        img_raw = entry.get("image", [])
        if isinstance(img_raw, str):
            try:
                img_raw = eval(img_raw)
            except Exception:
                img_raw = [img_raw]
        if not img_raw:
            continue

        # Resolve all image paths
        img_paths = []
        for img_rel in img_raw:
            p = data_dir / img_rel.replace("\\", "/")
            if p.exists():
                img_paths.append(str(p))
        if not img_paths:
            continue
        # Primary image = first one; extra images stored separately
        img_path = img_paths[0]

        # Parse answer
        raw_answer = str(entry.get("answer", ""))
        answer_type = entry.get("answer_type", "")
        choices = entry.get("choices", [])

        # Strip analysis/explanation suffix that follows the actual answer
        # Analysis markers appear in both English and Chinese source data.
        answer_core = re.split(
            r'\n\n+\s*(?:\[Analysis\]|【Analysis】|【分析】|\[分析\]|##)',
            raw_answer, maxsplit=1)[0].strip()

        if answer_type == "choice":
            # MC: extract letter from answer (may contain LaTeX/analysis text)
            label_match = re.search(r'\b([A-E])\b', answer_core)
            answer_label = label_match.group(1) if label_match else answer_core.strip()
            # For MC, also set expected to the letter so eval pipelines that check
            # `expected` see a non-None value; primary judge should use answer_label.
            expected = answer_label if answer_label else None
        else:
            # Open-ended: try to parse numeric value from the cleaned answer core
            answer_label = ""
            clean = _clean_gt_value(answer_core)
            try:
                expected = float(clean)
            except ValueError:
                try:
                    val = eval_symbolic(clean)
                    expected = float(val)
                except Exception:
                    # Last resort: extract a single numeric token, but only if the
                    # answer doesn't contain free variables (R, L, x, etc.) or LaTeX
                    # operators that signal a symbolic formula.
                    has_symbolic = bool(re.search(
                        r'[A-Za-z]\s*[\(\)+\-\*/]|\\(pi|frac|sqrt|sum|int|cdot)',
                        answer_core))
                    if has_symbolic:
                        expected = None  # leave to LLM judge / manual review
                    else:
                        num_match = re.search(r'-?\d+\.?\d*', answer_core)
                        if num_match:
                            try:
                                expected = float(num_match.group(0))
                            except ValueError:
                                expected = None
                        else:
                            expected = None

        # Question: replace <ImageHere> with [Image N] markers for text mode,
        # actual images are passed separately via "images" field.
        question = entry.get("question", "")
        if not question.strip():
            question = "Solve the solid geometry problem shown in the image."

        # Fallback: extract MC choices from question text when JSON `choices`
        # is empty but options are inline. SolidGeo's raw JSON often leaves
        # choices=[] even for MC problems. Try two formats in order:
        # (1) multi-line "A.\n content B.\n content"; (2) inline "(A) ... (B) ..."
        if answer_type == "choice" and not choices:
            # Format 1: "A. content\nB. content..." (newline-delimited)
            pattern1 = r'(?:^|\n)\s*([A-E])[.\s]+(.*?)(?=\n\s*[A-E][.\s]|\n\n|\Z)'
            matches = re.findall(pattern1, question, re.DOTALL)
            if len(matches) >= 2:
                choices = [content.strip() for letter, content in matches]
            else:
                # Format 2: inline "(A) content (B) content (C) content"
                pattern2 = r'\(([A-E])\)\s*(.+?)(?=\s*\([A-E]\)|$)'
                matches2 = re.findall(pattern2, question, re.DOTALL)
                if len(matches2) >= 2:
                    choices = [content.strip().rstrip('.,;') for letter, content in matches2]

        # Build image-interleaved question:
        # Replace <ImageHere> tags with [Image 1], [Image 2], ...
        # so LLM knows which image corresponds to which position.
        if "<ImageHere>" in question and len(img_paths) > 1:
            parts = question.split("<ImageHere>")
            rebuilt = parts[0]
            for i, part in enumerate(parts[1:], 1):
                rebuilt += f"[Image {i}]" + part
            question = rebuilt

        qa_id = str(entry.get("qa_id", ""))

        problems.append({
            "id":               qa_id,
            "dataset":          "solidgeo",
            "split":            "test",
            "question":         question,
            "choices":          [str(c) for c in choices] if choices else [],
            "answer_label":     answer_label,
            "expected":         expected,
            "expected_raw":     raw_answer,
            "image":            str(img_path),       # primary image (backward compat)
            "images":           img_paths,           # all images (multi-image support)
            "tolerance":        0.01,
            "hint_mode":        hint,
            # SolidGeo-specific
            "complexity_level": entry.get("complexity_level", ""),
            "problem_type":     entry.get("problem_type", []),
            "answer_type":      answer_type,
            "source":           entry.get("source", ""),
            "metadata":         {"subject": "Solid Geometry"},  # always 3D
        })
    return problems
