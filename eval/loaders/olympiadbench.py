"""OlympiadBench geometry GPS loader. See eval/loaders/__init__.py."""

import os
import re
import random
from pathlib import Path

from eval_common import _clean_gt_value, eval_symbolic


def load_olympiadbench(data_dir: Path, sample: int | None = None, seed: int = 42,
                       problem_id: str | None = None, hint: str = "none",
                       split: str = "en_comp",
                       subfield_filter: str = "Geometry|Plane|Solid|Conic",
                       gps_only: bool = True) -> list[dict]:
    """
    Load geometry GPS problems from OlympiadBench.

    data_dir:        root of OlympiadBench (contains OE_MM_maths_*/  dirs)
    split:           which subset to load:
                       "en_comp"  → OE_MM_maths_en_COMP  (112 geo, 107 GPS)
                       "zh_comp"  → OE_MM_maths_zh_COMP  (48 geo, 37 GPS)
                       "zh_cee"   → OE_MM_maths_zh_CEE   (1339 geo, 1155 GPS)
                       "all"      → union of all three
    subfield_filter: regex to filter subfield column (default: geometry-related)
    gps_only:        if True, keep only Numerical/Expression/Tuple answers (drop Interval/Equation)
    """
    import pandas as pd

    split_map = {
        "en_comp": ["OE_MM_maths_en_COMP"],
        "zh_comp": ["OE_MM_maths_zh_COMP"],
        "zh_cee":  ["OE_MM_maths_zh_CEE"],
        "all":     ["OE_MM_maths_en_COMP", "OE_MM_maths_zh_COMP", "OE_MM_maths_zh_CEE"],
    }
    split_dirs = split_map.get(split, [split])  # allow raw dir name too

    frames = []
    for sdir in split_dirs:
        parquet = data_dir / sdir / f"{sdir}.parquet"
        if not parquet.exists():
            print(f"  [WARN] Missing: {parquet}")
            continue
        df = pd.read_parquet(parquet)
        df["_split"] = sdir
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {data_dir} for split={split}")

    df = pd.concat(frames, ignore_index=True)

    # Filter by subfield (geometry-related)
    if subfield_filter:
        df = df[df["subfield"].str.contains(subfield_filter, case=False, na=False)]

    # GPS filter: keep only Numerical / Expression answer types
    if gps_only:
        gps_types = ["Numerical", "Expression", "Tuple", "Numerical,Tuple", "Tuple,Numerical"]
        df = df[df["answer_type"].isin(gps_types)]

    items = list(df.iterrows())

    # Filter by problem_id
    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        items = [(i, row) for i, row in items if str(row["id"]) in id_set]

    # Sample
    if sample and len(items) > sample:
        items = random.Random(seed).sample(items, sample)
        items.sort(key=lambda x: x[1]["id"])

    # Cache dir for extracted images
    img_cache = data_dir / "_img_cache"
    img_cache.mkdir(exist_ok=True)

    problems = []
    for _, row in items:
        pid = str(row["id"])
        split_name = row["_split"]

        # Extract image from parquet bytes → PNG file
        img_path = _extract_image(row, img_cache, pid, split_name)
        if img_path is None:
            continue

        # Parse answer — always produce a list; preserve raw LaTeX for math_verify
        final_ans = row["final_answer"]
        if hasattr(final_ans, '__len__') and not isinstance(final_ans, str):
            final_ans_str = str(final_ans[0]).strip() if len(final_ans) > 0 else ""
        else:
            final_ans_str = str(final_ans).strip()

        # Split multi-answer string into individual answer parts
        # Handles: '$25538$,$2053$' → ['25538', '2053']
        #          '7,4'           → ['7', '4']
        #          '$\\frac{1}{2}$' → ['\\frac{1}{2}']
        #          '$(4,0),(0,4)$' → ['(4,0)', '(0,4)']  (tuple-aware)
        answer_parts = _split_answer_parts(final_ans_str)

        # Parse each part to float where possible
        expected_list = [_parse_answer(p) for p in answer_parts]
        # Primary expected: first parseable float (for numeric comparison)
        expected = next((v for v in expected_list if v is not None), None)
        # Raw: full original string for math_verify fallback
        answer_str = final_ans_str.strip("$").strip()

        # Tolerance: use error field if present, else auto-detect from question
        tol = 0.01
        if row.get("error") is not None and str(row["error"]).strip():
            try:
                tol = float(row["error"])
            except (ValueError, TypeError):
                pass
        else:
            q_lower = str(row["question"]).lower()
            if "nearest tenth" in q_lower or "1 decimal place" in q_lower:
                tol = 0.05
            elif "nearest hundredth" in q_lower or "2 decimal place" in q_lower:
                tol = 0.005
            elif "nearest integer" in q_lower or "nearest whole" in q_lower:
                tol = 0.5

        # Unit info (append to question if present)
        question = str(row["question"])
        unit = row.get("unit")
        if unit and str(unit).strip():
            unit_str = str(unit).strip()
            # Don't duplicate if already in question
            if unit_str not in question:
                question += f"\n(Express your answer in {unit_str}.)"

        # Additional images (some problems have image_2, image_3, etc.)
        extra_images = []
        for j in range(2, 10):
            if row.get(f"image_{j}") is not None:
                extra = _extract_image_field(row[f"image_{j}"], img_cache, f"{split_name}_{pid}_img{j}")
                if extra:
                    extra_images.append(extra)

        problems.append({
            "id":            pid,
            "dataset":       "olympiadbench",
            "split":         split_name,
            "question":      question,
            "choices":       [],
            "answer_label":  "",
            "expected":      expected,
            "expected_raw":  answer_str,
            "image":         str(img_path),
            "extra_images":  extra_images,
            "tolerance":     tol,
            "hint_mode":     hint,
            # OlympiadBench-specific
            "subfield":      row.get("subfield", ""),
            "difficulty":    row.get("difficulty", ""),
            "language":      row.get("language", ""),
            "answer_type":   row.get("answer_type", ""),
            "is_multiple_answer": bool(row.get("is_multiple_answer", False)),
            "unit":          str(unit) if unit else None,
        })

    return problems


def _extract_image(row, cache_dir: Path, pid: str, split_name: str) -> str | None:
    """Extract image_1 from parquet row bytes → PNG file."""
    img_data = row.get("image_1")
    if img_data is None:
        return None
    return _extract_image_field(img_data, cache_dir, f"{split_name}_{pid}")


def _extract_image_field(img_data, cache_dir: Path, name: str) -> str | None:
    """Write image bytes dict to PNG file, return path."""
    if img_data is None:
        return None
    if isinstance(img_data, dict) and "bytes" in img_data:
        img_bytes = img_data["bytes"]
    elif isinstance(img_data, bytes):
        img_bytes = img_data
    else:
        return None

    out_path = cache_dir / f"{name}.png"
    if not out_path.exists():
        # Detect format from magic bytes
        ext = ".png"
        if img_bytes[:2] == b'\xff\xd8':
            ext = ".jpg"
        out_path = cache_dir / f"{name}{ext}"
        if not out_path.exists():
            out_path.write_bytes(img_bytes)
    return str(out_path)


def _split_answer_parts(raw: str) -> list[str]:
    """Split a multi-answer string into individual answer parts.

    Examples:
        '$25538$,$2053$'         → ['25538', '2053']
        '7,4'                    → ['7', '4']
        '$2$, $\\frac{2}{15}$'   → ['2', '\\frac{2}{15}']
        '$4,(4,16)$'             → ['4', '(4,16)']
        '$(4,0),(0,4),(2,0)$'    → ['(4,0)', '(0,4)', '(2,0)']
        '$\\frac{1}{2}$'         → ['\\frac{1}{2}']
    """
    s = raw.strip()
    # Remove outer $ if entire string is wrapped
    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1].strip()

    # Split on $,$ or $, $ boundaries (each value wrapped in $)
    if '$,$' in s or '$, $' in s:
        parts = re.split(r'\$\s*,\s*\$', s)
        return [p.strip().strip('$').strip() for p in parts if p.strip()]

    # Split on comma, but respect parentheses (tuples like (4,0))
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    # Clean up $ from each part
    return [p.strip().strip('$').strip() for p in parts if p.strip()]


def _parse_answer(answer_str: str) -> float | None:
    """Parse OlympiadBench answer string to float."""
    s = answer_str.strip().strip("$").strip()
    # Remove leading \quad
    s = re.sub(r'^\\quad\s*', '', s)

    # Skip answers with variables (not pure constants)
    if re.search(r'(?<![a-z])([xyznabt]|\\theta)(?![a-z])', s):
        return None

    # Direct numeric
    clean = _clean_gt_value(s)
    try:
        return float(clean)
    except (ValueError, TypeError):
        pass

    # Symbolic (e.g. "\\frac{1}{2}", "6\\sqrt{2-\\sqrt{3}}", "2\\sqrt{3}\\pi")
    try:
        return float(eval_symbolic(s))
    except Exception:
        pass

    return None
