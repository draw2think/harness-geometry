"""GeoGoal-SGVR loader. 256 test problems with NL construction instructions +
T_i query list + FL predicate ground truth. See eval/loaders/__init__.py."""

import ast
import random
from fractions import Fraction
from pathlib import Path

import pandas as pd


def _parse_answer_list(raw: str) -> list[str]:
    """SGVR stores answer as a stringified Python list, e.g.
       "['1', '0', '2/1', '90']".  Return list[str]; downstream code
       normalizes via Fraction for numeric equivalence."""
    try:
        parsed = ast.literal_eval(raw)
        return [str(x) for x in parsed]
    except Exception:
        return []


def load_geogoal_sgvr(data_dir: Path, sample: int | None = None, seed: int = 42,
                      problem_id: str | None = None, hint: str = "none",
                      split: str = "test") -> list[dict]:
    """
    Load GeoGoal-SGVR (256 test / 256 train).

    data_dir:  root containing data/{split}-00000-of-00001.parquet + images/
    split:     "test" (default) or "train"

    Each problem dict includes FL fields used by the predicate verifier:
      problem_FL   — AG-style construction plan
      solution_FL  — expanded predicate set (Premise/Numerical Check/Derived tiers)
    """
    parquet_path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")

    df = pd.read_parquet(parquet_path)

    explicit_ids = problem_id is not None
    if explicit_ids:
        id_set = {x.strip() for x in str(problem_id).split(",")}
        df = df[df["id"].isin(id_set)]

    rows = df.to_dict("records")

    # When --id is explicit, honor the exact list; otherwise allow sampling
    if sample and not explicit_ids and len(rows) > sample:
        rows = random.Random(seed).sample(rows, sample)
    rows.sort(key=lambda r: r["id"])

    images_dir = data_dir / "images"
    problems: list[dict] = []
    for r in rows:
        qid = r["id"]
        img_path = images_dir / f"{qid}.png"
        if not img_path.exists():
            continue

        answer_list = _parse_answer_list(r["answer"])
        problems.append({
            "id": qid,
            "dataset": "geogoal_sgvr",
            "split": split,
            "question": r["question"],
            "choices": [],
            "answer_label": None,
            "expected": answer_list,       # ordered list of T_i values
            "expected_raw": r["answer"],   # original stringified list
            "image": str(img_path),
            "tolerance": 0.01,
            "hint_mode": hint,
            # SGVR-specific: formal-language ground truth for the verifier
            "problem_FL": r.get("problem_FL", ""),
            "solution_FL": r.get("solution_FL", ""),
        })

    return problems
