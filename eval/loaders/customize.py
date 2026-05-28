"""Custom single-image construction loader. See eval/loaders/__init__.py.

Lets you run the full Draw2Think bench pipeline on your OWN image + text,
instead of a benchmark problem.  Mirrors the prob-dict shape produced by the
other loaders so test_agentic_geo_constructer.run_problem works unchanged.

Spec format
-----------
Each spec is a JSON file living directly under ``data_dir`` (top level, not in
a per-problem output subdir).  Minimal example::

    {
      "id": "draw_sketch",
      "image": "draw_sketch.png",
      "question": "Reproduce this sketch on the canvas ..."
    }

Fields:
  id        (required) problem id; output lands in eval/customize/<id>/
  image     (required) path to the input image; resolved relative to the spec
            file first, then tried as an absolute/CWD path
  question  (required) free-form instruction shown to the model
  choices   (optional) MC choices; usually empty for a construction task
  expected  (optional) ground-truth value; None for an open construction
  problem_type_graph / problem_type_goal (optional) tags for the run header

Usage
-----
  python eval/test_agentic_geo_constructer.py \\
      --dataset customize --data_dir eval/customize \\
      --id draw_sketch --mode construct \\
      --model gemini-3-flash-preview@medium
"""

import json
from pathlib import Path


def load_customize(data_dir: Path, sample: int | None = None, seed: int = 42,
                   problem_id: str | None = None, **_) -> list[dict]:
    """Load custom construction specs from ``data_dir`` (a dir of *.json specs,
    or a single .json spec file)."""
    data_dir = Path(data_dir)

    # Collect spec files: a single .json, or every top-level *.json in a dir
    # (skipping the pipeline's own result/summary artifacts).
    if data_dir.is_file() and data_dir.suffix == ".json":
        spec_files = [data_dir]
        base_dir = data_dir.parent
    else:
        spec_files = sorted(
            p for p in data_dir.glob("*.json")
            if not p.name.endswith("_result.json")
            and not p.name.startswith("summary_")
        )
        base_dir = data_dir

    if problem_id is not None:
        id_set = {x.strip() for x in str(problem_id).split(",")}
    else:
        id_set = None

    problems = []
    for sf in spec_files:
        try:
            entries = json.loads(sf.read_text())
        except Exception:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            pid = str(entry.get("id") or sf.stem)
            if id_set is not None and pid not in id_set:
                continue

            # Resolve image: relative to spec dir, else absolute/CWD.
            img_raw = entry.get("image", "")
            img = (base_dir / img_raw)
            if not img.exists():
                img = Path(img_raw)
            if not img.exists():
                print(f"[customize] WARN: image not found for '{pid}': {img_raw}")
                continue

            problems.append({
                "id":                 pid,
                "dataset":            "customize",
                "split":              "custom",
                "question":           entry.get("question", ""),
                "choices":            entry.get("choices", []),
                "answer_label":       entry.get("answer_label", ""),
                "expected":           entry.get("expected", None),
                "image":              str(img),
                "tolerance":          0.01,
                "problem_type_graph": entry.get("problem_type_graph", ["Construction"]),
                "problem_type_goal":  entry.get("problem_type_goal", ["Reproduction"]),
                "hint_mode":          "none",
            })

    if id_set is None and sample and len(problems) > sample:
        import random
        problems = random.Random(seed).sample(problems, sample)
    problems.sort(key=lambda p: p["id"])
    return problems
