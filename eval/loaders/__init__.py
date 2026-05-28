"""Dataset loaders for geometry eval benchmarks.

Each loader returns a list of problem dicts with standardized keys:
  id, dataset, split, question, choices, answer_label, expected,
  expected_raw, image, tolerance, hint_mode, ...

Usage:
    from loaders import DATASET_LOADERS
    loader = DATASET_LOADERS["geometry3k"]
    problems = loader(data_dir=Path("/data/geometry3k/test"), sample=10)
"""

from loaders.geometry3k import load_geometry3k
from loaders.pgps9k import load_pgps9k
from loaders.unigeo import load_unigeo
from loaders.mathverse import load_mathverse
from loaders.solidgeo import load_solidgeo
from loaders.mathcanvas import load_mathcanvas
from loaders.ggbench import load_ggbench
from loaders.geolaux import load_geolaux
from loaders.geosketch import load_geosketch
from loaders.genexam import load_genexam
from loaders.mathvista import load_mathvista
from loaders.olympiadbench import load_olympiadbench
from loaders.geogoal_sgvr import load_geogoal_sgvr
from loaders.customize import load_customize

DATASET_LOADERS = {
    "geometry3k":     load_geometry3k,
    "pgps9k":         load_pgps9k,
    "unigeo":         load_unigeo,
    "mathverse":      load_mathverse,
    "mathvista":      load_mathvista,
    "solidgeo":       load_solidgeo,
    "mathcanvas":     load_mathcanvas,
    "ggbench":        load_ggbench,
    "geolaux":        load_geolaux,
    "geosketch":      load_geosketch,
    "genexam":        load_genexam,
    "olympiadbench":  load_olympiadbench,
    "geogoal_sgvr":   load_geogoal_sgvr,
    "customize":      load_customize,
}

__all__ = ["DATASET_LOADERS"] + list(DATASET_LOADERS.keys())
