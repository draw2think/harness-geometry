# `tests/` — tool / engine / pipeline tests

Sanity and coverage tests for the GeoGebra integration and the tool layer. Most
spin up a real GeoGebra session via Selenium and save PNGs, so they need a headless browser
and the offline bundle (see the [top-level README](../README.md)). Run from the
repo root.

## Tests

| Test | Checks |
|---|---|
| `test_offline_bundle.py`, `test_bundle_variants.py` | offline GeoGebra bundle works via Selenium (web / web3d / webSimple) |
| `test_geogebra_basic.py` | GeoGebra API integration smoke test |
| `test_geogebra_commands.py` | command coverage across CAS categories |
| `test_geogebra_algebra.py` | algebra / CAS commands |
| `test_geogebra_query.py`, `test_geogebra_complex_queries.py` | query interface + complex geometric queries |
| `test_geogebra_construct.py` | manual construction drawing |
| `test_run_checked.py` | post-execution validation of construction tool calls |
| `test_tools_r5.py`, `test_tools_v6_fixes.py`, `test_algebra_tools_v7.py` | tool additions / fixes per release |
| `test_solid_tools.py` | 3D solid-geometry tools (one session per solid type) |
| `test_text_latex.py` | LaTeX rendering in the `add_text` tool |
| `test_llm_to_draw.py`, `test_llm_stepwise_canvas.py`, `test_llm_text_in_QA.py` | LLM → GeoGebra render pipelines |
| `geogebra_render_common.py` | shared rendering helpers (not a test) |

## Run

```bash
python tests/test_offline_bundle.py      # quickest smoke test (bundle + Selenium)
python tests/test_geogebra_commands.py   # CAS command coverage
python tests/test_solid_tools.py         # 3D tools
```

## Notes

- Need a headless browser (Chromium/Chrome) + the GeoGebra offline bundle (`python setup.py download_bundle`).
- LLM-pipeline tests (`test_llm_*`) additionally need API keys in `.env`.
- Tests render to PNG; GeoGebra errors surface in stdout/logs, not in the image.
