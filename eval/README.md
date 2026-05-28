# `eval/`: evaluation harness

Runs the Draw2Think PDV constructor and baselines across 14 geometry benchmarks,
plus rendering (GenExam) and construction-fidelity (GeoGoal) evals. Imports the
core library from [`../symbolic/`](../symbolic/). For install + quick start, see
the [top-level README](../README.md).

## Scripts

| Script | What it runs |
|---|---|
| `test_agentic_geo_constructer.py` | ★ main agentic eval: PDV construct loop over 14 benchmarks (`--mode construct` / `direct`) |
| `eval_baseline.py` | single-turn, no-tool baseline (BL); online by default, Gemini Batch API via `--collect` |
| `eval_common.py`, `eval_config.py` | shared answer parsing / validation / judge + default run config |
| `eval_genexam.py` | GenExam text→figure: construct → render → GPT vision judge (phased via `--generate-only` / `--judge-only` / `--score-only`) |
| `geogoal/` | GeoGoal suite: runner + Newclid predicate check + T_i expression eval |
| `ablation_*.py` | ToolSpec-description / query / delete ablations (paper §5) |
| `analyze_answer_source.py`, `compute_*.py` | answer-source taxonomy and paper-number scripts |
| `rescore_*.py` | offline re-scoring from saved logs (no model calls) |
| `download_datasets.py` | dataset fetch / status check |
| `loaders/` | 14 dataset loaders (`DATASET_LOADERS` registry) |
| `prompts/` | per-benchmark system + judge prompts (JSON) |
| `genexam/` | released GenExam-math result summaries (`genexam_math_{high,medium}.json`) |

## Output layout

Each run writes per-problem files under `eval/<dataset>/<id>/`, distinguished by filename:

```
eval/pgps9k/prob_13/
  <model>_result.json      # answer, pass/fail, token usage, per-turn process
  <model>_log.txt          # full transcript: LLM output + tool calls + results
  <model>_canvas.png       # final GeoGebra construction
  <model>_baseline_*.json  # baseline (BL) counterparts
```

Re-runs overwrite same-named files; `summary_*.json` is rebuilt from all `*_result.json`. Data directories and run products are git-ignored; only the released summaries under `genexam/` are tracked.

## Configuration (`eval_config.py`)

CLI flags override these defaults:

| Key | Default | Note |
|---|---|---|
| `DEFAULT_MODEL` | `gemini-3-flash-preview@medium` | `@<level>` suffix = thinking level |
| `DEFAULT_DATASET` / `DEFAULT_DATA_DIR` | `geometry3k` / `/data/geometry3k/val` | |
| `DEFAULT_MODE` | `construct` | `construct` (hide choices, force build) or `direct` |
| `MAX_TURNS` / `TEMPERATURE` | `30` / `0.0` | agentic only |
| `THINKING_LEVEL` | `medium` | Gemini: minimal/low/medium/high; DashScope: on/off |

## Datasets

| `--dataset` | `--data_dir` | Size |
|---|---|---|
| `geometry3k` | `/data/geometry3k/{val,test}` | val 300 / test 601 |
| `pgps9k` | `/data/PGPS9K` | test 1000 |
| `unigeo` | `/data/UniGeo` | 754 (4-way MC) |
| `mathverse` | `/data/mathverse` | 510 (Vision-Only × Plane) |
| `geolaux` | `/data/geolaux` | 221 (calculation split) |
| `genexam` / `geogoal` / `solidgeo` / … | `/data/<name>` | full registry in `loaders/` |

`--id a,b,c` runs specific problems; `--exclude-book Geometry3K` drops the PGPS9K↔Geo3K overlap.

## Notes

- **Headless browser.** GeoGebra runs through Selenium. `python setup.py bootstrap` prepares Chrome for Testing and ChromeDriver; `webdriver-manager` remains available for system Chrome installs.
- **GeoGebra runtime.** Prepare the runtime from the repo root before canvas/eval runs; see the top-level installation notes and [`../NOTICE`](../NOTICE).
- **GeoGoal needs Newclid.** `geogoal/geogoal_verifier.py` imports a local Newclid clone for predicate checks; set the `NEWCLID_SRC` env var to its `src/` directory.
- **Datasets live under `/data`.** Override paths with `--data_dir`, or fetch datasets with `python eval/download_datasets.py`.
- **Baseline batch mode.** `eval_baseline.py` runs online by default; the Gemini Batch API path (~50% cost) is reached via `--collect <job>`.
- **Silent GeoGebra errors.** Runs suppress GeoGebra browser popups; failed commands are captured in `*_log.txt`.
