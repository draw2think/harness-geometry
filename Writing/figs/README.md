# Paper Figure Scripts

This directory contains curated figure-generation scripts and small released
artifacts used for the Draw2Think paper. The scripts read saved evaluation
outputs under `eval/<dataset>/...` and default dataset files under `/data/...`;
they do not call model APIs.

Included scripts:

- `fig2_case/render_staged.py` — staged PDV canvas snapshots for the main case figure.
- `geogoal_sgvr/fidelity_bars.py` — GeoGoal fidelity-conditioned answer quality bars.
- `geogoal_sgvr/tol_sweep.py` — GeoGoal `T_i` tolerance sweep.
- `tool_trajectory_position/plot_tool_position.py` — tool-position distribution plots.
- `walltime/plot_walltime_lv3.py` — SolidGeo Level-3 wall-time distribution.

`harness_ablation/` contains the released micro-ablation traces and figure
assets for ToolSpec wording variants.
