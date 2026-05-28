# `symbolic/`: core library

The Draw2Think harness: a frozen VLM drives the **GeoGebra constraint engine**
through typed tools. This package is the engine bridge, the tool layer, and the
model registry. The evaluation scripts in [`../eval/`](../eval/) import from here.

## Modules

| Module | What it is |
|---|---|
| `integrations/geogebra_api.py` | Selenium-driven bridge to GeoGebra (the constraint engine `E`). Executes construction commands, returns exact values via GeoGebra's Giac CAS, captures canvas PNGs. |
| `tools/geogebra_tools.py` | 92 typed 2D **ToolSpecs** + execution dispatch + `CanvasTracker`. Per-provider schemas via `build_openai_tools()` / `build_anthropic_tools()` / `build_gemini_tools()`. |
| `tools/geogebra_tools_solid.py` | 21 solid-geometry (3D) tools; `exec_with_solid_routing()` is the shared 2D/3D dispatch. |
| `utils/model_registry.py` | LLM registry (Gemini / Claude / GPT / Qwen / ...). `get_model()`, `make_client()`. List PDV-ready models: `python -m symbolic.utils.model_registry --vision --tool-calling --thinking`. |
| `utils/env_loader.py` | Loads `.env`, builds per-provider clients (incl. proxy routing via `USE_PROXY`). |

## Action types

All tools live in `geogebra_tools.py` (2D) and `geogebra_tools_solid.py` (3D):

| Type | State effect | N (2D) | Example |
|---|---|---|---|
| Construction (`add_*`) | `S' = S ∪ {o_new}` | 54 | `add_perpendicular_line` |
| Delete (`delete_object`) | `S' = S \ {target ∪ dependents}` | 1 | `delete_object` |
| Query (`query_*`) | returns exact `v ∈ ℝ`, leaves `S` unchanged | 24 | `query_angle` |
| Render (`render_*`) | visual style only | 13 | `render_set_color` |

`Construction` also covers `transform_* / set_* / rename_*`. The paper's Table 2
folds `delete` into construction for a single count of 55.

## Notes

- **Headless browser:** GeoGebra runs headless via Selenium. `python setup.py bootstrap` prepares Chrome for Testing and ChromeDriver; `webdriver-manager` remains available for system Chrome installs.
- **Offline bundle:** run `python setup.py bootstrap --offline-bundle` from the repo root for the local GeoGebra Math Apps bundle (~115 MB).
- **Selenium only:** `geogebra_api` exposes a `mode` argument, but only `selenium` is implemented.
- **Silent GeoGebra errors:** browser JS popups are suppressed at init; failed commands are captured in the process logs.
