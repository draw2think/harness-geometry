"""
Default configuration for eval runs.

CLI arguments override these defaults. Keep frequently changed run settings
here so the main scripts stay stable.

List available models:
    python -m symbolic.utils.model_registry
    python -m symbolic.utils.model_registry --vision --tool-calling --thinking

Examples:
    # Use all defaults from this file.
    python eval/test_agentic_geo_constructer.py

    # Override from CLI.
    python eval/test_agentic_geo_constructer.py --model kimi-k2.5 --sample 5
"""

# ── Model ────────────────────────────────────────────────────────────────────
# Model registry ID. See symbolic/utils/model_registry.py for the full list.
DEFAULT_MODEL = "gemini-3-flash-preview@medium"

# ── Dataset ──────────────────────────────────────────────────────────────────
DEFAULT_DATASET  = "geometry3k"
DEFAULT_DATA_DIR = "/data/geometry3k/val"
DEFAULT_SAMPLE   = 10              # None = full set; number = random sample
DEFAULT_SEED     = 42

# ── PGPS9K defaults ──────────────────────────────────────────────────────────
PGPS9K_DATA_DIR  = "/data/PGPS9K"
PGPS9K_IMAGE_DIR = "Diagram_Visual"   # "Diagram" for clean (no labels)

# ── Eval parameters ──────────────────────────────────────────────────────────
MAX_TURNS   = 30          # Maximum conversation turns
TEMPERATURE = 0.0         # Generation temperature
STREAM      = False       # Non-streaming evals are more stable; --no-stream=False enables streaming.
THINKING_LEVEL = "medium"   # Gemini 3 thinking: minimal/low/medium/high
REASONING_EFFORT = "high" # OpenAI reasoning_effort: none/low/medium/high/xhigh
MAX_COMPLETION_TOKENS = 128000  # OpenAI max_completion_tokens; GPT-5.4 upper bound is 128K.
TTFT_TIMEOUT = 120        # Per-call LLM timeout in seconds; None = unlimited

# ── Mode ─────────────────────────────────────────────────────────────────────
DEFAULT_MODE = "construct"  # "direct" | "construct"
DEFAULT_HINT = "none"       # "none" | "points" | "logic_form"

# ── Debugging / figure generation ────────────────────────────────────────────
SAVE_PER_TURN = False       # --save-screenshot-per-turn: save a canvas PNG after each turn
