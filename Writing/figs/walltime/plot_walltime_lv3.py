"""Wall-time distribution for Gemini-3-Flash @high on SolidGeo-hard (Lv.3, N=177).

Three distributions side-by-side:
- BL: single LLM call, hard-capped at 120 s (Google GenAI SDK http timeout).
- CT total: multi-turn PDV cumulative wall-time (per-turn 120 s cap, total unbounded).
- CT per-turn: average per-turn time = t_final / turns.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType (ICML/IEEE-compliant), not Type 3 bitmap
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "Liberation Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

DATA_JSON = Path("/data/solidgeo/SolidGeo.json")
EVAL_DIR = Path(__file__).resolve().parents[3] / "eval" / "solidgeo"
OUT_DIR = Path(__file__).resolve().parent

MODEL_SLUG = "gemini-3-flash-preview@high"

# ── Load Lv.3 qa_ids ──────────────────────────────────────────────────
raw = json.loads(DATA_JSON.read_text())
items = list(raw.values()) if isinstance(raw, dict) else raw
lv3_ids = [str(d.get("qa_id", "")) for d in items
           if d.get("complexity_level") == "Level 3"]

# ── Collect walltimes ─────────────────────────────────────────────────
bl_times: list[float] = []
ct_times: list[float] = []
ct_per_turn: list[float] = []
ct_timeout = 0

for qid in lv3_ids:
    bl_f = EVAL_DIR / qid / f"{MODEL_SLUG}_baseline_result.json"
    ct_f = EVAL_DIR / qid / f"{MODEL_SLUG}_result.json"
    if bl_f.exists():
        bl = json.loads(bl_f.read_text())
        bl_times.append(bl.get("t_last_sec", 0) or 0)
    if ct_f.exists():
        ct = json.loads(ct_f.read_text())
        t = float(ct.get("t_final_sec") or ct.get("t_last_sec") or 0)
        ct_times.append(t)
        turns = int(ct.get("process", {}).get("total_turns") or 0)
        _s = json.dumps(ct).lower()
        had_timeout = ("ttft>120" in _s) or ("exceeded 120" in _s)
        # Count the timed-out turn itself as one attempted turn, so 9 turn-1
        # deaths (turns=0, timeout) and other timeout cases are included with
        # consistent divisor accounting.
        eff_turns = turns + (1 if had_timeout else 0)
        if eff_turns > 0 and t > 0:
            ct_per_turn.append(t / eff_turns)
        s = json.dumps(ct).lower()
        if "ttft>120" in s or "exceeded 120" in s:
            ct_timeout += 1

print(f"Lv.3 IDs: {len(lv3_ids)}  |  BL N={len(bl_times)}  "
      f"CT N={len(ct_times)}  CT-per-turn N={len(ct_per_turn)}")
print(f"BL           median={np.median(bl_times):6.1f}s  "
      f"max={max(bl_times):.1f}s")
print(f"CT total     median={np.median(ct_times):6.1f}s  "
      f"max={max(ct_times):.1f}s  (120s-skipped={ct_timeout})")
print(f"CT per-turn  median={np.median(ct_per_turn):6.1f}s  "
      f"max={max(ct_per_turn):.1f}s")

# ── Bucketing ────────────────────────────────────────────────────────
BINS = [0, 10, 20, 30, 60, 90, 120, float("inf")]
LABELS = ["0–10", "10–20", "20–30", "30–60", "60–90", "90–120", ">120"]

bl_h = np.histogram(bl_times, bins=BINS)[0]
ct_h = np.histogram(ct_times, bins=BINS)[0]
ctpt_h = np.histogram(ct_per_turn, bins=BINS)[0]

print("\nbucket    | BL   CT-tot  CT-per-turn")
for i, lab in enumerate(LABELS):
    print(f"{lab:>8}s | {bl_h[i]:>3}   {ct_h[i]:>3}     {ctpt_h[i]:>3}")

# ── Plot ──────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(5.6, 2.6))

x = np.arange(len(LABELS))
w = 0.27

bl_color = "#c44e52"    # BL muted red
ct_color = "#2b7bba"    # CT muted blue (matches other figs in this paper)
ctpt_color = "#a8c8e0"  # CT per-turn pale muted blue

b1 = ax.bar(x - w, bl_h, w, label="BL single call",
            color=bl_color, alpha=0.9, edgecolor="white", linewidth=0.5)
b2 = ax.bar(x,     ct_h, w, label="CT total",
            color=ct_color, alpha=0.9, edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + w, ctpt_h, w, label="CT per-turn",
            color=ctpt_color, alpha=0.95, edgecolor="white", linewidth=0.5,
            hatch="///")

# Bar-top counts (only non-zero)
for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        if h == 0:
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{int(h)}",
                ha="center", va="bottom", fontsize=6.5)

# 120s hard cap line (annotation just above the dashed line, top of plot)
y_top = max(max(bl_h), max(ct_h), max(ctpt_h)) * 1.30
ax.axvline(x=5.5, color="gray", linestyle=":", linewidth=0.8)
ax.text(5.48, y_top * 0.22, "120 s cap",
        fontsize=6.5, color="gray", ha="right", va="bottom", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}s" for lab in LABELS])
ax.set_ylabel(f"SolidGeo-hard (Lv.3) problems (N={len(lv3_ids)})")
ax.set_ylim(0, y_top)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper right", fontsize=7, frameon=False,
          handlelength=1.6, handletextpad=0.5,
          borderaxespad=0.3)

plt.tight_layout()
pdf_out = OUT_DIR / "walltime_solidgeo_lv3.pdf"
png_out = OUT_DIR / "walltime_solidgeo_lv3.png"
fig.savefig(pdf_out, bbox_inches="tight")
fig.savefig(png_out, bbox_inches="tight", dpi=180)
print(f"\nSaved: {pdf_out}\n       {png_out}")
