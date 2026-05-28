"""Sweep T_i tolerance threshold; split by logical vs numerical T_i.
Produce a line plot showing the fidelity gradient.

Input:
  <repo>/eval/geogoal/{id}/*_ti.json
  /data/geogoal_sgvr/data/test-00000-of-00001.parquet

Output (both under this script's directory):
  tol_sweep.json          — raw sweep data
  tol_sweep.{pdf,png}     — the line plot

Run: python plot_tol_sweep.py
"""
import ast
import json
from pathlib import Path
import pandas as pd

# Absolute paths so the script works from anywhere
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[3] / "eval" / "geogoal"
SLUG = "gemini-3-flash-preview@medium"
DATA = Path("/data/geogoal_sgvr/data/test-00000-of-00001.parquet")
FIG_DIR = HERE  # save alongside the script


def _is_logical(raw_s: str) -> bool:
    """GT strings '0', '1', '90', '180' are logical encodings of cong/para/perp/etc.
    Fractions 'n/m' (and integers outside {0,1,90,180}) are numerical."""
    s = raw_s.strip()
    if s in ("0", "0.0", "1", "1.0", "90", "180"):
        return True
    return False


def compute_at_tol(ti_data: list, tol: float) -> dict:
    """Recount Track A metrics at given abs tolerance (with mod180 circular)."""
    logical_pass = logical_total = 0
    numerical_pass = numerical_total = 0
    all_pass = all_total = 0
    for rec in ti_data:
        pred = rec["pred"]
        gt = rec["gt"]
        mod180 = rec["mod180"]
        raw_s = rec["raw_gt"]
        logical = _is_logical(raw_s)

        all_total += 1
        if logical:
            logical_total += 1
        else:
            numerical_total += 1

        if pred is None or gt is None:
            continue

        if mod180:
            d = abs(pred - gt) % 180
            eff = min(d, 180 - d)
            ok = eff < tol
        else:
            ok = abs(pred - gt) < tol

        if ok:
            all_pass += 1
            if logical:
                logical_pass += 1
            else:
                numerical_pass += 1

    return {
        "tol": tol,
        "all_pass": all_pass, "all_total": all_total,
        "all_sr": all_pass / all_total if all_total else 0,
        "logical_pass": logical_pass, "logical_total": logical_total,
        "logical_sr": logical_pass / logical_total if logical_total else 0,
        "numerical_pass": numerical_pass, "numerical_total": numerical_total,
        "numerical_sr": numerical_pass / numerical_total if numerical_total else 0,
    }


def collect_records() -> list[dict]:
    """Walk all ti.json and collect per-T_i records with raw GT."""
    df = pd.read_parquet(DATA)
    raw_map = {row["id"]: ast.literal_eval(row["answer"]) for _, row in df.iterrows()}

    records: list[dict] = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.name.startswith("geogal_"):
            continue
        ti = json.loads((d / f"{SLUG}_ti.json").read_text())
        raw_gt = raw_map.get(d.name, [])
        for i, c in enumerate(ti["compare"]["per_idx"]):
            records.append({
                "pred": c.get("pred"),
                "gt": c.get("gt"),
                "mod180": bool(c.get("mod180")),
                "raw_gt": str(raw_gt[i]) if i < len(raw_gt) else "",
            })
    return records


def main():
    cache = FIG_DIR / "tol_sweep.json"
    if cache.exists():
        results = json.loads(cache.read_text())
        print(f"Loaded cached sweep from {cache}")
    else:
        records = collect_records()
        print(f"Collected {len(records)} T_i records")
        # Sweep tolerances (log scale): sparse in the extremes, DENSE in [1e-2, 1]
        # where both curves change meaningfully. 17 points total.
        tols = [
            1e-4, 3e-4, 1e-3, 3e-3,           # sparse head
            1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 7e-2,  # dense middle (lower decade)
            1e-1, 1.5e-1, 2e-1, 3e-1, 5e-1, 7e-1,  # dense middle (upper decade)
            1.0, 2.0, 3.0,                    # sparse tail
        ]
        results = [compute_at_tol(records, t) for t in tols]
        cache.write_text(json.dumps(results, indent=2))
        print(f"Saved sweep data: {cache}")

    # Pretty-print
    print()
    print(f"{'tol':>8}  {'All SR':>7}  {'Logical SR':>11}  {'Numerical SR':>13}")
    for r in results:
        print(f"  {r['tol']:>6.0e}  {r['all_sr']*100:6.2f}%  "
              f"{r['logical_sr']*100:9.2f}%  {r['numerical_sr']*100:11.2f}%")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'Liberation Serif']
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        xs = [r["tol"] for r in results]

        ms = 2.5   # small markers; denser sweep needs compact symbols
        ax.plot(xs, [100 * r["logical_sr"] for r in results],
                marker="o", markersize=ms, color="#2b7bba", linewidth=1.5,
                label=f"Structural T$_i$ ({results[0]['logical_total']} items)")
        ax.plot(xs, [100 * r["numerical_sr"] for r in results],
                marker="s", markersize=ms, color="#c25757", linewidth=1.5,
                label=f"Numerical T$_i$ ({results[0]['numerical_total']} items)")
        ax.plot(xs, [100 * r["all_sr"] for r in results],
                marker="^", markersize=ms, color="#555555", linewidth=1.1, linestyle="--",
                label=f"All T$_i$ ({results[0]['all_total']} items)")

        # Reference line: predicate-level ceiling from engine-exact verifier
        ax.axhline(95.94, color="#2e9b4a", linewidth=1.1, linestyle=":")
        # Explicit in-plot label so the reader doesn't have to parse the legend
        ax.text(3.0, 97.5, "Predicate-level SR = 95.94% (engine-exact ceiling)",
                fontsize=7, color="#1f7038", ha="right", va="center")

        ax.set_xscale("log")
        ax.set_xlabel("Numerical-match tolerance on T$_i$ (abs.)")
        ax.set_ylabel("Pass rate (%)")
        ax.grid(alpha=0.3)
        # Float legend up so it does not collide with the "tol=0.01 (default)"
        # callout near the bottom of the axvline.
        ax.legend(fontsize=7.5, loc="lower right",
                  bbox_to_anchor=(0.99, 0.10), framealpha=0.92)
        ax.set_ylim(40, 102)  # zoom; a bit of headroom for ceiling label

        # Vertical line at our chosen 0.01 tol; split callout into two labels,
        # one on each side of the line, so the legend can sit just above.
        ax.axvline(0.01, color="#888", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(0.0093, 42, "(default)", fontsize=7, color="#666",
                ha="right", va="bottom")
        ax.text(0.0108, 42, "tol = 0.01", fontsize=7, color="#666",
                ha="left", va="bottom")

        # Annotate the plateau explicitly (the paper's headline)
        ax.annotate("structural plateau\n(tolerance-invariant)",
                    xy=(5e-1, 88.3), xytext=(2.5e0, 74),
                    fontsize=7.5, color="#1a4a6a", ha="right",
                    arrowprops=dict(arrowstyle="->", color="#1a4a6a",
                                    linewidth=0.8, alpha=0.7))

        fig.tight_layout()

        png = FIG_DIR / "tol_sweep.png"
        pdf = FIG_DIR / "tol_sweep.pdf"
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        print(f"\nSaved plots: {png} and {pdf}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
