"""
Tool Appearance Position Distribution in Construction Trajectories.

Two views:
  1. Strip plot: each tool as a row, dots at trajectory percentile (ok=green, fail=red)
  2. KDE curves: y=density, x=trajectory percentile, one curve per tool (top N)

Usage:
    python Writing/figs/tool_trajectory_position/plot_tool_position.py
"""

import json
import glob
import os
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'Liberation Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── Config ──────────────────────────────────────────────────
BASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "eval"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = [
    'geometry3k', 'pgps9k', 'unigeo', 'mathverse',
    'mathvista', 'geolaux', 'geosketch', 'olympiadbench',
]

TOP_N_STRIP = 25   # strip plot: top N tools
TOP_N_KDE = 12     # KDE plot: top N tools (too many curves = unreadable)


# ── Data collection ────────────────────────────────────────
def collect_tool_positions():
    """Return {tool_name: [(position_pct, ok_bool), ...]}."""
    tool_positions = defaultdict(list)
    for bench in BENCHMARKS:
        files = (glob.glob(f"{BASE_ROOT}/{bench}/*/gemini*_result.json")
                 + glob.glob(f"{BASE_ROOT}/{bench}/*/gpt*_result.json"))
        ct_files = [f for f in files
                    if 'baseline' not in f and 'hint' not in f]
        for f in ct_files:
            try:
                d = json.load(open(f))
                turns = d.get('process', {}).get('turns', [])
                all_calls = []
                for turn in turns:
                    for tc in turn.get('tool_calls', []):
                        fn = tc.get('fn', '')
                        ok = tc.get('ok', True)
                        if fn:
                            all_calls.append((fn, ok))
                total = len(all_calls)
                if total == 0:
                    continue
                for idx, (fn, ok) in enumerate(all_calls):
                    pct = idx / max(total - 1, 1) * 100
                    tool_positions[fn].append((pct, ok))
            except Exception:
                pass
    return tool_positions


# ── Plot 1: Strip plot ─────────────────────────────────────
def make_strip_plot(tool_positions, top_n=TOP_N_STRIP):
    freq = {fn: len(pos) for fn, pos in tool_positions.items()}
    top_tools = sorted(freq, key=freq.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, tool in enumerate(reversed(top_tools)):
        positions = tool_positions[tool]
        ok_pcts = [p for p, ok in positions if ok]
        fail_pcts = [p for p, ok in positions if not ok]

        y_ok = [i + np.random.uniform(-0.2, 0.2) for _ in ok_pcts]
        y_fail = [i + np.random.uniform(-0.2, 0.2) for _ in fail_pcts]

        ax.scatter(ok_pcts, y_ok, c='#43A047', s=1.5, alpha=0.15, rasterized=True)
        if fail_pcts:
            ax.scatter(fail_pcts, y_fail, c='#E53935', s=4, alpha=0.4,
                       marker='x', rasterized=True)

        median = np.median([p for p, _ in positions])
        ax.plot([median], [i], 'k|', markersize=12, markeredgewidth=2)
        q25, q75 = np.percentile([p for p, _ in positions], [25, 75])
        ax.plot([q25, q75], [i, i], color='#333333', linewidth=2,
                solid_capstyle='round')

    ax.set_yticks(range(len(top_tools)))
    ax.set_yticklabels(list(reversed(top_tools)), fontsize=8)
    ax.set_xlabel('Position in trajectory (%)', fontsize=11)
    ax.set_title(f'Tool Appearance Position in Construction Trajectory (top {top_n} tools)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-2, 102)
    ax.grid(axis='x', alpha=0.2)
    ax.axvline(x=50, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#43A047',
               markersize=6, label='OK call'),
        Line2D([0], [0], marker='x', color='#E53935', markersize=6,
               label='Failed call', linestyle='None'),
        Line2D([0], [0], color='#333333', linewidth=2,
               label='IQR (25th–75th pct)'),
        Line2D([0], [0], marker='|', color='k', markersize=10,
               markeredgewidth=2, label='Median', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_strip.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved strip plot to {OUT_DIR}/tool_trajectory_strip.pdf/.png")
    plt.close(fig)


# ── Plot 2: KDE curves ────────────────────────────────────
def make_kde_plot(tool_positions, top_n=TOP_N_KDE):
    freq = {fn: len(pos) for fn, pos in tool_positions.items()}
    top_tools = sorted(freq, key=freq.get, reverse=True)[:top_n]

    bins = np.linspace(0, 100, 101)  # 1% resolution
    sigma = 3  # Gaussian smoothing

    # Assign colors from a qualitative colormap
    cmap = cm.get_cmap('tab20', top_n)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, tool in enumerate(top_tools):
        positions = [p for p, _ in tool_positions[tool]]
        hist, _ = np.histogram(positions, bins=bins, density=True)
        smooth = gaussian_filter1d(hist, sigma)
        x = (bins[:-1] + bins[1:]) / 2
        color = cmap(i)
        ax.plot(x, smooth, linewidth=2, color=color, alpha=0.85,
                label=tool.replace('_', ' '))
        # Fill lightly
        ax.fill_between(x, 0, smooth, alpha=0.08, color=color)

    ax.set_xlabel('Position in trajectory (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Tool Appearance Density across Trajectory Position',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)

    # Vertical guides
    ax.axvline(x=20, color='gray', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.axvline(x=50, color='gray', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.axvline(x=80, color='gray', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.text(10, ax.get_ylim()[1]*0.95, 'Construction\nphase', fontsize=8,
            ha='center', va='top', color='gray')
    ax.text(50, ax.get_ylim()[1]*0.95, 'Assembly\nphase', fontsize=8,
            ha='center', va='top', color='gray')
    ax.text(90, ax.get_ylim()[1]*0.95, 'Verification\nphase', fontsize=8,
            ha='center', va='top', color='gray')

    ax.legend(fontsize=8, loc='upper center', ncol=4, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_kde.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved KDE plot to {OUT_DIR}/tool_trajectory_kde.pdf/.png")
    plt.close(fig)


# ── Plot 2b: KDE with zoom panel ───────────────────────────
def make_kde_zoom(tool_positions, top_n=TOP_N_KDE):
    """Two-panel: top = full range, bottom = zoom into Assembly phase (20-80%, y≤0.03)."""
    freq = {fn: len(pos) for fn, pos in tool_positions.items()}
    top_tools = sorted(freq, key=freq.get, reverse=True)[:top_n]

    bins = np.linspace(0, 100, 101)
    sigma = 3
    cmap = plt.colormaps.get_cmap('tab20')

    # Pre-compute curves
    curves = []
    for i, tool in enumerate(top_tools):
        positions = [p for p, _ in tool_positions[tool]]
        hist, _ = np.histogram(positions, bins=bins, density=True)
        smooth = gaussian_filter1d(hist, sigma)
        x = (bins[:-1] + bins[1:]) / 2
        curves.append((tool, x, smooth, cmap(i % 20)))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9),
                                          gridspec_kw={"height_ratios": [2, 1.5]},
                                          sharex=False)

    # ── Top panel: full range ──
    for tool, x, smooth, color in curves:
        ax_top.plot(x, smooth, linewidth=2, color=color, alpha=0.85,
                    label=tool.replace('_', ' '))
        ax_top.fill_between(x, 0, smooth, alpha=0.06, color=color)

    ax_top.set_ylabel('Density', fontsize=11)
    ax_top.set_title('Tool Appearance Density across Trajectory Position',
                     fontsize=13, fontweight='bold')
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, 0.15)
    ax_top.grid(alpha=0.15)
    for vx in (20, 80):
        ax_top.axvline(x=vx, color='gray', linewidth=0.6, linestyle=':', alpha=0.4)
    ax_top.text(10, 0.14, 'Construction', fontsize=8, ha='center', color='gray')
    ax_top.text(50, 0.14, 'Assembly', fontsize=8, ha='center', color='gray')
    ax_top.text(90, 0.14, 'Verification', fontsize=8, ha='center', color='gray')

    # Zoom rectangle indicator
    from matplotlib.patches import Rectangle
    rect = Rectangle((20, 0), 60, 0.03, linewidth=1.5, edgecolor='#E65100',
                      facecolor='#FFF3E0', alpha=0.3, linestyle='--')
    ax_top.add_patch(rect)
    ax_top.annotate('zoom below', xy=(50, 0.03), fontsize=8, color='#E65100',
                    ha='center', va='bottom')

    ax_top.legend(fontsize=7, loc='upper center', ncol=4, framealpha=0.9,
                  bbox_to_anchor=(0.5, -0.06))

    # ── Bottom panel: zoom into Assembly (20-80%, y 0-0.03) ──
    for tool, x, smooth, color in curves:
        ax_bot.plot(x, smooth, linewidth=2.5, color=color, alpha=0.9)
        ax_bot.fill_between(x, 0, smooth, alpha=0.1, color=color)

    ax_bot.set_xlim(15, 85)
    ax_bot.set_ylim(0, 0.03)
    ax_bot.set_xlabel('Position in trajectory (%)', fontsize=11)
    ax_bot.set_ylabel('Density (zoomed)', fontsize=11)
    ax_bot.set_title('Assembly Phase Detail (20%–80%)', fontsize=11)
    ax_bot.grid(alpha=0.15)
    for vx in (20, 50, 80):
        ax_bot.axvline(x=vx, color='gray', linewidth=0.6, linestyle=':', alpha=0.3)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_kde_zoom.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved KDE zoom plot to {OUT_DIR}/tool_trajectory_kde_zoom.pdf/.png")
    plt.close(fig)


# ── Plot 2c: Grouped KDE ──────────────────────────────────
TOOL_GROUPS = {
    'Primitive Construction': {
        'tools': ['add_point', 'add_line', 'add_segment', 'add_ray',
                  'add_vector', 'add_circle', 'add_arc', 'add_sector',
                  'add_semicircle', 'add_circle_3_points', 'add_ellipse',
                  'add_hyperbola', 'add_function', 'add_text',
                  # 3D primitives
                  'add_point3d', 'add_cube', 'add_cylinder', 'add_cone',
                  'add_prism', 'add_pyramid', 'add_sphere'],
        'color': '#1565C0', 'light': '#BBDEFB',
    },
    'Derived Construction': {
        'tools': ['add_intersect', 'add_midpoint', 'add_perpendicular_line',
                  'add_perpendicular_bisector', 'add_parallel_line',
                  'add_angle_bisector', 'add_polygon', 'add_regular_polygon',
                  'add_tangent', 'add_tangent_conic_conic', 'add_vertex',
                  'add_incircle', 'add_center', 'add_triangle_center',
                  'add_point_on', 'add_roots', 'add_turning_point',
                  'add_angle', 'add_distance', 'add_area'],
        'color': '#E65100', 'light': '#FFE0B2',
    },
    'Transform & Utility': {
        'tools': ['transform_rotate', 'transform_reflect_point',
                  'transform_reflect_line', 'transform_translate',
                  'transform_dilate', 'add_slider', 'set_value',
                  'delete_object', 'rename_object',
                  'set_label_visible', 'set_object_visible'],
        'color': '#2E7D32', 'light': '#C8E6C9',
    },
    'Query & Verification': {
        'tools': ['query_angle', 'query_distance', 'query_area',
                  'query_length', 'query_perimeter', 'query_solve',
                  'query_nsolve', 'query_radius', 'query_slope',
                  'query_x_coord', 'query_y_coord',
                  'query_are_parallel', 'query_are_perpendicular',
                  'query_are_collinear', 'query_are_concyclic',
                  'query_are_congruent', 'query_are_equal',
                  'query_is_tangent', 'query_is_in_region',
                  'query_is_defined', 'query_dependents',
                  'query_definite_integral', 'query_function_max',
                  'query_function_min',
                  # 3D queries
                  'query_volume', 'query_surface_area', 'query_coords3d'],
        'color': '#8E24AA', 'light': '#E1BEE7',
    },
}


def make_grouped_kde(tool_positions):
    """4-group KDE: thick group curve only."""
    bins = np.linspace(0, 100, 201)  # finer bins to spread discrete positions
    sigma = 4  # enough smoothing to absorb discretization from short trajectories
    x = (bins[:-1] + bins[1:]) / 2

    plt.rcParams.update({'font.size': 19, 'axes.labelsize': 20,
                          'axes.titlesize': 22, 'xtick.labelsize': 18,
                          'ytick.labelsize': 18, 'legend.fontsize': 17})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [2, 1.5]})

    # Pre-compute group curves
    group_curves = {}
    for gname, ginfo in TOOL_GROUPS.items():
        group_all = []
        per_tool = []
        for tool in ginfo['tools']:
            if tool not in tool_positions:
                continue
            positions = [p for p, _ in tool_positions[tool]]
            group_all.extend(positions)
            if len(positions) > 50:
                hist, _ = np.histogram(positions, bins=bins, density=True)
                per_tool.append(gaussian_filter1d(hist, sigma))
        if group_all:
            hist, _ = np.histogram(group_all, bins=bins, density=True)
            group_curves[gname] = {
                'aggregate': gaussian_filter1d(hist, sigma),
                'per_tool': per_tool,
                'color': ginfo['color'],
                'n': len(group_all),
            }

    # Normalize to percentage (sum of all groups = 100% at each bin)
    all_agg = np.zeros_like(x)
    for gc in group_curves.values():
        all_agg += gc['aggregate']
    for gc in group_curves.values():
        gc['aggregate_pct'] = gc['aggregate'] / np.clip(all_agg, 1e-10, None) * 100
        gc['per_tool_pct'] = []
        for pt in gc['per_tool']:
            gc['per_tool_pct'].append(pt / np.clip(all_agg, 1e-10, None) * 100)

    for ax_idx, (ax, xlim, ylim, use_pct) in enumerate([
            (axes[0], (0, 100), 100, True),
            (axes[1], (15, 85), 50, True)]):

        for gname, gc in group_curves.items():
            agg = gc['aggregate_pct']
            # Group aggregate thick curve only (no per-tool thin lines in % view)
            ax.plot(x, agg, linewidth=3, color=gc['color'],
                    alpha=0.9, label=f"{gname} (n={gc['n']:,})")
            ax.fill_between(x, 0, agg, alpha=0.12, color=gc['color'])

        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel('Share (%)', fontsize=20)
        ax.grid(alpha=0.12)

        if ax_idx == 0:
            ax.set_title('Tool Usage Share by Functional Group',
                         fontsize=22, fontweight='bold')
            top_y = ax.get_ylim()[1]
            # Phase background shading
            ax.axvspan(0, 25, alpha=0.06, color='#1565C0')
            ax.axvspan(25, 75, alpha=0.04, color='#FF9800')
            ax.axvspan(75, 100, alpha=0.06, color='#8E24AA')
            ax.text(12.5, top_y * 0.94, 'Initialization', fontsize=18,
                    ha='center', color='#222', fontstyle='italic')
            ax.text(50, top_y * 0.94, 'Composition', fontsize=18,
                    ha='center', color='#222', fontstyle='italic')
            ax.text(87.5, top_y * 0.94, 'Readout', fontsize=18,
                    ha='center', color='#222', fontstyle='italic')
            ax.legend(fontsize=17, loc='upper center', framealpha=0.9, ncol=1,
                      bbox_to_anchor=(0.5, 0.88))
        else:
            ax.set_xlabel('Position in trajectory (%)', fontsize=20)

    # ── Zoom connection lines between panels ──
    top_ax = axes[0]
    bot_ax = axes[1]
    # Mark the zoom range (15--85) on the upper panel so the reader sees
    # exactly which interval is expanded below.
    for x_pos in (15, 85):
        top_ax.axvline(x_pos, color='#E65100', linewidth=1.6,
                       linestyle='--', alpha=0.65, zorder=2.5)
    # Force explicit ticks at 15 and 85 on the upper panel.
    top_xticks = sorted(set(list(top_ax.get_xticks()) + [15, 85]))
    top_ax.set_xticks([t for t in top_xticks if 0 <= t <= 100])
    bot_ax.set_xticks([15, 25, 50, 75, 85])

    # Connection lines from upper-panel zoom-edge to lower-panel top corners,
    # solid + thicker so the zoom relationship is visually obvious.
    from matplotlib.patches import ConnectionPatch
    for x_pos in (15, 85):
        con = ConnectionPatch(
            xyA=(x_pos, 0), coordsA=top_ax.transData,
            xyB=(x_pos, bot_ax.get_ylim()[1]), coordsB=bot_ax.transData,
            color='#E65100', linewidth=2.4, linestyle='--', alpha=0.85)
        fig.add_artist(con)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_grouped.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved grouped KDE to {OUT_DIR}/tool_trajectory_grouped.pdf/.png")
    plt.close(fig)


def make_grouped_kde_1row(tool_positions):
    """Single-row variant of make_grouped_kde for the short/preprint version
    (drops the zoomed-in lower panel)."""
    bins = np.linspace(0, 100, 201)
    sigma = 4
    x = (bins[:-1] + bins[1:]) / 2

    plt.rcParams.update({'font.size': 19, 'axes.labelsize': 20,
                          'axes.titlesize': 22, 'xtick.labelsize': 18,
                          'ytick.labelsize': 18, 'legend.fontsize': 17})
    fig, ax = plt.subplots(figsize=(14, 6))

    group_curves = {}
    for gname, ginfo in TOOL_GROUPS.items():
        group_all = []
        for tool in ginfo['tools']:
            if tool not in tool_positions:
                continue
            group_all.extend(p for p, _ in tool_positions[tool])
        if group_all:
            hist, _ = np.histogram(group_all, bins=bins, density=True)
            group_curves[gname] = {
                'aggregate': gaussian_filter1d(hist, sigma),
                'color': ginfo['color'],
                'n': len(group_all),
            }

    all_agg = np.zeros_like(x)
    for gc in group_curves.values():
        all_agg += gc['aggregate']
    for gc in group_curves.values():
        gc['aggregate_pct'] = gc['aggregate'] / np.clip(all_agg, 1e-10, None) * 100

    for gname, gc in group_curves.items():
        agg = gc['aggregate_pct']
        ax.plot(x, agg, linewidth=3, color=gc['color'],
                alpha=0.9, label=f"{gname} (n={gc['n']:,})")
        ax.fill_between(x, 0, agg, alpha=0.12, color=gc['color'])

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share (%)', fontsize=20)
    ax.set_xlabel('Position in trajectory (%)', fontsize=20)
    ax.grid(alpha=0.12)
    ax.set_title('Tool Usage Share by Functional Group',
                 fontsize=22, fontweight='bold')

    top_y = ax.get_ylim()[1]
    ax.axvspan(0, 25, alpha=0.06, color='#1565C0')
    ax.axvspan(25, 75, alpha=0.04, color='#FF9800')
    ax.axvspan(75, 100, alpha=0.06, color='#8E24AA')
    ax.text(12.5, top_y * 0.94, 'Initialization', fontsize=18,
            ha='center', color='#222', fontstyle='italic')
    ax.text(50, top_y * 0.94, 'Composition', fontsize=18,
            ha='center', color='#222', fontstyle='italic')
    ax.text(87.5, top_y * 0.94, 'Readout', fontsize=18,
            ha='center', color='#222', fontstyle='italic')
    ax.legend(fontsize=17, loc='upper center', framealpha=0.9, ncol=1,
              bbox_to_anchor=(0.5, 0.88))

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_grouped_1row.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved 1-row grouped KDE to {OUT_DIR}/tool_trajectory_grouped_1row.pdf/.png")
    plt.close(fig)


# ── Plot 3: Stacked area (all tools as continuous canopy) ──
def make_stacked_area(tool_positions, top_n=TOP_N_KDE):
    freq = {fn: len(pos) for fn, pos in tool_positions.items()}
    top_tools = sorted(freq, key=freq.get, reverse=True)[:top_n]
    other_tools = [fn for fn in freq if fn not in top_tools]

    bins = np.linspace(0, 100, 51)  # 2% resolution
    sigma = 2
    cmap = cm.get_cmap('tab20', top_n + 1)

    # Build smoothed histograms
    curves = []
    labels = []
    for i, tool in enumerate(top_tools):
        positions = [p for p, _ in tool_positions[tool]]
        hist, _ = np.histogram(positions, bins=bins)
        smooth = gaussian_filter1d(hist.astype(float), sigma)
        curves.append(smooth)
        labels.append(tool.replace('_', ' '))

    # "Other" category
    other_positions = []
    for fn in other_tools:
        other_positions.extend([p for p, _ in tool_positions[fn]])
    if other_positions:
        hist, _ = np.histogram(other_positions, bins=bins)
        smooth = gaussian_filter1d(hist.astype(float), sigma)
        curves.append(smooth)
        labels.append('other')

    x = (bins[:-1] + bins[1:]) / 2
    colors = [cmap(i) for i in range(len(curves))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(x, *curves, labels=labels, colors=colors, alpha=0.85)

    ax.set_xlabel('Position in trajectory (%)', fontsize=12)
    ax.set_ylabel('Tool call count (per 2% bin)', fontsize=12)
    ax.set_title('Tool Usage Canopy across Trajectory Position',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)

    ax.legend(fontsize=7, loc='upper center', ncol=5, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{OUT_DIR}/tool_trajectory_canopy.{ext}",
                    bbox_inches='tight', dpi=150)
    print(f"Saved canopy plot to {OUT_DIR}/tool_trajectory_canopy.pdf/.png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting tool positions from all benchmarks...")
    tool_positions = collect_tool_positions()
    total_calls = sum(len(v) for v in tool_positions.values())
    print(f"  {total_calls} tool calls across {len(tool_positions)} unique tools")

    make_strip_plot(tool_positions)
    make_kde_plot(tool_positions)
    make_kde_zoom(tool_positions)
    make_grouped_kde(tool_positions)
    make_grouped_kde_1row(tool_positions)
    make_stacked_area(tool_positions)
    print("Done.")
