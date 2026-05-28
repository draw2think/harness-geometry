"""Generate fidelity-conditional answer quality bars (§4.5).

Reads per-problem result.json from code/eval/geogoal/ and cross-tabulates
SC (all predicates pass) × FA threshold (strict/≥90%/≥80%) to produce
a grouped bar chart showing the fidelity-conditional lift.
"""
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'Liberation Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ROOT = Path(__file__).resolve().parents[3] / 'eval' / 'geogoal'
OUT_PDF = Path(__file__).parent / 'fidelity_bars.pdf'
OUT_PNG = Path(__file__).parent / 'fidelity_bars.png'


def load_data():
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not (d.is_dir() and d.name.startswith('geogal_')):
            continue
        rf = d / 'gemini-3-flash-preview@medium_result.json'
        if not rf.exists():
            continue
        r = json.load(open(rf))
        fa = r.get('FA_local') or {}
        m, t = fa.get('match', 0), fa.get('total', 0)
        rows.append({
            'ir': bool(r.get('IR', False)),
            'rate': m/t if t else 0.0,
            'strict': (m == t and t > 0),
        })
    return rows


def counts(rows, thresholds):
    faithful = [r for r in rows if r['ir']]
    unfaithful = [r for r in rows if not r['ir']]
    out = {'faithful_n': len(faithful), 'unfaithful_n': len(unfaithful)}
    for label, cond in thresholds:
        out[f'faithful_{label}'] = sum(1 for r in faithful if cond(r))
        out[f'unfaithful_{label}'] = sum(1 for r in unfaithful if cond(r))
    return out


def plot(stats, thresholds):
    labels = [label for label, _ in thresholds]
    n_f = stats['faithful_n']
    n_u = stats['unfaithful_n']
    faithful_pct = [100 * stats[f'faithful_{l}'] / n_f for l in labels]
    unfaithful_pct = [100 * stats[f'unfaithful_{l}'] / n_u for l in labels]

    group_gap = 0.7
    x = np.arange(len(labels)) * group_gap
    w = 0.28

    fig, ax = plt.subplots(figsize=(2.8, 3.0))
    b1 = ax.bar(x - w/2, faithful_pct, w,
                label=f'Faithful canvas (SC$=$T, $n{{=}}${n_f})',
                color='#2b7bba', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, unfaithful_pct, w,
                label=f'Unfaithful canvas (SC$=$F, $n{{=}}${n_u})',
                color='#d68a46', edgecolor='black', linewidth=0.5)

    for bars, pct, cnt_key in [(b1, faithful_pct, 'faithful'),
                                (b2, unfaithful_pct, 'unfaithful')]:
        for i, (bar, p) in enumerate(zip(bars, pct)):
            n = stats[f'{cnt_key}_{labels[i]}']
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{p:.1f}%',
                    ha='center', va='bottom', fontsize=8)
            # Skip the in-bar count label for n==0: there is no bar region to
            # anchor it, so it would float as an orphan digit next to the
            # already-shown 0% label on top.
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height()/2, 4),
                        f'{n}', ha='center', va='center',
                        color='white' if p > 15 else 'black', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(['strict\n(all $T_i$)', r'$\geq 90\%$ $T_i$', r'$\geq 80\%$ $T_i$'])
    ax.set_ylabel('Answer-quality pass rate (\\%)')
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(x[0] - group_gap / 2, x[-1] + group_gap / 2)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=8, ncol=1,
              framealpha=0.95, handlelength=1.2, labelspacing=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate 4.3x lift at ≥80%
    i80 = labels.index('ge80')
    xi = x[i80]
    ax.annotate('', xy=(xi - w/2, faithful_pct[i80]),
                xytext=(xi + w/2, unfaithful_pct[i80]),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=0.9))
    ax.text(xi, (faithful_pct[i80] + unfaithful_pct[i80]) / 2 + 2,
            r'$4.3\times$', ha='center', fontsize=9, color='#222',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#888', lw=0.5))

    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(OUT_PNG, bbox_inches='tight', pad_inches=0.05, dpi=200)
    return fig


def main():
    rows = load_data()
    thresholds = [
        ('strict', lambda r: r['strict']),
        ('ge90',   lambda r: r['rate'] >= 0.90),
        ('ge80',   lambda r: r['rate'] >= 0.80),
    ]
    stats = counts(rows, thresholds)
    print(f'Loaded N={len(rows)} problems')
    print(f'Faithful: {stats["faithful_n"]}, Unfaithful: {stats["unfaithful_n"]}')
    for label, _ in thresholds:
        f_pct = 100 * stats[f'faithful_{label}'] / stats['faithful_n']
        u_pct = 100 * stats[f'unfaithful_{label}'] / stats['unfaithful_n']
        print(f'  {label}: faithful {stats[f"faithful_{label}"]}/{stats["faithful_n"]} ({f_pct:.1f}%) | '
              f'unfaithful {stats[f"unfaithful_{label}"]}/{stats["unfaithful_n"]} ({u_pct:.1f}%)')
    plot(stats, thresholds)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
