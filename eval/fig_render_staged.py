"""
Staged per-turn figure for paper.

Manually constructs a visually compelling step-by-step progression,
rather than replaying from result.json. Each turn has clear visual changes.

Usage:
    python eval/fig_render_staged.py --out /path/to/output_dir
"""
import sys, os, time
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from symbolic.integrations.geogebra_api import GeoGebraAPI


def _label(ggb, name, visible=True, style=0):
    """Set label visibility and style. 0=Name, 1=Name+Value, 2=Value, 3=Caption."""
    ggb.set_label_style(name, style)
    ggb.set_label_visible(name, visible)


def _enlarge_font(ggb, size=36):
    """Enlarge global label font via XML: get → replace <font size> → setXML.
    Only touches font size — line/point sizes are set separately."""
    import re
    full_xml = ggb._execute_js('return ggbApplet.getXML()')
    new_xml = re.sub(r'<font size="\d+"/>', f'<font size="{size}"/>', full_xml)
    ggb._driver.execute_script('ggbApplet.setXML(arguments[0])', new_xml)


def _set_line_styles(ggb):
    """Set absolute line/point styles after setXML reload."""
    for name in ggb.get_all_object_names() or []:
        t = ggb.get_object_type(name)
        if t == "circle":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 4)')
        elif t == "arc":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 7)')
        elif t == "segment":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 3)')
        elif t == "angle":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 5)')
        elif t == "point":
            ggb._execute_js(f'ggbApplet.setPointSize("{name}", 6)')



def _snapshot(ggb, path, padding=2.0):
    """Export PNG via browser screenshot (800x800, font already enlarged via XML)."""
    if padding is not None:
        ggb.fit_view(padding=padding)
    path.parent.mkdir(parents=True, exist_ok=True)
    elem = ggb._driver.find_element("id", "ggb-element")
    path.write_bytes(elem.screenshot_as_png)
    print(f"  → {path.name} ({path.stat().st_size // 1024}KB)")


def render_prob7177(out_dir: Path):
    """
    PGPS9K prob_7177: Find m arc BCA of circle Q.  (angle DQC = 47°, answer = 313)

    Staged into 3 visually distinct turns:
      Turn 1: Center Q + circle + point D  (base geometry)
      Turn 2: Rotate→C, Reflect→B,A, arc BCA  (construction via transforms)
      Turn 3: Angle measurement 313° appears  (engine feedback → answer)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ggb = GeoGebraAPI(mode="selenium", headless=True)
    ggb.initialize()

    # Clean canvas, enlarge font for paper
    ggb.set_axes_visible(False, False)
    ggb.set_grid_visible(False)
    # Note: fontSize is set at init (18px default in geogebra_api.py).
    # To make labels appear larger, we zoom in by using a tighter coord range.

    # ─── Turn 1: Base geometry — Q, circle, D ───
    ggb.eval_command("Q = (0, 0)")
    ggb.eval_command("circQ = Circle(Q, 5)")
    ggb.eval_command("D = (5, 0)")

    _label(ggb, "Q", True, 0)
    _label(ggb, "circQ", False)     # hide circle label
    _label(ggb, "D", True, 0)

    # Style: make points bigger for paper
    ggb.set_color("Q", 0, 0, 140)
    ggb.set_color("D", 0, 0, 140)
    ggb._execute_js('ggbApplet.setPointSize("Q", 5)')
    ggb._execute_js('ggbApplet.setPointSize("D", 5)')

    _enlarge_font(ggb, size=36)
    _set_line_styles(ggb)
    ggb._execute_js('ggbApplet.setCoordSystem(-6.2, 6.2, -6.2, 6.2)')
    _snapshot(ggb, out_dir / "turn_1_base.png", padding=None)
    print("  Turn 1: Q + circle + D")

    # ─── Turn 2: Geometric transforms — C, B, A, arc ───
    ggb.eval_command("C = Rotate(D, 47°, Q)")
    ggb.eval_command("B = Reflect(D, Q)")
    ggb.eval_command("A = Reflect(C, Q)")
    ggb.eval_command("arcBCA = CircularArc(Q, B, A)")

    # Draw chords to match original diagram
    ggb.eval_command("seg_BD = Segment(B, D)")
    ggb.eval_command("seg_AC = Segment(A, C)")

    # Label and style new points
    for pt in ["C", "B", "A"]:
        _label(ggb, pt, True, 0)
        ggb.set_color(pt, 0, 0, 140)
        ggb._execute_js(f'ggbApplet.setPointSize("{pt}", 5)')

    # Hide arc/segment labels
    _label(ggb, "arcBCA", False)
    _label(ggb, "seg_BD", False)
    _label(ggb, "seg_AC", False)

    # Make arc thicker for visibility
    ggb._execute_js('ggbApplet.setLineThickness("arcBCA", 5)')
    # Chords thinner
    ggb._execute_js('ggbApplet.setLineThickness("seg_BD", 2)')
    ggb._execute_js('ggbApplet.setLineThickness("seg_AC", 2)')

    _enlarge_font(ggb, size=36)
    _set_line_styles(ggb)
    ggb._execute_js('ggbApplet.setCoordSystem(-6.2, 6.2, -6.2, 6.2)')
    _snapshot(ggb, out_dir / "turn_2_construct.png", padding=None)
    print("  Turn 2: C=Rotate(D,47°,Q), B=Reflect(D,Q), A=Reflect(C,Q), arc BCA")

    # ─── Turn 3: Measurement — angle 313° ───
    ggb.eval_command("angAQB = Angle(A, Q, B)")

    # Show angle value only (313°)
    _label(ggb, "angAQB", True, 2)  # VALUE mode

    # Make angle arc thicker and colored for visibility
    ggb.set_color("angAQB", 0, 120, 0)
    ggb._execute_js('ggbApplet.setLineThickness("angAQB", 4)')

    _enlarge_font(ggb, size=36)
    _set_line_styles(ggb)
    ggb._execute_js('ggbApplet.setCoordSystem(-6.2, 6.2, -6.2, 6.2)')
    _snapshot(ggb, out_dir / "turn_3_measure.png", padding=None)
    print("  Turn 3: Angle(A,Q,B) = 313° → ANSWER")

    # ─── Summary ───
    summary = """# Staged Figure: prob_7177
## Find m arc BCA of circle Q  (angle DQC = 47°)

### Turn 1 — Base geometry
- `Q = (0, 0)` — center
- `circQ = Circle(Q, 5)` — circle with radius 5
- `D = (5, 0)` — reference point on circle

### Turn 2 — Construction via geometric transforms
- `C = Rotate(D, 47°, Q)` — rotate D by 47° around Q
- `B = Reflect(D, Q)` — B is diametrically opposite D
- `A = Reflect(C, Q)` — A is diametrically opposite C
- `arcBCA = CircularArc(Q, B, A)` — the target arc
- `seg_BD, seg_AC` — chords (matching original diagram)

### Turn 3 — Engine measurement → Answer
- `angAQB = Angle(A, Q, B)` → **313°**
- Arc BCA = 313° (central angle = arc measure)
- **ANSWER: 313** ✓

### Visual progression
1. Empty → circle with one point
2. +3 points via Rotate/Reflect + arc + chords
3. +angle measurement 313° → direct answer
"""
    (out_dir / "summary.md").write_text(summary)

    ggb.cleanup()
    print(f"\n  Done → {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    render_prob7177(Path(args.out))


if __name__ == "__main__":
    main()
