"""
Staged canvas snapshots for Fig 2 — mathvista/290.
Uses CONSTRAINT-BASED construction (rotate + parallel) for correct demonstration.

Stage 1: Transversal AE + Rotate to construct AB at 105°
Stage 2: + Parallel line CD through E
Stage 3: + Query angles → 105°, 75°

Usage:
    python Writing/figs/fig2_case/render_staged.py
"""
import sys, re, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT))

from symbolic.integrations.geogebra_api import GeoGebraAPI

OUT_DIR = Path(__file__).parent / "staged"


def _enlarge_font(ggb, size=36):
    full_xml = ggb._execute_js('return ggbApplet.getXML()')
    new_xml = re.sub(r'<font size="\d+"/>', f'<font size="{size}"/>', full_xml)
    ggb._driver.execute_script('ggbApplet.setXML(arguments[0])', new_xml)


def _style(ggb):
    """Style all objects for paper-quality screenshot."""
    for name in (ggb.get_all_object_names() or []):
        otype = (ggb.get_object_type(name) or "").lower()
        if otype == "point":
            if "_" in name:
                ggb.set_color(name, 140, 140, 140)
                ggb._execute_js(f'ggbApplet.setPointSize("{name}", 5)')
                parts = name.split("_")
                cap = parts[0] + "".join("_{" + p + "}" for p in parts[1:])
                ggb._execute_js(f'ggbApplet.setCaption("{name}","{cap}")')
                ggb.set_label_style(name, 3)
                ggb.set_label_visible(name, True)
            else:
                ggb.set_color(name, 0, 0, 180)
                ggb._execute_js(f'ggbApplet.setPointSize("{name}", 9)')
                ggb.set_label_style(name, 0)
                ggb.set_label_visible(name, True)
        elif otype == "angle":
            ggb.set_label_visible(name, True)
            ggb.set_label_style(name, 2)  # VALUE only
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 7)')
            ggb.set_color(name, 0, 140, 0)
        elif otype in ("line", "ray"):
            ggb.set_label_visible(name, False)
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 6)')
        elif otype == "segment":
            ggb.set_label_visible(name, False)
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 6)')
        else:
            ggb.set_label_visible(name, False)

    ggb.set_axes_visible(False, False)
    ggb.set_grid_visible(False)
    ggb.fit_view(padding=2.0)


def _snapshot(ggb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    elem = ggb._driver.find_element("id", "ggb-element")
    path.write_bytes(elem.screenshot_as_png)
    print(f"  → {path.name} ({path.stat().st_size // 1024}KB)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ggb = GeoGebraAPI(mode="selenium", headless=True)
    ggb.initialize()

    # ═══ Stage 1: Construct transversal + AB via Rotate (constraint-based) ═══
    print("\nStage 1: Transversal + Rotate → AB")
    ggb.eval_command("A = (0, 0)")
    ggb.eval_command("E = (5, 0)")
    ggb.eval_command("AE = Line(A, E)")                   # transversal
    ggb.eval_command("B = Rotate(E, -105°, A)")             # B via rotate: ∠BAE = 105° exact
    ggb.eval_command("AB = Ray(A, B)")                     # ray AB

    # Dump canvas state (same format the model sees)
    from symbolic.tools.geogebra_tools import build_rich_canvas
    canvas1 = build_rich_canvas(ggb)
    print("  Canvas JSON after Stage 1 (model sees this):")
    import json as _json
    print("  " + _json.dumps(canvas1, indent=2))

    _style(ggb)
    _enlarge_font(ggb, 36)
    _snapshot(ggb, OUT_DIR / "stage_1_base.png")

    # ═══ Stage 2: + Parallel line CD through E ═══
    print("\nStage 2: + Parallel CD = Line(E, AB)")
    ggb.eval_command("CD = Line(E, AB)")                   # CD ∥ AB by construction
    ggb.eval_command("D = Point(CD)")                      # free point on CD (arbitrary)

    canvas2 = build_rich_canvas(ggb)
    print("  Canvas JSON after Stage 2 (model sees this):")
    print("  " + _json.dumps(canvas2, indent=2))

    _style(ggb)
    _enlarge_font(ggb, 36)
    _snapshot(ggb, OUT_DIR / "stage_2_parallel.png")

    # ═══ Stage 3: + Query angles ═══
    print("\nStage 3: + Query angles")
    ggb.eval_command("ang1 = Angle(B, A, E)")              # verify ∠1
    ggb.eval_command("ang2 = Angle(A, E, D)")              # ∠AED = ∠2

    canvas3 = build_rich_canvas(ggb)
    print("  Canvas JSON after Stage 3 (model sees this):")
    print("  " + _json.dumps(canvas3, indent=2))

    _style(ggb)
    _enlarge_font(ggb, 36)
    time.sleep(0.3)
    _snapshot(ggb, OUT_DIR / "stage_3_query.png")

    # Print values
    v1 = ggb.get_value("ang1")
    v2 = ggb.get_value("ang2")
    print(f"  ang1 = {v1}°")
    print(f"  ang2 = {v2}°")

    ggb.cleanup()
    print(f"\nDone → {OUT_DIR}/")


if __name__ == "__main__":
    main()
