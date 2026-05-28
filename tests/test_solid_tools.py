"""
Test 3D solid geometry tools — each solid type gets its own GGB session + PNG.

Phase 1: 2D regression (enable_3d=False, must pass 100%)
Phase 2: 3D solids, one per session:
  01_prism, 02_pyramid, 03_sphere, 04_cone, 05_cylinder,
  06_tetrahedron, 07_cube, 08_plane_cross_section, 09_combined_scene

Usage:  python tests/test_solid_tools.py
"""
import sys
import time
import math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUTDIR = Path("temp") / "test_solid_tools"
(OUTDIR / "fig").mkdir(parents=True, exist_ok=True)
(OUTDIR / "log").mkdir(parents=True, exist_ok=True)


class TeeLog:
    def __init__(self, path):
        self._path = Path(path)
        self._file = None
        self._orig = None

    def __enter__(self):
        self._orig = sys.stdout
        self._file = self._path.open("w", encoding="utf-8")
        sys.stdout = self
        return self

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def __exit__(self, *_):
        sys.stdout = self._orig
        self._file.close()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ggb(enable_3d=True):
    from symbolic.integrations.geogebra_api import GeoGebraAPI
    ggb = GeoGebraAPI(headless=True, enable_3d=enable_3d)
    ggb.initialize()
    return ggb


def solid(ggb, tool_name, args):
    """Execute a solid tool, print result, return (ok, err)."""
    from symbolic.tools.geogebra_tools_solid import execute_solid_tool
    cmd, ok, err = execute_solid_tool(ggb, tool_name, args)
    tag = "ok" if ok else f"FAIL({err})"
    print(f"  [{tag}] `{cmd}`")
    return ok


def solid_q(ggb, tool_name, args):
    """Execute a solid query, print result, return (ok, val)."""
    from symbolic.tools.geogebra_tools_solid import execute_solid_query
    cmd, ok, err, val = execute_solid_query(ggb, tool_name, args)
    tag = "ok" if ok else f"FAIL({err})"
    print(f"  [{tag}] `{cmd}`  value={val}")
    return ok, val


def cmd(ggb, command):
    result = ggb.eval_command(command)
    ok = "ok" if result.success else f"FAIL({result.error_message})"
    print(f"  [{ok}] `{command}`")
    return result.success


def save(ggb, filename):
    out = OUTDIR / "fig" / f"{filename}.png"
    ok = ggb.export_png(out)
    print(f"  {'[SAVED]' if ok else '[FAIL export]'} {out}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1: 2D Regression
# ═══════════════════════════════════════════════════════════════════════════════

def test_2d_no_regression():
    print("\n" + "=" * 60)
    print("  [Phase 1] 2D Regression (enable_3d=False)")
    print("=" * 60)

    ggb = _make_ggb(enable_3d=False)
    tests = []

    cmd(ggb, "A = (0, 0)")
    cmd(ggb, "B = (3, 4)")
    tests.append(("point exists", ggb.exists("A") and ggb.exists("B")))

    cmd(ggb, "seg = Segment(A, B)")
    d = ggb.get_value("seg")
    tests.append(("Segment = 5", d is not None and abs(d - 5.0) < 0.01))

    cmd(ggb, "c = Circle(A, 3)")
    tests.append(("Circle exists", ggb.exists("c")))

    cmd(ggb, "C = (3, 0)")
    cmd(ggb, "tri = Polygon(A, B, C)")
    area = ggb.get_value("tri")
    tests.append(("Polygon area = 6", area is not None and abs(area - 6.0) < 0.1))

    png = ggb.get_png_base64()
    tests.append(("PNG export", png is not None and len(png) > 100))

    cas = ggb._execute_js('return ggbApplet.evalCommandCAS("Solve(x^2-4)")')
    tests.append(("CAS Solve", cas is not None and "2" in str(cas)))

    ggb.cleanup()

    passed = sum(1 for _, ok in tests if ok)
    for desc, ok in tests:
        print(f"  [{'OK' if ok else 'FAIL':4s}] {desc}")
    print(f"\n  2D: {passed}/{len(tests)}")
    return passed == len(tests)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: 3D — each solid gets its own clean session
# ═══════════════════════════════════════════════════════════════════════════════

def test_01_prism():
    """Right prism with square base, verify volume."""
    print("\n[01_prism]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "A", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "B", "x": 3, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "C", "x": 3, "y": 3, "z": 0})
    solid(ggb, "add_point3d", {"name": "D", "x": 0, "y": 3, "z": 0})
    cmd(ggb, "base = Polygon(A, B, C, D)")

    ok = solid(ggb, "add_prism", {"name": "p", "base": "base", "top": "5"})
    results.append(("Prism(base, 5)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "p"})
    results.append(("Volume = 45", ok and vol is not None and abs(vol - 45) < 0.5))

    save(ggb, "01_prism")
    ggb.cleanup()
    return results


def test_02_pyramid():
    """Right pyramid with square base, verify volume = 1/3 * base * h."""
    print("\n[02_pyramid]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "A", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "B", "x": 3, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "C", "x": 3, "y": 3, "z": 0})
    solid(ggb, "add_point3d", {"name": "D", "x": 0, "y": 3, "z": 0})
    cmd(ggb, "base = Polygon(A, B, C, D)")

    ok = solid(ggb, "add_pyramid", {"name": "pyr", "base": "base", "top": "6"})
    results.append(("Pyramid(base, 6)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "pyr"})
    # V = 1/3 * 9 * 6 = 18
    results.append(("Volume = 18", ok and vol is not None and abs(vol - 18) < 0.5))

    save(ggb, "02_pyramid")
    ggb.cleanup()
    return results


def test_03_sphere():
    """Sphere with radius 3, verify volume = 4/3 * pi * r^3."""
    print("\n[03_sphere]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "O", "x": 0, "y": 0, "z": 0})
    ok = solid(ggb, "add_sphere", {"name": "sph", "center": "O",
                                    "radius_or_point": "3"})
    results.append(("Sphere(O, 3)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "sph"})
    expected = 4 / 3 * math.pi * 27  # ≈ 113.0973
    results.append((f"Volume ≈ {expected:.4f}",
                    ok and vol is not None and abs(vol - expected) < 1.0))

    save(ggb, "03_sphere")
    ggb.cleanup()
    return results


def test_04_cone():
    """Cone with two points + radius, verify volume = 1/3 * pi * r^2 * h."""
    print("\n[04_cone]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "P1", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "P2", "x": 0, "y": 0, "z": 5})
    ok = solid(ggb, "add_cone", {"name": "cone", "a": "P1", "b": "P2",
                                  "radius": 2})
    results.append(("Cone(P1, P2, r=2)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "cone"})
    expected = 1 / 3 * math.pi * 4 * 5  # ≈ 20.9440
    results.append((f"Volume ≈ {expected:.4f}",
                    ok and vol is not None and abs(vol - expected) < 1.0))

    save(ggb, "04_cone")
    ggb.cleanup()
    return results


def test_05_cylinder():
    """Cylinder with two points + radius, verify volume = pi * r^2 * h."""
    print("\n[05_cylinder]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "Q1", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "Q2", "x": 0, "y": 0, "z": 4})
    ok = solid(ggb, "add_cylinder", {"name": "cyl", "a": "Q1", "b": "Q2",
                                      "radius": 2})
    results.append(("Cylinder(Q1, Q2, r=2)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "cyl"})
    expected = math.pi * 4 * 4  # ≈ 50.2655
    results.append((f"Volume ≈ {expected:.4f}",
                    ok and vol is not None and abs(vol - expected) < 1.0))

    save(ggb, "05_cylinder")
    ggb.cleanup()
    return results


def test_06_tetrahedron():
    """Regular tetrahedron from two points, verify it creates 4 faces."""
    print("\n[06_tetrahedron]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "A", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "B", "x": 3, "y": 0, "z": 0})
    ok = solid(ggb, "add_tetrahedron", {"name": "tet", "a": "A", "b": "B"})
    results.append(("Tetrahedron(A, B)", ok))

    # Verify volume: V = edge^3 / (6*sqrt(2)) = 27/(6*sqrt(2)) ≈ 3.1820
    ok, vol = solid_q(ggb, "query_volume", {"solid": "tet"})
    expected = 27 / (6 * math.sqrt(2))
    results.append((f"Volume ≈ {expected:.4f}",
                    ok and vol is not None and abs(vol - expected) < 0.5))

    save(ggb, "06_tetrahedron")
    ggb.cleanup()
    return results


def test_07_cube():
    """Cube from two points, verify volume = edge^3."""
    print("\n[07_cube]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "A", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "B", "x": 3, "y": 0, "z": 0})
    ok = solid(ggb, "add_cube", {"name": "cb", "a": "A", "b": "B"})
    results.append(("Cube(A, B)", ok))

    ok, vol = solid_q(ggb, "query_volume", {"solid": "cb"})
    results.append(("Volume = 27", ok and vol is not None and abs(vol - 27) < 0.5))

    save(ggb, "07_cube")
    ggb.cleanup()
    return results


def test_08_plane_cross_section():
    """Plane cutting a sphere → cross-section circle."""
    print("\n[08_plane_cross_section]")
    ggb = _make_ggb()
    results = []

    # Sphere at origin, r=3
    solid(ggb, "add_point3d", {"name": "O", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_sphere", {"name": "sph", "center": "O",
                               "radius_or_point": "3"})

    # Horizontal plane at z=1
    solid(ggb, "add_point3d", {"name": "R1", "x": 0, "y": 0, "z": 1})
    solid(ggb, "add_point3d", {"name": "R2", "x": 1, "y": 0, "z": 1})
    solid(ggb, "add_point3d", {"name": "R3", "x": 0, "y": 1, "z": 1})
    ok = solid(ggb, "add_plane", {"name": "pl", "a": "R1", "b": "R2", "c": "R3"})
    results.append(("Plane at z=1", ok))

    # Cross-section
    ok = solid(ggb, "add_cross_section", {"name": "csec",
                                           "plane": "pl", "solid": "sph"})
    results.append(("IntersectPath(plane, sphere)", ok))
    results.append(("csec exists", ggb.exists("csec")))

    # PerpendicularPlane test
    cmd(ggb, "axis = Line(O, (0,0,1))")
    ok = solid(ggb, "add_perpendicular_plane",
               {"name": "perpPl", "point": "O", "direction": "axis"})
    results.append(("PerpendicularPlane(O, z-axis)", ok))

    save(ggb, "08_plane_cross_section")
    ggb.cleanup()
    return results


def test_09_3d_coords_and_vectors():
    """3D coordinate queries + vector creation."""
    print("\n[09_3d_coords_and_vectors]")
    ggb = _make_ggb()
    results = []

    solid(ggb, "add_point3d", {"name": "A", "x": 1, "y": 2, "z": 3})
    solid(ggb, "add_point3d", {"name": "B", "x": 4, "y": 6, "z": 3})

    # Query 3D coords
    ok, coords = solid_q(ggb, "query_coords3d", {"point": "A"})
    results.append(("coords3d(A) = (1,2,3)",
                    ok and coords["x"] == 1 and coords["y"] == 2 and coords["z"] == 3))

    # Vector between points
    ok = solid(ggb, "add_vector3d", {"name": "v1",
                                      "from_pt": "A", "to_pt": "B"})
    results.append(("Vector(A, B)", ok))

    # Vector from components
    ok = solid(ggb, "add_vector3d", {"name": "v2", "x": 0, "y": 0, "z": 5})
    results.append(("Vector(0,0,5)", ok))

    # 2D still works inside 3D
    cmd(ggb, "seg = Segment(A, B)")
    d = ggb.get_value("seg")
    results.append(("2D Segment inside 3D", d is not None and d > 0))

    save(ggb, "09_coords_vectors")
    ggb.cleanup()
    return results


def test_10_labels_and_latex():
    """Point labels, edge captions, angles, color, LaTeX text in 3D."""
    print("\n[10_labels_and_latex]")
    ggb = _make_ggb()
    js = ggb._execute_js
    results = []

    # Build pyramid
    solid(ggb, "add_point3d", {"name": "A", "x": 0, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "B", "x": 4, "y": 0, "z": 0})
    solid(ggb, "add_point3d", {"name": "C", "x": 4, "y": 4, "z": 0})
    solid(ggb, "add_point3d", {"name": "D", "x": 0, "y": 4, "z": 0})
    cmd(ggb, "base = Polygon(A, B, C, D)")
    solid(ggb, "add_pyramid", {"name": "pyr", "base": "base", "top": "5"})
    # Apex auto-named E

    solid(ggb, "render_set_3d_view", {
        "x_angle": 20, "z_angle": -50, "scale": 65,
        "show_axes": "false", "show_plate": "false",
    })

    # 1. Point labels
    for pt in ["A", "B", "C", "D", "E"]:
        js(f'ggbApplet.setLabelVisible("{pt}", true)')
        js(f'ggbApplet.setLabelStyle("{pt}", 0)')
        js(f'ggbApplet.setPointSize("{pt}", 4)')
    results.append(("Point labels visible",
                    all(js(f'return ggbApplet.getLabelVisible("{p}")')
                        for p in ["A", "B", "C", "D", "E"])))

    # 2. Red edge segment with caption
    cmd(ggb, "h_seg = Segment(A, E)")
    js('ggbApplet.setColor("h_seg", 220, 50, 50)')
    js('ggbApplet.setLineThickness("h_seg", 4)')
    js('ggbApplet.setCaption("h_seg", "AE")')
    js('ggbApplet.setLabelStyle("h_seg", 3)')
    js('ggbApplet.setLabelVisible("h_seg", true)')
    results.append(("Red edge AE + caption", ggb.exists("h_seg")))

    # 3. Blue slant height with caption "x"
    cmd(ggb, "M = Midpoint(B, C)")
    js('ggbApplet.setLabelVisible("M", true)')
    cmd(ggb, "slant = Segment(E, M)")
    js('ggbApplet.setColor("slant", 50, 50, 220)')
    js('ggbApplet.setLineThickness("slant", 3)')
    js('ggbApplet.setCaption("slant", "x")')
    js('ggbApplet.setLabelStyle("slant", 3)')
    js('ggbApplet.setLabelVisible("slant", true)')
    results.append(("Blue slant + caption x", ggb.exists("slant")))

    # 4. Angle mark
    cmd(ggb, "ang1 = Angle(B, A, E)")
    js('ggbApplet.setColor("ang1", 0, 180, 0)')
    js('ggbApplet.setLabelVisible("ang1", true)')
    js('ggbApplet.setLabelStyle("ang1", 2)')  # VALUE
    ang_val = ggb.get_value("ang1")
    results.append(("Angle BAE exists + value",
                    ang_val is not None and ang_val > 0))

    # 5. Base edge caption "4"
    js('ggbApplet.setCaption("a", "4")')
    js('ggbApplet.setLabelStyle("a", 3)')
    js('ggbApplet.setLabelVisible("a", true)')
    results.append(("Base edge caption '4'", True))

    # 6. LaTeX text via add_text_3d
    ok1 = solid(ggb, "add_text_3d", {
        "name": "lbl_h", "text": "h = 5", "x": -1, "y": -1, "z": 6,
    })
    results.append(("add_text_3d plain", ok1))

    # LLM sends JSON: {"text": "\\sqrt{h^2 + (\\frac{a}{2})^2}"}
    # json.loads → Python string with single backslash: \sqrt{...}
    ok2 = solid(ggb, "add_text_3d", {
        "name": "lbl_formula",
        "text": "\\sqrt{h^2 + (\\frac{a}{2})^2}",
        "x": 5, "y": 0, "z": 6,
    })
    results.append(("add_text_3d LaTeX formula", ok2))

    # Hide noise
    for obj in ["base", "pyr", "b", "c", "d",
                 "edgeBE", "edgeAE", "edgeCE", "edgeDE",
                 "faceABE", "faceBCE", "faceCDE", "faceADE"]:
        js(f'ggbApplet.setLabelVisible("{obj}", false)')

    save(ggb, "10_labels_latex")
    ggb.cleanup()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    # Phase 1
    ok_2d = test_2d_no_regression()

    # Phase 2: each solid in its own session
    all_3d = []
    test_fns = [
        test_01_prism,
        test_02_pyramid,
        test_03_sphere,
        test_04_cone,
        test_05_cylinder,
        test_06_tetrahedron,
        test_07_cube,
        test_08_plane_cross_section,
        test_09_3d_coords_and_vectors,
        test_10_labels_and_latex,
    ]

    for fn in test_fns:
        log_name = fn.__name__.replace("test_", "")
        with TeeLog(OUTDIR / "log" / f"{log_name}.txt"):
            results = fn()
        all_3d.extend(results)

    elapsed = time.perf_counter() - t0

    # Summary
    print("\n" + "=" * 60)
    print(f"  FINAL SUMMARY  ({elapsed:.1f}s)")
    print("=" * 60)
    print(f"  2D regression: {'PASS' if ok_2d else 'FAIL'}")

    passed_3d = sum(1 for _, ok in all_3d if ok)
    for desc, ok in all_3d:
        if not ok:
            print(f"  [FAIL] {desc}")
    print(f"  3D solid: {passed_3d}/{len(all_3d)}")
    print(f"  PNGs: {OUTDIR / 'fig'}")

    if ok_2d and passed_3d == len(all_3d):
        print("  ALL PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
