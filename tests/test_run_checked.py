"""
Test run_checked() post-execution validation for GeoGebra construction tools.

Verifies that degenerate inputs (coincident points, collinear points, etc.)
are detected and reported as errors, while normal inputs succeed.

Run: python tests/test_run_checked.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import execute_geogebra_tool as ex


def main():
    ctx = GeoGebraAPI()
    ggb = ctx.__enter__()
    ggb.set_axes_visible(False, False)

    results = []

    def test(name, tool, args, expect_ok):
        cmd, ok, err = ex(ggb, tool, args)
        status = "PASS" if ok == expect_ok else "FAIL"
        results.append((status, name, ok, expect_ok, err))
        tag = "[+]" if status == "PASS" else "[-]"
        print(f"  {tag} {name:40s}  ok={ok}  expect={expect_ok}  err={err!r:.60s}")

    # ── Foundation points ─────────────────────────────────────────────
    print("=== Foundation ===")
    test("point A(0,0)",       "add_point", {"name": "A", "x": 0, "y": 0}, True)
    test("point B(3,0)",       "add_point", {"name": "B", "x": 3, "y": 0}, True)
    test("point C(0,4)",       "add_point", {"name": "C", "x": 0, "y": 4}, True)
    test("point D(6,0)",       "add_point", {"name": "D", "x": 6, "y": 0}, True)
    test("point E(0,0) [=A]",  "add_point", {"name": "E", "x": 0, "y": 0}, True)

    # ── Normal construction (all should succeed) ─────────────────────
    print("\n=== Normal construction (expect ok=True) ===")
    test("segment A-B",              "add_segment",  {"name": "sAB", "p1": "A", "p2": "B"}, True)
    test("line A-C",                 "add_line",     {"name": "lAC", "p1": "A", "p2": "C"}, True)
    test("ray B-C",                  "add_ray",      {"name": "rBC", "p1": "B", "p2": "C"}, True)
    test("perp line C->sAB",        "add_perpendicular_line", {"name": "perpC", "point": "C", "line": "sAB"}, True)
    test("perp bisector A-B",       "add_perpendicular_bisector", {"name": "pbAB", "p1": "A", "p2": "B"}, True)
    test("angle bisector B-A-C",    "add_angle_bisector", {"name": "bisBAC", "p1": "B", "vertex": "A", "p2": "C"}, True)
    test("parallel to sAB thru C",  "add_parallel_line", {"name": "parC", "point": "C", "line": "sAB"}, True)
    test("circle center A r=5",     "add_circle", {"name": "c1", "center": "A", "radius": 5, "point": None}, True)
    test("circle 3pt A-B-C",        "add_circle_3_points", {"name": "c3", "a": "A", "b": "B", "c": "C"}, True)
    test("incircle A-B-C",          "add_incircle", {"name": "ic", "a": "A", "b": "B", "c": "C"}, True)
    test("arc A center B-C",        "add_arc", {"name": "arc1", "center": "A", "start_pt": "B", "end_pt": "C"}, True)
    test("sector A center B-C",     "add_sector", {"name": "sec1", "center": "A", "start_pt": "B", "end_pt": "C"}, True)
    test("semicircle A-B",          "add_semicircle", {"name": "semi1", "p1": "A", "p2": "B"}, True)
    test("reflect C over sAB",      "transform_reflect_line", {"name": "Cr", "obj": "C", "line": "sAB"}, True)
    test("reflect C over A",        "transform_reflect_point", {"name": "Cp", "obj": "C", "point": "A"}, True)
    test("rotate B 90° around A",   "transform_rotate", {"name": "Br", "obj": "B", "angle": "90°", "center": "A"}, True)
    test("slider s1 [0,10]",        "add_slider", {"name": "s1", "min": 0, "max": 10, "step": 0.5}, True)

    # Tangent from outside: GeoGebra creates tanD_{1}, tanD_{2} — tool now detects this
    test("tangent D->c1 (outside, 2 lines)", "add_tangent", {"name": "tanD", "point": "D", "conic": "c1"}, True)

    # ── Degenerate cases (should fail with run_checked) ──────────────
    print("\n=== Degenerate cases (expect ok=False) ===")
    # Coincident points: A=(0,0) E=(0,0)
    test("line A-E (coincident)",    "add_line", {"name": "lDeg1", "p1": "A", "p2": "E"}, False)
    test("ray A-E (coincident)",     "add_ray",  {"name": "rDeg1", "p1": "A", "p2": "E"}, False)
    test("perp bisector A-E",        "add_perpendicular_bisector", {"name": "pbDeg", "p1": "A", "p2": "E"}, False)

    # Collinear: A(0,0), B(3,0), D(6,0) all on x-axis
    # GeoGebra treats 180° bisector as perpendicular → is_defined=True
    test("bisector B-A-D (collinear→perp)", "add_angle_bisector", {"name": "bisDeg", "p1": "B", "vertex": "A", "p2": "D"}, True)

    # Circle 3pt with collinear points → GeoGebra creates degenerate line, is_defined=True
    test("circle3pt A-B-D (collinear→line)", "add_circle_3_points", {"name": "c3Deg", "a": "A", "b": "B", "c": "D"}, True)

    # Incircle with collinear → GeoGebra still creates object, is_defined=True
    test("incircle A-B-D (collinear→deg)", "add_incircle", {"name": "icDeg", "a": "A", "b": "B", "c": "D"}, True)

    # Tangent from inside circle: B(3,0) is inside c1(center=A,r=5)
    test("tangent B->c1 (inside)",   "add_tangent", {"name": "tanIn", "point": "B", "conic": "c1"}, False)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    n_pass = sum(1 for s, *_ in results if s == "PASS")
    n_fail = sum(1 for s, *_ in results if s == "FAIL")
    print(f"  TOTAL: {n_pass} PASS, {n_fail} FAIL out of {len(results)} tests")

    if n_fail:
        print("\n  FAILURES:")
        for status, name, ok, expect_ok, err in results:
            if status == "FAIL":
                print(f"    {name}: got ok={ok}, expected ok={expect_ok}, err={err!r:.80s}")

    ctx.__exit__(None, None, None)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
