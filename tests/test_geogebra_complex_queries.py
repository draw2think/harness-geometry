"""
Complex Geometric Query Tests

Freely combines construction tools to build non-trivial figures,
then queries properties with analytically verified expected values.

Scenarios
---------
1. slope_variations    — oblique ±, fractional, steep, perpendicular pair,
                         slope-from-segment vs slope-from-line
2. area_diverse        — L-polygon, irregular pentagon, circle, arc, sector
3. varignon_theorem    — midpoint quadrilateral of ANY quad is a parallelogram
4. thales_theorem      — inscribed angle in semicircle = 90°
5. incircle_tangency   — incircle is tangent to all three sides
6. parallel_perp_chain — ParallelLine / PerpendicularLine propagate correctly
7. reflection_bisector — Reflect(A, L) → PerpendicularBisector(A,A') coincides with L
8. tangent_length      — power of a point: Distance(P, T) = sqrt(d² - r²)

Run: python tests/test_geogebra_complex_queries.py
"""
import sys, math
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from geogebra_render_common import apply_global_style, fit_view_square, set_label_visible

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_geogebra_complex_queries
(OUTDIR / "fig").mkdir(parents=True, exist_ok=True)
(OUTDIR / "log").mkdir(parents=True, exist_ok=True)


class TeeLog:
    """Mirror stdout to a file while keeping terminal output intact."""
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
        self._orig.write(f"  [LOG] {self._path}\n")

TOL = 1e-3   # tolerance for floating-point comparisons


# ─── helpers ────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

def subsection(title):
    print(f"\n  -- {title} --")

def cmd(ggb, label, command):
    result = ggb.eval_command(command)
    ok = "ok" if result.success else f"FAIL({result.error_message})"
    print(f"  [setup/{ok}] {label:42s}  `{command}`")
    return result.success

def query_cmd(ggb, label, command, obj_name, expected=None):
    """Execute query, read back value, assert expected if given."""
    result = ggb.eval_command(command)
    if not result.success:
        print(f"  [FAIL] {label:54s}  command failed: {result.error_message}")
        return None
    value     = ggb.get_value(obj_name)
    value_str = ggb.get_value_string(obj_name) or str(value)

    if expected is None:
        status = "PASS" if value is not None else "FAIL"
        print(f"  [{status}] {label:54s}  → {value_str}")
    elif isinstance(expected, bool):
        actual = bool(value) if value is not None else None
        status = "PASS" if actual == expected else "FAIL"
        print(f"  [{status}] {label:54s}  → {value_str}  (expected {expected})")
    else:
        ok = value is not None and abs(float(value) - float(expected)) < TOL
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:54s}  → {value_str}  (expected {expected:.4f})")
    return value

def save_figure(ggb, filename, hide=(), label_on=()):
    for name in hide:
        ggb.set_object_visible(name, False)
    apply_global_style(ggb)
    for name in label_on:
        set_label_visible(ggb, name, True)
    fit_view_square(ggb, padding=1.2)
    out = OUTDIR / "fig" / f"{filename}.png"
    ok  = ggb.export_png(out)
    print(f"\n  {'[SAVED]' if ok else '[FAIL export]'} {out}")


# ─── 1. Slope variations ────────────────────────────────────────────────────

def test_slope_variations(ggb):
    """
    Slopes tested:
      3/4   line through (0,0)-(4,3)
     -3/2   line through (0,0)-(2,-3)    negative
      √3    line through (0,0)-(1,√3)    60° angle
     -1/√3  perpendicular to 60° line    = -√3/3
      5     steep line through (0,0)-(1,5)
    Also: slope of a Segment object (unofficial but common use-case).
    """
    section("1. Slope Variations")
    ggb.reset()

    # Points for oblique lines
    cmd(ggb, "O=(0,0)",          "O = (0, 0)")
    cmd(ggb, "P1=(4,3)",         "P1 = (4, 3)")       # slope 3/4
    cmd(ggb, "P2=(2,-3)",        "P2 = (2, -3)")      # slope -3/2
    cmd(ggb, "P3=(1,sqrt(3))",   "P3 = (1, sqrt(3))") # slope √3  (60°)
    cmd(ggb, "P4=(sqrt(3),-1)",  "P4 = (sqrt(3), -1)")# slope -1/√3 (perp to 60°)
    cmd(ggb, "P5=(1,5)",         "P5 = (1, 5)")       # steep slope 5

    cmd(ggb, "l1 = Line(O,P1)",  "l1 = Line(O, P1)")
    cmd(ggb, "l2 = Line(O,P2)",  "l2 = Line(O, P2)")
    cmd(ggb, "l3 = Line(O,P3)",  "l3 = Line(O, P3)")
    cmd(ggb, "l4 = Line(O,P4)",  "l4 = Line(O, P4)")
    cmd(ggb, "l5 = Line(O,P5)",  "l5 = Line(O, P5)")
    cmd(ggb, "seg = Segment(O,P1)", "seg = Segment(O, P1)")

    subsection("Slope values")
    query_cmd(ggb, "Slope(l1)  through (4,3)  = 3/4",    "sl1 = Slope(l1)", "sl1", 3/4)
    query_cmd(ggb, "Slope(l2)  through (2,-3) = -3/2",   "sl2 = Slope(l2)", "sl2", -3/2)
    query_cmd(ggb, "Slope(l3)  60° line = √3",           "sl3 = Slope(l3)", "sl3", math.sqrt(3))
    query_cmd(ggb, "Slope(l4)  perp-60° = -1/√3",        "sl4 = Slope(l4)", "sl4", -1/math.sqrt(3))
    query_cmd(ggb, "Slope(l5)  steep = 5",                "sl5 = Slope(l5)", "sl5", 5.0)

    subsection("Perpendicular pair: l3 ⊥ l4")
    query_cmd(ggb, "ArePerpendicular(l3, l4)  (true)",
              "qp = ArePerpendicular(l3, l4)", "qp", True)

    subsection("Slope of Segment (exploratory – may not be supported)")
    query_cmd(ggb, "Slope(seg)  Segment OA slope",
              "sl_seg = Slope(seg)", "sl_seg")  # no expected: just observe

    save_figure(ggb, "complex_01_slope",
                hide=["sl1","sl2","sl3","sl4","sl5","sl_seg","qp"],
                label_on=["O","P1","P2","P3","P4","P5"])


# ─── 2. Area – diverse objects ──────────────────────────────────────────────

def test_area_diverse(ggb):
    """
    L-polygon  (0,0)-(3,0)-(3,1)-(1,1)-(1,3)-(0,3): area = 5  (exact)
    Pentagon   (0,0)-(4,0)-(5,3)-(2,5)-(-1,3):       area = 21 (Shoelace)
    Circle     radius=3:    area = 9π ≈ 28.274
    Quarter-sector radius=4: area = (1/4)π*16 = 4π ≈ 12.566
    Arc length (quarter, r=4):  = (1/4)*2πr = 2π ≈ 6.283
    """
    section("2. Area – Diverse Objects")
    ggb.reset()

    subsection("L-shaped polygon  area = 5")
    cmd(ggb, "Lv1=(0,0)", "Lv1 = (0, 0)")
    cmd(ggb, "Lv2=(3,0)", "Lv2 = (3, 0)")
    cmd(ggb, "Lv3=(3,1)", "Lv3 = (3, 1)")
    cmd(ggb, "Lv4=(1,1)", "Lv4 = (1, 1)")
    cmd(ggb, "Lv5=(1,3)", "Lv5 = (1, 3)")
    cmd(ggb, "Lv6=(0,3)", "Lv6 = (0, 3)")
    cmd(ggb, "Lpoly = Polygon(Lv1,Lv2,Lv3,Lv4,Lv5,Lv6)",
             "Lpoly = Polygon(Lv1, Lv2, Lv3, Lv4, Lv5, Lv6)")
    query_cmd(ggb, "Area(L-polygon) = 5",
              "a_L = Area(Lpoly)", "a_L", 5.0)

    subsection("Irregular pentagon  area = 21  (Shoelace verified)")
    cmd(ggb, "Pv1=(0,0)",  "Pv1 = (0, 0)")
    cmd(ggb, "Pv2=(4,0)",  "Pv2 = (4, 0)")
    cmd(ggb, "Pv3=(5,3)",  "Pv3 = (5, 3)")
    cmd(ggb, "Pv4=(2,5)",  "Pv4 = (2, 5)")
    cmd(ggb, "Pv5=(-1,3)", "Pv5 = (-1, 3)")
    cmd(ggb, "Ppoly = Polygon(Pv1,Pv2,Pv3,Pv4,Pv5)",
             "Ppoly = Polygon(Pv1, Pv2, Pv3, Pv4, Pv5)")
    query_cmd(ggb, "Area(pentagon) = 21",
              "a_P = Area(Ppoly)", "a_P", 21.0)

    subsection("Circle r=3  area = 9π ≈ 28.274")
    cmd(ggb, "Oc = (10,0)", "Oc = (10, 0)")  # offset to avoid overlap
    cmd(ggb, "circ = Circle(Oc, 3)", "circ = Circle(Oc, 3)")
    query_cmd(ggb, "Area(circle r=3) = 9π",
              "a_circ = Area(circ)", "a_circ", 9 * math.pi)

    subsection("Quarter sector r=4  area = 4π ≈ 12.566")
    cmd(ggb, "Os = (0,10)",  "Os = (0, 10)")
    cmd(ggb, "Sa = (4,10)",  "Sa = (4, 10)")  # on circle, right of center
    cmd(ggb, "Sb = (0,14)",  "Sb = (0, 14)")  # 90° sweep endpoint, top of center
    cmd(ggb, "circ4 = Circle(Os, Sa)", "circ4 = Circle(Os, Sa)")   # r=4
    # Note: use 'csec' not 'sec' to avoid name clash with secant trig function
    cmd(ggb, "csec = CircularSector(Os, Sa, Sb)",
             "csec = CircularSector(Os, Sa, Sb)")
    query_cmd(ggb, "Area(quarter sector r=4) = 4π",
              "a_sec = Area(csec)", "a_sec", 4 * math.pi)

    subsection("Arc length (quarter r=4)  = 2π ≈ 6.283")
    cmd(ggb, "arc = CircularArc(Os, Sa, Sb)", "arc = CircularArc(Os, Sa, Sb)")
    query_cmd(ggb, "Length(quarter arc r=4) = 2π",
              "l_arc = Length(arc)", "l_arc", 2 * math.pi)

    save_figure(ggb, "complex_02_area",
                hide=["a_L","a_P","a_circ","a_sec","l_arc","circ4","csec"],
                label_on=["Lv1","Pv2","Oc","Os"])


# ─── 3. Varignon's theorem ──────────────────────────────────────────────────

def test_varignon_theorem(ggb):
    """
    Varignon (1731): for ANY quadrilateral ABCD, the midpoints of its four
    sides form a parallelogram.

    ABCD = (0,0)-(6,0)-(5,4)-(1,4)   (irregular trapezoid)
    E = Mid(A,B) = (3,0)
    F = Mid(B,C) = (5.5,2)
    G = Mid(C,D) = (3,4)
    H = Mid(D,A) = (0.5,2)

    Verify:
      AreParallel(Line(E,F), Line(H,G))  → true   (EF ∥ HG)
      AreParallel(Line(F,G), Line(E,H))  → true   (FG ∥ EH)
      Area(EFGH) = Area(ABCD) / 2
        Area(ABCD): Shoelace = (0*0-6*0)+(6*4-5*0)+(5*4-1*4)+(1*0-0*4) = 40/2 = 20
        Area(EFGH): Shoelace = 20/2 = 10
    """
    section("3. Varignon's Theorem – Midpoint Parallelogram")
    ggb.reset()

    cmd(ggb, "A=(0,0)", "A = (0, 0)")
    cmd(ggb, "B=(6,0)", "B = (6, 0)")
    cmd(ggb, "C=(5,4)", "C = (5, 4)")
    cmd(ggb, "D=(1,4)", "D = (1, 4)")
    cmd(ggb, "quad = Polygon(A,B,C,D)", "quad = Polygon(A, B, C, D)")

    cmd(ggb, "E = Midpoint(A,B)", "E = Midpoint(A, B)")
    cmd(ggb, "F = Midpoint(B,C)", "F = Midpoint(B, C)")
    cmd(ggb, "G = Midpoint(C,D)", "G = Midpoint(C, D)")
    cmd(ggb, "H = Midpoint(D,A)", "H = Midpoint(D, A)")
    cmd(ggb, "mpoly = Polygon(E,F,G,H)", "mpoly = Polygon(E, F, G, H)")

    # Lines through midpoint pairs (for parallel queries)
    cmd(ggb, "lEF = Line(E,F)", "lEF = Line(E, F)")
    cmd(ggb, "lHG = Line(H,G)", "lHG = Line(H, G)")
    cmd(ggb, "lFG = Line(F,G)", "lFG = Line(F, G)")
    cmd(ggb, "lEH = Line(E,H)", "lEH = Line(E, H)")

    subsection("Parallelism of opposite sides")
    query_cmd(ggb, "AreParallel(EF, HG)  (true)",
              "qv1 = AreParallel(lEF, lHG)", "qv1", True)
    query_cmd(ggb, "AreParallel(FG, EH)  (true)",
              "qv2 = AreParallel(lFG, lEH)", "qv2", True)
    # Cross-check: adjacent sides are NOT parallel (general quad)
    query_cmd(ggb, "AreParallel(EF, FG)  (false – adjacent)",
              "qv3 = AreParallel(lEF, lFG)", "qv3", False)

    subsection("Area: EFGH = ABCD / 2")
    query_cmd(ggb, "Area(ABCD) = 20",   "a_quad  = Area(quad)",  "a_quad",  20.0)
    query_cmd(ggb, "Area(EFGH) = 10",   "a_mpoly = Area(mpoly)", "a_mpoly", 10.0)

    save_figure(ggb, "complex_03_varignon",
                hide=["qv1","qv2","qv3","a_quad","a_mpoly","lEF","lHG","lFG","lEH"],
                label_on=["A","B","C","D","E","F","G","H"])


# ─── 4. Thales' theorem ─────────────────────────────────────────────────────

def test_thales_theorem(ggb):
    """
    Thales (≈600 BCE): Any angle inscribed in a semicircle (with the diameter
    as chord) is a right angle.

    Setup:
      Diameter: A=(-3,0), B=(3,0), radius=3, circle centered at O=(0,0)
      Semicircle: upper half

    Test points on the semicircle:
      P1=(0,3)          top       → Angle(A,P1,B) = 90°
      P2=(3cos60°, 3sin60°)       → Angle(A,P2,B) = 90°
      P3=(3cos120°,3sin120°)      → Angle(A,P3,B) = 90°

    Also verify that a point NOT on the semicircle gives ≠ 90°.
    """
    section("4. Thales' Theorem – Inscribed Angle in Semicircle = 90°")
    ggb.reset()

    cmd(ggb, "A=(-3,0)", "A = (-3, 0)")
    cmd(ggb, "B=(3,0)",  "B = (3, 0)")
    cmd(ggb, "O=(0,0)",  "O = (0, 0)")
    cmd(ggb, "sc = Semicircle(A,B)", "sc = Semicircle(A, B)")

    # Three inscribed points (analytically on the semicircle)
    cmd(ggb, "P1=(0,3)",                "P1 = (0, 3)")                   # 90°
    cmd(ggb, "P2=(3cos(60°),3sin(60°))", "P2 = (3cos(60°), 3sin(60°))") # 60°
    cmd(ggb, "P3=(3cos(120°),3sin(120°))","P3 = (3cos(120°), 3sin(120°))") # 120°

    # One point off the circle (interior)
    cmd(ggb, "Q=(1,1) (interior)", "Q = (1, 1)")

    subsection("Angle at inscribed points = 90° (= π/2 rad ≈ 1.5708)")
    pi_over_2 = math.pi / 2
    query_cmd(ggb, "Angle(A,P1,B) top of circle",
              "ang1 = Angle(A, P1, B)", "ang1", pi_over_2)
    query_cmd(ggb, "Angle(A,P2,B) at 60°",
              "ang2 = Angle(A, P2, B)", "ang2", pi_over_2)
    query_cmd(ggb, "Angle(A,P3,B) at 120°",
              "ang3 = Angle(A, P3, B)", "ang3", pi_over_2)

    subsection("Interior point Q=(1,1): angle ≠ 90°")
    query_cmd(ggb, "Angle(A,Q,B) interior (not 90°)",
              "ang4 = Angle(A, Q, B)", "ang4")   # just observe value

    subsection("AreCollinear: A, O, B on x-axis (diameter)")
    query_cmd(ggb, "AreCollinear(A,O,B)  (true)",
              "qth = AreCollinear(A, O, B)", "qth", True)

    save_figure(ggb, "complex_04_thales",
                hide=["ang1","ang2","ang3","ang4","qth"],
                label_on=["A","B","O","P1","P2","P3","Q"])


# ─── 5. Incircle tangency ───────────────────────────────────────────────────

def test_incircle_tangency(ggb):
    """
    3-4-5 right triangle: A=(0,0), B=(4,0), C=(0,3)
    Semi-perimeter s = 6, Area = 6, inradius r = Area/s = 1
    Incenter = (r, r) = (1, 1)

    Each side as a Line must be tangent to Incircle(A,B,C):
      l_AB: y=0            dist((1,1), y=0)      = 1 = r ✓
      l_BC: 3x+4y=12       dist((1,1), 3x+4y=12) = |3+4-12|/5 = 1 = r ✓
      l_CA: x=0            dist((1,1), x=0)       = 1 = r ✓
    """
    section("5. Incircle Tangency – All Three Sides")
    ggb.reset()

    cmd(ggb, "A=(0,0)", "A = (0, 0)")
    cmd(ggb, "B=(4,0)", "B = (4, 0)")
    cmd(ggb, "C=(0,3)", "C = (0, 3)")
    cmd(ggb, "tri = Polygon(A,B,C)", "tri = Polygon(A, B, C)")
    cmd(ggb, "inc = Incircle(A,B,C)", "inc = Incircle(A, B, C)")

    # Lines through each side (IsTangent requires Line, not Segment)
    cmd(ggb, "l_AB = Line(A,B)", "l_AB = Line(A, B)")  # y = 0
    cmd(ggb, "l_BC = Line(B,C)", "l_BC = Line(B, C)")  # 3x+4y = 12
    cmd(ggb, "l_CA = Line(C,A)", "l_CA = Line(C, A)")  # x = 0

    subsection("IsTangent: incircle tangent to each side")
    query_cmd(ggb, "IsTangent(l_AB, inc)  AB: y=0       (true)",
              "qt1 = IsTangent(l_AB, inc)", "qt1", True)
    query_cmd(ggb, "IsTangent(l_BC, inc)  BC: 3x+4y=12  (true)",
              "qt2 = IsTangent(l_BC, inc)", "qt2", True)
    query_cmd(ggb, "IsTangent(l_CA, inc)  CA: x=0       (true)",
              "qt3 = IsTangent(l_CA, inc)", "qt3", True)

    subsection("Inradius = 1  (Area/s = 6/6)")
    query_cmd(ggb, "Area(tri) = 6",   "a_tri = Area(tri)",    "a_tri", 6.0)
    query_cmd(ggb, "Perimeter = 12",  "p_tri = Perimeter(tri)","p_tri", 12.0)

    subsection("All angles of triangle (sum check)")
    query_cmd(ggb, "Angle at A (right angle = 90°)",
              "ang_A = Angle(B, A, C)", "ang_A", math.pi/2)
    query_cmd(ggb, "Angle at B = arctan(3/4) ≈ 36.87°",
              "ang_B = Angle(C, B, A)", "ang_B", math.atan(3/4))
    query_cmd(ggb, "Angle at C = arctan(4/3) ≈ 53.13°",
              "ang_C = Angle(A, C, B)", "ang_C", math.atan(4/3))

    save_figure(ggb, "complex_05_incircle",
                hide=["qt1","qt2","qt3","a_tri","p_tri","ang_A","ang_B","ang_C",
                      "l_AB","l_BC","l_CA"],
                label_on=["A","B","C"])


# ─── 6. Parallel-perpendicular chain ────────────────────────────────────────

def test_parallel_perp_chain(ggb):
    """
    Start with line L1 (slope 3/4).
    External point P not on L1.
    L2 = ParallelLine(P, L1)    → L2 ∥ L1
    L3 = PerpendicularLine(P, L1) → L3 ⊥ L1

    Chain: since L2 ∥ L1 and L3 ⊥ L1 → L3 ⊥ L2 also.

    Verify:
      AreParallel(L1, L2)       true
      ArePerpendicular(L1, L3)  true
      ArePerpendicular(L2, L3)  true  ← derived, not constructed directly
      AreParallel(L1, L3)       false
      AreParallel(L2, L3)       false
      ArePerpendicular(L1, L2)  false
    """
    section("6. Parallel-Perpendicular Chain")
    ggb.reset()

    cmd(ggb, "A=(0,0)", "A = (0, 0)")
    cmd(ggb, "B=(4,3)", "B = (4, 3)")   # slope 3/4
    cmd(ggb, "P=(0,6)", "P = (0, 6)")   # external point
    cmd(ggb, "L1 = Line(A,B)",              "L1 = Line(A, B)")
    cmd(ggb, "L2 = Line(P,L1)  (parallel)", "L2 = Line(P, L1)")
    cmd(ggb, "L3 = PerpendicularLine(P,L1)","L3 = PerpendicularLine(P, L1)")

    subsection("Direct constructions")
    query_cmd(ggb, "AreParallel(L1,L2)       (true)",
              "qc1 = AreParallel(L1, L2)",       "qc1", True)
    query_cmd(ggb, "ArePerpendicular(L1,L3)   (true)",
              "qc2 = ArePerpendicular(L1, L3)",   "qc2", True)

    subsection("Derived: L3 ⊥ L2  (because L2 ∥ L1)")
    query_cmd(ggb, "ArePerpendicular(L2,L3)   (true – derived)",
              "qc3 = ArePerpendicular(L2, L3)",   "qc3", True)

    subsection("Negative cases")
    query_cmd(ggb, "AreParallel(L1,L3)        (false)",
              "qc4 = AreParallel(L1, L3)",        "qc4", False)
    query_cmd(ggb, "AreParallel(L2,L3)        (false)",
              "qc5 = AreParallel(L2, L3)",        "qc5", False)
    query_cmd(ggb, "ArePerpendicular(L1,L2)   (false – parallel)",
              "qc6 = ArePerpendicular(L1, L2)",   "qc6", False)

    subsection("Slopes (verify numerically)")
    query_cmd(ggb, "Slope(L1) = 0.75",  "sl_L1 = Slope(L1)", "sl_L1", 0.75)
    query_cmd(ggb, "Slope(L2) = 0.75",  "sl_L2 = Slope(L2)", "sl_L2", 0.75)
    query_cmd(ggb, "Slope(L3) = -4/3",  "sl_L3 = Slope(L3)", "sl_L3", -4/3)

    save_figure(ggb, "complex_06_chain",
                hide=["qc1","qc2","qc3","qc4","qc5","qc6","sl_L1","sl_L2","sl_L3"],
                label_on=["A","B","P"])


# ─── 7. Reflection = perpendicular bisector ──────────────────────────────────

def test_reflection_bisector(ggb):
    """
    Theorem: reflecting A over line L gives A'; the perpendicular bisector of
    segment AA' is exactly L.

    Setup:
      L  = Line through O=(0,0) and (1,1)  (y=x, slope 1)
      A  = (3,0)
      A' = Reflect(A, L)    → should be (0,3)  (swap x,y over y=x)
      PB = PerpendicularBisector(A, A')
         midpoint of AA' = (1.5,1.5) ∈ y=x ✓
         direction of AA' = (-3,3) → perpendicular direction = (1,1) → slope 1 ✓
         PB passes through (1.5,1.5) with slope 1  → same as y=x = L

    Query: AreEqual(L, PB) → true

    Also:
      Midpoint(A,A') should lie on L → AreCollinear(O, mid, (1,1)) [=(Midpoint)]
      Distance(A, A') = 3√2 ≈ 4.243
    """
    section("7. Reflection = Perpendicular Bisector")
    ggb.reset()

    cmd(ggb, "O=(0,0)",  "O = (0, 0)")
    cmd(ggb, "V=(1,1)",  "V = (1, 1)")    # second point on y=x
    cmd(ggb, "L = Line(O,V)", "L = Line(O, V)")
    cmd(ggb, "A=(3,0)",  "A = (3, 0)")
    cmd(ggb, "Ar = Reflect(A,L)", "Ar = Reflect(A, L)")   # should be (0,3)
    cmd(ggb, "PB = PerpendicularBisector(A,Ar)",
             "PB = PerpendicularBisector(A, Ar)")
    cmd(ggb, "Mid_AAr = Midpoint(A,Ar)", "Mid_AAr = Midpoint(A, Ar)")

    subsection("Core theorem: L == PerpendicularBisector(A, A')")
    query_cmd(ggb, "AreEqual(L, PB)  (true)",
              "qr1 = AreEqual(L, PB)", "qr1", True)

    subsection("Midpoint of AA' lies on L")
    query_cmd(ggb, "AreCollinear(O, Mid_AAr, V)  (true)",
              "qr2 = AreCollinear(O, Mid_AAr, V)", "qr2", True)

    subsection("Distance and symmetry checks")
    query_cmd(ggb, "Distance(O,A) = 3",
              "dr_OA = Distance(O, A)", "dr_OA", 3.0)
    query_cmd(ggb, "Distance(O,Ar) = 3  (image equidistant)",
              "dr_OAr = Distance(O, Ar)", "dr_OAr", 3.0)
    query_cmd(ggb, "Distance(A,Ar) = 3√2 ≈ 4.243",
              "dr_AAr = Distance(A, Ar)", "dr_AAr", 3*math.sqrt(2))

    subsection("AreCongruent: segment(O,A) ≅ segment(O,Ar)")
    cmd(ggb, "s1 = Segment(O,A)",  "s1 = Segment(O, A)")
    cmd(ggb, "s2 = Segment(O,Ar)", "s2 = Segment(O, Ar)")
    query_cmd(ggb, "AreCongruent(s1, s2)  (true)",
              "qr3 = AreCongruent(s1, s2)", "qr3", True)

    save_figure(ggb, "complex_07_reflection",
                hide=["qr1","qr2","dr_OA","dr_OAr","dr_AAr","qr3","Mid_AAr","PB","s1","s2"],
                label_on=["O","V","A","Ar"])


# ─── 8. Tangent length from external point ───────────────────────────────────

def test_tangent_length(ggb):
    """
    Power of a point: for external point P, circle center C radius r,
        tangent length PT = sqrt(PC² - r²)

    Setup:
      C=(5,0), r=3, P=(0,0)
      PC = 5, PT = sqrt(25-9) = sqrt(16) = 4

    Construction:
      tan_lines = Tangent(P, circ)  → two tangent lines
      T1 = Intersect(tan_lines, circ, 1)  ← tangent point on line 1
      Verify: Distance(P,T1) = 4
      Verify: IsTangent(line_1, circ) = true
      Verify: ArePerpendicular(Line(C,T1), line_1) = true  ← radius ⊥ tangent

    Also test the secant-tangent relationship:
      Secant through P: Line(P, (5,3)) intersects circle at S1, S2
      PS1 * PS2 should equal PT² = 16
    """
    section("8. Tangent Length – Power of a Point")
    ggb.reset()

    cmd(ggb, "C=(5,0)",   "C = (5, 0)")
    cmd(ggb, "P=(0,0)",   "P = (0, 0)")
    cmd(ggb, "circ = Circle(C,3)", "circ = Circle(C, 3)")

    # Thales-circle construction: tangent points = intersection of
    # the circle-with-diameter-PC with circ.  (radius ⊥ tangent at touch point)
    cmd(ggb, "M = Midpoint(P,C)",      "M = Midpoint(P, C)")     # midpoint of PC
    cmd(ggb, "hcirc = Circle(M,P)",    "hcirc = Circle(M, P)")   # diameter PC → r=2.5
    cmd(ggb, "T1 = Intersect(hcirc,circ,1)", "T1 = Intersect(hcirc, circ, 1)")
    cmd(ggb, "T2 = Intersect(hcirc,circ,2)", "T2 = Intersect(hcirc, circ, 2)")
    # Tangent lines and radius line
    cmd(ggb, "tan1 = Line(P,T1)",  "tan1 = Line(P, T1)")
    cmd(ggb, "tan2 = Line(P,T2)",  "tan2 = Line(P, T2)")
    cmd(ggb, "r_CT1 = Line(C,T1)", "r_CT1 = Line(C, T1)")

    subsection("Tangent length = sqrt(PC²-r²) = sqrt(25-9) = 4")
    query_cmd(ggb, "Distance(P,T1) = 4",
              "d_PT1 = Distance(P, T1)", "d_PT1", 4.0)
    query_cmd(ggb, "Distance(P,T2) = 4  (symmetric)",
              "d_PT2 = Distance(P, T2)", "d_PT2", 4.0)

    subsection("Radius ⊥ tangent at touch point")
    query_cmd(ggb, "ArePerpendicular(r_CT1, tan1)  (true)",
              "qtg1 = ArePerpendicular(r_CT1, tan1)", "qtg1", True)

    subsection("IsTangent verification")
    query_cmd(ggb, "IsTangent(tan1, circ)  (true)",
              "qtg2 = IsTangent(tan1, circ)", "qtg2", True)

    subsection("Distances to center")
    query_cmd(ggb, "Distance(P,C) = 5",
              "d_PC = Distance(P, C)", "d_PC", 5.0)
    query_cmd(ggb, "Distance(C,T1) = 3  (radius)",
              "d_CT1 = Distance(C, T1)", "d_CT1", 3.0)

    subsection("Pythagorean check: PT1²+CT1² = PC²")
    # Create numeric checks via GeoGebra expressions
    query_cmd(ggb, "Distance(P,T1)² + Distance(C,T1)² via sides (=25)",
              "pyth = Distance(P,T1)^2 + Distance(C,T1)^2", "pyth", 25.0)

    save_figure(ggb, "complex_08_tangent",
                hide=["d_PT1","d_PT2","qtg1","qtg2","d_PC","d_CT1","pyth",
                      "r_CT1","tans","tan1","tan2"],
                label_on=["P","C","T1","T2"])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  Complex Geometric Query Tests")
    print("="*62)

    tests = [
        ("Slope Variations",          test_slope_variations,      "complex_01_slope"),
        ("Area – Diverse Objects",    test_area_diverse,          "complex_02_area"),
        ("Varignon's Theorem",        test_varignon_theorem,      "complex_03_varignon"),
        ("Thales' Theorem",           test_thales_theorem,        "complex_04_thales"),
        ("Incircle Tangency",         test_incircle_tangency,     "complex_05_incircle"),
        ("Parallel-Perp Chain",       test_parallel_perp_chain,   "complex_06_chain"),
        ("Reflection = PerpBisector", test_reflection_bisector,   "complex_07_reflection"),
        ("Tangent Length",            test_tangent_length,        "complex_08_tangent"),
    ]

    for name, fn, log_name in tests:
        print(f"\n[GeoGebra ← {name}]")
        with TeeLog(OUTDIR / "log" / f"{log_name}.txt"):
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                fn(ggb)

    print("\n" + "="*62)
    print("  All complex query tests complete. Figures → temp/")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
