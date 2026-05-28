"""
GeoGebra Query Interface Test.

Tests boolean predicates and scalar queries as documented in docs/query_guide.md:
  - Boolean predicates: AreParallel, ArePerpendicular, IsTangent, AreEqual,
                        AreCollinear, AreConcyclic, AreCongruent, IsInRegion
  - Scalar queries:     Length, Distance, Area, Perimeter, Slope, Angle

Each section saves a rendered PNG to temp/.

Run: python tests/test_geogebra_query.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from geogebra_render_common import apply_global_style, fit_view_square, set_label_visible

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_geogebra_query
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


# ─── helpers ────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subsection(title):
    print(f"\n  -- {title} --")


def cmd(ggb, label, command):
    """Execute a construction command; print setup status."""
    result = ggb.eval_command(command)
    status = "PASS" if result.success else "FAIL"
    reason = f"  ({result.error_message})" if not result.success else ""
    print(f"  [setup] {label:45s}  `{command}`{reason}")
    return result.success


def query_cmd(ggb, label, command, obj_name, expected=None):
    """
    Execute a query command, read back the value, print PASS/FAIL.

    expected:
      bool  → compare against GeoGebra bool (1.0=true / 0.0=false)
      float → compare with tolerance 1e-4
      None  → just check the value exists (no assertion)
    """
    result = ggb.eval_command(command)
    if not result.success:
        print(f"  [FAIL] {label:52s}  command failed: {result.error_message}")
        return None

    value     = ggb.get_value(obj_name)
    value_str = ggb.get_value_string(obj_name) or str(value)

    if expected is None:
        status = "PASS" if value is not None else "FAIL"
        print(f"  [{status}] {label:52s}  → {value_str}")
    elif isinstance(expected, bool):
        actual = bool(value) if value is not None else None
        status = "PASS" if actual == expected else "FAIL"
        print(f"  [{status}] {label:52s}  → {value_str}  (expected {expected})")
    else:
        ok = value is not None and abs(float(value) - float(expected)) < 1e-4
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:52s}  → {value_str}  (expected {expected})")

    return value


def save_figure(ggb, filename, hide=(), label_on=()):
    """
    Apply style, fit viewport, export PNG to temp/<filename>.png.

    hide     : object names to hide from the graphics view before export
    label_on : object names whose labels should be shown (all others hidden)
    """
    # Hide clutter objects (query result booleans/numbers, auxiliary points)
    for name in hide:
        ggb.set_object_visible(name, False)

    # Apply global style (sets colors, thickness; hides all labels)
    apply_global_style(ggb)

    # Selectively show labels for key objects
    for name in label_on:
        set_label_visible(ggb, name, True)

    # Fit viewport to visible points with square aspect ratio
    fit_view_square(ggb, padding=1.2)

    out = OUTDIR / "fig" / f"{filename}.png"
    ok  = ggb.export_png(out)
    print(f"\n  {'[SAVED]' if ok else '[FAIL export]'} {out}")
    return ok


# ─── Test: Boolean Predicates ────────────────────────────────────────────────

def test_boolean_predicates(ggb):
    section("Boolean Predicates")
    ggb.reset()

    # ── setup ────────────────────────────────────────────────────────────────
    cmd(ggb, "O = (0,0)",    "O = (0, 0)")
    cmd(ggb, "A = (4,0)",    "A = (4, 0)")
    cmd(ggb, "B = (0,4)",    "B = (0, 4)")
    cmd(ggb, "E = (8,0)",    "E = (8, 0)")      # collinear with O, A on x-axis

    cmd(ggb, "P_in  = (1,1)",   "P_in  = (1, 1)")   # inside triangle OAB
    cmd(ggb, "P_out = (5,5)",   "P_out = (5, 5)")   # outside triangle OAB (hide in fig)

    # AreConcyclic: four points on the unit circle
    cmd(ggb, "Q1 = (1,0)",   "Q1 = (1, 0)")
    cmd(ggb, "Q2 = (0,1)",   "Q2 = (0, 1)")
    cmd(ggb, "Q3 = (-1,0)",  "Q3 = (-1, 0)")
    cmd(ggb, "Q4 = (0,-1)",  "Q4 = (0, -1)")
    cmd(ggb, "unit_c = Circle((0,0), 1)", "unit_c = Circle((0,0), 1)")

    # Lines
    cmd(ggb, "l_h1 = horizontal y=0",  "l_h1 = Line((0,0),(1,0))")
    cmd(ggb, "l_h2 = horizontal y=3",  "l_h2 = Line((0,3),(1,3))")   # parallel to l_h1
    cmd(ggb, "l_v  = vertical  x=0",   "l_v  = Line((0,0),(0,1))")   # perp to l_h1

    # Circle centered at A=(4,0) radius 2; l_tan: y=2 is tangent (dist=2 from center)
    cmd(ggb, "cir = Circle(A, 2)",          "cir = Circle(A, 2)")
    cmd(ggb, "l_tan = Line y=2 (tangent)",  "l_tan = Line((0,2),(1,2))")

    # Triangle
    cmd(ggb, "tri = Polygon(O,A,B)",  "tri = Polygon(O, A, B)")

    # AreEqual: Midpoint(O,E) == A=(4,0)
    cmd(ggb, "M   = Midpoint(O,E)",   "M   = Midpoint(O, E)")
    cmd(ggb, "Meq = (4,0)",           "Meq = (4, 0)")

    # AreCongruent: two circles same radius, different centers
    cmd(ggb, "cc1 = Circle(O, 3)",  "cc1 = Circle(O, 3)")
    cmd(ggb, "cc2 = Circle(A, 3)",  "cc2 = Circle(A, 3)")

    # ── query tests ──────────────────────────────────────────────────────────
    subsection("AreParallel")
    query_cmd(ggb, "l_h1 ∥ l_h2  (true) ",  "q_par1 = AreParallel(l_h1, l_h2)", "q_par1", True)
    query_cmd(ggb, "l_h1 ∥ l_v   (false)",  "q_par2 = AreParallel(l_h1, l_v)",  "q_par2", False)

    subsection("ArePerpendicular")
    query_cmd(ggb, "l_h1 ⊥ l_v   (true) ",  "q_perp1 = ArePerpendicular(l_h1, l_v)",   "q_perp1", True)
    query_cmd(ggb, "l_h1 ⊥ l_h2  (false)",  "q_perp2 = ArePerpendicular(l_h1, l_h2)",  "q_perp2", False)

    subsection("IsTangent")
    query_cmd(ggb, "l_tan tangent to cir  (true) ",  "q_tan1 = IsTangent(l_tan, cir)", "q_tan1", True)
    query_cmd(ggb, "l_h1  secant of cir   (false)",  "q_tan2 = IsTangent(l_h1, cir)",  "q_tan2", False)

    subsection("AreEqual")
    query_cmd(ggb, "M == Meq == A=(4,0)  (true)",  "q_eq = AreEqual(M, Meq)", "q_eq", True)

    subsection("AreCollinear")
    query_cmd(ggb, "O, A, E on x-axis  (true) ", "q_col1 = AreCollinear(O, A, E)", "q_col1", True)
    query_cmd(ggb, "O, A, B            (false)", "q_col2 = AreCollinear(O, A, B)", "q_col2", False)

    subsection("AreConcyclic")
    query_cmd(ggb, "Q1..Q4 on unit circle  (true)", "q_cyc = AreConcyclic(Q1, Q2, Q3, Q4)", "q_cyc", True)

    subsection("AreCongruent")
    query_cmd(ggb, "cc1 ≅ cc2 (same radius=3)  (true)", "q_cong = AreCongruent(cc1, cc2)", "q_cong", True)

    subsection("IsInRegion")
    query_cmd(ggb, "P_in  inside tri  (true) ",  "q_reg1 = IsInRegion(P_in, tri)",  "q_reg1", True)
    query_cmd(ggb, "P_out inside tri  (false)",  "q_reg2 = IsInRegion(P_out, tri)", "q_reg2", False)

    # ── render ───────────────────────────────────────────────────────────────
    # Hide: query result objects, auxiliary/off-frame points
    QUERY_OBJS = [
        "q_par1","q_par2","q_perp1","q_perp2",
        "q_tan1","q_tan2","q_eq","q_col1","q_col2",
        "q_cyc","q_cong","q_reg1","q_reg2",
    ]
    AUX_OBJS = ["P_out", "M", "Meq"]   # P_in stays to illustrate IsInRegion

    save_figure(
        ggb,
        filename  = "query_01_boolean",
        hide      = QUERY_OBJS + AUX_OBJS,
        label_on  = ["O", "A", "B", "E", "P_in"],
    )


# ─── Test: Scalar Queries ────────────────────────────────────────────────────

def test_scalar_queries(ggb):
    section("Scalar Queries")
    ggb.reset()

    # ── setup ────────────────────────────────────────────────────────────────
    cmd(ggb, "O = (0,0)",  "O = (0, 0)")
    cmd(ggb, "A = (4,0)",  "A = (4, 0)")
    cmd(ggb, "B = (0,3)",  "B = (0, 3)")   # 3-4-5 right triangle

    cmd(ggb, "seg = Segment(O,A)",    "seg = Segment(O, A)")
    cmd(ggb, "tri = Polygon(O,A,B)",  "tri = Polygon(O, A, B)")
    cmd(ggb, "ang = Angle(A,O,B)",    "ang = Angle(A, O, B)")   # 90° arc at O
    cmd(ggb, "l_h  = Line((0,0),(1,0))",   "l_h  = Line((0,0),(1,0))")   # slope 0
    cmd(ggb, "l_d  = Line((0,0),(1,1))",   "l_d  = Line((0,0),(1,1))")   # slope 1
    cmd(ggb, "l_AB = Line(A,B)",           "l_AB = Line(A, B)")           # A=(4,0) B=(0,3) -> slope -3/4
    cmd(ggb, "seg_AB = Segment(A,B)",      "seg_AB = Segment(A, B)")      # same slope via Segment

    # ── query tests ──────────────────────────────────────────────────────────
    subsection("Length / Distance")
    query_cmd(ggb, "Length(seg)   OA = 4",   "qs_len  = Length(seg)",    "qs_len",  4.0)
    query_cmd(ggb, "Distance(O,A) = 4",      "qs_dist = Distance(O, A)", "qs_dist", 4.0)
    query_cmd(ggb, "Distance(O,B) = 3",      "qs_db   = Distance(O, B)", "qs_db",   3.0)

    subsection("Area")
    # base=4, height=3 → area = 0.5 × 4 × 3 = 6
    query_cmd(ggb, "Area(tri) OAB = 6",  "qs_area = Area(tri)", "qs_area", 6.0)

    subsection("Perimeter")
    # OA=4, OB=3, AB=5 → perimeter = 12
    query_cmd(ggb, "Perimeter(tri) 3-4-5 = 12",  "qs_peri = Perimeter(tri)", "qs_peri", 12.0)

    subsection("Slope")
    query_cmd(ggb, "Slope(l_h)    horizontal = 0",     "qs_sl1 = Slope(l_h)",    "qs_sl1", 0.0)
    query_cmd(ggb, "Slope(l_d)    diagonal   = 1",     "qs_sl2 = Slope(l_d)",    "qs_sl2", 1.0)
    query_cmd(ggb, "Slope(l_AB)   oblique    = -3/4",  "qs_sl3 = Slope(l_AB)",   "qs_sl3", -3/4)
    query_cmd(ggb, "Slope(seg_AB) Segment    = -3/4",  "qs_sl4 = Slope(seg_AB)", "qs_sl4", -3/4)

    subsection("Angle")
    # Right angle at O: OA along x, OB along y → 90°
    query_cmd(ggb, "Angle(A,O,B) = 90°",  "qs_ang = Angle(A, O, B)", "qs_ang")
    query_cmd(ggb, "Angle(B,A,O)", "qs_ang2 = Angle(B, A, O)", "qs_ang2")
    query_cmd(ggb, "Angle(O,B,A)", "qs_ang3 = Angle(O, B, A)", "qs_ang3")

    # ── render ───────────────────────────────────────────────────────────────
    QUERY_OBJS = [
        "qs_len","qs_dist","qs_db","qs_area","qs_peri",
        "qs_sl1","qs_sl2","qs_sl3","qs_sl4",
        "qs_ang","qs_ang2","qs_ang3",
        "l_AB","seg_AB",   # hide infinite line and duplicate segment to keep figure clean
    ]

    save_figure(
        ggb,
        filename  = "query_02_scalar",
        hide      = QUERY_OBJS,
        label_on  = ["O", "A", "B"],
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  GeoGebra Query Interface Test")
    print("="*60)

    for name, test_fn, log_name in [
        ("Boolean Predicates", test_boolean_predicates, "query_01_boolean"),
        ("Scalar Queries",     test_scalar_queries,     "query_02_scalar"),
    ]:
        print(f"\n[Initializing GeoGebra for: {name}]")
        with TeeLog(OUTDIR / "log" / f"{log_name}.txt"):
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                test_fn(ggb)

    print("\n" + "="*60)
    print("  Query tests complete. Figures saved to temp/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
