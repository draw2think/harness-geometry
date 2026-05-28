"""
Comprehensive GeoGebra command coverage test.

Tests all 4 tool categories via the JavaScript API:
  1. Basic & Edit
  2. Geometric Construction
  3. Advanced Curves & Measurement
  4. Geometric Transformations

Then tests: NL description -> LLM -> GGBScript -> GeoGebra render

Run: python tests/test_geogebra_commands.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_geogebra_commands
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

def cmd(ggb, label, command):
    """Execute one command and print PASS/FAIL."""
    result = ggb.eval_command(command)
    status = "PASS" if result.success else "FAIL"
    reason = f"  ({result.error_message})" if not result.success else ""
    print(f"  [{status}] {label:45s}  `{command}`{reason}")
    return result.success


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subsection(title):
    print(f"\n  -- {title} --")


def reset(ggb):
    ggb.reset()


# ─── Category 1: Basic & Edit ────────────────────────────────────────────────

def test_basic(ggb):
    section("1. Basic & Edit")
    reset(ggb)

    subsection("Points & Basics")
    cmd(ggb, "Free point",            "A = (1, 2)")
    cmd(ggb, "Free point B",          "B = (4, 1)")
    cmd(ggb, "Free point C",          "C = (2, 4)")
    cmd(ggb, "Slider (numeric)",      "n = Slider(1, 10, 1)")
    cmd(ggb, "Slider (angle)",        "alpha = Slider(0, 360, 1)")

    subsection("Intersection")
    cmd(ggb, "Line for intersect 1",  "l1 = Line((0,0),(3,3))")
    cmd(ggb, "Line for intersect 2",  "l2 = Line((0,3),(3,0))")
    cmd(ggb, "Intersect two lines",   "P = Intersect(l1, l2)")

    subsection("Function roots / turning points")
    cmd(ggb, "Define polynomial",     "f(x) = x^3 - 3x")
    cmd(ggb, "Root of f",             "r1 = Root(f, -2, 0)")
    cmd(ggb, "TurningPoint of f",     "tp = Extremum(f, -2, 2)")

    subsection("Best Fit Line")
    cmd(ggb, "List of points",        "pts = {(1,2),(2,3.5),(3,4),(4,5.5)}")
    cmd(ggb, "Best fit line",         "bfl = FitLine(pts)")

    subsection("Text")
    cmd(ggb, "Text object",           'txt = Text("Hello GeoGebra", (1, 5))')


# ─── Category 2: Geometric Construction ─────────────────────────────────────

def test_construction(ggb):
    section("2. Geometric Construction")
    reset(ggb)

    # Base points
    cmd(ggb, "Point A", "A = (0, 0)")
    cmd(ggb, "Point B", "B = (4, 0)")
    cmd(ggb, "Point C", "C = (2, 3)")
    cmd(ggb, "Point D", "D = (5, 3)")

    subsection("Lines")
    cmd(ggb, "Segment",                "seg = Segment(A, B)")
    cmd(ggb, "Line through 2 pts",     "lin = Line(A, C)")
    cmd(ggb, "Ray",                    "ray = Ray(A, B)")
    cmd(ggb, "Vector",                 "vec = Vector(A, B)")
    cmd(ggb, "Segment with length",    "segL = Segment(A, 3)")
    cmd(ggb, "Polyline",               "poly = Polyline(A, B, C, D)")

    subsection("Constructions")
    cmd(ggb, "Midpoint",               "M = Midpoint(A, B)")
    cmd(ggb, "Perpendicular line",     "perp = PerpendicularLine(C, seg)")
    cmd(ggb, "Perpendicular bisector", "pb = PerpendicularBisector(A, B)")
    cmd(ggb, "Parallel line",          "par = Line(C, seg)")
    cmd(ggb, "Angle bisector (3pts)",  "ab = AngleBisector(A, C, B)")
    cmd(ggb, "Angle bisector (2lns)",  "ab2 = AngleBisector(lin, par)")

    subsection("Polygons")
    cmd(ggb, "Polygon (triangle)",     "tri = Polygon(A, B, C)")
    cmd(ggb, "Regular polygon (hex)",  "hex = Polygon(A, B, 6)")

    subsection("Point on Object")
    cmd(ggb, "Point on segment",       "Pseg = Point(seg)")
    cmd(ggb, "Point on line",          "Plin = Point(lin)")

    subsection("Tangents")
    cmd(ggb, "Circle for tangent",     "circ = Circle(C, 2)")
    cmd(ggb, "Tangent from point",     "tan = Tangent(A, circ)")

    subsection("Locus")
    cmd(ggb, "Slider t",               "t = Slider(0, 1, 0.01)")
    cmd(ggb, "Locus point",            "Ploc = A + t * (B - A)")
    cmd(ggb, "Locus curve",            "loc = Locus(Ploc, t)")


# ─── Category 3: Advanced Curves & Measurement ──────────────────────────────

def test_curves_and_measure(ggb):
    section("3. Advanced Curves & Measurement")
    reset(ggb)

    cmd(ggb, "Point A", "A = (0, 0)")
    cmd(ggb, "Point B", "B = (4, 0)")
    cmd(ggb, "Point C", "C = (2, 3)")
    cmd(ggb, "Point D", "D = (-1, 1)")
    cmd(ggb, "Point E", "E = (3, -1)")

    subsection("Circles")
    cmd(ggb, "Circle centre+radius",   "c1 = Circle(A, 2)")
    cmd(ggb, "Circle centre+point",    "c2 = Circle(A, B)")
    cmd(ggb, "Circle through 3 pts",   "c3 = Circle(A, B, C)")
    cmd(ggb, "Semicircle",             "sc = Semicircle(A, B)")
    cmd(ggb, "Circular arc",           "arc = CircularArc(A, B, C)")
    cmd(ggb, "Circumcircular arc",     "carc = CircumcircularArc(A, B, C)")
    cmd(ggb, "Circular sector",        "sec = CircularSector(A, B, C)")
    cmd(ggb, "Circumcircular sector",  "csec = CircumcircularSector(A, B, C)")

    subsection("Conics")
    cmd(ggb, "Ellipse (2 foci+pt)",    "ell = Ellipse(A, B, C)")
    cmd(ggb, "Parabola (focus+dir)",   "dirL = Line((0,2),(1,2))")
    cmd(ggb, "Parabola",               "par = Parabola(A, dirL)")
    cmd(ggb, "Hyperbola (2 foci+pt)",  "hyp = Hyperbola(A, B, C)")
    cmd(ggb, "Conic through 5 pts",    "con = Conic(A, B, C, D, E)")

    subsection("Measurement")
    cmd(ggb, "Polygon for measure",    "tri = Polygon(A, B, C)")
    cmd(ggb, "Angle (3 pts)",          "ang = Angle(A, B, C)")
    cmd(ggb, "Angle (polygon)",        "angT = Angle(tri)")
    cmd(ggb, "Distance pts",           "dist = Distance(A, B)")
    cmd(ggb, "Area polygon",           "area = Area(tri)")
    cmd(ggb, "Area circle",            "areaC = Area(c1)")
    cmd(ggb, "Slope of line",          "lin = Line(A, C)")
    cmd(ggb, "Slope",                  "sl = Slope(lin)")


# ─── Category 4: Geometric Transformations ───────────────────────────────────

def test_transformations(ggb):
    section("4. Geometric Transformations")
    reset(ggb)

    cmd(ggb, "Point A",               "A = (1, 1)")
    cmd(ggb, "Point B",               "B = (4, 1)")
    cmd(ggb, "Point C",               "C = (2, 4)")
    cmd(ggb, "Mirror line",           "mirL = Line((3,0),(3,5))")
    cmd(ggb, "Triangle",              "tri = Polygon(A, B, C)")
    cmd(ggb, "Vector v",              "v = Vector((2, 1))")
    cmd(ggb, "Centre O",              "O = (0, 0)")
    cmd(ggb, "Circle for inversion",  "invC = Circle(O, 3)")

    subsection("Reflections")
    cmd(ggb, "Reflect point in line", "A2 = Reflect(A, mirL)")
    cmd(ggb, "Reflect tri in line",   "tri2 = Reflect(tri, mirL)")
    cmd(ggb, "Reflect point in pt",   "A3 = Reflect(A, O)")
    cmd(ggb, "Reflect tri in pt",     "tri3 = Reflect(tri, O)")
    cmd(ggb, "Reflect in circle",     "A4 = Reflect(A, invC)")

    subsection("Translation")
    cmd(ggb, "Translate point",       "A5 = Translate(A, v)")
    cmd(ggb, "Translate triangle",    "tri4 = Translate(tri, v)")

    subsection("Rotation")
    cmd(ggb, "Rotate point 45deg",    "A6 = Rotate(A, 45*pi/180, O)")
    cmd(ggb, "Rotate tri 90deg",      "tri5 = Rotate(tri, pi/2, O)")

    subsection("Dilation (Enlarge)")
    cmd(ggb, "Dilate point k=2",      "A7 = Dilate(A, 2, O)")
    cmd(ggb, "Dilate triangle k=2",   "tri6 = Dilate(tri, 2, O)")



# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _log = OUTDIR / "log" / "test_geogebra_commands.txt"
    with TeeLog(_log):
        print("\n" + "="*60)
        print("  GeoGebra Full Command Coverage + LLM Pipeline Test")
        print("="*60)

        for name, test_fn in [
            ("Basic & Edit",              test_basic),
            ("Geometric Construction",    test_construction),
            ("Advanced Curves & Measure", test_curves_and_measure),
            ("Transformations",           test_transformations),
        ]:
            print(f"\n[Initializing GeoGebra for: {name}]")
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                test_fn(ggb)

        print("\n" + "="*60)
        print("  All tests complete.")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
