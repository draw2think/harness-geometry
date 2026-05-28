"""
GeoGebra manual construction draw tests.

Builds representative figures from each tool category,
applies styling via the JavaScript API, fits the viewport,
and exports PNG to temp/.

Run: python tests/test_geogebra_draw.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from geogebra_render_common import (
    fit_view_square,
    set_color,
    set_thickness,
    set_visible,
    set_label_visible,
)

OUTDIR = Path("temp") / Path(__file__).stem   # temp/test_geogebra_construct
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


# ── Shared style + viewport helpers ─────────────────────────────────────────

def fit_view(ggb, padding=1.5):
    """Compatibility wrapper to shared square viewport fitter."""
    fit_view_square(ggb, padding=padding)


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


# ── Figure builders ──────────────────────────────────────────────────────────

def draw_isosceles_perp_bisector(ggb):
    """
    Isosceles triangle ABC (AB=AC=5, BC=6).
    Perpendicular bisector of BC meets BC at M.
    Show AM perp BC, label 90° angle.
    """
    ggb.reset()
    # Construct by intersection of two circles (constraint-based, not coords)
    cmd(ggb, "B = (0, 0)")
    cmd(ggb, "C = (6, 0)")
    cmd(ggb, "cB = Circle(B, 5)")
    cmd(ggb, "cC = Circle(C, 5)")
    cmd(ggb, "A = Intersect(cB, cC, 1)")   # upper intersection
    cmd(ggb, "tri = Polygon(A, B, C)")
    cmd(ggb, "pb = PerpendicularBisector(B, C)")
    cmd(ggb, "M = Intersect(pb, Segment(B, C))")
    cmd(ggb, "am = Segment(A, M)")
    cmd(ggb, "ang = Angle(B, M, A)")

    # Styling via JS API
    set_visible(ggb, "cB", False)
    set_visible(ggb, "cC", False)
    set_color(ggb, "tri", 100, 149, 237)   # cornflower blue fill
    set_color(ggb, "pb",  220, 50,  50)    # red bisector
    set_thickness(ggb, "pb", 3)
    set_color(ggb, "am",  50,  180, 50)    # green median
    set_thickness(ggb, "am", 3)

    fit_view(ggb, padding=1.0)
    save(ggb, "01_isosceles_perp_bisector")


def draw_inscribed_angle_theorem(ggb):
    """
    Circle O. A, B, C on circle.
    Inscribed angle ABC vs central angle AOC.
    """
    ggb.reset()
    cmd(ggb, "O = (0, 0)")
    cmd(ggb, "A = (3, 0)")
    cmd(ggb, "circ = Circle(O, A)")
    cmd(ggb, "B = Point(circ)")          # free point on circle
    cmd(ggb, "SetCoords(B, -1.5, -2.6)")
    cmd(ggb, "C = Point(circ)")
    cmd(ggb, "SetCoords(C, -3, 0)")
    cmd(ggb, "inscribed = Angle(A, B, C)")
    cmd(ggb, "central = Angle(A, O, C)")
    cmd(ggb, "sBA = Segment(B, A)")
    cmd(ggb, "sBC = Segment(B, C)")
    cmd(ggb, "sOA = Segment(O, A)")
    cmd(ggb, "sOC = Segment(O, C)")

    set_color(ggb, "inscribed", 220, 50, 50)
    set_color(ggb, "central",   50, 150, 220)
    set_color(ggb, "circ", 80, 80, 80)
    set_thickness(ggb, "circ", 2)

    fit_view(ggb, padding=1.5)
    save(ggb, "02_inscribed_angle_theorem")


def draw_tangent_radius_perp(ggb):
    """
    Circle O radius 3. External point P=(6,0).
    Two tangents from P touching circle at T1, T2.
    Show OT1 perp PT1.
    """
    ggb.reset()
    cmd(ggb, "O = (0, 0)")
    cmd(ggb, "P = (6, 0)")
    cmd(ggb, "circ = Circle(O, 3)")

    # Robust tangent-point construction:
    # In right triangle OPT, OP=6 and OT=3 => PT=sqrt(6^2-3^2)=sqrt(27).
    # Tangency points are intersections of circle(O,3) and circle(P,sqrt(27)).
    cmd(ggb, "cP = Circle(P, sqrt(27))")
    cmd(ggb, "T1 = Intersect(circ, cP, 1)")
    cmd(ggb, "T2 = Intersect(circ, cP, 2)")
    cmd(ggb, "l1 = Line(P, T1)")
    cmd(ggb, "l2 = Line(P, T2)")
    cmd(ggb, "r1 = Segment(O, T1)")
    cmd(ggb, "t1 = Segment(P, T1)")
    cmd(ggb, "r2 = Segment(O, T2)")
    cmd(ggb, "t2 = Segment(P, T2)")
    # Point order is chosen to prefer the interior right angle (90°) display.
    cmd(ggb, "ang1 = Angle(O, T1, P)")

    set_color(ggb, "circ", 60, 60, 200)
    set_thickness(ggb, "circ", 2)
    set_color(ggb, "l1", 200, 60, 60)
    set_color(ggb, "l2", 200, 60, 60)
    set_thickness(ggb, "l1", 2)
    set_thickness(ggb, "l2", 2)
    set_color(ggb, "r1",  50, 180, 50)
    set_color(ggb, "r2",  50, 180, 50)
    set_color(ggb, "ang1", 220, 140, 0)
    set_visible(ggb, "cP", False)

    fit_view(ggb, padding=1.0)
    save(ggb, "03_tangent_radius_perp")


def draw_midsegment_theorem(ggb):
    """
    Triangle ABC. D=midpoint(AB), E=midpoint(AC).
    DE parallel to BC and DE=BC/2.
    """
    ggb.reset()
    cmd(ggb, "A = (0, 4)")
    cmd(ggb, "B = (-3, 0)")
    cmd(ggb, "C = (3, 0)")
    cmd(ggb, "tri = Polygon(A, B, C)")
    cmd(ggb, "D = Midpoint(A, B)")
    cmd(ggb, "E = Midpoint(A, C)")
    cmd(ggb, "de = Segment(D, E)")
    cmd(ggb, "bc = Segment(B, C)")
    cmd(ggb, "lenDE = Distance(D, E)")
    cmd(ggb, "lenBC = Distance(B, C)")

    set_color(ggb, "tri", 200, 220, 255)
    set_color(ggb, "de",  220, 50, 50)
    set_thickness(ggb, "de", 4)
    set_color(ggb, "bc",  50, 150, 50)
    set_thickness(ggb, "bc", 4)

    fit_view(ggb, padding=1.0)
    save(ggb, "04_midsegment_theorem")


def draw_transformations(ggb):
    """
    Triangle + reflection in line + rotation + dilation.
    """
    ggb.reset()
    cmd(ggb, "A = (1, 1)")
    cmd(ggb, "B = (3, 1)")
    cmd(ggb, "C = (2, 3)")
    cmd(ggb, "tri = Polygon(A, B, C)")
    cmd(ggb, "mirL = Line((4,0),(4,5))")
    cmd(ggb, "triRef = Reflect(tri, mirL)")
    cmd(ggb, "O = (0, 0)")
    cmd(ggb, "triRot = Rotate(tri, pi/2, O)")
    cmd(ggb, "triDil = Dilate(tri, 2, O)")

    set_color(ggb, "tri",    70,  130, 220)
    set_color(ggb, "triRef", 220, 80,  80)
    set_color(ggb, "triRot", 80,  200, 80)
    set_color(ggb, "triDil", 200, 160, 0)
    set_visible(ggb, "mirL", False)

    fit_view(ggb, padding=1.5)
    save(ggb, "05_transformations")


def draw_conic_sections(ggb):
    """
    Ellipse, hyperbola, parabola in one figure.
    """
    ggb.reset()
    cmd(ggb, "F1 = (-2, 0)")
    cmd(ggb, "F2 = (2, 0)")
    cmd(ggb, "Pe = (0, 2)")
    cmd(ggb, "ell = Ellipse(F1, F2, Pe)")
    cmd(ggb, "hyp = Hyperbola(F1, F2, Pe)")
    cmd(ggb, "Fp = (0, 2)")
    cmd(ggb, "dirL = Line((-5,-2),(5,-2))")
    cmd(ggb, "par = Parabola(Fp, dirL)")

    set_color(ggb, "ell", 50, 50, 220)
    set_thickness(ggb, "ell", 2)
    set_color(ggb, "hyp", 200, 50, 50)
    set_thickness(ggb, "hyp", 2)
    set_color(ggb, "par", 50, 180, 50)
    set_thickness(ggb, "par", 2)
    set_visible(ggb, "dirL", False)

    fit_view(ggb, padding=1.5)
    save(ggb, "06_conic_sections")


# ── Main ─────────────────────────────────────────────────────────────────────

FIGURES = [
    ("Isosceles triangle + perp bisector",   draw_isosceles_perp_bisector, "01_isosceles_perp_bisector"),
    ("Inscribed angle theorem",              draw_inscribed_angle_theorem,  "02_inscribed_angle_theorem"),
    ("Tangent-radius perpendicularity",      draw_tangent_radius_perp,      "03_tangent_radius_perp"),
    ("Midsegment theorem",                   draw_midsegment_theorem,       "04_midsegment_theorem"),
    ("Transformations (reflect/rotate/dil)", draw_transformations,          "05_transformations"),
    ("Conic sections",                       draw_conic_sections,           "06_conic_sections"),
]


def main():
    print("\n" + "="*60)
    print("  GeoGebra Manual Construction Draw Tests")
    print("="*60)

    for name, fn, log_name in FIGURES:
        print(f"\n[{name}]")
        with TeeLog(OUTDIR / "log" / f"{log_name}.txt"):
            with GeoGebraAPI(mode="selenium", headless=True) as ggb:
                fn(ggb)

    print("\n" + "="*60)
    print(f"  Done. Figures saved to {OUTDIR / 'fig'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
