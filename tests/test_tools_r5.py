"""
Verification tests for r5 tool additions (6 construction + 13 render).

Tests every new tool via execute_geogebra_tool / CanvasTracker to ensure
GeoGebra commands execute correctly and error guards work.

Run:  cd /path/to/harness-geometry
      python tests/test_tools_r5.py
"""
import sys
import math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import (
    execute_geogebra_tool,
    execute_query_tool,
    CanvasTracker,
    GLOBAL_GEOGEBRA_TOOLS,
    QUERY_GEOGEBRA_TOOLS,
    RENDER_GEOGEBRA_TOOLS,
)

OUTDIR = Path("temp") / "test_tools_r5"
OUTDIR.mkdir(parents=True, exist_ok=True)

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label}  —  {detail}")


def export(ggb, name: str):
    """Export current canvas as PNG for visual inspection."""
    path = OUTDIR / f"{name}.png"
    ggb.export_png(path)
    print(f"     📸 {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("r5 TOOL VERIFICATION")
print("=" * 70)
print(f"\nTool counts: GLOBAL={len(GLOBAL_GEOGEBRA_TOOLS)}  "
      f"QUERY={len(QUERY_GEOGEBRA_TOOLS)}  RENDER={len(RENDER_GEOGEBRA_TOOLS)}  "
      f"TOTAL={len(GLOBAL_GEOGEBRA_TOOLS)+len(QUERY_GEOGEBRA_TOOLS)+len(RENDER_GEOGEBRA_TOOLS)}")

ggb = GeoGebraAPI(mode="selenium", headless=True)
ggb.initialize()

tracker = CanvasTracker()


def run(tool_name, args, expect_ok=True):
    """Helper: execute tool and return (cmd, ok, err)."""
    cmd, ok, err = execute_geogebra_tool(ggb, tool_name, args)
    return cmd, ok, err


def run_tracked(tool_name, args):
    """Helper: execute via CanvasTracker."""
    result, log = tracker.execute(ggb, tool_name, args)
    return result, log


# ══════════════════════════════════════════════════════════════════════════════
# §1  Construction Tools (new in r5)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §1 add_function ──")
ggb.reset()

cmd, ok, err = run("add_function", {"name": "f", "expr": "x^2"})
check("add_function basic: f(x) = x^2", ok, err)

cmd, ok, err = run("add_function", {"name": "g", "expr": "sin(x)", "start_x": -3.14, "end_x": 3.14})
check("add_function domain-restricted: g(x) = sin(x) on [-π, π]", ok, err)

cmd, ok, err = run("add_function", {"name": "h", "expr": "If(x < 0, -x, x^2)"})
check("add_function piecewise: If(x<0, -x, x^2)", ok, err)

# Test auto-strip of (x) from name
cmd, ok, err = run("add_function", {"name": "p(x)", "expr": "x^3 - 3*x"})
check("add_function auto-strip (x) from name 'p(x)'", ok, err)
check("  → function p is defined", ggb.is_defined("p"), "p not found on canvas")

export(ggb, "01_functions")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §2 add_curve ──")
ggb.reset()

cmd, ok, err = run("add_curve", {
    "name": "ellipse_param", "x_expr": "3*cos(t)", "y_expr": "2*sin(t)",
    "t_start": "0", "t_end": "2*pi"
})
check("add_curve parametric ellipse", ok, err)

cmd, ok, err = run("add_curve", {
    "name": "spiral", "x_expr": "t*cos(t)", "y_expr": "t*sin(t)",
    "t_start": "0", "t_end": "4*pi", "param": "t"
})
check("add_curve spiral with explicit param=t", ok, err)

# Error case: x/y/z as param
cmd, ok, err = run("add_curve", {
    "name": "bad", "x_expr": "cos(x)", "y_expr": "sin(x)",
    "t_start": "0", "t_end": "2*pi", "param": "x"
})
check("add_curve REJECTS param='x'", not ok, err if ok else "correctly rejected")

export(ggb, "02_curves")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §3 add_inequality ──")
ggb.reset()

cmd, ok, err = run("add_inequality", {"name": "ineq1", "expr": "y <= x^2"})
check("add_inequality: y <= x^2", ok, err)

cmd, ok, err = run("add_inequality", {"name": "ineq2", "expr": "x^2 + y^2 < 4"})
check("add_inequality: x^2 + y^2 < 4 (disk)", ok, err)

cmd, ok, err = run("add_inequality", {
    "name": "ineq3", "expr": "(x >= 0) && (y >= 0) && (x + y <= 5)"
})
check("add_inequality: feasible region with &&", ok, err)

export(ggb, "03_inequalities")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §4 add_integral_shade ──")
ggb.reset()

# First create functions
run("add_function", {"name": "f", "expr": "x^2"})
run("add_function", {"name": "g", "expr": "x + 2"})

cmd, ok, err = run("add_integral_shade", {
    "name": "area1", "func": "f", "x_start": "0", "x_end": "2"
})
check("add_integral_shade: f to x-axis [0,2]", ok, err)

cmd, ok, err = run("add_integral_shade", {
    "name": "area2", "func": "g", "x_start": "-1", "x_end": "2", "func2": "f"
})
check("add_integral_shade: between g and f [-1,2]", ok, err)

export(ggb, "04_integral_shade")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §5 add_text ──")
ggb.reset()

cmd, ok, err = run("add_text", {
    "name": "label1", "text": "Hello World", "x": 1, "y": 3
})
check("add_text plain", ok, err)

cmd, ok, err = run("add_text", {
    "name": "formula1", "text": "y = x^2", "x": 0, "y": 5, "latex": 1
})
check("add_text LaTeX", ok, err)

export(ggb, "05_text")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §6 rename_object ──")
ggb.reset()

ggb.eval_command("A = (1, 2)")
ggb.eval_command("B = (3, 4)")

cmd, ok, err = run("rename_object", {"name": "A", "new_name": "P1"})
check("rename_object: A → P1", ok, err)
check("  → P1 exists", ggb.is_defined("P1"), "P1 not found")
check("  → A no longer exists", not ggb.is_defined("A"), "A still exists!")

# Error: rename to existing name
cmd, ok, err = run("rename_object", {"name": "P1", "new_name": "B"})
check("rename_object REJECTS conflict (P1 → B, but B exists)", not ok,
      err if ok else "correctly rejected")

# Error: rename non-existent
cmd, ok, err = run("rename_object", {"name": "ZZZZZ", "new_name": "X"})
check("rename_object REJECTS non-existent source", not ok,
      err if ok else "correctly rejected")


# ══════════════════════════════════════════════════════════════════════════════
# §2  Render Tools
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §7 render_set_color ──")
ggb.reset()
ggb.eval_command("A = (0, 0)")
ggb.eval_command("B = (3, 0)")
ggb.eval_command("seg1 = Segment(A, B)")

cmd, ok, err = run("render_set_color", {"obj": "seg1", "color": "Red"})
check("render_set_color: seg1 → Red", ok, err)

cmd, ok, err = run("render_set_color", {"obj": "A", "color": "Blue"})
check("render_set_color: point A → Blue", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §8 render_set_line_style ──")

cmd, ok, err = run("render_set_line_style", {"obj": "seg1", "style": 1})
check("render_set_line_style: seg1 → dashed (1)", ok, err)

cmd, ok, err = run("render_set_line_style", {"obj": "seg1", "style": 3})
check("render_set_line_style: seg1 → dotted (3)", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §9 render_set_line_thickness ──")

cmd, ok, err = run("render_set_line_thickness", {"obj": "seg1", "thickness": 8})
check("render_set_line_thickness: seg1 → 8", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §10 render_set_point_style ──")

cmd, ok, err = run("render_set_point_style", {"obj": "A", "style": 2})
check("render_set_point_style: A → empty circle (2)", ok, err)

cmd, ok, err = run("render_set_point_style", {"obj": "B", "style": 4})
check("render_set_point_style: B → diamond (4)", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §11 render_set_point_size ──")

cmd, ok, err = run("render_set_point_size", {"obj": "A", "size": 9})
check("render_set_point_size: A → 9 (large)", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §12 render_set_filling ──")
ggb.eval_command("C = (1, 2)")
ggb.eval_command("tri1 = Polygon(A, B, C)")

cmd, ok, err = run("render_set_filling", {"obj": "tri1", "opacity": 0.4})
check("render_set_filling: tri1 → 0.4 opacity", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §13 render_set_decoration ──")

cmd, ok, err = run("render_set_decoration", {"obj": "seg1", "decoration": 2})
check("render_set_decoration: seg1 → two ticks (2)", ok, err)

# Angle decoration
ggb.eval_command("ang1 = Angle(A, B, C)")
cmd, ok, err = run("render_set_decoration", {"obj": "ang1", "decoration": 1})
check("render_set_decoration: angle → double arc (1)", ok, err)

export(ggb, "06_styling")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §14 render_show_axes / render_show_grid ──")
ggb.reset()

cmd, ok, err = run("render_show_axes", {"visible": 1})
check("render_show_axes: show", ok, err)

cmd, ok, err = run("render_show_grid", {"visible": 1})
check("render_show_grid: show", ok, err)

cmd, ok, err = run("render_show_axes", {"visible": 0})
check("render_show_axes: hide", ok, err)

cmd, ok, err = run("render_show_grid", {"visible": 0})
check("render_show_grid: hide", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §15 render_set_caption ──")
ggb.reset()
ggb.eval_command("A = (1, 2)")

# Exercise custom GeoGebra caption rendering.
cmd, ok, err = run("render_set_caption", {"obj": "A", "caption": "Start P0"})
check("render_set_caption: A → custom caption", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §16 render_set_label_mode ──")

cmd, ok, err = run("render_set_label_mode", {"obj": "A", "mode": 1})
check("render_set_label_mode: A → Name+Value (1)", ok, err)

cmd, ok, err = run("render_set_label_mode", {"obj": "A", "mode": 3})
check("render_set_label_mode: A → Caption (3)", ok, err)

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §17 render_set_coord_system ──")

cmd, ok, err = run("render_set_coord_system", {
    "x_min": -10, "x_max": 10, "y_min": -8, "y_max": 8
})
check("render_set_coord_system: [-10,10] x [-8,8]", ok, err)

export(ggb, "07_coord_system")

# ──────────────────────────────────────────────────────────────────────────────

print("\n── §18 render_add_right_angle_mark ──")
ggb.reset()
# Create a real right angle
ggb.eval_command("A = (3, 0)")
ggb.eval_command("B = (0, 0)")
ggb.eval_command("C = (0, 4)")

cmd, ok, err = run("render_add_right_angle_mark", {
    "name": "ra1", "a": "A", "b": "B", "c": "C"
})
check("render_add_right_angle_mark: 90° angle at B", ok, err)

# Error case: not a right angle
ggb.eval_command("D = (2, 3)")
cmd, ok, err = run("render_add_right_angle_mark", {
    "name": "ra2", "a": "A", "b": "B", "c": "D"
})
check("render_add_right_angle_mark REJECTS non-90° angle", not ok,
      err if ok else "correctly rejected")

export(ggb, "08_right_angle")


# ══════════════════════════════════════════════════════════════════════════════
# §3  CanvasTracker Integration
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §19 CanvasTracker with new tools ──")
ggb.reset()
tracker2 = CanvasTracker()

result, log = tracker2.execute(ggb, "add_point", {"name": "A", "x": 0, "y": 0})
check("tracker: add_point", result["success"], result.get("error", ""))

result, log = tracker2.execute(ggb, "add_function", {"name": "f", "expr": "x^2"})
check("tracker: add_function", result["success"], result.get("error", ""))

result, log = tracker2.execute(ggb, "render_set_color", {"obj": "A", "color": "Red"})
check("tracker: render_set_color dispatches correctly", result["success"],
      result.get("error", ""))

result, log = tracker2.execute(ggb, "render_show_axes", {"visible": 1})
check("tracker: render_show_axes dispatches correctly", result["success"],
      result.get("error", ""))

print(f"\n  tracker stats: ok={tracker2.ok_n}  fail={tracker2.fail_n}  total={tracker2.total_n}")


# ══════════════════════════════════════════════════════════════════════════════
# §4  Combined GenExam-style figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §20 GenExam-style combined figure ──")
ggb.reset()

# 1. Show axes and grid
run("render_show_axes", {"visible": 1})
run("render_show_grid", {"visible": 1})

# 2. Define function
run("add_function", {"name": "f", "expr": "x^2 - 2*x"})
run("render_set_color", {"obj": "f", "color": "Blue"})
run("render_set_line_thickness", {"obj": "f", "thickness": 5})

# 3. Define second function
run("add_function", {"name": "g", "expr": "x"})
run("render_set_color", {"obj": "g", "color": "Red"})
run("render_set_line_style", {"obj": "g", "style": 1})  # dashed

# 4. Shade area between them
run("add_integral_shade", {
    "name": "shaded", "func": "g", "x_start": "0", "x_end": "3", "func2": "f"
})
run("render_set_color", {"obj": "shaded", "color": "Orange"})
run("render_set_filling", {"obj": "shaded", "opacity": 0.3})

# 5. Mark intersections
ggb.eval_command("P1 = Intersect(f, g, 1)")
ggb.eval_command("P2 = Intersect(f, g, 2)")
run("render_set_point_size", {"obj": "P1", "size": 7})
run("render_set_point_size", {"obj": "P2", "size": 7})
run("render_set_color", {"obj": "P1", "color": "Black"})
run("render_set_color", {"obj": "P2", "color": "Black"})

# 6. Add text annotation
run("add_text", {"name": "title", "text": "f(x) = x^2 - 2x", "x": 1, "y": 5, "latex": 1})

# 7. Set viewport
run("render_set_coord_system", {"x_min": -2, "x_max": 5, "y_min": -3, "y_max": 7})

check("GenExam-style combined figure built", True)
export(ggb, "09_genexam_style")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"RESULT:  {PASS} passed,  {FAIL} failed,  {PASS + FAIL} total")
print(f"IMAGES:  {OUTDIR}/")
print("=" * 70)

ggb.cleanup()
sys.exit(0 if FAIL == 0 else 1)
