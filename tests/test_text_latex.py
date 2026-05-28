"""
Test LaTeX rendering in GeoGebra add_text tool.

Verifies that various LaTeX formulas render correctly (or fail gracefully)
in GeoGebra text objects, and exports PNGs for visual inspection.

Run:  cd /path/to/harness-geometry
      python tests/test_text_latex.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import execute_geogebra_tool

OUTDIR = Path("temp") / "test_text_latex"
OUTDIR.mkdir(parents=True, exist_ok=True)

PASS = 0
FAIL = 0


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  OK  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  --  {detail}")


def run(tool_name, args):
    cmd, ok, err = execute_geogebra_tool(ggb, tool_name, args)
    return cmd, ok, err


def export(name):
    path = OUTDIR / f"{name}.png"
    ggb.export_png(path)
    print(f"     -> {path}")


# ── Setup ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("TEXT & LATEX RENDERING TESTS")
print("=" * 70)

ggb = GeoGebraAPI(mode="selenium", headless=True)
ggb.initialize()

# ── Test 1: Plain text ───────────────────────────────────────────────────────

print("\n-- Test 1: Plain text --")
ggb.reset()

cmd, ok, err = run("add_text", {
    "name": "t1", "text": "Hello World", "x": -3, "y": 4
})
check("plain text", ok, err)

cmd, ok, err = run("add_text", {
    "name": "t2", "text": "AB = 5, CD = 3", "x": -3, "y": 3
})
check("plain text with equals", ok, err)

export("01_plain_text")

# ── Test 2: LaTeX basic formulas ─────────────────────────────────────────────

print("\n-- Test 2: LaTeX basic formulas --")
ggb.reset()

# Simple power
cmd, ok, err = run("add_text", {
    "name": "f1", "text": "y = x^2", "x": -4, "y": 5, "latex": 1
})
check("latex: y = x^2", ok, err)

# Fraction with \\frac
cmd, ok, err = run("add_text", {
    "name": "f2", "text": "y = \\frac{1}{2}x + 3", "x": -4, "y": 3, "latex": 1
})
check("latex: \\frac{1}{2}", ok, err)

# Square root
cmd, ok, err = run("add_text", {
    "name": "f3", "text": "c = \\sqrt{a^2 + b^2}", "x": -4, "y": 1, "latex": 1
})
check("latex: \\sqrt{}", ok, err)

# Greek letters
cmd, ok, err = run("add_text", {
    "name": "f4", "text": "\\alpha + \\beta = \\pi", "x": -4, "y": -1, "latex": 1
})
check("latex: greek letters", ok, err)

export("02_latex_basic")

# ── Test 3: LaTeX advanced (the ones that broke in eval) ─────────────────────

print("\n-- Test 3: LaTeX advanced (known problematic) --")
ggb.reset()

# \begin{cases} — piecewise (Mathematics_139 failure)
cmd, ok, err = run("add_text", {
    "name": "pw1", "text": "f(x) = \\begin{cases} 1 & x \\geq 0 \\\\ -1 & x < 0 \\end{cases}",
    "x": -4, "y": 5, "latex": 1
})
check("latex: \\begin{cases} piecewise", ok, err)

# \text{} inside formula (Mathematics_140 failure)
cmd, ok, err = run("add_text", {
    "name": "tx1", "text": "\\text{Concave Up}", "x": -4, "y": 3, "latex": 1
})
check("latex: \\text{} command", ok, err)

# Complex fraction
cmd, ok, err = run("add_text", {
    "name": "cf1", "text": "f(x) = \\frac{x^2 - 1}{x + 1}",
    "x": -4, "y": 1, "latex": 1
})
check("latex: complex fraction", ok, err)

# Subscript + superscript
cmd, ok, err = run("add_text", {
    "name": "ss1", "text": "a_{n+1} = a_n^2 + 1",
    "x": -4, "y": -1, "latex": 1
})
check("latex: subscript + superscript", ok, err)

export("03_latex_advanced")

# ── Test 4: Plain text alternatives (recommended for GGB) ───────────────────

print("\n-- Test 4: Plain text alternatives (no LaTeX) --")
ggb.reset()

# Piecewise without LaTeX
cmd, ok, err = run("add_text", {
    "name": "p1", "text": "f(x) = { 1 if x >= 0, -1 if x < 0 }",
    "x": -4, "y": 5
})
check("plain: piecewise alternative", ok, err)

# Fraction as 1/2
cmd, ok, err = run("add_text", {
    "name": "p2", "text": "y = (1/2)x + 3", "x": -4, "y": 3
})
check("plain: fraction as 1/2", ok, err)

# Unicode alternatives
cmd, ok, err = run("add_text", {
    "name": "p3", "text": "c = sqrt(a^2 + b^2)", "x": -4, "y": 1
})
check("plain: sqrt()", ok, err)

export("04_plain_alternatives")

# ── Test 5: GeoGebra native LaTeX syntax ─────────────────────────────────────

print("\n-- Test 5: GeoGebra FormulaText approach --")
ggb.reset()

# Try using FormulaText with a defined function
ggb.eval_command("f(x) = x^2 / 2 + 1")
res = ggb.eval_command('ft = FormulaText(f)')
check("FormulaText(f) created", ggb.is_defined("ft"), str(res))

# Try Text with substitution for object value
ggb.eval_command("A = (3, 4)")
res = ggb.eval_command('tA = Text("A = " + A)')
check("Text with object substitution", ggb.is_defined("tA"), str(res))

export("05_formula_text")

# ── Cleanup ──────────────────────────────────────────────────────────────────

ggb.cleanup()

print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"PNGs saved to {OUTDIR}/")
print("=" * 70)

sys.exit(1 if FAIL > 0 else 0)
