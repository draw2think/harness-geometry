"""
GeoGebra Algebra / CAS Command Test.

Tests algebraic and equation-solving commands to verify which work in our
GeoGebra Classic 5 (Graphing) applet and which require the CAS engine.

Background: Solve/NSolve/Solutions are CAS commands and return '?' in the
Graphing applet. This file documents the exact behavior and tests alternatives
(Root, Substitute, numeric workarounds).

Run: python tests/test_geogebra_algebra.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI

OUTDIR = Path("temp") / Path(__file__).stem
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def subsection(title):
    print(f"\n  -- {title} --")


def test_cmd(ggb, label, command, obj_name, *,
             expect_value=None, expect_str=None, expect_fail=False):
    """
    Execute a command, read back getValue / getValueString, print PASS/FAIL.

    expect_value : expected float from getValue() (tolerance 1e-4)
    expect_str   : expected substring in getValueString()
    expect_fail  : True if we expect the command itself to fail
    """
    result = ggb.eval_command(command)

    if expect_fail:
        status = "PASS" if not result.success else "FAIL"
        err = result.error_message or "(no error msg)"
        print(f"  [{status}] {label:55s}  expected fail → {err}")
        return

    if not result.success:
        print(f"  [FAIL] {label:55s}  command error: {result.error_message}")
        return

    value     = ggb.get_value(obj_name)
    value_str = ggb.get_value_string(obj_name)
    obj_type  = ggb._driver.execute_script(
        f'return ggbApplet.getObjectType("{obj_name}")')

    if expect_value is not None:
        ok = value is not None and abs(float(value) - float(expect_value)) < 1e-4
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:55s}  "
              f"type={obj_type}  val={value}  str={value_str}"
              f"  (expected {expect_value})")
    elif expect_str is not None:
        ok = value_str is not None and expect_str in value_str
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:55s}  "
              f"type={obj_type}  str={value_str}"
              f"  (expected *{expect_str}*)")
    else:
        # Just report what we got
        print(f"  [INFO] {label:55s}  "
              f"type={obj_type}  val={value}  str={value_str}")


def test_x_coord(ggb, label, command, obj_name, expect_x):
    """For Root() which returns a point — check x coordinate."""
    result = ggb.eval_command(command)
    if not result.success:
        print(f"  [FAIL] {label:55s}  command error: {result.error_message}")
        return

    x = ggb._driver.execute_script(
        f'return ggbApplet.getXcoord("{obj_name}")')
    ok = x is not None and abs(float(x) - float(expect_x)) < 1e-3
    status = "PASS" if ok else "FAIL"
    vs = ggb.get_value_string(obj_name)
    print(f"  [{status}] {label:55s}  x={x}  str={vs}  (expected x={expect_x})")


# ─── Test 1: Solve / NSolve / Solutions (CAS commands — expect '?') ─────────

def test_cas_commands(ggb):
    """Verify that CAS-only commands return '?' in Graphing applet."""
    section("CAS Commands (Solve / NSolve / Solutions) — expect '?'")
    ggb.reset()

    subsection("Solve — linear")
    test_cmd(ggb, "Solve(2*x + 3 = 9, x)",
             "s1 = Solve(2*x + 3 = 9, x)", "s1",
             expect_str="?")

    subsection("Solve — quadratic")
    test_cmd(ggb, "Solve(x^2 = 36, x)",
             "s2 = Solve(x^2 = 36, x)", "s2",
             expect_str="?")

    subsection("Solve — fraction / proportion")
    test_cmd(ggb, "Solve(2*x/9 = 8/x, x)",
             "s3 = Solve(2*x/9 = 8/x, x)", "s3",
             expect_str="?")

    subsection("Solve — moved to zero form")
    test_cmd(ggb, "Solve(2*x/9 - 8/x = 0, x)",
             "s4 = Solve(2*x/9 - 8/x = 0, x)", "s4",
             expect_str="?")

    subsection("Solve — set notation {eq}, {var}")
    test_cmd(ggb, "Solve({2*x + 3 = 9}, {x})",
             "s5 = Solve({2*x + 3 = 9}, {x})", "s5",
             expect_str="?")

    subsection("NSolve — linear")
    test_cmd(ggb, "NSolve(2*x + 3 = 9, x)",
             "ns1 = NSolve(2*x + 3 = 9, x)", "ns1",
             expect_str="?")

    subsection("NSolve — trig")
    test_cmd(ggb, "NSolve(sin(x) = 0.5, x)",
             "ns2 = NSolve(sin(x) = 0.5, x)", "ns2",
             expect_str="?")

    subsection("Solutions — quadratic")
    test_cmd(ggb, "Solutions(x^2 - 36 = 0, x)",
             "sol1 = Solutions(x^2 - 36 = 0, x)", "sol1",
             expect_str="?")

    subsection("evalCommandCAS")
    cas_result = ggb._driver.execute_script(
        'return ggbApplet.evalCommandCAS("Solve(2*x + 3 = 9, x)")')
    ok = cas_result in (None, "?", "")
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'evalCommandCAS(Solve(2*x+3=9, x))':55s}  → {cas_result!r}"
          f"  (expected '?')")

    subsection("Element(Solve(...), 1) — extracting from broken list")
    ggb.eval_command("sl = Solve(x^2 = 36, x)")
    test_cmd(ggb, "Element(sl, 1)",
             "e1 = Element(sl, 1)", "e1",
             expect_str="?")


# ─── Test 2: Root() — numeric root finder (works in Graphing) ───────────────

def test_root_command(ggb):
    """Root(f, xStart, xEnd) works without CAS — it's a numeric root finder."""
    section("Root() — Numeric Root Finder (Graphing engine)")
    ggb.reset()

    subsection("Linear: 2x + 3 - 9 = 0 → x = 3")
    ggb.eval_command("f1(x) = 2*x + 3 - 9")
    test_x_coord(ggb, "Root(f1, -100, 100)",
                 "r1 = Root(f1, -100, 100)", "r1", 3.0)

    subsection("Quadratic positive root: x^2 - 36 = 0 → x = 6")
    ggb.eval_command("f2(x) = x^2 - 36")
    test_x_coord(ggb, "Root(f2, 0, 100)  positive root",
                 "r2 = Root(f2, 0, 100)", "r2", 6.0)

    subsection("Quadratic negative root: x^2 - 36 = 0 → x = -6")
    test_x_coord(ggb, "Root(f2, -100, 0)  negative root",
                 "r2n = Root(f2, -100, 0)", "r2n", -6.0)

    subsection("Fraction: 2x/9 - 8/x = 0 → x = 6")
    ggb.eval_command("f3(x) = 2*x/9 - 8/x")
    test_x_coord(ggb, "Root(f3, 0.1, 100)  positive root",
                 "r3 = Root(f3, 0.1, 100)", "r3", 6.0)

    subsection("Trig: sin(x) - 0.5 = 0 → x ≈ π/6 (RADIANS — applet default)")
    ggb.eval_command("f4(x) = sin(x) - 0.5")
    # NOTE: GeoGebra Classic 5 web applet uses RADIANS for trig functions.
    #       sin(30) = sin(30 rad) ≈ -0.988, NOT sin(30°) = 0.5.
    #       To get degrees, use sin(30°) with the degree symbol.
    import math
    test_x_coord(ggb, "Root(f4, 0, 1)  → π/6 ≈ 0.5236 (radians)",
                 "r4 = Root(f4, 0, 1)", "r4", math.pi / 6)

    subsection("Geometric mean: x^2 - 6*8 = 0 → x ≈ 6.928")
    ggb.eval_command("f5(x) = x^2 - 48")
    test_x_coord(ggb, "Root(f5, 0, 100)  → sqrt(48)",
                 "r5 = Root(f5, 0, 100)", "r5", 6.9282)

    subsection("Proportion: 3/x = x/12 → x^2 = 36 → x = 6")
    ggb.eval_command("f6(x) = 3/x - x/12")
    test_x_coord(ggb, "Root(f6, 0.1, 100)  → 6",
                 "r6 = Root(f6, 0.1, 100)", "r6", 6.0)


# ─── Test 3: Substitute — evaluating expressions ────────────────────────────

def test_substitute(ggb):
    """Substitute(expr, var, val) — CAS command, fails in Graphing applet."""
    section("Substitute() — CAS command (expect FAIL)")
    ggb.reset()

    subsection("Substitute into polynomial")
    test_cmd(ggb, "Substitute(x^2 + 3*x, x, 4) → expect fail",
             "sub1 = Substitute(x^2 + 3*x, x, 4)", "sub1",
             expect_fail=True)

    subsection("Alternative: use function eval instead")
    ggb.eval_command("fsub(x) = x^2 + 3*x")
    test_cmd(ggb, "fsub(4) = 16 + 12 = 28",
             "sub1b = fsub(4)", "sub1b",
             expect_value=28.0)


# ─── Test 4: Function definition + evaluation ───────────────────────────────

def test_function_eval(ggb):
    """Define f(x), then evaluate f(val) — always works in Graphing."""
    section("Function Definition + Evaluation")
    ggb.reset()

    subsection("Define and evaluate polynomial")
    ggb.eval_command("g(x) = x^2 + 3*x - 10")
    test_cmd(ggb, "g(2) = 4 + 6 - 10 = 0",
             "gv1 = g(2)", "gv1", expect_value=0.0)
    test_cmd(ggb, "g(5) = 25 + 15 - 10 = 30",
             "gv2 = g(5)", "gv2", expect_value=30.0)

    subsection("Trig — applet uses RADIANS, not degrees")
    ggb.eval_command("h(x) = 2*sin(x)")
    import math
    test_cmd(ggb, "h(30) = 2*sin(30 rad) ≈ -1.976  (NOT 1.0)",
             "hv1 = h(30)", "hv1", expect_value=2 * math.sin(30))
    # Use pi/2 for 90°
    test_cmd(ggb, "h(pi/2) = 2*sin(π/2) = 2.0",
             "hv2 = h(pi/2)", "hv2", expect_value=2.0)
    # Use degree symbol for degrees
    test_cmd(ggb, "sin(30°) = 0.5  (degree symbol forces degrees)",
             "hv3 = sin(30°)", "hv3", expect_value=0.5)


# ─── Test 5: Simplify / Expand / Factor (CAS? or Graphing?) ─────────────────

def test_simplify_expand(ggb):
    """Test whether Simplify/Expand/Factor work in Graphing mode."""
    section("Simplify / Expand / Factor")
    ggb.reset()

    subsection("Simplify")
    test_cmd(ggb, "Simplify(x + x + x)",
             "sim1 = Simplify(x + x + x)", "sim1")
    test_cmd(ggb, "Simplify(x^2 / x)",
             "sim2 = Simplify(x^2 / x)", "sim2")

    subsection("Expand")
    test_cmd(ggb, "Expand((x + 1)^2)",
             "exp1 = Expand((x + 1)^2)", "exp1")
    test_cmd(ggb, "Expand((x + 2)*(x - 3))",
             "exp2 = Expand((x + 2)*(x - 3))", "exp2")

    subsection("Factor")
    test_cmd(ggb, "Factor(x^2 - 9)",
             "fac1 = Factor(x^2 - 9)", "fac1")
    test_cmd(ggb, "Factor(x^2 + 5*x + 6)",
             "fac2 = Factor(x^2 + 5*x + 6)", "fac2")


# ─── Test 6: Numeric computation commands ────────────────────────────────────

def test_numeric(ggb):
    """Direct numeric computation — always works."""
    section("Numeric Computation")
    ggb.reset()

    subsection("Direct arithmetic via named variable")
    test_cmd(ggb, "a = sqrt(48) → 6.928",
             "a = sqrt(48)", "a", expect_value=6.9282)
    test_cmd(ggb, "b = 3 * 4 + 5 → 17",
             "b = 3 * 4 + 5", "b", expect_value=17.0)

    subsection("Trig uses RADIANS — sin(30) = sin(30 rad) ≈ -0.988")
    import math
    test_cmd(ggb, "sin(30) = sin(30 rad) ≈ -0.988",
             "c = sin(30)", "c", expect_value=math.sin(30))
    test_cmd(ggb, "atan(1) = π/4 ≈ 0.785 (radians)",
             "d = atan(1)", "d", expect_value=math.pi / 4)

    subsection("abs (works) / max, min (CAS — fail)")
    test_cmd(ggb, "abs(-7) → 7",
             "e = abs(-7)", "e", expect_value=7.0)
    test_cmd(ggb, "max(3, 8) → expect fail (CAS cmd)",
             "f = max(3, 8)", "f", expect_fail=True)
    test_cmd(ggb, "min(3, 8) → expect fail (CAS cmd)",
             "g = min(3, 8)", "g", expect_fail=True)
    test_cmd(ggb, "Max({3, 8}) → list form works?",
             "f2 = Max({3, 8})", "f2", expect_value=8.0)
    test_cmd(ggb, "Min({3, 8}) → list form works?",
             "g2 = Min({3, 8})", "g2", expect_value=3.0)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*64)
    print("  GeoGebra Algebra / CAS Command Test")
    print("  GeoGebra applet: Classic 5 (CAS loads async, ~2s after applet ready)")
    print("="*64)

    pass_count = 0
    fail_count = 0

    tests = [
        ("CAS Commands (expect ?)",       test_cas_commands),
        ("Root (numeric root finder)",     test_root_command),
        ("Substitute",                     test_substitute),
        ("Function eval",                  test_function_eval),
        ("Simplify / Expand / Factor",     test_simplify_expand),
        ("Numeric computation",            test_numeric),
    ]

    with GeoGebraAPI(mode="selenium", headless=True) as ggb:
        for name, fn in tests:
            fn(ggb)

    print("\n" + "="*64)
    print("  Algebra tests complete.")
    print("="*64)

    print("""
Summary of GeoGebra Classic 5 (Graphing applet) algebra capabilities:

  CAS commands (BROKEN — CAS engine not loaded):
    ❌ Solve(), NSolve(), Solutions()   — always returns '?'
    ❌ evalCommandCAS()                 — always returns '?'
    ❌ Element(Solve(...), n)           — broken list, returns '?'
    ❌ Substitute(expr, var, val)       — command fails
    ❌ max(a, b), min(a, b)            — command fails (use Max/Min with list)

  Graphing commands (WORK):
    ✅ Root(f, xStart, xEnd)           — numeric root finder
    ✅ f(x) = expr; f(val)             — function definition + evaluation
    ✅ Numeric: sqrt, abs              — direct computation
    ✅ Max({a,b}), Min({a,b})          — list form works
    ✅ Expand(), Factor()              — return correct string representation
    ⚠️ Simplify()                      — may not actually simplify

  CRITICAL: Trig functions use RADIANS, not degrees:
    ⚠️ sin(30)  = sin(30 rad) ≈ -0.988  (NOT 0.5)
    ⚠️ sin(30°) = 0.5  (degree symbol forces degrees)
    ⚠️ atan(1)  = π/4 ≈ 0.785  (NOT 45)
    ⚠️ Angle(A,B,C) getValue() returns RADIANS, valueString() shows degrees
""")


if __name__ == "__main__":
    main()
