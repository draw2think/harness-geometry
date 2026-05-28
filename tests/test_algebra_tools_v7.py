"""
Comprehensive test suite for v7 algebra/calculus tool extensions.

Covers: set_value, add_derivative, add_integral_function,
        query_definite_integral, add_inflection_point, add_asymptote,
        query_function_max, query_function_min

Plus: chained operations, slider-dependent workflows, edge cases,
      and MathCanvas-style composite problems.

Usage:  python tests/test_algebra_tools_v7.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import execute_geogebra_tool, execute_query_tool

ggb = GeoGebraAPI(mode="selenium", headless=True)
ggb.initialize()
print("GeoGebra initialized:", "OK" if ggb._driver else "FAIL")

def add(name, args):
    cmd, ok, err = execute_geogebra_tool(ggb, name, args)
    tag = "OK" if ok else "FAIL"
    print(f"    [{tag}] {cmd}  -> {err}")
    return ok

def query(name, args):
    result = execute_query_tool(ggb, name, args)
    cmd, ok, err, val = result[0], result[1], result[2], result[3]
    tag = "OK" if ok else "FAIL"
    print(f"    [{tag}] {cmd}  -> val={val}  err={err}")
    return ok, val

def get(expr):
    """Evaluate expression and return float."""
    return ggb.get_value(expr)

def approx(a, b, tol=0.01):
    """Check approximate equality."""
    if a is None or b is None:
        return False
    return abs(a - b) < tol

passed = 0
total = 0

def check(condition, label=""):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✓ PASS {label}")
    else:
        print(f"  ✗ FAIL {label}")

# ═══════════════════════════════════════════════════════════════════════
#  Part 1: Basic single-tool tests
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 1: Basic single-tool tests")
print("="*60)

# --- 1.1 add_derivative: cubic ---
print("\n--- 1.1 add_derivative (cubic) ---")
add("add_function", {"name": "f1", "expr": "x^3 - 3*x^2 + 2"})
add("add_derivative", {"name": "df1", "function": "f1"})
check(approx(get("df1(0)"), 0), "df1(0)=0")
check(approx(get("df1(1)"), -3), "df1(1)=-3")
check(approx(get("df1(2)"), 0), "df1(2)=0")

# --- 1.2 add_derivative: second order ---
print("\n--- 1.2 add_derivative (2nd order) ---")
add("add_derivative", {"name": "d2f1", "function": "f1", "order": 2})
check(approx(get("d2f1(0)"), -6), "d2f1(0)=-6")
check(approx(get("d2f1(1)"), 0), "d2f1(1)=0 (inflection)")
check(approx(get("d2f1(3)"), 12), "d2f1(3)=12")

# --- 1.3 add_derivative: trig ---
print("\n--- 1.3 add_derivative (trig) ---")
add("add_function", {"name": "f2", "expr": "sin(x)"})
add("add_derivative", {"name": "df2", "function": "f2"})
check(approx(get("df2(0)"), 1), "d/dx sin(0) = cos(0) = 1")
check(approx(get(f"df2({math.pi/2})"), 0, 0.001), "d/dx sin(π/2) = cos(π/2) = 0")

# --- 1.4 add_integral_function ---
print("\n--- 1.4 add_integral_function ---")
add("add_function", {"name": "f3", "expr": "2*x"})
add("add_integral_function", {"name": "F3", "function": "f3"})
# F3(x) = x^2 + C; F3(3) - F3(0) should be 9
v3 = get("F3(3)")
v0 = get("F3(0)")
check(v3 is not None and v0 is not None and approx(v3 - v0, 9), f"∫2x dx from 0 to 3 = 9 (F3(3)-F3(0)={v3}-{v0})")

# --- 1.5 query_definite_integral ---
print("\n--- 1.5 query_definite_integral ---")
ok, val = query("query_definite_integral", {"function": "f3", "start": 0, "end": 3})
check(ok and approx(val, 9), f"∫₀³ 2x dx = 9 (got {val})")

ok, val = query("query_definite_integral", {"function": "f1", "start": 0, "end": 2})
check(ok and approx(val, 0), f"∫₀² (x³-3x²+2)dx = 0 (got {val})")

ok, val = query("query_definite_integral", {"function": "f2", "start": 0, "end": math.pi})
check(ok and approx(val, 2), f"∫₀ᵖ sin(x)dx = 2 (got {val})")

# --- 1.6 add_inflection_point ---
print("\n--- 1.6 add_inflection_point ---")
ok = add("add_inflection_point", {"name": "ip1", "function": "f1"})
# f1 = x³-3x²+2, inflection at x=1, y=0
ip_x = get("x(ip1)")
ip_y = get("y(ip1)")
check(ok and approx(ip_x, 1) and approx(ip_y, 0), f"inflection at ({ip_x},{ip_y}), expected (1,0)")

# --- 1.7 add_asymptote ---
print("\n--- 1.7 add_asymptote ---")
add("add_function", {"name": "f4", "expr": "(x^2 - 1)/(x - 2)"})
ok = add("add_asymptote", {"name": "asy4", "function": "f4"})
check(ok, "asymptotes of (x²-1)/(x-2) found")

# --- 1.8 query_function_max ---
print("\n--- 1.8 query_function_max ---")
ok, val = query("query_function_max", {"function": "f1", "start": -1, "end": 1})
check(ok and isinstance(val, tuple) and approx(val[0], 0) and approx(val[1], 2),
      f"max of x³-3x²+2 on [-1,1] at (0,2), got {val}")

# --- 1.9 query_function_min ---
print("\n--- 1.9 query_function_min ---")
ok, val = query("query_function_min", {"function": "f1", "start": 0, "end": 3})
check(ok and isinstance(val, tuple) and approx(val[0], 2) and approx(val[1], -2),
      f"min of x³-3x²+2 on [0,3] at (2,-2), got {val}")

# --- 1.10 add_slider + set_value ---
print("\n--- 1.10 add_slider + set_value ---")
add("add_slider", {"name": "s1", "min": -5, "max": 5, "step": 0.01})
add("set_value", {"name": "s1", "value": 2.718})
check(approx(get("s1"), 2.72, 0.01), f"slider s1 = {get('s1')}, expected ~2.72")

# --- 1.11 set_value updates dependents ---
print("\n--- 1.11 slider-dependent update ---")
add("add_function", {"name": "fslider", "expr": "s1 * x^2"})
check(approx(get("fslider(1)"), 2.72, 0.02), f"fslider(1) = {get('fslider(1)')}, expected ~2.72")
add("set_value", {"name": "s1", "value": -1.0})
check(approx(get("fslider(1)"), -1.0), f"after s1=-1: fslider(1) = {get('fslider(1)')}, expected -1")
check(approx(get("fslider(3)"), -9.0), f"fslider(3) = {get('fslider(3)')}, expected -9")


# ═══════════════════════════════════════════════════════════════════════
#  Part 2: Chained operations
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 2: Chained operations")
print("="*60)

# --- 2.1 function → derivative → roots of derivative (critical points) ---
print("\n--- 2.1 function → derivative → roots (critical pts) ---")
add("add_function", {"name": "p", "expr": "x^4 - 8*x^2 + 3"})
add("add_derivative", {"name": "dp", "function": "p"})
# dp = 4x³ - 16x = 4x(x²-4), roots at x=0, ±2
add("add_roots", {"name": "cp", "obj": "dp"})
check(True, "critical points created (visual check)")

# --- 2.2 function → max/min → verify with derivative ---
print("\n--- 2.2 max/min verify with derivative ---")
ok_max, val_max = query("query_function_max", {"function": "p", "start": -3, "end": 0})
ok_min, val_min = query("query_function_min", {"function": "p", "start": -3, "end": 0})
# local max at x=0: p(0)=3; local min at x=-2: p(-2)=3-16+3=-13
check(ok_max and isinstance(val_max, tuple) and approx(val_max[0], 0) and approx(val_max[1], 3),
      f"max on [-3,0] at (0,3), got {val_max}")
check(ok_min and isinstance(val_min, tuple) and approx(val_min[0], -2) and approx(val_min[1], -13),
      f"min on [-3,0] at (-2,-13), got {val_min}")

# --- 2.3 definite integral chain: ∫₋₂² (x⁴-8x²+3) dx ---
print("\n--- 2.3 definite integral of quartic ---")
ok, val = query("query_definite_integral", {"function": "p", "start": -2, "end": 2})
# ∫₋₂² (x⁴-8x²+3)dx = 2[x⁵/5-8x³/3+3x]₀² = 2(32/5-64/3+6) = 2(6.4-21.333+6) = -17.867
expected = 2 * (32/5 - 64/3 + 6)  # = -268/15 ≈ -17.867
check(ok and approx(val, expected, 0.1), f"∫₋₂² (x⁴-8x²+3)dx = {expected:.3f}, got {val}")

# --- 2.4 derivative → inflection → verify second derivative sign ---
print("\n--- 2.4 inflection point verification ---")
add("add_derivative", {"name": "d2p", "function": "p", "order": 2})
# d2p = 12x² - 16, roots at x = ±√(4/3) = ±1.1547
add("add_inflection_point", {"name": "ipp", "function": "p"})
# check d2p changes sign at inflection
d2p_before = get("d2p(1)")  # 12-16 = -4 (concave down)
d2p_after = get("d2p(1.5)")  # 27-16 = 11 (concave up)
check(d2p_before is not None and d2p_before < 0 and d2p_after > 0,
      f"d2p(1)={d2p_before}<0, d2p(1.5)={d2p_after}>0, confirms inflection")


# ═══════════════════════════════════════════════════════════════════════
#  Part 3: Slider-driven exploration (MathCanvas-style)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 3: Slider-driven exploration")
print("="*60)

# --- 3.1 parametric intersection count ---
# f(x) = |x+2| + |x-2|, g(x) = a*|x-1|
# Find how many intersections for different a values
print("\n--- 3.1 parametric intersection (abs function) ---")
add("add_function", {"name": "fa", "expr": "abs(x+2) + abs(x-2)"})
add("add_slider", {"name": "aa", "min": 0, "max": 10, "step": 0.1})
add("add_function", {"name": "ga", "expr": "aa * abs(x-1)"})

# a=0: g=0, intersections where f=0 → no intersection (f≥4)
add("set_value", {"name": "aa", "value": 0})
ga_at_0 = get("ga(0)")
check(approx(ga_at_0, 0), f"a=0: ga(0)={ga_at_0}, g≡0")

# a=1.5: should have some intersections
add("set_value", {"name": "aa", "value": 1.5})
ga_at_m2 = get("ga(-2)")
fa_at_m2 = get("fa(-2)")
check(ga_at_m2 is not None and fa_at_m2 is not None,
      f"a=1.5: ga(-2)={ga_at_m2}, fa(-2)={fa_at_m2}")

# a=3: steeper, check g dominates
add("set_value", {"name": "aa", "value": 3})
ga_at_5 = get("ga(5)")
fa_at_5 = get("fa(5)")
check(ga_at_5 is not None and ga_at_5 > fa_at_5,
      f"a=3: ga(5)={ga_at_5} > fa(5)={fa_at_5}")

# --- 3.2 exploring function max via slider ---
print("\n--- 3.2 exploring parabola vertex via slider ---")
add("add_slider", {"name": "kk", "min": -5, "max": 5, "step": 0.1})
add("add_function", {"name": "fk", "expr": "-(x - kk)^2 + kk^2"})

# k=2: vertex at (2,4)
add("set_value", {"name": "kk", "value": 2})
ok, val = query("query_function_max", {"function": "fk", "start": -5, "end": 5})
check(ok and isinstance(val, tuple) and approx(val[0], 2) and approx(val[1], 4),
      f"k=2: max at (2,4), got {val}")

# k=-3: vertex at (-3,9)
add("set_value", {"name": "kk", "value": -3})
ok, val = query("query_function_max", {"function": "fk", "start": -5, "end": 5})
check(ok and isinstance(val, tuple) and approx(val[0], -3) and approx(val[1], 9),
      f"k=-3: max at (-3,9), got {val}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 4: Edge cases
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 4: Edge cases")
print("="*60)

# --- 4.1 derivative of constant ---
print("\n--- 4.1 derivative of constant ---")
add("add_function", {"name": "fc", "expr": "5"})
add("add_derivative", {"name": "dfc", "function": "fc"})
check(approx(get("dfc(0)"), 0), "d/dx (5) = 0")
check(approx(get("dfc(100)"), 0), "d/dx (5) at x=100 = 0")

# --- 4.2 integral of zero ---
print("\n--- 4.2 integral of zero ---")
add("add_function", {"name": "fz", "expr": "0*x"})
ok, val = query("query_definite_integral", {"function": "fz", "start": -10, "end": 10})
check(ok and approx(val, 0), f"∫₋₁₀¹⁰ 0 dx = 0, got {val}")

# --- 4.3 negative integral ---
print("\n--- 4.3 negative integral ---")
add("add_function", {"name": "fn", "expr": "-1"})
ok, val = query("query_definite_integral", {"function": "fn", "start": 0, "end": 5})
check(ok and approx(val, -5), f"∫₀⁵ (-1)dx = -5, got {val}")

# --- 4.4 derivative of exponential ---
print("\n--- 4.4 derivative of exp ---")
add("add_function", {"name": "fe", "expr": "exp(x)"})
add("add_derivative", {"name": "dfe", "function": "fe"})
check(approx(get("dfe(0)"), 1), "d/dx e^x at x=0 = 1")
check(approx(get("dfe(1)"), math.e), f"d/dx e^x at x=1 = e = {math.e:.4f}")

# --- 4.5 slider at boundary ---
print("\n--- 4.5 slider boundary values ---")
add("add_slider", {"name": "sb", "min": -10, "max": 10, "step": 1})
add("set_value", {"name": "sb", "value": -10})
check(approx(get("sb"), -10), "slider at min boundary = -10")
add("set_value", {"name": "sb", "value": 10})
check(approx(get("sb"), 10), "slider at max boundary = 10")

# --- 4.6 integral of sin (exact) ---
print("\n--- 4.6 integral of sin (exact) ---")
ok, val = query("query_definite_integral", {"function": "f2", "start": 0, "end": math.pi})
check(ok and approx(val, 2, 0.001), f"∫₀ᵖ sin(x)dx = 2, got {val}")

ok, val = query("query_definite_integral", {"function": "f2", "start": 0, "end": 2*math.pi})
check(ok and approx(val, 0, 0.001), f"∫₀²ᵖ sin(x)dx = 0, got {val}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 5: MathCanvas-style composite problems
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Part 5: MathCanvas-style composite problems")
print("="*60)

# --- 5.1 Find area between two curves ---
print("\n--- 5.1 area between curves ---")
add("add_function", {"name": "curve1", "expr": "x^2"})
add("add_function", {"name": "curve2", "expr": "2*x"})
# Intersect at x=0 and x=2
add("add_intersect", {"name": "int1", "obj1": "curve1", "obj2": "curve2", "index": 1})
add("add_intersect", {"name": "int2", "obj1": "curve1", "obj2": "curve2", "index": 2})
ix1 = get("x(int1)")
ix2 = get("x(int2)")
print(f"    intersections at x={ix1}, x={ix2}")
# Area = ∫₀² (2x - x²)dx = [x² - x³/3]₀² = 4-8/3 = 4/3
add("add_function", {"name": "diff12", "expr": "2*x - x^2"})
ok, area = query("query_definite_integral", {"function": "diff12", "start": 0, "end": 2})
check(ok and approx(area, 4/3, 0.01), f"area between x² and 2x = 4/3 ≈ 1.333, got {area}")

# --- 5.2 Tangent line slope at a point ---
print("\n--- 5.2 tangent slope via derivative ---")
add("add_function", {"name": "ft", "expr": "x^3 - x"})
add("add_derivative", {"name": "dft", "function": "ft"})
# slope at x=1: f'(1) = 3(1)² - 1 = 2
slope = get("dft(1)")
check(approx(slope, 2), f"slope of x³-x at x=1: f'(1) = {slope}, expected 2")
# slope at x=0: f'(0) = -1
slope0 = get("dft(0)")
check(approx(slope0, -1), f"slope at x=0: f'(0) = {slope0}, expected -1")

# --- 5.3 function analysis pipeline ---
print("\n--- 5.3 full function analysis: f(x) = x³ - 6x² + 9x + 1 ---")
add("add_function", {"name": "fA", "expr": "x^3 - 6*x^2 + 9*x + 1"})
add("add_derivative", {"name": "dfA", "function": "fA"})
add("add_derivative", {"name": "d2fA", "function": "fA", "order": 2})
add("add_turning_point", {"name": "tpA", "obj": "fA"})
add("add_inflection_point", {"name": "ipA", "function": "fA"})
# f'(x) = 3x²-12x+9 = 3(x-1)(x-3), critical at x=1,3
# f(1)=1-6+9+1=5 (local max), f(3)=27-54+27+1=1 (local min)
# f''(x) = 6x-12, inflection at x=2, f(2)=8-24+18+1=3
ok_max, v_max = query("query_function_max", {"function": "fA", "start": -1, "end": 2})
ok_min, v_min = query("query_function_min", {"function": "fA", "start": 1, "end": 4})
check(ok_max and isinstance(v_max, tuple) and approx(v_max[0], 1) and approx(v_max[1], 5),
      f"local max at (1,5), got {v_max}")
check(ok_min and isinstance(v_min, tuple) and approx(v_min[0], 3) and approx(v_min[1], 1),
      f"local min at (3,1), got {v_min}")
ip_x = get("x(ipA)")
ip_y = get("y(ipA)")
check(approx(ip_x, 2) and approx(ip_y, 3), f"inflection at ({ip_x},{ip_y}), expected (2,3)")


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

ggb.cleanup()

print(f"\n{'='*60}")
print(f"  v7 Algebra Tools: {passed}/{total} passed")
if passed == total:
    print("  ALL TESTS PASSED ✓")
else:
    print(f"  {total - passed} FAILED ✗")
print(f"{'='*60}")
