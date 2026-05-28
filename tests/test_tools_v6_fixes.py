"""
Verification tests for v6 tool fixes:
  Fix 1: add_tangent_conic_conic — common tangent lines between two circles/conics
  Fix 2: CanvasTracker._llm_viewport — viewport interception for render_set_coord_system
  Fix 3: add_semicircle — direction feedback via arc probe

Run:  cd /path/to/harness-geometry
      python tests/test_tools_v6_fixes.py
"""
import sys
import math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import (
    execute_geogebra_tool,
    CanvasTracker,
    GLOBAL_GEOGEBRA_TOOLS,
    QUERY_GEOGEBRA_TOOLS,
    RENDER_GEOGEBRA_TOOLS,
)

OUTDIR = Path("temp") / "test_tools_v6_fixes"
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
print("v6 TOOL FIXES VERIFICATION")
print("=" * 70)
print(f"\nTool counts: GLOBAL={len(GLOBAL_GEOGEBRA_TOOLS)}  "
      f"QUERY={len(QUERY_GEOGEBRA_TOOLS)}  RENDER={len(RENDER_GEOGEBRA_TOOLS)}  "
      f"TOTAL={len(GLOBAL_GEOGEBRA_TOOLS)+len(QUERY_GEOGEBRA_TOOLS)+len(RENDER_GEOGEBRA_TOOLS)}")

ggb = GeoGebraAPI(mode="selenium", headless=True)
ggb.initialize()

tracker = CanvasTracker()


def run(tool_name, args):
    """Helper: execute tool and return (cmd, ok, err)."""
    cmd, ok, err = execute_geogebra_tool(ggb, tool_name, args)
    return cmd, ok, err


def run_tracked(tool_name, args):
    """Helper: execute via CanvasTracker."""
    result, log = tracker.execute(ggb, tool_name, args)
    return result, log


# ══════════════════════════════════════════════════════════════════════════════
# §1  Fix 1: add_tangent_conic_conic
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §1.1  add_tangent_conic_conic: ToolSpec exists ──")

tool_names = [t.name for t in GLOBAL_GEOGEBRA_TOOLS]
check("add_tangent_conic_conic is registered", "add_tangent_conic_conic" in tool_names)

idx_tangent = tool_names.index("add_tangent")
idx_conic_conic = tool_names.index("add_tangent_conic_conic")
check("add_tangent_conic_conic comes right after add_tangent",
      idx_conic_conic == idx_tangent + 1,
      f"index {idx_conic_conic} vs add_tangent at {idx_tangent}")

spec = [t for t in GLOBAL_GEOGEBRA_TOOLS if t.name == "add_tangent_conic_conic"][0]
check("has 'name' param", "name" in spec.params)
check("has 'conic1' param", "conic1" in spec.params)
check("has 'conic2' param", "conic2" in spec.params)


# ── §1.2  Two separate circles — expect 4 common tangent lines ──────────────

print("\n── §1.2  Two separate circles → 4 tangent lines ──")
ggb.reset()

run("add_point", {"name": "O1", "x": 0, "y": 0})
run("add_point", {"name": "O2", "x": 6, "y": 0})
run("add_circle", {"name": "c1", "center": "O1", "radius": 1})
run("add_circle", {"name": "c2", "center": "O2", "radius": 1})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "ct", "conic1": "c1", "conic2": "c2"})
check("add_tangent_conic_conic succeeds for separated circles", ok, err)

# Count auto-created tangent lines
n_found = sum(1 for i in range(1, 5) if ggb.is_defined(f"ct_{{{i}}}"))
check(f"4 tangent lines created (found {n_found})", n_found == 4,
      f"expected 4, got {n_found}")

# Feedback message should mention the sub-names
if err:
    check("feedback mentions ct_{1}", "ct_{1}" in err, err)

export(ggb, "01_tangent_conic_conic_separate")


# ── §1.3  Two overlapping circles — 2 external tangents only ────────────────

print("\n── §1.3  Overlapping circles → 2 tangent lines ──")
ggb.reset()

run("add_point", {"name": "P1", "x": 0, "y": 0})
run("add_point", {"name": "P2", "x": 2, "y": 0})
run("add_circle", {"name": "k1", "center": "P1", "radius": 1.5})
run("add_circle", {"name": "k2", "center": "P2", "radius": 1.5})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "ot", "conic1": "k1", "conic2": "k2"})
check("add_tangent_conic_conic succeeds for overlapping circles", ok, err)

n_found = sum(1 for i in range(1, 5) if ggb.is_defined(f"ot_{{{i}}}"))
check(f"2 tangent lines created (found {n_found})", n_found == 2,
      f"expected 2, got {n_found}")

export(ggb, "02_tangent_conic_conic_overlap")


# ── §1.4  Concentric circles — no tangent lines ────────────────────────────

print("\n── §1.4  Concentric circles → 0 tangent lines ──")
ggb.reset()

run("add_point", {"name": "CC", "x": 0, "y": 0})
run("add_circle", {"name": "cc1", "center": "CC", "radius": 1})
run("add_circle", {"name": "cc2", "center": "CC", "radius": 3})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "zz", "conic1": "cc1", "conic2": "cc2"})
check("add_tangent_conic_conic fails for concentric circles", not ok,
      "should have failed" if ok else "correctly failed")
if not ok:
    check("error mentions 'concentric'", "concentric" in err.lower(), err)

export(ggb, "03_tangent_conic_conic_concentric")


# ── §1.5  Circle vs Ellipse ─────────────────────────────────────────────────

print("\n── §1.5  Circle vs Ellipse → tangent lines ──")
ggb.reset()

run("add_point", {"name": "Fc1", "x": -1, "y": 0})
run("add_point", {"name": "Fc2", "x": 1, "y": 0})
run("add_point", {"name": "Pe", "x": 2, "y": 0})
run("add_ellipse", {"name": "ell", "f1": "Fc1", "f2": "Fc2", "p": "Pe"})

run("add_point", {"name": "Oc", "x": 6, "y": 0})
run("add_circle", {"name": "cExt", "center": "Oc", "radius": 1})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "ce", "conic1": "ell", "conic2": "cExt"})
check("circle-ellipse tangent succeeds", ok, err)
n_found = sum(1 for i in range(1, 5) if ggb.is_defined(f"ce_{{{i}}}"))
check(f"tangent lines created (found {n_found})", n_found >= 2,
      f"expected ≥2, got {n_found}")

export(ggb, "04_tangent_circle_ellipse")


# ── §1.6  Two Ellipses ──────────────────────────────────────────────────────

print("\n── §1.6  Two Ellipses → tangent lines ──")
ggb.reset()

run("add_point", {"name": "E1f1", "x": -1, "y": 0})
run("add_point", {"name": "E1f2", "x": 1, "y": 0})
run("add_point", {"name": "E1p", "x": 2, "y": 0})
run("add_ellipse", {"name": "e1", "f1": "E1f1", "f2": "E1f2", "p": "E1p"})

run("add_point", {"name": "E2f1", "x": 7, "y": 0})
run("add_point", {"name": "E2f2", "x": 9, "y": 0})
run("add_point", {"name": "E2p", "x": 10, "y": 0})
run("add_ellipse", {"name": "e2", "f1": "E2f1", "f2": "E2f2", "p": "E2p"})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "ee", "conic1": "e1", "conic2": "e2"})
check("ellipse-ellipse tangent succeeds", ok, err)
n_found = sum(1 for i in range(1, 5) if ggb.is_defined(f"ee_{{{i}}}"))
check(f"tangent lines created (found {n_found})", n_found >= 2,
      f"expected ≥2, got {n_found}")

export(ggb, "05_tangent_ellipse_ellipse")


# ── §1.7  Parabola vs Circle — GGB limitation: graceful failure ─────────────

print("\n── §1.7  Parabola vs Circle → graceful failure (GGB unsupported) ──")
ggb.reset()

run("add_point", {"name": "Fp", "x": 1, "y": 0})
run("add_point", {"name": "Dpt1", "x": -1, "y": 0})
run("add_point", {"name": "Dpt2", "x": -1, "y": 1})
run("add_line", {"name": "dxl", "p1": "Dpt1", "p2": "Dpt2"})
run("add_parabola", {"name": "par", "focus": "Fp", "directrix": "dxl"})

run("add_point", {"name": "Opc", "x": 6, "y": 0})
run("add_circle", {"name": "cPar", "center": "Opc", "radius": 1})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "pc", "conic1": "par", "conic2": "cPar"})
# GeoGebra Tangent(Conic,Conic) only supports circles & ellipses
check("parabola-circle: fails gracefully (GGB limitation)", not ok,
      "unexpectedly succeeded" if ok else "correctly failed")

export(ggb, "06_tangent_parabola_circle")


# ── §1.8  Hyperbola vs Circle — GGB limitation: graceful failure ────────────

print("\n── §1.8  Hyperbola vs Circle → graceful failure (GGB unsupported) ──")
ggb.reset()

run("add_point", {"name": "Hf1", "x": -3, "y": 0})
run("add_point", {"name": "Hf2", "x": 3, "y": 0})
run("add_point", {"name": "Hp", "x": 2, "y": 0})
run("add_hyperbola", {"name": "hyp", "f1": "Hf1", "f2": "Hf2", "p": "Hp"})

run("add_point", {"name": "Ohc", "x": 8, "y": 0})
run("add_circle", {"name": "cHyp", "center": "Ohc", "radius": 1})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "hc", "conic1": "hyp", "conic2": "cHyp"})
check("hyperbola-circle: fails gracefully (GGB limitation)", not ok,
      "unexpectedly succeeded" if ok else "correctly failed")

export(ggb, "07_tangent_hyperbola_circle")


# ── §1.9  Ellipse vs Hyperbola — GGB limitation: graceful failure ───────────

print("\n── §1.9  Ellipse vs Hyperbola → graceful failure (GGB unsupported) ──")
ggb.reset()

run("add_point", {"name": "Xf1", "x": -1, "y": 3})
run("add_point", {"name": "Xf2", "x": 1, "y": 3})
run("add_point", {"name": "Xp", "x": 2, "y": 3})
run("add_ellipse", {"name": "xe", "f1": "Xf1", "f2": "Xf2", "p": "Xp"})

run("add_point", {"name": "Yf1", "x": 7, "y": 3})
run("add_point", {"name": "Yf2", "x": 13, "y": 3})
run("add_point", {"name": "Yp", "x": 9, "y": 3})
run("add_hyperbola", {"name": "yh", "f1": "Yf1", "f2": "Yf2", "p": "Yp"})

cmd, ok, err = run("add_tangent_conic_conic",
                    {"name": "eh", "conic1": "xe", "conic2": "yh"})
check("ellipse-hyperbola: fails gracefully (GGB limitation)", not ok,
      "unexpectedly succeeded" if ok else "correctly failed")

export(ggb, "08_tangent_ellipse_hyperbola")


# ── §1.10  Regression: original add_tangent (point-to-conic) still works ────

print("\n── §1.10  Regression: add_tangent (point, conic) unchanged ──")
ggb.reset()

run("add_point", {"name": "ExtP", "x": 4, "y": 0})
run("add_point", {"name": "Ctr", "x": 0, "y": 0})
run("add_circle", {"name": "cReg", "center": "Ctr", "radius": 2})

cmd, ok, err = run("add_tangent",
                    {"name": "tReg", "point": "ExtP", "conic": "cReg"})
check("original add_tangent still works", ok, err)

export(ggb, "09_tangent_regression")


# ══════════════════════════════════════════════════════════════════════════════
# §2  Fix 2: CanvasTracker._llm_viewport interception
# ══════════════════════════════════════════════════════════════════════════════

print("\n── §2.1  CanvasTracker._llm_viewport starts as None ──")

t2 = CanvasTracker()
check("_llm_viewport is None initially", t2._llm_viewport is None)


print("\n── §2.2  render_set_coord_system updates _llm_viewport ──")
ggb.reset()

# Create a point so the canvas is non-empty
run("add_point", {"name": "Anchor", "x": 0, "y": 0})

result, log = t2.execute(ggb, "render_set_coord_system",
                         {"x_min": -10, "x_max": 20, "y_min": -5, "y_max": 15})
check("render_set_coord_system succeeds", result.get("success", False),
      result.get("error", ""))

vp = t2._llm_viewport
check("_llm_viewport is set after render_set_coord_system", vp is not None)
if vp:
    check(f"_llm_viewport == (-10, 20, -5, 15) (got {vp})",
          vp == (-10.0, 20.0, -5.0, 15.0), str(vp))


print("\n── §2.3  Second render_set_coord_system overwrites ──")

result2, _ = t2.execute(ggb, "render_set_coord_system",
                        {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100})
vp2 = t2._llm_viewport
check("_llm_viewport updated to (0, 100, 0, 100)",
      vp2 == (0.0, 100.0, 0.0, 100.0), str(vp2))


print("\n── §2.4  Non-viewport tools do NOT change _llm_viewport ──")

t3 = CanvasTracker()
run("add_point", {"name": "Q", "x": 5, "y": 5})
result3, _ = t3.execute(ggb, "add_point", {"name": "R", "x": 3, "y": 3})
check("_llm_viewport still None after add_point", t3._llm_viewport is None)


# ══════════════════════════════════════════════════════════════════════════════
# §3  Fix 3: add_semicircle direction feedback
# ══════════════════════════════════════════════════════════════════════════════

# Reset viewport from §2's [0,100]×[0,100] — ggb.reset() only clears objects
ggb.set_coord_system(-3, 7, -5, 5)

print("\n── §3.1  Horizontal: p1=left, p2=right → arc ABOVE ──")
ggb.reset()
ggb.set_coord_system(-3, 7, -5, 5)

run("add_point", {"name": "L", "x": 0, "y": 0})
run("add_point", {"name": "R", "x": 4, "y": 0})

cmd, ok, err = run("add_semicircle", {"name": "sAB", "p1": "L", "p2": "R"})
check("add_semicircle succeeds", ok, err)
check("feedback mentions ABOVE", "ABOVE" in (err or ""),
      f"expected 'ABOVE' in feedback, got: {err}")

export(ggb, "10_semicircle_above")


print("\n── §3.2  Horizontal: p1=right, p2=left → arc BELOW ──")
ggb.reset()
ggb.set_coord_system(-3, 7, -5, 5)

run("add_point", {"name": "L2", "x": 0, "y": 0})
run("add_point", {"name": "R2", "x": 4, "y": 0})

cmd, ok, err = run("add_semicircle", {"name": "sBA", "p1": "R2", "p2": "L2"})
check("add_semicircle succeeds (reversed)", ok, err)
check("feedback mentions BELOW", "BELOW" in (err or ""),
      f"expected 'BELOW' in feedback, got: {err}")

export(ggb, "11_semicircle_below")


print("\n── §3.3  Vertical: p1=bottom, p2=top → arc LEFT ──")
ggb.reset()
ggb.set_coord_system(-5, 5, -3, 7)

run("add_point", {"name": "Bot", "x": 0, "y": 0})
run("add_point", {"name": "Top", "x": 0, "y": 4})

cmd, ok, err = run("add_semicircle", {"name": "sVert", "p1": "Bot", "p2": "Top"})
check("add_semicircle succeeds (vertical)", ok, err)
check("feedback mentions LEFT", "LEFT" in (err or ""),
      f"expected 'LEFT' in feedback, got: {err}")

export(ggb, "12_semicircle_left")


print("\n── §3.4  Vertical: p1=top, p2=bottom → arc RIGHT ──")
ggb.reset()
ggb.set_coord_system(-5, 5, -3, 7)

run("add_point", {"name": "Bot2", "x": 0, "y": 0})
run("add_point", {"name": "Top2", "x": 0, "y": 4})

cmd, ok, err = run("add_semicircle", {"name": "sVertR", "p1": "Top2", "p2": "Bot2"})
check("add_semicircle succeeds (vertical reversed)", ok, err)
check("feedback mentions RIGHT", "RIGHT" in (err or ""),
      f"expected 'RIGHT' in feedback, got: {err}")

export(ggb, "13_semicircle_right")


print("\n── §3.5  Semicircle description updated ──")

sc_spec = [t for t in GLOBAL_GEOGEBRA_TOOLS if t.name == "add_semicircle"][0]
check("description mentions 'LEFT side'", "LEFT side" in sc_spec.description,
      sc_spec.description[:80])
check("description mentions 'SWAP p1 and p2'", "SWAP p1 and p2" in sc_spec.description,
      sc_spec.description[:80])


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

ggb.cleanup()

print("\n" + "=" * 70)
print(f"RESULT:  {PASS} passed,  {FAIL} failed,  {PASS + FAIL} total")
print("=" * 70)
if FAIL:
    sys.exit(1)
