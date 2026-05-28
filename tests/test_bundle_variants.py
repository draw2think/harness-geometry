"""
Test each GeoGebra bundle variant (web / web3d / webSimple) independently.

Measures: startup time, API completeness, CAS availability, PNG export,
and (for web3d) basic 3D construction.

Usage:  python tests/test_bundle_variants.py
        python tests/test_bundle_variants.py web3d       # single variant
"""
import sys
import time
import math
import tempfile
from pathlib import Path

BUNDLE_DIR = (Path(__file__).resolve().parent.parent
              / "symbolic" / "integrations" / "geogebra_bundle" / "GeoGebra")
DEPLOY_JS = BUNDLE_DIR / "deployggb.js"

# ── Variant config ──────────────────────────────────────────────────────
# appName controls which GeoGebra app is loaded;
# codebase subfolder must match the nocache entry-point script name.
VARIANTS = {
    "web": {
        "codebase": "web",
        "appName": "classic",
        "description": "2D Classic (no 3D support)",
    },
    "web3d": {
        "codebase": "web3d",
        "appName": "classic",
        "description": "2D+3D Classic (current default)",
    },
    "webSimple": {
        "codebase": "webSimple",
        "appName": "classic",
        "description": "Lightweight 2D (fewer features)",
    },
}


def _build_html(variant: str) -> str:
    cfg = VARIANTS[variant]
    codebase_path = BUNDLE_DIR / "HTML5" / "5.0" / cfg["codebase"]
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GGB {variant}</title>
    <script src="file://{DEPLOY_JS.absolute()}"></script>
    <style>
        html, body {{ margin:0; padding:0; width:800px; height:800px;
                      overflow:hidden; background:white; }}
        #ggb-element {{ width:800px; height:800px; }}
    </style>
</head>
<body>
    <div id="ggb-element"></div>
    <script>
        window.ggbApplet = null;
        window.casReady = false;

        function appletOnLoad(api) {{
            window.ggbApplet = api;
            function checkCAS() {{
                try {{
                    var r = api.evalCommandCAS("1+1");
                    if (r && r !== "?" && r !== "") {{ window.casReady = true; return; }}
                }} catch(e) {{}}
                setTimeout(checkCAS, 200);
            }}
            checkCAS();
        }}

        var ggbApp = new GGBApplet({{
            "appName": "{cfg['appName']}",
            "perspective": "G",
            "width": 800,
            "height": 800,
            "showToolBar": false,
            "showAlgebraView": false,
            "showAlgebraInput": false,
            "showMenuBar": false,
            "enableRightClick": false,
            "showResetIcon": false,
            "appletOnLoad": appletOnLoad
        }}, true);

        window.addEventListener("load", function() {{
            ggbApp.setHTML5Codebase(
                'file://{codebase_path.absolute()}/', true);
            ggbApp.inject('ggb-element');
        }});
    </script>
</body>
</html>"""


# ── Test helpers ────────────────────────────────────────────────────────

class Result:
    def __init__(self, name):
        self.name = name
        self.tests = []   # (test_name, passed, detail)

    def add(self, name, passed, detail=""):
        tag = "OK" if passed else "FAIL"
        self.tests.append((name, passed, detail))
        print(f"    [{tag:4s}] {name}  {detail}")

    @property
    def passed(self):
        return sum(1 for _, p, _ in self.tests if p)

    @property
    def total(self):
        return len(self.tests)

    @property
    def all_passed(self):
        return self.passed == self.total


def _run_variant(variant: str) -> Result:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    cfg = VARIANTS[variant]
    res = Result(variant)
    print(f"\n{'='*60}")
    print(f"  [{variant}]  {cfg['description']}")
    print(f"  codebase: HTML5/5.0/{cfg['codebase']}/")
    print(f"{'='*60}")

    # Write HTML
    tmp = Path(tempfile.mkdtemp(prefix=f"ggb_{variant}_"))
    html_path = tmp / "test.html"
    html_path.write_text(_build_html(variant))

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-web-security")
    opts.add_argument("--allow-file-access-from-files")
    opts.add_argument("--window-size=1280,1280")

    driver = webdriver.Chrome(options=opts)
    try:
        # ── 1. Startup time ─────────────────────────────────────────────
        t0 = time.perf_counter()
        driver.get(f"file://{html_path.absolute()}")

        try:
            WebDriverWait(driver, 45).until(
                lambda d: d.execute_script(
                    "return window.ggbApplet !== null"
                )
            )
            t_applet = time.perf_counter() - t0
            res.add("applet_load", True, f"{t_applet:.2f}s")
        except Exception as e:
            res.add("applet_load", False, str(e))
            return res   # can't continue without applet

        js = driver.execute_script

        # ── 2. CAS availability ─────────────────────────────────────────
        t1 = time.perf_counter()
        cas_ok = False
        try:
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return window.casReady === true")
            )
            cas_ok = True
            t_cas = time.perf_counter() - t1
            res.add("cas_ready", True, f"{t_cas:.2f}s")
        except Exception:
            res.add("cas_ready", False, "timeout 15s")

        # ── 3. Basic point construction ─────────────────────────────────
        ok = js('return ggbApplet.evalCommand("A = (2, 3)")')
        ax = js('return ggbApplet.getXcoord("A")')
        ay = js('return ggbApplet.getYcoord("A")')
        res.add("add_point", ok and abs(ax - 2) < 0.01 and abs(ay - 3) < 0.01,
                f"A=({ax},{ay})")

        # ── 4. Segment + distance ───────────────────────────────────────
        js('ggbApplet.evalCommand("B = (5, 7)")')
        js('ggbApplet.evalCommand("seg = Segment(A, B)")')
        dist = js('return ggbApplet.getValue("seg")')
        expected = math.sqrt((5-2)**2 + (7-3)**2)
        res.add("segment_length", dist is not None and abs(dist - expected) < 0.01,
                f"{dist:.4f} (expected {expected:.4f})")

        # ── 5. Circle + Intersect ───────────────────────────────────────
        js('ggbApplet.evalCommand("c = Circle(A, 3)")')
        js('ggbApplet.evalCommand("line1 = Line((0,0), (10,10))")')
        js('ggbApplet.evalCommand("P = Intersect(c, line1, 1)")')
        p_exists = js('return ggbApplet.exists("P")')
        res.add("circle_intersect", p_exists is True,
                f"exists={p_exists}")

        # ── 6. Angle measurement ────────────────────────────────────────
        js('ggbApplet.evalCommand("C = (5, 0)")')
        js('ggbApplet.evalCommand("ang = Angle(C, A, B)")')
        ang_val = js('return ggbApplet.getValue("ang")')
        if ang_val is not None:
            ang_deg = math.degrees(ang_val)
            res.add("angle_measure", 0 < ang_deg < 360,
                    f"{ang_deg:.1f}°")
        else:
            res.add("angle_measure", False, "getValue returned None")

        # ── 7. Midpoint / derived construction ──────────────────────────
        js('ggbApplet.evalCommand("M = Midpoint(A, B)")')
        mx = js('return ggbApplet.getXcoord("M")')
        my = js('return ggbApplet.getYcoord("M")')
        res.add("midpoint", mx is not None and abs(mx - 3.5) < 0.01
                and abs(my - 5) < 0.01,
                f"M=({mx},{my})")

        # ── 8. Perpendicular line ───────────────────────────────────────
        js('ggbApplet.evalCommand("perp = PerpendicularLine(M, seg)")')
        perp_exists = js('return ggbApplet.exists("perp")')
        res.add("perpendicular_line", perp_exists is True)

        # ── 9. Polygon (triangle) ───────────────────────────────────────
        js('ggbApplet.evalCommand("tri = Polygon(A, B, C)")')
        tri_area = js('return ggbApplet.getValue("tri")')
        # A=(2,3), B=(5,7), C=(5,0) → area = |det| / 2
        exp_area = abs((5-2)*(0-3) - (5-2)*(7-3)) / 2
        res.add("polygon_area",
                tri_area is not None and abs(tri_area - exp_area) < 0.1,
                f"{tri_area} (expected {exp_area})")

        # ── 10. CAS: Solve ──────────────────────────────────────────────
        if cas_ok:
            cas_r = js('return ggbApplet.evalCommandCAS("Solve(x^2 - 9)")')
            res.add("cas_solve", cas_r is not None and "3" in str(cas_r),
                    str(cas_r))

            cas_r2 = js('return ggbApplet.evalCommandCAS("Factor(x^3 - 1)")')
            res.add("cas_factor", cas_r2 is not None and len(str(cas_r2)) > 3,
                    str(cas_r2))
        else:
            res.add("cas_solve", False, "CAS unavailable")
            res.add("cas_factor", False, "CAS unavailable")

        # ── 11. PNG export ──────────────────────────────────────────────
        png_b64 = js('return ggbApplet.getPNGBase64(1.0, false, 72)')
        res.add("png_export",
                png_b64 is not None and len(png_b64) > 100,
                f"{len(png_b64) if png_b64 else 0} chars")

        # ── 12. Delete + cascade ────────────────────────────────────────
        n_before = js('return ggbApplet.getObjectNumber()')
        js('ggbApplet.deleteObject("c")')   # circle — should cascade P
        n_after = js('return ggbApplet.getObjectNumber()')
        p_gone = not js('return ggbApplet.exists("P")')
        res.add("delete_cascade",
                n_after < n_before and p_gone,
                f"objects {n_before}→{n_after}, P removed={p_gone}")

        # ── 13. setCoordSystem ──────────────────────────────────────────
        try:
            js('ggbApplet.setCoordSystem(-10, 10, -10, 10)')
            res.add("set_coord_system", True)
        except Exception as e:
            res.add("set_coord_system", False, str(e))

        # ── 14. getValueString ──────────────────────────────────────────
        vs = js('return ggbApplet.getValueString("seg")')
        res.add("get_value_string", vs is not None and len(vs) > 0,
                repr(vs))

        # ── 15. 3D tests (web3d only) ──────────────────────────────────
        if variant == "web3d":
            print(f"\n  -- 3D-specific tests --")
            # Point3D
            ok_3d = js('return ggbApplet.evalCommand("D = (1, 2, 3)")')
            dz = js("""
                try { return ggbApplet.getZcoord("D"); }
                catch(e) { return null; }
            """)
            res.add("3d_point", ok_3d and dz is not None,
                    f"D z-coord={dz}")

            # Plane
            js('ggbApplet.evalCommand("E = (4, 0, 0)")')
            js('ggbApplet.evalCommand("F = (0, 5, 0)")')
            ok_plane = js('return ggbApplet.evalCommand("pl = Plane(D, E, F)")')
            pl_exists = js('return ggbApplet.exists("pl")')
            res.add("3d_plane", pl_exists is True,
                    f"evalCommand={ok_plane}, exists={pl_exists}")

            # Sphere
            ok_sph = js('return ggbApplet.evalCommand("sph = Sphere(D, 2)")')
            sph_exists = js('return ggbApplet.exists("sph")')
            res.add("3d_sphere", sph_exists is True)

            # Prism / Pyramid (need polygon base)
            js('ggbApplet.evalCommand("G = (0, 0, 0)")')
            js('ggbApplet.evalCommand("H = (3, 0, 0)")')
            js('ggbApplet.evalCommand("I = (3, 3, 0)")')
            js('ggbApplet.evalCommand("J = (0, 3, 0)")')
            ok_prism = js(
                'return ggbApplet.evalCommand("prism1 = Prism(G, H, I, J, 5)")')
            prism_exists = js('return ggbApplet.exists("prism1")')
            res.add("3d_prism", prism_exists is True,
                    f"evalCommand={ok_prism}, exists={prism_exists}")

            ok_pyr = js(
                'return ggbApplet.evalCommand("pyr1 = Pyramid(G, H, I, J, 6)")')
            pyr_exists = js('return ggbApplet.exists("pyr1")')
            res.add("3d_pyramid", pyr_exists is True,
                    f"evalCommand={ok_pyr}, exists={pyr_exists}")

            # Volume query
            vol = js('return ggbApplet.getValue("prism1")')
            # base area = 3*3 = 9, height = 5 → volume = 45
            res.add("3d_volume", vol is not None and abs(vol - 45) < 0.5,
                    f"prism volume={vol} (expected 45)")

            # Cone
            ok_cone = js(
                'return ggbApplet.evalCommand("cone1 = Cone(G, D, 2)")')
            cone_exists = js('return ggbApplet.exists("cone1")')
            res.add("3d_cone", cone_exists is True,
                    f"exists={cone_exists}")

            # IntersectPath (cross-section)
            ok_sec = js(
                'return ggbApplet.evalCommand("csec = IntersectPath(prism1, pl)")')
            sec_exists = js('return ggbApplet.exists("csec")')
            res.add("3d_cross_section", sec_exists is True,
                    f"IntersectPath(prism, plane) exists={sec_exists}")

            # Net (unfolded surface)
            ok_net = js('return ggbApplet.evalCommand("net1 = Net(prism1, 1)")')
            net_exists = js('return ggbApplet.exists("net1")')
            res.add("3d_net", net_exists is True,
                    f"Net(prism) exists={net_exists}")

        t_total = time.perf_counter() - t0

    except Exception as e:
        res.add("unexpected_error", False, f"{type(e).__name__}: {e}")
        t_total = time.perf_counter() - t0
    finally:
        driver.quit()

    print(f"\n  Summary [{variant}]: {res.passed}/{res.total} passed  "
          f"({t_total:.2f}s total)")
    return res


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if not DEPLOY_JS.exists():
        sys.exit(f"Bundle not found. Run:  python setup.py download_bundle")

    # Allow selecting specific variant(s) via CLI
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(VARIANTS.keys())
    for t in targets:
        if t not in VARIANTS:
            sys.exit(f"Unknown variant: {t}. Choose from {list(VARIANTS.keys())}")

    results = {}
    for variant in targets:
        results[variant] = _run_variant(variant)

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, r in results.items():
        status = "PASS" if r.all_passed else "FAIL"
        print(f"  [{status}] {name:12s}  {r.passed}/{r.total}")
        if not r.all_passed:
            all_ok = False
            for tname, ok, detail in r.tests:
                if not ok:
                    print(f"         FAIL: {tname}  {detail}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
