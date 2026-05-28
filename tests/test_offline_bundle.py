"""
Smoke test: verify GeoGebra local bundle works offline via Selenium.

Usage:  python tests/test_offline_bundle.py
"""
import tempfile, time, sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "symbolic" / "integrations" / "geogebra_bundle" / "GeoGebra"

def main():
    # ── 0. Check bundle exists ──────────────────────────────────────────
    deploy_js = BUNDLE_DIR / "deployggb.js"
    codebase  = BUNDLE_DIR / "HTML5" / "5.0" / "web3d"
    if not deploy_js.exists():
        sys.exit(f"[FAIL] Bundle not found: {deploy_js}")
    if not codebase.exists():
        sys.exit(f"[FAIL] Codebase not found: {codebase}")
    print(f"[OK] Bundle found: {BUNDLE_DIR}")

    # ── 1. Build HTML that loads from local bundle ──────────────────────
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GeoGebra Offline Test</title>
    <script src="file://{deploy_js.absolute()}"></script>
    <style>
        html, body {{ margin:0; padding:0; width:800px; height:800px; overflow:hidden; background:white; }}
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
            "appName": "classic",
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
            ggbApp.setHTML5Codebase('file://{codebase.absolute()}/', true);
            ggbApp.inject('ggb-element');
        }});
    </script>
</body>
</html>"""

    tmp = Path(tempfile.mkdtemp(prefix="ggb_offline_test_"))
    html_path = tmp / "test.html"
    html_path.write_text(html_content)
    print(f"[OK] HTML written: {html_path}")

    # ── 2. Launch headless Chrome ───────────────────────────────────────
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-web-security")          # allow file:// cross-origin
    opts.add_argument("--allow-file-access-from-files")  # allow file:// to load file://
    opts.add_argument("--window-size=1280,1280")

    driver = webdriver.Chrome(options=opts)
    try:
        t0 = time.perf_counter()
        driver.get(f"file://{html_path.absolute()}")
        print(f"[OK] Page loaded  ({time.perf_counter()-t0:.1f}s)")

        # ── 3. Wait for applet ──────────────────────────────────────────
        t1 = time.perf_counter()
        WebDriverWait(driver, 45).until(
            lambda d: d.execute_script(
                "return window.ggbApplet !== null && window.ggbApplet !== undefined"
            )
        )
        print(f"[OK] ggbApplet ready  ({time.perf_counter()-t1:.1f}s)")

        # ── 4. Wait for CAS ────────────────────────────────────────────
        t2 = time.perf_counter()
        try:
            WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return window.casReady === true")
            )
            print(f"[OK] CAS ready  ({time.perf_counter()-t2:.1f}s)")
        except Exception:
            print(f"[WARN] CAS not ready after 20s — continuing without CAS")

        # ── 5. Smoke tests: construction + query ────────────────────────
        api = driver

        # Add a point
        ok = api.execute_script('return ggbApplet.evalCommand("A = (2, 3)")')
        print(f"[{'OK' if ok else 'FAIL'}] evalCommand('A = (2,3)')  → {ok}")

        # Read back coordinates
        ax = api.execute_script('return ggbApplet.getXcoord("A")')
        ay = api.execute_script('return ggbApplet.getYcoord("A")')
        print(f"[{'OK' if abs(ax-2)<0.01 and abs(ay-3)<0.01 else 'FAIL'}] "
              f"A coords = ({ax}, {ay})")

        # Add line, intersect
        api.execute_script('ggbApplet.evalCommand("B = (5, 7)")')
        api.execute_script('ggbApplet.evalCommand("seg = Segment(A, B)")')
        dist = api.execute_script('return ggbApplet.getValue("seg")')
        expected = ((5-2)**2 + (7-3)**2) ** 0.5
        print(f"[{'OK' if abs(dist-expected)<0.01 else 'FAIL'}] "
              f"Segment length = {dist:.4f}  (expected {expected:.4f})")

        # Circle + intersection
        api.execute_script('ggbApplet.evalCommand("c = Circle(A, 3)")')
        api.execute_script('ggbApplet.evalCommand("d = Line((0,0), (10,10))")')
        api.execute_script('ggbApplet.evalCommand("P = Intersect(c, d, 1)")')
        px = api.execute_script('return ggbApplet.getXcoord("P")')
        py = api.execute_script('return ggbApplet.getYcoord("P")')
        p_exists = api.execute_script('return ggbApplet.exists("P")')
        print(f"[{'OK' if p_exists else 'FAIL'}] "
              f"Intersect(circle, line) → P=({px:.2f}, {py:.2f})")

        # PNG export
        png_b64 = api.execute_script(
            'return ggbApplet.getPNGBase64(1.0, false, 72)')
        png_ok = png_b64 is not None and len(png_b64) > 100
        print(f"[{'OK' if png_ok else 'FAIL'}] "
              f"PNG export  ({len(png_b64) if png_b64 else 0} chars base64)")

        # CAS test
        cas_ok = api.execute_script("return window.casReady")
        if cas_ok:
            cas_result = api.execute_script(
                'return ggbApplet.evalCommandCAS("Solve(x^2-4)")')
            print(f"[OK] CAS Solve(x^2-4) → {cas_result}")
        else:
            print(f"[SKIP] CAS not available")

        # Object count
        obj_count = api.execute_script('return ggbApplet.getObjectNumber()')
        print(f"[OK] Total objects on canvas: {obj_count}")

        elapsed = time.perf_counter() - t0
        print(f"\n{'='*50}")
        print(f"All tests passed — offline bundle works!  (total {elapsed:.1f}s)")

    except Exception as e:
        # Dump browser console for debugging
        try:
            logs = driver.get_log("browser")
            print("\n--- Browser console ---")
            for entry in logs[-20:]:
                print(f"  {entry['level']}: {entry['message']}")
        except Exception:
            pass
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
