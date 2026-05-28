"""
GeoGebra API Integration

Provides an interface to GeoGebra via its JavaScript API, driven by Selenium.
Based on: https://geogebra.github.io/docs/reference/en/GeoGebra_Apps_API/
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import base64
import os
import platform
import shutil
import sys
import tempfile
import subprocess
import time

# ── Local GeoGebra bundle (offline mode) ────────────────────────────────
# Downloaded from https://download.geogebra.org/package/geogebra-math-apps-bundle
# Contains deployggb.js + HTML5/5.0/web3d/ (~115 MB unzipped).
_BUNDLE_DIR = Path(__file__).resolve().parent / "geogebra_bundle" / "GeoGebra"
_BUNDLE_DEPLOY_JS = _BUNDLE_DIR / "deployggb.js"
_BUNDLE_CODEBASE  = _BUNDLE_DIR / "HTML5" / "5.0" / "web3d"
_CFT_ROOT = Path(
    os.environ.get(
        "DRAW2THINK_BROWSER_ROOT",
        Path(__file__).resolve().parents[2] / ".chrome-for-testing",
    )
)


def _cft_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux64"
    if sys.platform == "darwin":
        return "mac-arm64" if platform.machine() == "arm64" else "mac-x64"
    if sys.platform.startswith("win"):
        machine = platform.machine().lower()
        return "win64" if machine in {"amd64", "x86_64", "arm64"} else "win32"
    return ""


def _latest_cft_version_dir() -> Optional[Path]:
    if not _CFT_ROOT.exists():
        return None
    dirs = [path for path in _CFT_ROOT.iterdir() if path.is_dir()]
    return sorted(dirs)[-1] if dirs else None


def _cft_binary_paths(version_dir: Path) -> Tuple[Path, Path]:
    if sys.platform.startswith("linux"):
        return (
            version_dir / "chrome-linux64" / "chrome",
            version_dir / "chromedriver-linux64" / "chromedriver",
        )
    if sys.platform == "darwin":
        plat = _cft_platform()
        return (
            version_dir / f"chrome-{plat}" / "Google Chrome for Testing.app"
            / "Contents" / "MacOS" / "Google Chrome for Testing",
            version_dir / f"chromedriver-{plat}" / "chromedriver",
        )
    plat = _cft_platform()
    return (
        version_dir / f"chrome-{plat}" / "chrome.exe",
        version_dir / f"chromedriver-{plat}" / "chromedriver.exe",
    )


def _active_env_executable(*names: str) -> Optional[str]:
    bin_dir = Path(sys.executable).parent
    for name in names:
        path = bin_dir / name
        if path.exists():
            return str(path)
    return None


def _find_chrome_binary() -> Optional[str]:
    env = os.environ.get("DRAW2THINK_CHROME")
    if env:
        return env
    version_dir = _latest_cft_version_dir()
    if version_dir:
        chrome, _ = _cft_binary_paths(version_dir)
        if chrome.exists():
            return str(chrome)
    active = _active_env_executable(
        "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
        "chrome", "chrome.cmd",
    )
    if active:
        return active
    for name in ("chromium", "chromium-browser", "google-chrome",
                 "google-chrome-stable", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_chromedriver() -> Optional[str]:
    env = os.environ.get("DRAW2THINK_CHROMEDRIVER")
    if env:
        return env
    version_dir = _latest_cft_version_dir()
    if version_dir:
        _, driver = _cft_binary_paths(version_dir)
        if driver.exists():
            return str(driver)
    active = _active_env_executable("chromedriver", "chromedriver.exe", "chromedriver.cmd")
    if active:
        return active
    return shutil.which("chromedriver")


@dataclass
class GeoGebraResult:
    """Result from GeoGebra execution."""
    success: bool
    objects: Dict[str, Any]
    properties: Dict[str, Any]
    error_message: Optional[str] = None
    script_executed: str = ""
    created_labels: List[str] = None

    def __post_init__(self):
        if self.created_labels is None:
            self.created_labels = []


class GeoGebraAPI:
    """
    Interface to GeoGebra via JavaScript API.

    Uses Selenium WebDriver to control GeoGebra web app in headless browser.
    This provides full access to the GeoGebra Apps API.

    Alternative modes:
    1. Selenium (recommended): Full API access via browser automation
    2. HTTP Server: Local server with GeoGebra embedded
    3. Desktop App: Via command-line interface (limited)
    """

    def __init__(
        self,
        mode: str = "selenium",
        headless: bool = True,
        timeout: float = 10.0,
        enable_3d: bool = False,
        textbook_style: bool = False,
    ):
        """
        Initialize GeoGebra API.

        Args:
            mode: 'selenium', 'http', or 'desktop'
            headless: Run browser in headless mode (for selenium)
            timeout: Timeout for operations in seconds
            enable_3d: Activate 3D Graphics View (perspective "T").
                       When False (default), only the 2D Geometry view is shown.
                       The web3d engine is always loaded (backward-compatible),
                       but 3D solids (Prism, Pyramid) require this flag.
            textbook_style: When True, apply a textbook-grade rendering style
                       (black-and-white, thin 1px strokes, hidden axes/grid,
                       no polygon fill, italic NAME labels). Default False
                       preserves the standard GeoGebra appearance. The same
                       effect is reachable at runtime via apply_textbook_style().
        """
        self.mode = mode
        self.headless = headless
        self.timeout = timeout
        self.enable_3d = enable_3d
        self.textbook_style = textbook_style

        self._driver = None
        self._ggb_api = None
        self._temp_dir = Path(tempfile.mkdtemp(prefix="geogebra_"))
        self._textbook_style_applied = False

    def _execute_js(self, script: str):
        """Execute raw JavaScript on the embedded GeoGebra applet."""
        return self._driver.execute_script(script)

    def _call_api(self, method: str, *args):
        """
        Call ggbApplet.<method>(...args) safely.

        Returns:
            Value returned by the underlying JS call, or None on error.
        """
        try:
            args_json = ", ".join(json.dumps(arg) for arg in args)
            return self._execute_js(f"return ggbApplet.{method}({args_json})")
        except Exception:
            return None

    def _call_api_void(self, method: str, *args) -> bool:
        """Call a void API method (setColor, setLineStyle, etc.).

        These JS methods return undefined, so we cannot use the return value
        to determine success.  Instead, success = no exception.
        """
        try:
            args_json = ", ".join(json.dumps(arg) for arg in args)
            self._execute_js(f"ggbApplet.{method}({args_json})")
            return True
        except Exception:
            return False

    def _call_api_bool(self, method: str, *args) -> bool:
        """Call API method and cast result to bool."""
        result = self._call_api(method, *args)
        return bool(result) if result is not None else False

    def initialize(self):
        """Initialize GeoGebra connection."""
        if self.mode == "selenium":
            self._init_selenium()
        elif self.mode == "http":
            self._init_http_server()
        elif self.mode == "desktop":
            self._init_desktop()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _init_selenium(self):
        """Initialize Selenium WebDriver with GeoGebra."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError:
            raise RuntimeError(
                "Selenium not installed. Install with: pip install selenium"
            )

        # Set up Chrome options
        chrome_options = Options()
        chrome_binary = _find_chrome_binary()
        if chrome_binary:
            chrome_options.binary_location = chrome_binary
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,1280")
        # Allow file:// pages to load other file:// resources (offline bundle)
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-file-access-from-files")

        # Create HTML page with GeoGebra embedded
        html_path = self._create_geogebra_html()

        # Start WebDriver
        driver_path = _find_chromedriver()
        if driver_path:
            service = Service(driver_path)
        else:
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
            except ImportError:
                service = None

        if service:
            self._driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self._driver = webdriver.Chrome(options=chrome_options)

        self._driver.get(f"file://{html_path.absolute()}")

        # Wait for GeoGebra appletOnLoad callback to fire
        WebDriverWait(self._driver, 30).until(
            lambda d: d.execute_script(
                "return window.ggbApplet !== null && window.ggbApplet !== undefined"
            )
        )

        # Wait for CAS engine (loads asynchronously, typically ~1-3s)
        try:
            WebDriverWait(self._driver, 15).until(
                lambda d: d.execute_script("return window.casReady === true")
            )
        except Exception:
            print("  [WARN] CAS engine not ready after 15s — Solve/NSolve will return '?'")

        # Suppress GeoGebra error dialogs — errors are already captured by
        # the tool execution layer; popups would only occlude the canvas PNG.
        self._execute_js("ggbApplet.setErrorDialogsActive(false)")

        # 3D mode defaults: black points (smaller), thinner lines, reduced face opacity,
        # and auto-hide labels for auto-generated sub-objects (edges, faces).
        if self.enable_3d:
            self._execute_js("""
                ggbApplet.registerAddListener(function(name) {
                    var t = ggbApplet.getObjectType(name);
                    if (t === 'point') {
                        ggbApplet.setColor(name, 0, 0, 0);
                        ggbApplet.setPointSize(name, 3);
                        ggbApplet.setLabelVisible(name, true);
                    }
                    if (t === 'segment' || t === 'line' || t === 'ray' || t === 'vector') {
                        ggbApplet.setLineThickness(name, 2);
                        // Hide labels on auto-generated edges (edge*, face* prefixes)
                        if (name.indexOf('edge') === 0) {
                            ggbApplet.setLabelVisible(name, false);
                        }
                    }
                    if (t === 'triangle' || t === 'quadrilateral' || t === 'polygon') {
                        var cur = ggbApplet.getFilling(name);
                        ggbApplet.setFilling(name, cur * 0.25);
                        // Hide labels on auto-generated faces
                        if (name.indexOf('face') === 0) {
                            ggbApplet.setLabelVisible(name, false);
                        }
                    }
                    // Hide labels on auto-generated solids (prism, cube, pyramid names)
                    if (t === 'prism' || t === 'cube' || t === 'pyramid' || t === 'tetrahedron'
                        || t === 'sphere' || t === 'cylinder' || t === 'cone') {
                        ggbApplet.setLabelVisible(name, false);
                    }
                });
            """)

        # Optional textbook-grade visual style (default off).
        if self.textbook_style:
            self.apply_textbook_style()

        # Display rounding stays at GeoGebra default (2 decimal places)
        # for clean canvas labels. Full precision is injected at the
        # build_rich_canvas level for numeric/angle objects.
        print("GeoGebra initialized successfully via Selenium")

    def _create_geogebra_html(self) -> Path:
        """Create HTML page with GeoGebra embedded.

        Prefers local bundle (offline, faster) if available under
        ``symbolic/integrations/geogebra_bundle/GeoGebra/``.
        Falls back to CDN when the bundle is absent.
        """
        # ── Decide: local bundle vs CDN ─────────────────────────────────
        if _BUNDLE_DEPLOY_JS.exists() and _BUNDLE_CODEBASE.exists():
            script_src = f"file://{_BUNDLE_DEPLOY_JS.absolute()}"
            codebase_line = (
                f"ggbApp.setHTML5Codebase("
                f"'file://{_BUNDLE_CODEBASE.absolute()}/', true);")
            print("  [GGB] Using LOCAL bundle (offline)")
        else:
            script_src = "https://www.geogebra.org/apps/deployggb.js"
            codebase_line = "// codebase: CDN default"
            print("  [GGB] Using CDN (online)")

        # ── Perspective: "G" = 2D Geometry, "T" = 3D Graphics ────────────
        perspective = "T" if self.enable_3d else "G"
        if self.enable_3d:
            print("  [GGB] 3D perspective enabled")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GeoGebra API</title>
    <script src="{script_src}"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 800px;
            height: 800px;
            overflow: hidden;
            background: white;
        }}
        #ggb-element {{
            width: 800px;
            height: 800px;
        }}
    </style>
</head>
<body>
    <div id="ggb-element"></div>
    <script>
        window.ggbApplet = null;
        window.casReady = false;

        function appletOnLoad(api) {{
            window.ggbApplet = api;
            // CAS engine loads asynchronously — poll until ready
            function checkCAS() {{
                try {{
                    var r = api.evalCommandCAS("1+1");
                    if (r && r !== "?" && r !== "") {{
                        window.casReady = true;
                        return;
                    }}
                }} catch(e) {{}}
                setTimeout(checkCAS, 200);
            }}
            checkCAS();
        }}

        var ggbApp = new GGBApplet({{
            "appName": "classic",
            "perspective": "{perspective}",
            "width": 800,
            "height": 800,
            "fontSize": {24 if self.enable_3d else 18},
            "showToolBar": false,
            "showToolBarHelp": false,
            "showAlgebraView": false,
            "showAlgebraInput": false,
            "showMenuBar": false,
            "showAnimationButton": false,
            "showFullscreenButton": false,
            "showZoomButtons": false,
            "enableRightClick": false,
            "enableShiftDragZoom": true,
            "showResetIcon": false,
            "appletOnLoad": appletOnLoad
        }}, true);

        window.addEventListener("load", function() {{
            {codebase_line}
            ggbApp.inject('ggb-element');
            // Force a square Euclidian view export surface.
            setTimeout(function() {{
                if (window.ggbApplet && window.ggbApplet.setSize) {{
                    window.ggbApplet.setSize(800, 800);
                }}
                // Defensive: hide any remaining side panel in new UI variants.
                var candidates = document.querySelectorAll('[class*="algebra"], [class*="sidebar"], [data-test*="algebra"]');
                candidates.forEach(function(el) {{
                    if (el && el.style) {{
                        el.style.display = "none";
                        el.style.width = "0px";
                        el.style.minWidth = "0px";
                    }}
                }});
            }}, 50);
        }});
    </script>
</body>
</html>"""
        html_path = self._temp_dir / "geogebra.html"
        html_path.write_text(html_content)
        return html_path

    def _init_http_server(self):
        """Initialize local HTTP server with GeoGebra."""
        # TODO: Implement HTTP server mode
        raise NotImplementedError("HTTP server mode not yet implemented")

    def _init_desktop(self):
        """Initialize GeoGebra desktop app."""
        # TODO: Implement desktop app mode
        raise NotImplementedError("Desktop mode not yet implemented")

    def eval_command(self, command: str) -> GeoGebraResult:
        """
        Execute a GeoGebra command.

        Args:
            command: GeoGebra command (e.g., "A = (1, 2)")

        Returns:
            GeoGebraResult with execution status
        """
        try:
            # Execute command
            js_code = f'ggbApplet.evalCommand("{self._escape_js(command)}")'
            success = self._driver.execute_script(f"return {js_code}")

            return GeoGebraResult(
                success=bool(success),
                objects={},
                properties={},
                script_executed=command
            )
        except Exception as e:
            return GeoGebraResult(
                success=False,
                objects={},
                properties={},
                error_message=str(e),
                script_executed=command
            )

    def eval_command_get_labels(self, command: str) -> Tuple[bool, List[str]]:
        """
        Execute command and get labels of created objects.

        Args:
            command: GeoGebra command

        Returns:
            (success, list of created object labels)
        """
        try:
            js_code = f'ggbApplet.evalCommandGetLabels("{self._escape_js(command)}")'
            result = self._driver.execute_script(f"return {js_code}")

            if result:
                labels = result.split(",") if result else []
                return True, labels
            return False, []
        except Exception as e:
            return False, []

    def eval_commands(self, commands: List[str]) -> Tuple[int, int, List[Tuple[str, str]]]:
        """
        Execute multiple commands sequentially.

        Returns:
            (success_count, total_count, failed_items[(command, error_message)])
        """
        success = 0
        failed: List[Tuple[str, str]] = []
        for cmd in commands:
            result = self.eval_command(cmd)
            if result.success:
                success += 1
            else:
                failed.append((cmd, result.error_message or "unknown error"))
        return success, len(commands), failed

    def eval_latex(self, latex_input: str) -> bool:
        """Evaluate LaTeX to a construction element."""
        return self._call_api_bool("evalLaTeX", latex_input)

    def eval_cas(self, cas_input: str) -> Optional[str]:
        """Evaluate a CAS expression and return text result."""
        result = self._call_api("evalCommandCAS", cas_input)
        return str(result) if result is not None else None

    # -------------------- Object state setters --------------------

    def delete_object(self, obj_name: str) -> bool:
        """Delete an object by name."""
        result = self._call_api("deleteObject", obj_name)
        return result is not None

    def set_auxiliary(self, obj_name: str, is_auxiliary: bool) -> bool:
        """Set auxiliary flag for an object."""
        result = self._call_api("setAuxiliary", obj_name, bool(is_auxiliary))
        return result is not None

    def set_value(self, obj_name: str, value: float) -> bool:
        """Set numeric/boolean value of an object."""
        result = self._call_api("setValue", obj_name, value)
        return result is not None

    def set_text_value(self, obj_name: str, value: str) -> bool:
        """Set text value of an object."""
        result = self._call_api("setTextValue", obj_name, value)
        return result is not None

    def set_list_value(self, obj_name: str, index: int, value: float) -> bool:
        """Set list element at index to value."""
        result = self._call_api("setListValue", obj_name, int(index), value)
        return result is not None

    def set_caption(self, obj_name: str, caption: str) -> bool:
        """Set object caption."""
        return self._call_api_void("setCaption", obj_name, caption)

    def set_color(self, obj_name: str, red: int, green: int, blue: int) -> bool:
        """Set object RGB color."""
        return self._call_api_void("setColor", obj_name, int(red), int(green), int(blue))

    def set_label_style(self, obj_name: str, style: int) -> bool:
        """
        Set label style:
          0 NAME, 1 NAME_VALUE, 2 VALUE, 3 CAPTION
        """
        return self._call_api_void("setLabelStyle", obj_name, int(style))

    def set_fixed(self, obj_name: str, fixed: bool, selection_allowed: bool = True) -> bool:
        """Set object fixed state and selection allowed flag."""
        result = self._call_api("setFixed", obj_name, bool(fixed), bool(selection_allowed))
        return result is not None

    def set_trace(self, obj_name: str, enabled: bool) -> bool:
        """Enable/disable trace for object."""
        result = self._call_api("setTrace", obj_name, bool(enabled))
        return result is not None

    def rename_object(self, old_name: str, new_name: str) -> bool:
        """Rename object, returns success."""
        return self._call_api_bool("renameObject", old_name, new_name)

    def set_layer(self, obj_name: str, layer: int) -> bool:
        """Set object layer."""
        result = self._call_api("setLayer", obj_name, int(layer))
        return result is not None

    def set_layer_visible(self, layer: int, visible: bool) -> bool:
        """Show/hide all objects in a layer."""
        result = self._call_api("setLayerVisible", int(layer), bool(visible))
        return result is not None

    def set_line_style(self, obj_name: str, style: int) -> bool:
        """Set line style (0..4)."""
        return self._call_api_void("setLineStyle", obj_name, int(style))

    def set_line_thickness(self, obj_name: str, thickness: int) -> bool:
        """Set line thickness (1..13, -1 default)."""
        return self._call_api_void("setLineThickness", obj_name, int(thickness))

    def set_point_style(self, obj_name: str, style: int) -> bool:
        """Set point style."""
        return self._call_api_void("setPointStyle", obj_name, int(style))

    def set_point_size(self, obj_name: str, size: int) -> bool:
        """Set point size (1..9)."""
        return self._call_api_void("setPointSize", obj_name, int(size))

    def set_filling(self, obj_name: str, filling: float) -> bool:
        """Set filling [0, 1] for object."""
        return self._call_api_void("setFilling", obj_name, float(filling))

    # -------------------- Animation --------------------

    def set_animating(self, obj_name: str, animate: bool) -> bool:
        """Set whether object should animate."""
        result = self._call_api("setAnimating", obj_name, bool(animate))
        return result is not None

    def set_animation_speed(self, obj_name: str, speed: float) -> bool:
        """Set animation speed for object."""
        result = self._call_api("setAnimationSpeed", obj_name, float(speed))
        return result is not None

    def start_animation(self) -> bool:
        """Start animation for all animating objects."""
        result = self._call_api("startAnimation")
        return result is not None

    def stop_animation(self) -> bool:
        """Stop animation."""
        result = self._call_api("stopAnimation")
        return result is not None

    def is_animation_running(self) -> bool:
        """Return whether animation is running."""
        return self._call_api_bool("isAnimationRunning")

    def get_all_object_names(self, obj_type: Optional[str] = None) -> List[str]:
        """
        Get all object names, optionally filtered by type.

        Args:
            obj_type: Filter by type ("point", "line", "circle", etc.)

        Returns:
            List of object names
        """
        try:
            if obj_type:
                js_code = f'ggbApplet.getAllObjectNames("{obj_type}")'
            else:
                js_code = 'ggbApplet.getAllObjectNames()'

            result = self._driver.execute_script(f"return {js_code}")
            return result if result else []
        except Exception:
            return []

    def get_object_type(self, obj_name: str) -> Optional[str]:
        """Get the type of an object."""
        try:
            js_code = f'ggbApplet.getObjectType("{obj_name}")'
            return self._driver.execute_script(f"return {js_code}")
        except Exception:
            return None

    def get_value(self, obj_name: str) -> Optional[float]:
        """Get numeric value of an object."""
        try:
            js_code = f'ggbApplet.getValue("{obj_name}")'
            return self._driver.execute_script(f"return {js_code}")
        except Exception:
            return None

    def get_list_value(self, obj_name: str, index: int) -> Optional[float]:
        """Get numeric value at list index."""
        result = self._call_api("getListValue", obj_name, int(index))
        return float(result) if result is not None else None

    def get_color(self, obj_name: str) -> Optional[str]:
        """Get object color as hex string."""
        result = self._call_api("getColor", obj_name)
        return str(result) if result is not None else None

    def get_visible(self, obj_name: str, view: Optional[int] = None) -> bool:
        """Get object visibility (optionally for a specific view)."""
        if view is None:
            return self._call_api_bool("getVisible", obj_name)
        return self._call_api_bool("getVisible", obj_name, int(view))

    def get_coords(self, obj_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates of a point."""
        try:
            x = self._execute_js(f'return ggbApplet.getXcoord("{obj_name}")')
            y = self._execute_js(f'return ggbApplet.getYcoord("{obj_name}")')
            return (x, y) if x is not None and y is not None else None
        except Exception:
            return None

    def get_coords_3d(self, obj_name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D coordinates of a point/vector if available."""
        try:
            x = self._call_api("getXcoord", obj_name)
            y = self._call_api("getYcoord", obj_name)
            z = self._call_api("getZcoord", obj_name)
            if x is None or y is None or z is None:
                return None
            return float(x), float(y), float(z)
        except Exception:
            return None

    def set_coords(self, obj_name: str, x: float, y: float) -> bool:
        """Set coordinates of a point."""
        try:
            js_code = f'ggbApplet.setCoords("{obj_name}", {x}, {y})'
            self._driver.execute_script(js_code)
            return True
        except Exception:
            return False

    def get_value_string(self, obj_name: str) -> Optional[str]:
        """Get string representation of object value."""
        try:
            js_code = f'ggbApplet.getValueString("{obj_name}")'
            return self._driver.execute_script(f"return {js_code}")
        except Exception:
            return None

    def get_definition_string(self, obj_name: str) -> Optional[str]:
        """Get object definition string."""
        result = self._call_api("getDefinitionString", obj_name)
        return str(result) if result is not None else None

    def get_command_string(self, obj_name: str, use_localized_input: bool = False) -> Optional[str]:
        """Get command string for object."""
        result = self._call_api("getCommandString", obj_name, bool(use_localized_input))
        return str(result) if result is not None else None

    def get_latex_string(self, obj_name: str) -> Optional[str]:
        """Get LaTeX string representation of object value."""
        result = self._call_api("getLaTeXString", obj_name)
        return str(result) if result is not None else None

    def exists(self, obj_name: str) -> bool:
        """Check if object exists."""
        try:
            js_code = f'ggbApplet.exists("{obj_name}")'
            result = self._driver.execute_script(f"return {js_code}")
            return bool(result)
        except Exception:
            return False

    def is_defined(self, obj_name: str) -> bool:
        """Check if object is properly defined."""
        try:
            js_code = f'ggbApplet.isDefined("{obj_name}")'
            result = self._driver.execute_script(f"return {js_code}")
            return bool(result)
        except Exception:
            return False

    def is_independent(self, obj_name: str) -> bool:
        """Check if object is independent."""
        return self._call_api_bool("isIndependent", obj_name)

    def is_moveable(self, obj_name: str) -> bool:
        """Check if object is moveable."""
        return self._call_api_bool("isMoveable", obj_name)

    # ── Dependency graph helpers ──────────────────────────────────────

    def get_direct_dependents(self, obj_name: str) -> List[str]:
        """Return names of objects whose definition directly references *obj_name*.

        Iterates all objects and checks whether *obj_name* appears in their
        command string (the GeoGebra construction command, e.g. ``Midpoint(A, B)``).
        Only **direct** (depth-1) dependents are returned.
        """
        dependents: List[str] = []
        all_names = self.get_all_object_names()
        for name in all_names:
            if name == obj_name:
                continue
            if self.is_independent(name):
                continue
            cmd = self.get_command_string(name)
            if cmd and obj_name in cmd:
                # Exact word-boundary check to avoid "A" matching "AB"
                import re
                if re.search(rf'\b{re.escape(obj_name)}\b', cmd):
                    dependents.append(name)
        return dependents

    def get_dependency_tree(self, obj_name: str) -> Dict[str, list]:
        """Return the full hierarchical dependency tree rooted at *obj_name*.

        Returns a nested dict::

            {
                "circle_O": {
                    "tangent_1": {},
                    "arc_AB": {
                        "len_arc": {}
                    }
                }
            }

        Each key is an object that directly depends on its parent key.
        Leaves (objects with no further dependents) map to ``{}``.
        """
        visited: set = set()

        def _build(name: str) -> Dict[str, dict]:
            if name in visited:
                return {}
            visited.add(name)
            children = self.get_direct_dependents(name)
            return {child: _build(child) for child in children}

        return {obj_name: _build(obj_name)}

    def get_all_dependents(self, obj_name: str) -> List[str]:
        """Return a flat list of ALL objects that transitively depend on *obj_name*.

        Uses BFS; the result is ordered breadth-first (direct dependents first,
        then their dependents, etc.).  This is exactly the set of objects that
        would be cascade-deleted by ``Delete(obj_name)``.
        """
        from collections import deque
        visited: set = set()
        queue: deque = deque([obj_name])
        result: List[str] = []
        while queue:
            current = queue.popleft()
            for dep in self.get_direct_dependents(current):
                if dep not in visited:
                    visited.add(dep)
                    result.append(dep)
                    queue.append(dep)
        return result

    def get_parents(self, obj_name: str) -> List[str]:
        """Return names of objects that *obj_name* directly depends on.

        Parses the command string of *obj_name* and finds referenced object
        names that exist in the construction.
        """
        cmd = self.get_command_string(obj_name)
        if not cmd:
            return []
        import re
        all_names = set(self.get_all_object_names())
        # Extract candidate tokens from command arguments
        # e.g. "Midpoint(A, B)" → ["A", "B"]
        # Strip the command name, get content inside outermost parentheses
        inner = re.search(r'\((.+)\)', cmd)
        if not inner:
            return []
        tokens = [t.strip() for t in inner.group(1).split(',')]
        parents = []
        for token in tokens:
            # token may be a name or a numeric literal; only keep existing names
            clean = token.strip()
            if clean in all_names and clean != obj_name:
                parents.append(clean)
        return parents

    def print_dependency_tree(self, obj_name: str) -> str:
        """Return a human-readable tree string for the dependency hierarchy.

        Example output::

            A (point)
            ├── seg_AB (segment)
            │   └── M (point)
            └── circ_O (circle)
                └── tangent_1 (line)
        """
        lines: List[str] = []

        def _walk(name: str, prefix: str, is_last: bool, visited: set):
            connector = "└── " if is_last else "├── "
            obj_type = self.get_object_type(name) or "?"
            if prefix == "":
                lines.append(f"{name} ({obj_type})")
            else:
                lines.append(f"{prefix}{connector}{name} ({obj_type})")

            if name in visited:
                return
            visited.add(name)

            children = self.get_direct_dependents(name)
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                _walk(child, child_prefix if prefix != "" else "",
                      i == len(children) - 1, visited)

        _walk(obj_name, "", True, set())
        return "\n".join(lines)

    def get_object_number(self) -> Optional[int]:
        """Get number of objects in construction."""
        result = self._call_api("getObjectNumber")
        return int(result) if result is not None else None

    def get_object_name(self, index: int) -> Optional[str]:
        """Get object name by index."""
        result = self._call_api("getObjectName", int(index))
        return str(result) if result is not None else None

    def get_layer(self, obj_name: str) -> Optional[int]:
        """Get layer of object."""
        result = self._call_api("getLayer", obj_name)
        try:
            return int(result) if result is not None else None
        except Exception:
            return None

    def get_line_style(self, obj_name: str) -> Optional[int]:
        """Get line style."""
        result = self._call_api("getLineStyle", obj_name)
        return int(result) if result is not None else None

    def get_line_thickness(self, obj_name: str) -> Optional[int]:
        """Get line thickness."""
        result = self._call_api("getLineThickness", obj_name)
        return int(result) if result is not None else None

    def get_point_style(self, obj_name: str) -> Optional[int]:
        """Get point style."""
        result = self._call_api("getPointStyle", obj_name)
        return int(result) if result is not None else None

    def get_point_size(self, obj_name: str) -> Optional[int]:
        """Get point size."""
        result = self._call_api("getPointSize", obj_name)
        return int(result) if result is not None else None

    def get_filling(self, obj_name: str) -> Optional[float]:
        """Get filling [0, 1] value."""
        result = self._call_api("getFilling", obj_name)
        return float(result) if result is not None else None

    def get_png_base64(
        self,
        scale: float = 1.0,
        transparent: bool = False,
        dpi: float = 72.0
    ) -> Optional[str]:
        """
        Export current view as PNG (base64 encoded).

        Args:
            scale: Export scale factor
            transparent: Transparent background
            dpi: DPI resolution

        Returns:
            Base64 encoded PNG string
        """
        try:
            # Re-assert square output surface before export.
            self._execute_js(
                "if (window.ggbApplet && window.ggbApplet.setSize) { ggbApplet.setSize(800, 800); }"
            )
            js_code = f'ggbApplet.getPNGBase64({scale}, {str(transparent).lower()}, {dpi})'
            return self._driver.execute_script(f"return {js_code}")
        except Exception as e:
            print(f"Error exporting PNG: {e}")
            return None

    def get_screenshot_base64(self, timeout_ms: int = 10000) -> Optional[str]:
        """
        Get screenshot of whole applet as base64 PNG.

        Uses callback-based API via Selenium async script.
        """
        try:
            self._driver.set_script_timeout(timeout_ms / 1000.0)
            script = """
                const done = arguments[0];
                try {
                    ggbApplet.getScreenshotBase64(function(url){
                        done(url || null);
                    });
                } catch (e) {
                    done(null);
                }
            """
            result = self._driver.execute_async_script(script)
            return result if result else None
        except Exception:
            return None

    def set_coord_system(self, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
        """Set 2D coordinate system bounds."""
        return self._call_api_void("setCoordSystem", float(xmin), float(xmax), float(ymin), float(ymax))

    def export_png(
        self,
        output_path: Path,
        scale: float = 1.0,
        transparent: bool = False,
        dpi: float = 72.0
    ) -> bool:
        """
        Export current view to PNG file.

        Args:
            output_path: Where to save PNG
            scale: Export scale factor
            transparent: Transparent background
            dpi: DPI resolution

        Returns:
            True if successful
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer element screenshot for predictable, square canvas exports.
        # GeoGebra getPNGBase64() may return internal view bounds (non-square).
        try:
            elem = self._driver.find_element("id", "ggb-element")
            output_path.write_bytes(elem.screenshot_as_png)
            return True
        except Exception:
            pass

        # Fallback to GeoGebra API export.
        png_data = self.get_png_base64(scale, transparent, dpi)
        if not png_data:
            return False

        if png_data.startswith("data:image/png;base64,"):
            png_data = png_data.split(",", 1)[1]

        output_path.write_bytes(base64.b64decode(png_data))
        return True

    def get_xml(self) -> Optional[str]:
        """Get current construction as XML."""
        try:
            js_code = 'ggbApplet.getXML()'
            return self._driver.execute_script(f"return {js_code}")
        except Exception:
            return None

    def set_xml(self, xml: str) -> bool:
        """Load construction from XML (clears current)."""
        try:
            escaped_xml = self._escape_js(xml)
            js_code = f'ggbApplet.setXML("{escaped_xml}")'
            self._driver.execute_script(js_code)
            return True
        except Exception:
            return False

    def reset(self):
        """Reset construction to empty state."""
        try:
            self._execute_js('ggbApplet.reset()')
        except Exception:
            pass

    def new_construction(self) -> bool:
        """Clear all construction objects."""
        result = self._call_api("newConstruction")
        return result is not None

    def refresh_views(self) -> bool:
        """Refresh all views (clears traces)."""
        result = self._call_api("refreshViews")
        return result is not None

    def show_all_objects(self) -> bool:
        """Auto-adjust view to include visible objects."""
        result = self._call_api("showAllObjects")
        return result is not None

    def set_mode(self, mode: int) -> bool:
        """Set current toolbar mode."""
        result = self._call_api("setMode", int(mode))
        return result is not None

    def get_mode(self) -> Optional[int]:
        """Get current toolbar mode."""
        result = self._call_api("getMode")
        return int(result) if result is not None else None

    def set_perspective(self, perspective: str) -> bool:
        """Set perspective string (e.g., 'G' for Geometry)."""
        result = self._call_api("setPerspective", perspective)
        return result is not None

    def set_error_dialogs_active(self, enabled: bool) -> bool:
        """Enable/disable GeoGebra error dialogs."""
        result = self._call_api("setErrorDialogsActive", bool(enabled))
        return result is not None

    def set_repainting_active(self, enabled: bool) -> bool:
        """Enable/disable repainting for batch updates."""
        result = self._call_api("setRepaintingActive", bool(enabled))
        return result is not None

    def set_undo_point(self) -> bool:
        """Create an undo point."""
        result = self._call_api("setUndoPoint")
        return result is not None

    def undo(self) -> bool:
        """Undo one step."""
        result = self._call_api("undo")
        return result is not None

    def redo(self) -> bool:
        """Redo one step."""
        result = self._call_api("redo")
        return result is not None

    def set_grid_visible(self, visible: bool, view_number: Optional[int] = None) -> bool:
        """Show/hide grid globally or in a specific view."""
        if view_number is None:
            return self._call_api_void("setGridVisible", bool(visible))
        return self._call_api_void("setGridVisible", int(view_number), bool(visible))

    def get_grid_visible(self, view_number: Optional[int] = None) -> bool:
        """Return whether grid is visible in view."""
        if view_number is None:
            return self._call_api_bool("getGridVisible")
        return self._call_api_bool("getGridVisible", int(view_number))

    def set_axes_visible(
        self,
        x_axis: bool,
        y_axis: bool,
        view_number: Optional[int] = None,
        z_axis: Optional[bool] = None,
    ) -> bool:
        """
        Show/hide axes.

        - 2D usage: set_axes_visible(True, True)
        - View-specific usage: set_axes_visible(True, True, view_number=1)
        - 3D usage: set_axes_visible(True, True, view_number=3, z_axis=True)
        """
        if view_number is None:
            return self._call_api_void("setAxesVisible", bool(x_axis), bool(y_axis))

        if z_axis is None:
            z_axis = True
        return self._call_api_void(
            "setAxesVisible", int(view_number), bool(x_axis), bool(y_axis), bool(z_axis)
        )

    def set_size(self, width: int, height: int) -> bool:
        """Set applet size in pixels."""
        result = self._call_api("setSize", int(width), int(height))
        return result is not None

    def apply_textbook_style(self) -> None:
        """Switch the canvas to a textbook-grade rendering style.

        Mirrors the visual conventions of printed math textbooks (e.g.
        the We-Math 2.0 dataset, arXiv:2508.10433): pure black-and-white
        objects, thin 1px strokes, hidden coordinate axes and grid,
        unfilled polygons, italic NAME labels, and dashed hidden edges
        for 3D solids (the latter is GeoGebra's own default for the 3D
        view).

        This is opt-in (set ``textbook_style=True`` at construction or
        call this method later). Existing canvases keep the GeoGebra
        default styling unless this is invoked. Idempotent.
        """
        if self._textbook_style_applied:
            return

        # Hide axes & grid for the 2D Euclidian view.
        self._call_api_void("setAxesVisible", False, False)
        self._call_api_void("setGridVisible", False)
        # And for the 3D view when present (view 3 in GeoGebra's view ids).
        if self.enable_3d:
            self._call_api_void("setAxesVisible", 3, False, False, False)
            self._call_api_void("setGridVisible", 3, False)

        # Repaint every newly added object in textbook style. We attach
        # the listener as a property so a future invocation can locate
        # and skip re-registering it.
        self._execute_js("""
            ggbApplet.__textbookListener = function(name) {
                var t = ggbApplet.getObjectType(name);
                if (t === 'point') {
                    ggbApplet.setColor(name, 0, 0, 0);
                    ggbApplet.setPointSize(name, 3);
                    ggbApplet.setPointStyle(name, 0);
                    ggbApplet.setLabelVisible(name, true);
                    ggbApplet.setLabelStyle(name, 0);
                } else if (t === 'segment' || t === 'line' || t === 'ray' ||
                           t === 'vector' || t === 'conic' || t === 'circle' ||
                           t === 'arc' || t === 'angle') {
                    ggbApplet.setColor(name, 0, 0, 0);
                    ggbApplet.setLineThickness(name, 1);
                } else if (t === 'triangle' || t === 'quadrilateral' ||
                           t === 'polygon' || t === 'polyline') {
                    ggbApplet.setColor(name, 0, 0, 0);
                    ggbApplet.setLineThickness(name, 1);
                    ggbApplet.setFilling(name, 0);
                } else if (t === 'prism' || t === 'cube' || t === 'pyramid' ||
                           t === 'tetrahedron' || t === 'sphere' ||
                           t === 'cylinder' || t === 'cone') {
                    ggbApplet.setColor(name, 0, 0, 0);
                    ggbApplet.setFilling(name, 0);
                    ggbApplet.setLabelVisible(name, false);
                }
            };
            ggbApplet.registerAddListener(ggbApplet.__textbookListener);
        """)

        # Apply once to objects already on the canvas (the listener only
        # fires for additions made after registration).
        self._execute_js("""
            ggbApplet.getAllObjectNames().forEach(function(n) {
                ggbApplet.__textbookListener(n);
            });
        """)
        self._textbook_style_applied = True

    def get_view_properties(self, view_id: int = 1) -> Optional[Dict[str, Any]]:
        """Get view properties JSON (width, height, scales, bounds)."""
        raw = self._call_api("getViewProperties", int(view_id))
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_object_visible(self, obj_name: str, visible: bool) -> bool:
        """Show or hide a geometric object."""
        try:
            value = "true" if visible else "false"
            self._execute_js(f'ggbApplet.setVisible("{obj_name}", {value})')
            return True
        except Exception:
            return False

    def set_label_visible(self, obj_name: str, visible: bool) -> bool:
        """Show or hide an object's label."""
        try:
            value = "true" if visible else "false"
            self._execute_js(f'ggbApplet.setLabelVisible("{obj_name}", {value})')
            return True
        except Exception:
            return False

    def set_all_labels_visible(self, visible: bool) -> bool:
        """Show or hide labels for all current objects."""
        try:
            all_objects = self.get_all_object_names()
            for name in all_objects:
                self.set_label_visible(name, visible)
            return True
        except Exception:
            return False

    def fit_view(
        self,
        padding: float = 1.5,
        default_bounds: Tuple[float, float, float, float] = (-8.0, 8.0, -8.0, 8.0),
        target_aspect_ratio: float = 1.0
    ) -> bool:
        """
        Fit viewport to include all point objects with padding.

        Notes:
            - This mirrors test helper behavior and is intentionally point-based.
            - If no points exist, falls back to default bounds.
        """
        try:
            state = self.get_construction_state()
            pts = [
                (info["x"], info["y"])
                for info in state["objects"].values()
                if info.get("type") == "point" and "x" in info and "y" in info
            ]

            if not pts:
                xmin, xmax, ymin, ymax = default_bounds
                self._execute_js(f"ggbApplet.setCoordSystem({xmin}, {xmax}, {ymin}, {ymax})")
                return True

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, xmax = min(xs) - padding, max(xs) + padding
            ymin, ymax = min(ys) - padding, max(ys) + padding

            xspan = max(xmax - xmin, 1e-6)
            yspan = max(ymax - ymin, 1e-6)
            current_ratio = xspan / yspan

            if current_ratio < target_aspect_ratio:
                extra = (yspan * target_aspect_ratio - xspan) / 2.0
                xmin -= extra
                xmax += extra
            elif current_ratio > target_aspect_ratio:
                extra = (xspan / target_aspect_ratio - yspan) / 2.0
                ymin -= extra
                ymax += extra

            self._execute_js(
                f"ggbApplet.setCoordSystem({xmin:.4f}, {xmax:.4f}, {ymin:.4f}, {ymax:.4f})"
            )
            return True
        except Exception:
            return False

    def get_construction_state(self) -> Dict[str, Any]:
        """
        Get complete construction state.

        Returns:
            Dictionary with all objects and their properties
        """
        state = {
            "objects": {},
            "points": [],
            "lines": [],
            "circles": [],
            "other": []
        }

        try:
            all_objects = self.get_all_object_names()

            for obj_name in all_objects:
                obj_type = self.get_object_type(obj_name)

                obj_info = {
                    "name": obj_name,
                    "type": obj_type,
                    "defined": self.is_defined(obj_name)
                }

                # Get type-specific properties
                if obj_type == "point":
                    coords = self.get_coords(obj_name)
                    if coords:
                        obj_info["x"], obj_info["y"] = coords
                    state["points"].append(obj_name)
                elif obj_type == "line":
                    state["lines"].append(obj_name)
                elif obj_type == "circle":
                    state["circles"].append(obj_name)
                else:
                    state["other"].append(obj_name)

                # Get value string
                value_str = self.get_value_string(obj_name)
                if value_str:
                    obj_info["value_string"] = value_str

                state["objects"][obj_name] = obj_info

        except Exception as e:
            print(f"Error getting construction state: {e}")

        return state

    def execute_script(
        self,
        ggb_script: str,
        auto_fit: bool = False,
        hide_labels: bool = False,
        fit_padding: float = 1.5
    ) -> GeoGebraResult:
        """
        Execute a GeoGebra script (multiple commands).

        Args:
            ggb_script: Multi-line GeoGebra script
            auto_fit: Automatically fit viewport after successful execution
            hide_labels: Hide all labels after successful execution
            fit_padding: Padding used when auto_fit is enabled

        Returns:
            GeoGebraResult with execution status
        """
        commands = [cmd.strip() for cmd in ggb_script.split('\n') if cmd.strip()]
        all_labels = []

        for command in commands:
            success, labels = self.eval_command_get_labels(command)
            if not success:
                return GeoGebraResult(
                    success=False,
                    objects={},
                    properties={},
                    error_message=f"Failed to execute: {command}",
                    script_executed=ggb_script
                )
            all_labels.extend(labels)

        # Get state of created objects
        state = self.get_construction_state()

        if hide_labels:
            self.set_all_labels_visible(False)
        if auto_fit:
            self.fit_view(padding=fit_padding)

        return GeoGebraResult(
            success=True,
            objects=state["objects"],
            properties=state,
            script_executed=ggb_script,
            created_labels=all_labels
        )

    def _escape_js(self, text: str) -> str:
        """Escape string for JavaScript."""
        return text.replace('"', '\\"').replace('\n', '\\n')

    def cleanup(self):
        """Clean up resources."""
        if self._driver:
            self._driver.quit()

        # Clean up temp directory
        import shutil
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
