"""Setup script for symbolic reasoning package."""

from setuptools import setup, find_packages
from pathlib import Path
import os
import platform
import re
import stat
import subprocess
import sys
import zipfile
import urllib.request
import shutil

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# ── GeoGebra offline bundle ────────────────────────────────────────────
_BUNDLE_URL = "https://download.geogebra.org/package/geogebra-math-apps-bundle"
_BUNDLE_DIR = (Path(__file__).parent
               / "symbolic" / "integrations" / "geogebra_bundle" / "GeoGebra")


def download_geogebra_bundle(force: bool = False):
    """Download and extract GeoGebra Math Apps Bundle for offline use.

    Skips download if the bundle already exists (unless *force* is True).
    The bundle (~31 MB zip, ~115 MB extracted) provides the full GeoGebra
    Classic + 3D engine so the framework can run without network access.

    Source: https://download.geogebra.org/package/geogebra-math-apps-bundle
    """
    marker = _BUNDLE_DIR / "deployggb.js"
    if marker.exists() and not force:
        print(f"[GGB] Bundle already exists: {_BUNDLE_DIR}")
        return

    dest_parent = _BUNDLE_DIR.parent          # .../geogebra_bundle/
    dest_parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_parent / "geogebra-math-apps-bundle.zip"

    # Download
    print(f"[GGB] Downloading GeoGebra bundle from {_BUNDLE_URL} ...")
    try:
        urllib.request.urlretrieve(_BUNDLE_URL, zip_path)
    except Exception as e:
        print(f"[GGB] Download failed: {e}")
        print(f"[GGB] You can manually download and unzip into {dest_parent}/")
        return
    print(f"[GGB] Downloaded: {zip_path}  ({zip_path.stat().st_size // 1024 // 1024} MB)")

    # Extract
    print(f"[GGB] Extracting ...")
    # Remove old bundle if re-downloading
    if _BUNDLE_DIR.exists():
        shutil.rmtree(_BUNDLE_DIR)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_parent)
    zip_path.unlink()     # clean up zip

    if marker.exists():
        print(f"[GGB] Bundle ready: {_BUNDLE_DIR}")
    else:
        print(f"[GGB] WARNING: extraction succeeded but {marker} not found")


# ── .env API-key template ──────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"

_ENV_TEMPLATE = """\
# ── Draw2Think API keys ─────────────────────────────────────
# Fill in only the providers you actually use, then save.
# This file is git-ignored; never commit real keys.

# Core providers
GOOGLE_API_KEY=          # Gemini (default constructor)
OPENAI_API_KEY=          # GPT (also used as GenExam / GGBench judge)
ANTHROPIC_API_KEY=       # Claude

# Additional providers (optional; selected via the model registry)
DASHSCOPE_API_KEY=       # Alibaba Qwen (Singapore endpoint)
MOONSHOT_API_KEY=        # Kimi
ZHIPU_API_KEY=           # GLM
DEEPSEEK_API_KEY=        # DeepSeek
MINIMAX_API_KEY=         # MiniMax
STEPFUN_API_KEY=         # Step

# Optional proxy routing (leave blank to use official endpoints).
# Set USE_PROXY=1 to route Gemini through the proxy below.
USE_PROXY=
PROXY_BASE_URL=
PROXY_API_KEY=
GOOGLE_PROXY_KEY=
OPENAI_PROXY_KEY=
USE_JUDGE_PROXY=
"""


def create_env_template(force: bool = False):
    """Write a blank .env template to the project root for API keys.

    Skips if .env already exists (unless *force* is True) so user-filled
    keys are never overwritten. Writes empty placeholders only, never
    real credentials.
    """
    if _ENV_PATH.exists() and not force:
        print(f"[ENV] .env already exists, leaving it untouched: {_ENV_PATH}")
        return
    _ENV_PATH.write_text(_ENV_TEMPLATE)
    print(f"[ENV] Wrote blank API-key template: {_ENV_PATH}")
    print("[ENV] Edit it and fill in the keys you need.")


# ── GeoGebra command manual (optional reference docs) ──────────────────
_MANUAL_REPO = "geogebra/manual"
_MANUAL_REF = "main"
_MANUAL_SUBDIR = "en/modules/ROOT/pages"
_MANUAL_DEST = Path(__file__).parent / "docs" / "geogebra-manual"


def download_geogebra_manual(force: bool = False):
    """Download the GeoGebra command manual (en/modules/ROOT/pages, ~743 .adoc).

    Optional reference for looking up GeoGebra command syntax; not required at
    runtime. Licensed CC BY-NC-SA by GeoGebra (see NOTICE); we do not
    redistribute it, setup fetches it from the official repo on demand. Lists
    the subtree via the GitHub API, then downloads each .adoc from
    raw.githubusercontent.com concurrently. Skips if present (unless *force*);
    every failure is non-fatal so it never blocks installation.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    if _MANUAL_DEST.exists() and any(_MANUAL_DEST.iterdir()) and not force:
        print(f"[MANUAL] Already present: {_MANUAL_DEST}")
        return

    api = f"https://api.github.com/repos/{_MANUAL_REPO}/git/trees/{_MANUAL_REF}?recursive=1"
    try:
        req = urllib.request.Request(
            api, headers={"User-Agent": "draw2think-setup",
                          "Accept": "application/vnd.github+json"})
        tree = json.load(urllib.request.urlopen(req, timeout=60))["tree"]
    except Exception as e:
        print(f"[MANUAL] Skipped (could not list manual tree): {e}")
        print("[MANUAL] Retry later with:  python setup.py download_manual")
        return

    prefix = _MANUAL_SUBDIR + "/"
    files = [e["path"] for e in tree
             if e.get("type") == "blob"
             and e["path"].startswith(prefix) and e["path"].endswith(".adoc")]
    if not files:
        print("[MANUAL] Skipped (no .adoc pages found upstream)")
        return

    base = f"https://raw.githubusercontent.com/{_MANUAL_REPO}/{_MANUAL_REF}/"

    def _fetch(path):
        out = _MANUAL_DEST / path[len(prefix):]
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(base + path, out)
            return True
        except Exception:
            return False

    print(f"[MANUAL] Downloading {len(files)} manual pages from {_MANUAL_REPO} ...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        ok = sum(pool.map(_fetch, files))
    print(f"[MANUAL] Done: {ok}/{len(files)} pages -> {_MANUAL_DEST}")


# ── Browser runtime (Chrome for Testing) ───────────────────────────────
_CFT_INDEX_URL = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)
_BROWSER_ROOT = Path(
    os.environ.get(
        "DRAW2THINK_BROWSER_ROOT",
        Path(__file__).parent / ".chrome-for-testing",
    )
)


def _cft_platform() -> str:
    """Return the Chrome for Testing platform tag for this machine."""
    if sys.platform.startswith("linux"):
        return "linux64"
    if sys.platform == "darwin":
        return "mac-arm64" if platform.machine() == "arm64" else "mac-x64"
    if sys.platform.startswith("win"):
        machine = platform.machine().lower()
        return "win64" if machine in {"amd64", "x86_64", "arm64"} else "win32"
    raise RuntimeError(f"Unsupported platform for Chrome for Testing: {sys.platform}")


def _cft_binary_paths(version_dir: Path) -> tuple[Path, Path]:
    """Return (chrome, chromedriver) paths inside an extracted CFT version dir."""
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


def _latest_cft_version_dir() -> Path | None:
    if not _BROWSER_ROOT.exists():
        return None
    dirs = [path for path in _BROWSER_ROOT.iterdir() if path.is_dir()]
    return sorted(dirs)[-1] if dirs else None


def _active_env_executable(*names: str) -> str | None:
    bin_dir = Path(sys.executable).parent
    for name in names:
        path = bin_dir / name
        if path.exists():
            return str(path)
    return None


def _make_executable(path: Path):
    if path.exists() and not sys.platform.startswith("win"):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _which_browser() -> str | None:
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


def _which_chromedriver() -> str | None:
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
    found = shutil.which("chromedriver")
    if found:
        return found
    return None


def _version_output(path: str) -> str | None:
    try:
        out = subprocess.run(
            [path, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _major_version(path: str) -> str | None:
    out = _version_output(path)
    if not out:
        return None
    match = re.search(r"\b(\d+)\.", out)
    return match.group(1) if match else None


def _link_into_active_env(chrome: Path, chromedriver: Path):
    """Expose downloaded browser in the active Python env's bin/Scripts dir."""
    bin_dir = Path(sys.executable).parent
    if not bin_dir.exists() or not os.access(bin_dir, os.W_OK):
        print(f"[BROWSER] Active Python bin is not writable: {bin_dir}")
        print(f"[BROWSER] Set DRAW2THINK_CHROME={chrome}")
        print(f"[BROWSER] Set DRAW2THINK_CHROMEDRIVER={chromedriver}")
        return

    if sys.platform.startswith("win"):
        for name, src in {"chrome.cmd": chrome, "chromedriver.cmd": chromedriver}.items():
            dst = bin_dir / name
            if dst.exists():
                print(f"[BROWSER] Keeping existing {dst}")
                continue
            dst.write_text(f'@echo off\n"{src}" %*\n')
            print(f"[BROWSER] Wrote {dst}")
        return

    links = {
        "chrome": chrome,
        "google-chrome": chrome,
        "chromium": chrome,
        "chromedriver": chromedriver,
    }
    for name, src in links.items():
        dst = bin_dir / name
        try:
            if dst.exists() and not dst.is_symlink():
                print(f"[BROWSER] Keeping existing executable: {dst}")
                continue
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            print(f"[BROWSER] Linked {dst} -> {src}")
        except Exception as e:
            print(f"[BROWSER] Could not link {dst}: {e}")


def download_browser(force: bool = False):
    """Download Chrome for Testing and a matching ChromeDriver."""
    import json

    existing_chrome = _which_browser()
    existing_driver = _which_chromedriver()
    if existing_chrome and existing_driver and not force:
        chrome_major = _major_version(existing_chrome)
        driver_major = _major_version(existing_driver)
        if chrome_major and chrome_major == driver_major:
            print(f"[BROWSER] Browser already available: {existing_chrome}")
            print(f"[BROWSER] ChromeDriver already available: {existing_driver}")
            return

    platform_tag = _cft_platform()
    print(f"[BROWSER] Fetching Chrome for Testing index: {_CFT_INDEX_URL}")
    try:
        with urllib.request.urlopen(_CFT_INDEX_URL, timeout=60) as response:
            meta = json.load(response)
    except Exception as e:
        print(f"[BROWSER] Could not fetch Chrome for Testing index: {e}")
        return

    stable = meta["channels"]["Stable"]
    downloads = stable["downloads"]
    version_dir = _BROWSER_ROOT / stable["version"]

    def pick(kind: str) -> str:
        for item in downloads[kind]:
            if item["platform"] == platform_tag:
                return item["url"]
        raise RuntimeError(f"No {kind} download for platform {platform_tag}")

    chrome_bin, driver_bin = _cft_binary_paths(version_dir)
    if chrome_bin.exists() and driver_bin.exists() and not force:
        print(f"[BROWSER] Chrome for Testing already exists: {version_dir}")
        _link_into_active_env(chrome_bin, driver_bin)
        return

    version_dir.mkdir(parents=True, exist_ok=True)
    for name, url in (("chrome", pick("chrome")), ("chromedriver", pick("chromedriver"))):
        zip_path = version_dir / f"{name}-{platform_tag}.zip"
        print(f"[BROWSER] Downloading {name}: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[BROWSER] Download failed for {name}: {e}")
            return
        print(f"[BROWSER] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(version_dir)
        zip_path.unlink()

    _make_executable(chrome_bin)
    _make_executable(driver_bin)
    if not chrome_bin.exists() or not driver_bin.exists():
        print(f"[BROWSER] Expected binaries were not found under {version_dir}")
        return

    print(f"[BROWSER] Chrome ready: {chrome_bin}")
    print(f"[BROWSER] ChromeDriver ready: {driver_bin}")
    _link_into_active_env(chrome_bin, driver_bin)


def _ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
    except Exception:
        return None
    return None


def _selenium_smoke(chrome: str, driver: str) -> bool:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        opts = Options()
        opts.binary_location = chrome
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        with webdriver.Chrome(service=Service(driver), options=opts) as browser:
            browser.get("data:text/html,<title>Draw2Think</title><p>ok</p>")
            title = browser.title
        print(f"[DOCTOR] Selenium smoke test: ok ({title})")
        return True
    except Exception as e:
        print(f"[DOCTOR] Selenium smoke test: failed ({type(e).__name__}: {e})")
        return False


def doctor() -> bool:
    """Check whether the local install can run the Selenium/GeoGebra stack."""
    ok = True
    print("[DOCTOR] Python:", sys.version.split()[0], sys.executable)
    ok = ok and sys.version_info >= (3, 10)

    ram = _ram_gb()
    if ram is None:
        print("[DOCTOR] RAM: unknown")
    else:
        print(f"[DOCTOR] RAM: {ram:.1f} GB" + ("" if ram >= 4 else "  (recommended: >= 4 GB)"))
        ok = ok and ram >= 4

    print("[DOCTOR] .env:", "ok" if _ENV_PATH.exists() else "missing")
    ok = ok and _ENV_PATH.exists()

    bundle_marker = _BUNDLE_DIR / "deployggb.js"
    print("[DOCTOR] GeoGebra offline bundle:",
          "ok" if bundle_marker.exists() else "missing; CDN mode available")

    chrome = _which_browser()
    driver = _which_chromedriver()
    print("[DOCTOR] Chrome:", chrome or "missing")
    print("[DOCTOR] ChromeDriver:", driver or "missing")
    ok = ok and bool(chrome) and bool(driver)

    if chrome:
        print("[DOCTOR] Chrome version:", _version_output(chrome) or "unavailable")
    if driver:
        print("[DOCTOR] ChromeDriver version:", _version_output(driver) or "unavailable")
    if chrome and driver:
        ok = _selenium_smoke(chrome, driver) and ok
    else:
        print("[DOCTOR] Selenium smoke test: skipped")

    print("[DOCTOR] Result:", "ok" if ok else "needs attention")
    return ok


def bootstrap(force: bool = False, offline_bundle: bool = False, manual: bool = False):
    """Prepare a runnable local environment after `pip install -e .`."""
    create_env_template(force=force)
    download_browser(force=force)
    if offline_bundle:
        download_geogebra_bundle(force=force)
    else:
        print("[BOOTSTRAP] GeoGebra runtime: CDN mode selected. Use --offline-bundle for a local offline runtime.")
    if manual:
        download_geogebra_manual(force=force)
    return doctor()


# ── Allow standalone invocation ─────────────────────────────────────────
#   python setup.py bootstrap --offline-bundle # recommended: env + browser + offline GeoGebra
#   python setup.py bootstrap                  # lighter: env + browser + GeoGebra CDN
#   python setup.py download_browser           # fetch Chrome for Testing
#   python setup.py download_bundle            # fetch the GeoGebra offline bundle
#   python setup.py create_env                 # write a blank .env template
#   python setup.py download_manual            # fetch optional GeoGebra manual
#   python setup.py doctor                     # check local runtime requirements
if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
    force = "--force" in sys.argv
    offline_bundle = "--offline-bundle" in sys.argv
    manual = "--manual" in sys.argv
    sys.exit(0 if bootstrap(force=force, offline_bundle=offline_bundle, manual=manual) else 1)

if len(sys.argv) > 1 and sys.argv[1] == "download_browser":
    force = "--force" in sys.argv
    download_browser(force=force)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "download_bundle":
    force = "--force" in sys.argv
    download_geogebra_bundle(force=force)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "create_env":
    force = "--force" in sys.argv
    create_env_template(force=force)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "download_manual":
    force = "--force" in sys.argv
    download_geogebra_manual(force=force)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "doctor":
    sys.exit(0 if doctor() else 1)


setup(
    name="symbolic",
    version="0.1.0",
    description="Neuro-symbolic framework for geometric reasoning with test-time scaling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        # LLM APIs
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "google-genai",
        # Symbolic reasoning
        "z3-solver>=4.12.0",
        "sympy",
        "antlr4-python3-runtime>=4.11,<4.12",
        "math-verify",
        # GeoGebra integration
        "selenium>=4.10.0",
        "webdriver-manager>=4.0.0",
        # Dataset loading
        "huggingface_hub",
        "pyarrow",
        # Utilities
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        # Testing
        "pytest>=7.4.0",
    ],
    extras_require={
        "train": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.14.0",
            "hf-xet",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
