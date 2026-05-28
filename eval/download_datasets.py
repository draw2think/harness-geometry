#!/usr/bin/env python3
"""
download_datasets.py — geometry benchmark downloader / status checker

Usage:
    python eval/download_datasets.py                # show local dataset status
    python eval/download_datasets.py --download all # download all missing datasets
    python eval/download_datasets.py --download mathverse olympiadbench
    python eval/download_datasets.py --fast --download all   # faster HF downloads (hf-xet)
    python eval/download_datasets.py --data-root /data

Dependencies:
    pip install huggingface_hub hf-xet   # hf-xet is an optional fast backend
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_DATA_ROOT = "/data"

# ── Dataset registry ──
DATASETS = [
    # ── Local / already prepared datasets ──
    {
        "id": "geometry3k",
        "name": "Geometry3K",
        "local_dir": "geometry3k",
        "source": "local",
        "check_path": "test",
        "test_size": 601,
        "description": "Gold-standard English high-school geometry, 3002 problems",
    },
    {
        "id": "formalgeo7k",
        "name": "FormalGeo7K v2",
        "local_dir": "formalgeo7k_v2",
        "source": "local",
        "check_path": "problems",
        "test_size": 7000,
        "description": "7000 CDL-formalized problems, Chinese and English",
    },
    {
        "id": "geoqa",
        "name": "GeoQA",
        "local_dir": "GeoQA",
        "source": "local",
        "check_path": "data/GeoQA3",
        "test_size": 1001,
        "description": "4998 Chinese middle-school geometry problems, ACL 2021",
    },
    {
        "id": "unigeo",
        "name": "UniGeo",
        "local_dir": "UniGeo",
        "source": "local",
        "check_path": "calculation_test.pk",
        "test_size": 3353,
        "description": "9543 proving + 4998 calc, EMNLP 2022",
    },
    {
        "id": "pgps9k",
        "name": "PGPS9K",
        "local_dir": "PGPS9K",
        "source": "zip",
        "url": "https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/PGPS9K_all.zip",
        "check_path": "PGPS9K/test.json",
        "test_size": 1000,
        "description": "9021 problems (1000 test), 30 knowledge points; password is PAL_PGPS_2023",
    },
    # ── Hugging Face (snapshot_download -> /data/) ──
    {
        "id": "mathverse",
        "name": "MathVerse",
        "local_dir": "mathverse",
        "source": "huggingface",
        "hf_id": "AI4Math/MathVerse",
        "check_path": "",
        "test_size": 2612,
        "description": "6 variants (Vision_Only is hardest), Apache 2.0",
    },
    {
        "id": "mathvision",
        "name": "MathVision (MATH-V)",
        "local_dir": "mathvision",
        "source": "huggingface",
        "hf_id": "MathLLMs/MathVision",
        "check_path": "",
        "test_size": 3040,
        "description": "3040 competition-level visual math problems, 16 domains, NeurIPS 2024",
    },
    {
        "id": "olympiadbench",
        "name": "OlympiadBench",
        "local_dir": "olympiadbench",
        "source": "huggingface",
        "hf_id": "Hothan/OlympiadBench",
        "check_path": "",
        "test_size": 675,
        "description": "8476 olympiad problems (including ~675 geometry), Chinese and English",
    },
    {
        "id": "geo170k",
        "name": "Geo170K (G-LLaVA)",
        "local_dir": "geo170k",
        "source": "huggingface",
        "hf_id": "Luckyjhg/Geo170K",
        "check_path": "",
        "test_size": 0,  # training set, no standard test split
        "description": "170K geometry image-caption + QA examples, ICLR 2025, training use",
    },
    # ── Comparator benchmarks ──
    {
        "id": "geolaux",
        "name": "GeoLaux-mini",
        "local_dir": "geolaux",
        "source": "github",
        "github_repo": "Candice-yu/GeoLaux",
        "check_path": "data/GeoLaux_minidata.json",
        "test_size": 330,
        "description": "330 long-step geometry problems (calculation + proof), avg 6.51 steps, PCS/ACS/PQS eval",
    },
    {
        "id": "geosketch",
        "name": "GeoSketch Benchmark",
        "local_dir": "geosketch",
        "source": "huggingface",
        "hf_id": "datatune/GeoSketch",
        "check_path": "data",
        "test_size": 390,
        "description": "390 auxiliary-line construction problems (Num 201/Ratio 108/Desc 81), human 86.14%",
    },
    # ── Solid-geometry benchmarks ──
    {
        "id": "solidgeo",
        "name": "SolidGeo",
        "local_dir": "solidgeo",
        "source": "huggingface",
        "hf_id": "HarryYancy/SolidGeo",
        "check_path": "",
        "test_size": 3113,
        "description": "3113 solid-geometry problems (8 categories, 3 difficulty levels), NeurIPS 2025, EN&CN",
    },
    {
        "id": "dynasolidgeo",
        "name": "DynaSolidGeo",
        "local_dir": "dynasolidgeo",
        "source": "github",
        "github_repo": "ChangtiWu/DynaSolidGeo",
        "check_path": "",
        "test_size": 503,
        "description": "503 seed solid-geometry dynamic eval problems (8 categories, 3 levels), Dynamic + Process Eval",
    },
    {
        "id": "mathcanvas_bench",
        "name": "MathCanvas-Bench",
        "local_dir": "mathcanvas_bench",
        "source": "huggingface",
        "hf_id": "shiwk24/MathCanvas-Bench",
        "check_path": "",
        "test_size": 3087,
        "description": "3K visual reasoning-chain benchmark (8 math categories, VCoT), arXiv:2510.14958",
    },
    # ── Subgoal / Step Verification ──
    {
        "id": "geogoal_sgvr",
        "name": "GeoGoal-SGVR",
        "local_dir": "geogoal_sgvr",
        "source": "huggingface",
        "hf_id": "carpe002/GeoGoal-SGVR",
        "check_path": "",
        "test_size": 256,
        "description": "512 problems (256 train / 256 test) in 2D geometry, with formal-logic premises and proof steps (cong/perp/eqangle/ncoll predicates), for per-step construction-fidelity evaluation, Apache 2.0, arXiv:2601.05073",
    },
]


def check_dataset(ds: dict, data_root: str) -> dict:
    """Check whether a dataset is already available locally."""
    local_path = Path(data_root) / ds["local_dir"]
    check = local_path / ds["check_path"] if ds.get("check_path") else local_path

    if not local_path.exists():
        return {"status": "missing", "path": str(local_path), "detail": "directory not found"}

    if ds.get("check_path") and not check.exists():
        return {"status": "incomplete", "path": str(local_path),
                "detail": f"missing {ds['check_path']}"}

    # Count local files and size.
    n_files = sum(1 for _ in local_path.rglob("*") if _.is_file())
    size_mb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1e6

    return {
        "status": "ok",
        "path": str(local_path),
        "detail": f"{n_files} files, {size_mb:.0f} MB",
    }


def enable_fast_transfer():
    """Enable the hf-xet / hf_transfer fast download backend."""
    # New backend: hf-xet (Rust, auto-enabled if installed).
    try:
        import hf_xet  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[FAST] hf-xet backend enabled")
        return
    except ImportError:
        pass

    # Legacy fallback: hf_transfer.
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[FAST] hf_transfer backend enabled")
        return
    except ImportError:
        pass

    print("[WARN] Fast backend unavailable; install hf-xet for faster downloads")


def download_huggingface(ds: dict, data_root: str) -> bool:
    """Download a Hugging Face dataset snapshot directly into /data/xxx."""
    target = Path(data_root) / ds["local_dir"]
    hf_id = ds["hf_id"]

    print(f"  [HF] {hf_id} → {target}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] pip install huggingface_hub hf-xet")
        return False

    try:
        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        print(f"  [OK] download complete")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def download_zip(ds: dict, data_root: str) -> bool:
    """Download a zip file and extract it into the target directory."""
    target = Path(data_root) / ds["local_dir"]
    url = ds["url"]
    zip_path = Path(data_root) / f"{ds['local_dir']}_tmp.zip"

    print(f"  [ZIP] {url}")
    print(f"        → {target}")

    try:
        def progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(done * 100 / total, 100)
                print(f"\r  downloading... {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)

        urlretrieve(url, str(zip_path), reporthook=progress)
        print()  # newline after progress

        # Extract.
        print(f"  extracting...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(target))

        # Clean up.
        zip_path.unlink()
        print(f"  [OK] extraction complete")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def download_github(ds: dict, data_root: str) -> bool:
    """Clone a GitHub repository into the target directory."""
    target = Path(data_root) / ds["local_dir"]
    repo = ds["github_repo"]
    url = f"https://github.com/{repo}.git"

    print(f"  [GH] {url} → {target}")

    if target.exists():
        print(f"  [INFO] directory exists; trying git pull")
        try:
            subprocess.run(["git", "-C", str(target), "pull"], check=True)
            print(f"  [OK] update complete")
            return True
        except Exception as e:
            print(f"  [WARN] pull failed: {e}")
            return False

    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
        print(f"  [OK] clone complete")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def download_dataset(ds: dict, data_root: str) -> bool:
    """Dispatch downloads by source type."""
    src = ds["source"]
    if src == "local":
        print(f"  [INFO] local dataset; place it manually at {Path(data_root) / ds['local_dir']}")
        return False
    elif src == "huggingface":
        return download_huggingface(ds, data_root)
    elif src == "zip":
        return download_zip(ds, data_root)
    elif src == "github":
        return download_github(ds, data_root)
    else:
        print(f"  [ERROR] unknown source: {src}")
        return False


def print_status(statuses: list[dict]):
    """Print a status summary."""
    icons = {"ok": "✅", "missing": "❌", "incomplete": "⚠️ "}

    print()
    print("=" * 88)
    print(f"{'Dataset':<24} {'Status':<6} {'Test':>6}  {'Source':<12} {'Details'}")
    print("-" * 88)

    for s in statuses:
        icon = icons.get(s["status"], "?")
        test_str = str(s["test_size"]) if s["test_size"] > 0 else "train"
        print(
            f"{s['name']:<24} {icon:<4} {test_str:>6}  "
            f"{s['source']:<12} {s['detail']}"
        )

    print("=" * 88)

    ok = sum(1 for s in statuses if s["status"] == "ok")
    print(f"\nReady: {ok}/{len(statuses)}")

    downloadable = [
        s for s in statuses
        if s["status"] != "ok" and s["source"] not in ("local",)
    ]
    if downloadable:
        ids = " ".join(s["id"] for s in downloadable)
        print(f"Download missing datasets with: python eval/download_datasets.py --download {ids}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Geometry benchmark dataset manager")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download", nargs="*", default=None,
                        help="Download selected dataset ids (all = every missing dataset)")
    parser.add_argument("--fast", action="store_true",
                        help="Enable faster Hugging Face downloads via hf-xet")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.fast:
        enable_fast_transfer()

    # Status check.
    statuses = []
    for ds in DATASETS:
        info = check_dataset(ds, args.data_root)
        statuses.append({"id": ds["id"], "name": ds["name"],
                         "test_size": ds["test_size"], "source": ds["source"],
                         **info})

    if args.json:
        print(json.dumps(statuses, indent=2, ensure_ascii=False))
        return

    print_status(statuses)

    # Download.
    if args.download is not None:
        targets = args.download if args.download else []
        if not targets:
            print("Usage: --download all | --download mathverse olympiadbench")
            return

        if "all" in targets:
            to_dl = [ds for ds, s in zip(DATASETS, statuses) if s["status"] != "ok"]
        else:
            to_dl = [ds for ds in DATASETS if ds["id"] in targets]

        if not to_dl:
            print("No datasets need downloading.")
            return

        print(f"\nPreparing to download {len(to_dl)} dataset(s)...\n")
        for ds in to_dl:
            print(f"[{ds['id']}] {ds['name']}")
            download_dataset(ds, args.data_root)
            print()

        # Re-check after download.
        print("Re-checking:")
        statuses2 = []
        for ds in DATASETS:
            info = check_dataset(ds, args.data_root)
            statuses2.append({"id": ds["id"], "name": ds["name"],
                              "test_size": ds["test_size"], "source": ds["source"],
                              **info})
        print_status(statuses2)


if __name__ == "__main__":
    main()
