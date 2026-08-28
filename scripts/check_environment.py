#!/usr/bin/env python3
"""Check Python dependencies and local model resources without importing heavy models."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "core": {
        "tornado": "tornado",
        "numpy": "numpy",
        "pandas": "pandas",
        "anndata": "anndata",
        "h5py": "h5py",
        "Pillow": "PIL",
        "tifffile": "tifffile",
        "imagecodecs": "imagecodecs",
        "openpyxl": "openpyxl",
        "xlrd": "xlrd",
        "requests": "requests",
    },
    "analysis": {
        "torch": "torch",
        "torchvision": "torchvision",
        "opencv-python-headless": "cv2",
        "ultralytics": "ultralytics",
        "cellpose": "cellpose",
        "segment-anything": "segment_anything",
        "scipy": "scipy",
        "scikit-image": "skimage",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "readlif": "readlif",
    },
}

RESOURCES = {
    "viability checkpoint": (ROOT / "fluorescence_prediction" / "weights" / "viability_best.pth", 1_000_000),
    "YOLO checkpoint": (ROOT / "fluorescence_prediction" / "weights" / "yolo_organoid_best.pt", 1_000_000),
    "SAM checkpoint": (ROOT / "fluorescence_prediction" / "weights" / "sam_vit_b_01ec64.pth", 1_000_000),
}


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_packages(profile: str) -> bool:
    packages = dict(PROFILES["core"])
    if profile == "analysis":
        packages.update(PROFILES["analysis"])
    ok = True
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for distribution, module in packages.items():
        present = importlib.util.find_spec(module) is not None
        marker = "OK" if present else "MISSING"
        version = package_version(distribution) if present else "install required"
        print(f"[{marker:7}] {distribution:<22} {version}")
        ok &= present
    return ok


def check_resources(profile: str) -> bool:
    if profile != "analysis":
        return True
    ok = True
    print("\nModel resources:")
    for label, (path, minimum_size) in RESOURCES.items():
        size = path.stat().st_size if path.is_file() else 0
        present = size >= minimum_size
        marker = "OK" if present else "MISSING"
        print(f"[{marker:7}] {label:<22} {size / (1024 * 1024):8.1f} MiB  {path.relative_to(ROOT)}")
        ok &= present
    return ok


def print_runtime_notes(profile: str) -> None:
    if profile != "analysis" or importlib.util.find_spec("torch") is None:
        return
    import torch

    print(f"\nPyTorch CUDA build: {torch.version.cuda or 'none'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("NOTE: CPU startup and previews work, but model inference/training will be slow.")
    configured = os.environ.get("ORGANOID_CELLPOSE_PYTHON")
    if configured:
        print(f"ORGANOID_CELLPOSE_PYTHON: {configured}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "analysis"), default="analysis")
    args = parser.parse_args()
    ok = check_packages(args.profile)
    ok &= check_resources(args.profile)
    print_runtime_notes(args.profile)
    print("\nEnvironment is ready." if ok else "\nEnvironment is incomplete.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
