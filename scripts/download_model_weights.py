#!/usr/bin/env python3
"""Download the versioned OrganoidAgent model weights and verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = "models-v1.0.0"
BASE_URL = f"https://github.com/yhyh2270/OrganoidAgent/releases/download/{RELEASE_TAG}"
WEIGHTS = {
    "sam_vit_b_01ec64.pth": (
        375_042_383,
        "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    ),
    "viability_best.pth": (
        335_956_115,
        "cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b",
    ),
    "yolo_organoid_best.pt": (
        6_258_019,
        "2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected_size: int, expected_hash: str) -> bool:
    return path.is_file() and path.stat().st_size == expected_size and sha256(path) == expected_hash


def download(url: str, destination: Path, expected_size: int, expected_hash: str) -> None:
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "OrganoidAgent-weight-downloader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                print(f"\r  {received / 1024 / 1024:.1f} / {expected_size / 1024 / 1024:.1f} MiB", end="")
        print()
        actual_size = temporary.stat().st_size
        actual_hash = sha256(temporary)
        if actual_size != expected_size:
            raise ValueError(f"size mismatch: expected {expected_size}, received {actual_size}")
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch: expected {expected_hash}, received {actual_hash}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "fluorescence_prediction" / "weights",
        help="Weight destination (default: fluorescence_prediction/weights).",
    )
    parser.add_argument("--force", action="store_true", help="Download again even when a valid file exists.")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, (expected_size, expected_hash) in WEIGHTS.items():
            destination = output_dir / name
            if not args.force and is_valid(destination, expected_size, expected_hash):
                print(f"Verified existing {name}")
                continue
            print(f"Downloading {name} from {RELEASE_TAG}...")
            download(f"{BASE_URL}/{name}", destination, expected_size, expected_hash)
            print(f"Verified {name}")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"All model weights are ready in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
