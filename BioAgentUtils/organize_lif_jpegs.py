#!/usr/bin/env python3
"""Organize exported LIF JPEG files into per-object folders."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

FILE_PATTERN = re.compile(
    r"^(?P<index>\d+)_(?P<object>.+)_t(?P<t>\d+)_z(?P<z>\d+)_c(?P<c>\d+)\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "unknown_object"


def place_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "move":
        shutil.move(str(src), str(dst))
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def organize(input_dir: Path, output_dir: Path, mode: str) -> tuple[int, int]:
    placed = 0
    unmatched = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")):
        match = FILE_PATTERN.match(src.name)
        if not match:
            unmatched += 1
            dst = output_dir / "_unmatched" / src.name
            place_file(src, dst, mode)
            placed += 1
            continue

        object_name = sanitize_name(match.group("object"))
        dst = output_dir / object_name / src.name
        place_file(src, dst, mode)
        placed += 1

    return placed, unmatched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group exported LIF JPEG files by object name into subfolders."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing flat JPEG export")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>_by_object)",
    )
    parser.add_argument(
        "--mode",
        choices=["link", "copy", "move"],
        default="link",
        help="How files are placed into output (default: link)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_dir.with_name(f"{input_dir.name}_by_object")

    placed, unmatched = organize(input_dir, output_dir, args.mode)

    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Mode:      {args.mode}")
    print(f"Processed: {placed}")
    print(f"Unmatched: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
