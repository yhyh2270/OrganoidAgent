#!/usr/bin/env python3
"""Export Leica LIF image planes to JPEG files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from readlif.reader import LifFile
except ImportError:  # pragma: no cover - runtime environment dependency
    print(
        "Missing dependency: readlif.\n"
        "Install with: pip install readlif",
        file=sys.stderr,
    )
    raise


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "image"


def parse_indices(indices: List[int] | None, upper_bound: int, label: str) -> Iterable[int]:
    if not indices:
        return range(upper_bound)
    for idx in indices:
        if idx < 0 or idx >= upper_bound:
            raise ValueError(f"{label} index out of range: {idx} (allowed: 0..{upper_bound - 1})")
    return indices


def export_lif_to_jpeg(
    input_path: Path,
    output_dir: Path,
    quality: int,
    image_indices: List[int] | None,
    channels: List[int] | None,
    z_indices: List[int] | None,
    time_indices: List[int] | None,
    first_plane_only: bool,
    overwrite: bool,
) -> tuple[int, int]:
    lif = LifFile(str(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0

    selected_images = parse_indices(image_indices, len(lif.image_list), "Image")
    for image_index in selected_images:
        lif_image = lif.get_image(image_index)
        image_name = sanitize_name(getattr(lif_image, "name", f"image_{image_index}"))

        selected_channels = parse_indices(channels, lif_image.channels, "Channel")
        if first_plane_only:
            selected_z = [0]
            selected_t = [0]
        else:
            selected_z = parse_indices(z_indices, lif_image.nz, "Z")
            selected_t = parse_indices(time_indices, lif_image.nt, "Time")

        for time_index in selected_t:
            for z_index in selected_z:
                for channel_index in selected_channels:
                    out_name = (
                        f"{image_index:02d}_{image_name}_"
                        f"t{time_index:03d}_z{z_index:03d}_c{channel_index}.jpg"
                    )
                    out_path = output_dir / out_name
                    if out_path.exists() and not overwrite:
                        total_skipped += 1
                        continue

                    requested_dims = {}
                    if lif_image.nz > 1 or z_index != 0:
                        requested_dims[3] = z_index
                    if lif_image.nt > 1 or time_index != 0:
                        requested_dims[4] = time_index
                    plane = lif_image.get_plane(c=channel_index, requested_dims=requested_dims)
                    plane.save(out_path, format="JPEG", quality=quality)
                    total_saved += 1

    return total_saved, total_skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Leica LIF file to JPEG images.")
    parser.add_argument("input_lif", type=Path, help="Path to input .lif file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <lif_stem>_jpeg beside the input file)",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument(
        "--image-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional image indices to export (default: all)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Optional channel indices to export (default: all channels)",
    )
    parser.add_argument(
        "--z-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional Z-plane indices to export (default: all planes)",
    )
    parser.add_argument(
        "--time-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional time indices to export (default: all time points)",
    )
    parser.add_argument(
        "--first-plane-only",
        action="store_true",
        help="Export only t=0, z=0 per image/channel (legacy behavior)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_lif.expanduser().resolve()
    if input_path.suffix.lower() != ".lif":
        parser.error(f"Expected a .lif file, got: {input_path.name}")
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in the range 1-100")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_jpeg"
    else:
        output_dir = output_dir.expanduser().resolve()

    try:
        saved, skipped = export_lif_to_jpeg(
            input_path=input_path,
            output_dir=output_dir,
            quality=args.quality,
            image_indices=args.image_indices,
            channels=args.channels,
            z_indices=args.z_indices,
            time_indices=args.time_indices,
            first_plane_only=args.first_plane_only,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
