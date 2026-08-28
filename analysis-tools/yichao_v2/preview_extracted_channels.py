#!/usr/bin/env python3
"""Build quick channel previews for Yichao v2 extracted LIF JPEGs.

The v2 LIF files contain named acquisition series such as ALEXA488, DAPI,
ALEXA647, and mCherry. Each named series stores two internal channels, usually
the fluorescence-like image and a BF-like companion image. This script makes
contact sheets that expose that structure without making a channel decision.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


FILENAME_RE = re.compile(r"_t(?P<t>\d+)_z(?P<z>\d+)_c(?P<c>\d+)\.jpg$", re.IGNORECASE)
ACQ_ORDER = ("ALEXA488", "DAPI", "ALEXA647", "mCherry")


@dataclass(frozen=True)
class GroupInfo:
    field: str
    acquisition: str
    display_name: str


def natural_key(text: str) -> tuple:
    parts = re.split(r"(\d+)", text)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def parse_group_name(name: str) -> GroupInfo:
    if name.startswith("N39Rep_"):
        stem = name.removeprefix("N39Rep_")
    else:
        stem = name
    stem = stem.removesuffix("_BF")
    if "_" in stem:
        field, acquisition = stem.split("_", 1)
    else:
        field, acquisition = stem, "unknown"
    acquisition = acquisition.replace("ALEZA", "ALEXA")
    return GroupInfo(field=field, acquisition=acquisition, display_name=name)


def collect_group_planes(group_dir: Path) -> dict[tuple[int, int], Path]:
    planes: dict[tuple[int, int], Path] = {}
    for path in sorted(group_dir.glob("*.jpg"), key=lambda p: natural_key(p.name)):
        match = FILENAME_RE.search(path.name)
        if not match:
            continue
        if int(match.group("t")) != 0:
            continue
        planes[(int(match.group("z")), int(match.group("c")))] = path
    return planes


def choose_mid_z(planes: dict[tuple[int, int], Path]) -> int | None:
    z_values = sorted({z for (z, c) in planes if (z, 0) in planes or (z, 1) in planes})
    if not z_values:
        return None
    complete = [z for z in z_values if (z, 0) in planes and (z, 1) in planes]
    candidates = complete or z_values
    return candidates[len(candidates) // 2]


def load_cell(path: Path | None, size: int, tint: str | None) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGB", (size, size), (8, 8, 8))
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image, cutoff=0.1)
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)
    if tint == "green":
        return Image.merge("RGB", (Image.new("L", image.size), image, Image.new("L", image.size)))
    if tint == "blue":
        return Image.merge("RGB", (Image.new("L", image.size), Image.new("L", image.size), image))
    if tint == "red":
        return Image.merge("RGB", (image, Image.new("L", image.size), Image.new("L", image.size)))
    return Image.merge("RGB", (image, image, image))


def tint_for(acquisition: str, channel: int) -> str | None:
    if channel == 1:
        return None
    acq = acquisition.upper()
    if "488" in acq:
        return "green"
    if "DAPI" in acq:
        return "blue"
    if "647" in acq or "MCHERRY" in acq:
        return "red"
    return None


def short_label(text: str, max_chars: int = 18) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "."


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill=fill, font=font)


def build_dataset_grid(
    grouped_root: Path,
    output_path: Path,
    title: str,
    max_fields: int | None = None,
    cell: int = 128,
) -> dict:
    group_dirs = [p for p in grouped_root.iterdir() if p.is_dir()]
    groups: dict[str, dict[str, tuple[GroupInfo, dict[tuple[int, int], Path], int | None]]] = {}
    for group_dir in group_dirs:
        info = parse_group_name(group_dir.name)
        planes = collect_group_planes(group_dir)
        mid_z = choose_mid_z(planes)
        groups.setdefault(info.field, {})[info.acquisition] = (info, planes, mid_z)

    fields = sorted(groups, key=natural_key)
    if max_fields is not None:
        fields = fields[:max_fields]
    acquisitions = [acq for acq in ACQ_ORDER if any(acq in groups[f] for f in fields)]
    extra = sorted(
        {acq for f in fields for acq in groups[f] if acq not in acquisitions},
        key=natural_key,
    )
    acquisitions.extend(extra)

    columns = [(acq, c) for acq in acquisitions for c in (0, 1)]
    label_w = 120
    header_h = 74
    caption_h = 32
    gap = 4
    width = label_w + len(columns) * (cell + gap) + gap
    height = header_h + len(fields) * (cell + caption_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (13, 17, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw_text(draw, (8, 8), title, (235, 238, 240), font)
    draw_text(draw, (8, 26), f"{grouped_root}", (150, 160, 166), font)
    for col_idx, (acq, c) in enumerate(columns):
        x = label_w + col_idx * (cell + gap)
        draw_text(draw, (x + 4, 50), f"{short_label(acq, 10)} c{c}", (210, 216, 220), font)

    for row_idx, field in enumerate(fields):
        y = header_h + row_idx * (cell + caption_h + gap)
        draw.rectangle((0, y, width, y + cell + caption_h), fill=(17, 22, 26))
        draw_text(draw, (8, y + 8), f"field {field}", (235, 238, 240), font)
        for col_idx, (acq, c) in enumerate(columns):
            x = label_w + col_idx * (cell + gap)
            entry = groups[field].get(acq)
            path = None
            z_label = "missing"
            if entry is not None:
                info, planes, mid_z = entry
                if mid_z is not None:
                    path = planes.get((mid_z, c))
                    z_label = f"z{mid_z:03d}"
                else:
                    z_label = "no-z"
            tile = load_cell(path, cell, tint_for(acq, c))
            canvas.paste(tile, (x, y))
            draw_text(draw, (x + 4, y + cell + 4), z_label, (138, 148, 156), font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "grouped_root": str(grouped_root),
        "output_path": str(output_path),
        "field_count": len(fields),
        "group_folder_count": len(group_dirs),
        "acquisitions": acquisitions,
        "columns": [f"{acq}_c{c}" for acq, c in columns],
    }


def find_first_planes(grouped_root: Path, acquisition: str, channel: int, max_count: int) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for group_dir in sorted((p for p in grouped_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        info = parse_group_name(group_dir.name)
        if info.acquisition != acquisition:
            continue
        planes = collect_group_planes(group_dir)
        z = choose_mid_z(planes)
        if z is None:
            continue
        path = planes.get((z, channel))
        if path is not None:
            candidates.append((f"{grouped_root.parent.name} field {info.field} {acquisition} c{channel}", path))
        if len(candidates) >= max_count:
            break
    return candidates


def build_brightfield_candidate_comparison(
    old_root: Path | None,
    d3_root: Path,
    d4_root: Path,
    output_path: Path,
    cell: int = 150,
) -> dict:
    rows: list[tuple[str, list[Path]]] = []

    if old_root and old_root.exists():
        old_paths = []
        for group_dir in sorted((p for p in old_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
            planes = collect_group_planes(group_dir)
            z = choose_mid_z(planes)
            if z is None:
                continue
            path = planes.get((z, 1))
            if path:
                old_paths.append(path)
            if len(old_paths) >= 6:
                break
        if old_paths:
            rows.append(("previous Data-Yichao-11 c1", old_paths))

    for label, root in (("D3 ALEXA488 c1", d3_root), ("D3 DAPI c1", d3_root), ("D4 ALEXA488 c1", d4_root), ("D4 DAPI c1", d4_root)):
        acq = "ALEXA488" if "ALEXA488" in label else "DAPI"
        paths = [p for _, p in find_first_planes(root, acq, 1, 6)]
        if paths:
            rows.append((label, paths))

    if not rows:
        return {"output_path": str(output_path), "rows": 0}

    label_w = 190
    header_h = 42
    gap = 4
    width = label_w + 6 * (cell + gap) + gap
    height = header_h + len(rows) * (cell + gap) + gap
    canvas = Image.new("RGB", (width, height), (13, 17, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw_text(draw, (8, 8), "BF candidate comparison: old c1 vs v2 c1 channels", (235, 238, 240), font)

    for row_idx, (label, paths) in enumerate(rows):
        y = header_h + row_idx * (cell + gap)
        draw.rectangle((0, y, width, y + cell), fill=(17, 22, 26))
        draw_text(draw, (8, y + 8), short_label(label, 28), (235, 238, 240), font)
        for col_idx in range(6):
            x = label_w + col_idx * (cell + gap)
            path = paths[col_idx] if col_idx < len(paths) else None
            canvas.paste(load_cell(path, cell, None), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"output_path": str(output_path), "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3-root", type=Path, required=True)
    parser.add_argument("--d4-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "d3": build_dataset_grid(args.d3_root, out / "yichao_v2_d3_all_fields_channel_preview.png", "Yichao v2 D3: all fields, named acquisitions, c0/c1"),
        "d4": build_dataset_grid(args.d4_root, out / "yichao_v2_d4_all_fields_channel_preview.png", "Yichao v2 D4: all fields, named acquisitions, c0/c1"),
        "bf_candidate_comparison": build_brightfield_candidate_comparison(
            args.old_root,
            args.d3_root,
            args.d4_root,
            out / "yichao_v2_brightfield_candidate_comparison.png",
        ),
    }
    summary_path = out / "yichao_v2_preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    for value in summary.values():
        if isinstance(value, dict) and "output_path" in value:
            print(value["output_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
