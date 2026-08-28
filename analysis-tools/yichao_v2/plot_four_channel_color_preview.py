#!/usr/bin/env python3
"""Plot Yichao v2 named channels with biological color coding and BF context."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


FILENAME_RE = re.compile(r"_t(?P<t>\d+)_z(?P<z>\d+)_c(?P<c>\d+)\.jpg$", re.IGNORECASE)
ACQUISITIONS = ("ALEXA488", "DAPI", "ALEXA647", "mCherry")
CHANNEL_COLORS = {
    "ALEXA488": (0, 255, 0),
    "DAPI": (40, 80, 255),
    "ALEXA647": (255, 40, 40),
    "mCherry": (255, 0, 180),
}


def natural_key(text: str) -> tuple:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_group_name(name: str) -> tuple[str, str]:
    stem = name.removeprefix("N39Rep_").removesuffix("_BF")
    if "_" in stem:
        field, acquisition = stem.split("_", 1)
    else:
        field, acquisition = stem, "unknown"
    return field, acquisition.replace("ALEZA", "ALEXA")


def collect_planes(group_dir: Path) -> dict[tuple[int, int], Path]:
    planes: dict[tuple[int, int], Path] = {}
    for path in group_dir.glob("*.jpg"):
        match = FILENAME_RE.search(path.name)
        if not match or int(match.group("t")) != 0:
            continue
        planes[(int(match.group("z")), int(match.group("c")))] = path
    return planes


def choose_mid_z(planes: dict[tuple[int, int], Path]) -> int | None:
    z_values = sorted({z for z, _ in planes})
    complete = [z for z in z_values if (z, 0) in planes and (z, 1) in planes]
    candidates = complete or z_values
    return candidates[len(candidates) // 2] if candidates else None


def load_gray(path: Path | None, size: int) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("L", (size, size), 0)
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image, cutoff=0.1)
    return ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)


def colorize(gray: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    channels = [gray.point(lambda value, weight=weight: int(value * weight / 255)) for weight in color]
    return Image.merge("RGB", tuple(channels))


def build_preview(grouped_root: Path, output_path: Path, title: str, cell: int = 138) -> None:
    grouped: dict[str, dict[str, tuple[dict[tuple[int, int], Path], int | None]]] = {}
    for group_dir in sorted((p for p in grouped_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        field, acquisition = parse_group_name(group_dir.name)
        if acquisition not in ACQUISITIONS:
            continue
        planes = collect_planes(group_dir)
        grouped.setdefault(field, {})[acquisition] = (planes, choose_mid_z(planes))

    fields = sorted(grouped, key=natural_key)
    columns = ["BF: ALEXA488 c1", "ALEXA488 c0", "DAPI c0", "ALEXA647 c0", "mCherry c0", "c0 composite"]
    label_w = 110
    header_h = 72
    caption_h = 26
    gap = 4
    width = label_w + len(columns) * (cell + gap) + gap
    height = header_h + len(fields) * (cell + caption_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (12, 15, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((8, 8), title, fill=(235, 238, 240), font=font)
    draw.text((8, 26), str(grouped_root), fill=(145, 155, 162), font=font)
    for idx, label in enumerate(columns):
        x = label_w + idx * (cell + gap)
        draw.text((x + 4, 50), label, fill=(215, 220, 224), font=font)

    for row_idx, field in enumerate(fields):
        y = header_h + row_idx * (cell + caption_h + gap)
        draw.rectangle((0, y, width, y + cell + caption_h), fill=(17, 22, 26))
        draw.text((8, y + 8), f"field {field}", fill=(235, 238, 240), font=font)

        alexa_planes, alexa_z = grouped[field].get("ALEXA488", ({}, None))
        bf = load_gray(alexa_planes.get((alexa_z, 1)) if alexa_z is not None else None, cell)
        canvas.paste(Image.merge("RGB", (bf, bf, bf)), (label_w, y))
        draw.text((label_w + 4, y + cell + 4), f"ALEXA488 c1 z{alexa_z:03d}" if alexa_z is not None else "missing", fill=(150, 160, 168), font=font)

        composite_channels = []
        for col_offset, acquisition in enumerate(ACQUISITIONS, start=1):
            x = label_w + col_offset * (cell + gap)
            planes, z = grouped[field].get(acquisition, ({}, None))
            gray = load_gray(planes.get((z, 0)) if z is not None else None, cell)
            composite_channels.append((gray, CHANNEL_COLORS[acquisition]))
            canvas.paste(colorize(gray, CHANNEL_COLORS[acquisition]), (x, y))
            draw.text((x + 4, y + cell + 4), f"c0 z{z:03d}" if z is not None else "missing", fill=(150, 160, 168), font=font)

        comp_r = Image.new("L", (cell, cell), 0)
        comp_g = Image.new("L", (cell, cell), 0)
        comp_b = Image.new("L", (cell, cell), 0)
        for gray, color in composite_channels:
            red, green, blue = colorize(gray, color).split()
            comp_r = ImageChops.lighter(comp_r, red)
            comp_g = ImageChops.lighter(comp_g, green)
            comp_b = ImageChops.lighter(comp_b, blue)
        x = label_w + 5 * (cell + gap)
        canvas.paste(Image.merge("RGB", (comp_r, comp_g, comp_b)), (x, y))
        draw.text((x + 4, y + cell + 4), "c0 composite", fill=(150, 160, 168), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3-root", type=Path, required=True)
    parser.add_argument("--d4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_preview(
        args.d3_root,
        args.output_dir / "yichao_v2_d3_four_named_channels_plus_bf_color.png",
        "Yichao v2 D3: BF candidate plus four named c0 channels",
    )
    build_preview(
        args.d4_root,
        args.output_dir / "yichao_v2_d4_four_named_channels_plus_bf_color.png",
        "Yichao v2 D4: BF candidate plus four named c0 channels",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
