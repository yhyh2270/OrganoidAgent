#!/usr/bin/env python3
"""Create a 10x10 gallery of projected Yichao brightfield/fluorescence pairs."""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = (
    REPO_ROOT
    / "analysis-outputs"
    / "yichao_projected_instance_pairs"
    / "database"
    / "projected_instance_pairs.sqlite"
)
DEFAULT_OUTPUT = REPO_ROOT / "visualizations" / "yichao_projected_complete_pair_grid_10x10.png"


@dataclass(frozen=True)
class InstanceRow:
    dataset: str
    instance_id: str
    brightfield_crop_path: Path
    fluorescence_crop_path: Path
    area_px: float
    day_label: str
    position_label: str
    time_index: str


def natural_dataset_order(dataset: str) -> tuple[str, int]:
    prefix, _, suffix = dataset.rpartition("-")
    try:
        return prefix, int(suffix)
    except ValueError:
        return dataset, 0


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_evenly_spread(rows: list[InstanceRow], count: int) -> list[InstanceRow]:
    if len(rows) <= count:
        return rows
    rows = sorted(rows, key=lambda row: row.area_px)
    if count <= 1:
        return [rows[len(rows) // 2]]
    selected: list[InstanceRow] = []
    seen: set[int] = set()
    for i in range(count):
        idx = round(i * (len(rows) - 1) / (count - 1))
        while idx in seen and idx + 1 < len(rows):
            idx += 1
        seen.add(idx)
        selected.append(rows[idx])
    return selected


def fetch_rows(db_path: Path, per_dataset: int) -> tuple[list[str], dict[str, list[InstanceRow]], dict[str, int]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        datasets = [
            row["dataset"]
            for row in conn.execute("select distinct dataset from projected_instances order by dataset")
        ]
        datasets = sorted(datasets, key=natural_dataset_order)
        selected_by_dataset: dict[str, list[InstanceRow]] = {}
        complete_counts: dict[str, int] = {}
        for dataset in datasets:
            rows = conn.execute(
                """
                select
                    dataset,
                    instance_id,
                    brightfield_crop_path,
                    fluorescence_crop_path,
                    area_px,
                    coalesce(day_label, '') as day_label,
                    coalesce(position_label, '') as position_label,
                    coalesce(time_index, '') as time_index
                from projected_instances
                where dataset = ? and is_edge_padded = '0'
                order by cast(area_px as real), instance_id
                """,
                (dataset,),
            ).fetchall()
            instances = [
                InstanceRow(
                    dataset=row["dataset"],
                    instance_id=row["instance_id"],
                    brightfield_crop_path=Path(row["brightfield_crop_path"]),
                    fluorescence_crop_path=Path(row["fluorescence_crop_path"]),
                    area_px=coerce_float(row["area_px"]),
                    day_label=row["day_label"],
                    position_label=row["position_label"],
                    time_index=row["time_index"],
                )
                for row in rows
            ]
            complete_counts[dataset] = len(instances)
            selected_by_dataset[dataset] = select_evenly_spread(instances, per_dataset)
        return datasets, selected_by_dataset, complete_counts
    finally:
        conn.close()


def resize_contain(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (12, 14, 16))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_fluorescence_green(path: Path, size: int) -> Image.Image:
    image = Image.open(path)
    gray = image.convert("L")
    rgb = Image.merge("RGB", (Image.new("L", gray.size, 0), gray, Image.new("L", gray.size, 0)))
    return resize_contain(rgb, size)


def make_brightfield(path: Path, size: int) -> Image.Image:
    return resize_contain(Image.open(path), size)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (230, 236, 240),
) -> None:
    left, top, right, bottom = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + ((right - left) - (bbox[2] - bbox[0])) // 2
    y = top + ((bottom - top) - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def multiline_label(row: InstanceRow, pair_index: int) -> str:
    fields = [f"pair {pair_index + 1}", f"area {int(round(row.area_px))}"]
    if row.day_label:
        fields.append(row.day_label)
    if row.position_label:
        fields.append(row.position_label.replace("Position", "Pos"))
    if row.time_index:
        fields.append(f"t{row.time_index}")
    return " | ".join(fields)


def write_manifest(path: Path, selected_by_dataset: dict[str, list[InstanceRow]], complete_counts: dict[str, int]) -> None:
    lines = [
        "dataset\tcomplete_non_edge_count\tshown_index\tinstance_id\tarea_px\tday_label\tposition_label\ttime_index\tbrightfield_crop_path\tfluorescence_crop_path"
    ]
    for dataset, rows in selected_by_dataset.items():
        for idx, row in enumerate(rows, start=1):
            lines.append(
                "\t".join(
                    [
                        dataset,
                        str(complete_counts.get(dataset, 0)),
                        str(idx),
                        row.instance_id,
                        str(row.area_px),
                        row.day_label,
                        row.position_label,
                        row.time_index,
                        str(row.brightfield_crop_path),
                        str(row.fluorescence_crop_path),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gallery(
    datasets: Iterable[str],
    selected_by_dataset: dict[str, list[InstanceRow]],
    complete_counts: dict[str, int],
    output: Path,
    manifest: Path,
    tile_size: int,
    per_dataset: int,
) -> None:
    datasets = list(datasets)
    label_w = 190
    header_h = 74
    footer_h = 24
    gap = 6
    cols = per_dataset * 2
    rows = len(datasets)
    width = label_w + cols * tile_size + (cols + 1) * gap
    height = header_h + rows * (tile_size + footer_h) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (20, 23, 25))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(16)
    small_font = load_font(12)
    tiny_font = load_font(10)

    draw.text(
        (16, 12),
        "Projected Yichao complete organoid pairs (brightfield c1 projection / fluorescence c0 projection)",
        font=title_font,
        fill=(242, 246, 248),
    )
    draw.text(
        (16, 44),
        f"Rows=datasets; columns=5 organoids x BF/FL; filter=is_edge_padded='0'; tile={tile_size}px",
        font=small_font,
        fill=(170, 182, 190),
    )

    for pair_idx in range(per_dataset):
        for channel_idx, channel in enumerate(("BF", "FL")):
            col = pair_idx * 2 + channel_idx
            x0 = label_w + gap + col * (tile_size + gap)
            draw_centered_text(
                draw,
                (x0, header_h - 28, x0 + tile_size, header_h - 4),
                f"{pair_idx + 1} {channel}",
                small_font,
                (210, 220, 224),
            )

    for row_idx, dataset in enumerate(datasets):
        y0 = header_h + gap + row_idx * (tile_size + footer_h + gap)
        row_bottom = y0 + tile_size + footer_h
        draw.rectangle((0, y0 - 2, width, row_bottom + 2), fill=(24, 28, 31) if row_idx % 2 == 0 else (18, 21, 24))
        draw.text((14, y0 + 10), dataset, font=label_font, fill=(236, 241, 244))
        draw.text(
            (14, y0 + 34),
            f"complete: {complete_counts.get(dataset, 0)}",
            font=small_font,
            fill=(157, 175, 184),
        )
        rows_for_dataset = selected_by_dataset.get(dataset, [])
        for pair_idx in range(per_dataset):
            if pair_idx >= len(rows_for_dataset):
                for channel_idx in range(2):
                    col = pair_idx * 2 + channel_idx
                    x0 = label_w + gap + col * (tile_size + gap)
                    draw.rectangle((x0, y0, x0 + tile_size, y0 + tile_size), fill=(38, 42, 45))
                    draw_centered_text(draw, (x0, y0, x0 + tile_size, y0 + tile_size), "missing", small_font)
                continue

            instance = rows_for_dataset[pair_idx]
            images = [
                make_brightfield(instance.brightfield_crop_path, tile_size),
                make_fluorescence_green(instance.fluorescence_crop_path, tile_size),
            ]
            for channel_idx, image in enumerate(images):
                col = pair_idx * 2 + channel_idx
                x0 = label_w + gap + col * (tile_size + gap)
                canvas.paste(image, (x0, y0))
                draw.rectangle((x0, y0, x0 + tile_size, y0 + tile_size), outline=(70, 78, 82), width=1)
            label_x = label_w + gap + pair_idx * 2 * (tile_size + gap)
            draw.text(
                (label_x, y0 + tile_size + 4),
                multiline_label(instance, pair_idx)[:46],
                font=tiny_font,
                fill=(160, 174, 181),
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    write_manifest(manifest, selected_by_dataset, complete_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--per-dataset", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=144)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_dataset <= 0:
        raise SystemExit("--per-dataset must be positive")
    if not args.db_path.exists():
        raise SystemExit(f"Missing database: {args.db_path}")
    manifest = args.manifest or args.output.with_suffix(".tsv")
    datasets, selected_by_dataset, complete_counts = fetch_rows(args.db_path, args.per_dataset)
    build_gallery(
        datasets=datasets,
        selected_by_dataset=selected_by_dataset,
        complete_counts=complete_counts,
        output=args.output,
        manifest=manifest,
        tile_size=args.tile_size,
        per_dataset=args.per_dataset,
    )
    print(args.output)
    print(manifest)
    print(f"datasets={len(datasets)} organoids_shown={sum(len(v) for v in selected_by_dataset.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
