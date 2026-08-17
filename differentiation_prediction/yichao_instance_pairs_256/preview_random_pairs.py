#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_root = repo_root / "analysis-outputs" / "yichao_instance_pairs_resized_256"
    parser = argparse.ArgumentParser(description="Render a random preview grid from the prepared 256x256 Yichao instance-pair dataset.")
    parser.add_argument("--dataset-root", default=str(default_root))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--datasets", nargs="*", default=[])
    return parser.parse_args()


def load_manifest(dataset_root: Path) -> list[dict[str, str]]:
    manifest_path = dataset_root / "metadata" / "pairs_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.datasets:
        allowed = set(args.datasets)
        rows = [row for row in rows if row["dataset"] in allowed]
    if not rows:
        raise RuntimeError("No pair rows matched the requested preview filters.")
    if args.indices:
        selected = []
        for index in args.indices:
            if index < 0 or index >= len(rows):
                raise IndexError(f"Pair index out of range: {index}")
            selected.append(rows[index])
        return selected
    count = min(args.count, len(rows))
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[idx] for idx in selected_indices]


def add_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    draw.rectangle((x, y, x + 512, y + 38), fill=(255, 255, 255))
    draw.text((x + 8, y + 8), text, fill=(0, 0, 0), font=font)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    rows = load_manifest(dataset_root)
    selected = select_rows(rows, args)

    pair_width = 256 * 2
    label_height = 40
    cell_height = 256 + label_height
    columns = max(1, args.columns)
    rows_n = math.ceil(len(selected) / columns)
    margin = 20
    gutter = 16
    canvas_w = margin * 2 + columns * pair_width + (columns - 1) * gutter
    canvas_h = margin * 2 + rows_n * cell_height + (rows_n - 1) * gutter

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    preview_dir = dataset_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_rows = []
    for idx, row in enumerate(selected):
        grid_x = idx % columns
        grid_y = idx // columns
        x0 = margin + grid_x * (pair_width + gutter)
        y0 = margin + grid_y * (cell_height + gutter)
        label = f"idx {row['pair_index']} | {row['dataset']} | {row['object_name']}"
        add_label(draw, x0, y0, label, font)

        with Image.open(row["brightfield_256_path"]) as bf:
            bf_rgb = bf.convert("RGB")
        with Image.open(row["fluorescence_256_path"]) as fl:
            fl_rgb = fl.convert("RGB")

        canvas.paste(bf_rgb, (x0, y0 + label_height))
        canvas.paste(fl_rgb, (x0 + 256, y0 + label_height))
        draw.text((x0 + 8, y0 + label_height + 8), "brightfield", fill=(255, 255, 255), font=font)
        draw.text((x0 + 256 + 8, y0 + label_height + 8), "fluorescence", fill=(255, 255, 255), font=font)
        preview_rows.append(
            {
                "pair_index": int(row["pair_index"]),
                "dataset": row["dataset"],
                "object_name": row["object_name"],
                "brightfield_256_path": row["brightfield_256_path"],
                "fluorescence_256_path": row["fluorescence_256_path"],
            }
        )

    if args.indices:
        suffix = "indices_" + "_".join(str(value) for value in args.indices)
    else:
        suffix = f"random_{len(selected)}_seed_{args.seed}"
    preview_path = preview_dir / f"pair_preview_{suffix}.png"
    preview_meta_path = preview_dir / f"pair_preview_{suffix}.json"

    canvas.save(preview_path)
    preview_meta_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_root": str(dataset_root),
                "preview_path": str(preview_path),
                "rows": preview_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(preview_path)
    print(preview_meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
