#!/usr/bin/env python3
"""Prepare paired brightfield/fluorescence data as .npy tensors.

Default behavior matches the requested fast setup:
- Train root: Data-Yichao-2
- Test root: Data-Yichao-1
- Pick first c0/c1 pair per top-level train folder
- Crop each selected pair into 4 tiles of 512x512
- Save paired arrays to .npy with progress logs
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

CHANNEL_PATTERN = re.compile(r"^(?P<base>.+)_c(?P<channel>\d+)\.(?P<ext>jpe?g)$", re.IGNORECASE)


@dataclass(frozen=True)
class PairItem:
    input_path: Path
    target_path: Path
    key: str


def log(msg: str) -> None:
    print(msg, flush=True)


def scan_pairs(root: Path, input_channel: int, target_channel: int, scan_log_interval: int) -> List[PairItem]:
    root = root.resolve()
    grouped: Dict[str, Dict[int, Path]] = {}
    scanned = 0
    matched = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        if scan_log_interval > 0 and scanned % scan_log_interval == 0:
            log(f"[scan:{root.name}] scanned={scanned} matched={matched}")
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        m = CHANNEL_PATTERN.match(path.name)
        if not m:
            continue
        matched += 1
        base = m.group("base")
        channel = int(m.group("channel"))
        key = f"{path.parent.relative_to(root)}/{base}"
        grouped.setdefault(key, {})[channel] = path

    pairs: List[PairItem] = []
    for key in sorted(grouped.keys()):
        channels = grouped[key]
        if input_channel in channels and target_channel in channels:
            pairs.append(PairItem(channels[input_channel], channels[target_channel], key))

    log(
        f"[scan:{root.name}] done: files={scanned}, channel_files={matched}, "
        f"candidate_keys={len(grouped)}, pairs={len(pairs)}"
    )
    return pairs


def select_first_pair_per_object(pairs: Sequence[PairItem], dataset_label: str) -> List[PairItem]:
    by_top: Dict[str, List[PairItem]] = {}
    for p in pairs:
        top = p.key.split("/", 1)[0]
        by_top.setdefault(top, []).append(p)

    selected: List[PairItem] = []
    for top in sorted(by_top.keys()):
        first = sorted(by_top[top], key=lambda x: x.key)[0]
        selected.append(first)
        log(f"[select:{dataset_label}] {top} -> {first.key}")
    log(f"[select:{dataset_label}] selected {len(selected)} pairs from {len(pairs)}")
    return selected


def crop_boxes(width: int, height: int, crop_size: int, crops_per_image: int) -> List[Tuple[int, int, int, int]]:
    if width < crop_size or height < crop_size:
        return [(0, 0, width, height)]

    if crops_per_image == 4 and width >= crop_size * 2 and height >= crop_size * 2:
        return [
            (0, 0, crop_size, crop_size),
            (width - crop_size, 0, width, crop_size),
            (0, height - crop_size, crop_size, height),
            (width - crop_size, height - crop_size, width, height),
        ]

    xs = np.linspace(0, max(width - crop_size, 0), num=max(1, int(np.ceil(np.sqrt(crops_per_image)))), dtype=int)
    ys = np.linspace(0, max(height - crop_size, 0), num=max(1, int(np.ceil(crops_per_image / len(xs)))), dtype=int)

    boxes: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((int(x), int(y), int(x + crop_size), int(y + crop_size)))
            if len(boxes) >= crops_per_image:
                return boxes
    return boxes


def pair_to_arrays(
    pairs: Sequence[PairItem],
    dataset_label: str,
    crop_size: int,
    crops_per_image: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    meta: List[Dict[str, object]] = []

    total_pairs = len(pairs)
    for i, pair in enumerate(pairs, start=1):
        with Image.open(pair.input_path).convert("L") as in_probe:
            width, height = in_probe.size
        boxes = crop_boxes(width, height, crop_size, crops_per_image)
        log(f"[prep:{dataset_label}] pair {i}/{total_pairs} {pair.key} size={width}x{height} crops={len(boxes)}")

        with Image.open(pair.input_path).convert("L") as in_img, Image.open(pair.target_path).convert("L") as tgt_img:
            for j, box in enumerate(boxes, start=1):
                in_crop = in_img.crop(box)
                tgt_crop = tgt_img.crop(box)
                if image_size > 0 and in_crop.size != (image_size, image_size):
                    in_crop = in_crop.resize((image_size, image_size), Image.BILINEAR)
                    tgt_crop = tgt_crop.resize((image_size, image_size), Image.BILINEAR)

                inputs.append(np.asarray(in_crop, dtype=np.uint8))
                targets.append(np.asarray(tgt_crop, dtype=np.uint8))
                meta.append(
                    {
                        "key": pair.key,
                        "crop_index": j,
                        "crop_box": list(box),
                        "input_path": str(pair.input_path),
                        "target_path": str(pair.target_path),
                    }
                )

    if not inputs:
        raise RuntimeError(f"No samples prepared for {dataset_label}")

    x = np.stack(inputs, axis=0)[:, None, :, :]
    y = np.stack(targets, axis=0)[:, None, :, :]
    log(f"[prep:{dataset_label}] final tensor shape: X={x.shape} Y={y.shape} dtype={x.dtype}")
    return x, y, meta


def save_arrays(
    out_dir: Path,
    prefix: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_input.npy", x)
    np.save(out_dir / f"{prefix}_target.npy", y)
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        f"[save] {prefix}: "
        f"{out_dir / (prefix + '_input.npy')} | {out_dir / (prefix + '_target.npy')} | "
        f"{out_dir / (prefix + '_meta.json')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Yichao c0/c1 pairs into .npy")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object"),
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results/yichao_paired_npy"),
    )
    parser.add_argument("--input-channel", type=int, default=0)
    parser.add_argument("--target-channel", type=int, default=1)
    parser.add_argument("--scan-log-interval", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--train-crops-per-image", type=int, default=4)
    parser.add_argument("--test-crops-per-image", type=int, default=4)
    parser.add_argument("--train-first-pair-per-object", action="store_true", default=True)
    parser.add_argument("--test-first-pair-per-object", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    t0 = time.perf_counter()
    log("== Prepare Yichao pairs to NPY ==")
    log(f"Train root:  {args.train_root}")
    log(f"Test root:   {args.test_root}")
    log(f"Output dir:  {args.output_dir}")

    train_pairs = scan_pairs(args.train_root, args.input_channel, args.target_channel, args.scan_log_interval)
    test_pairs = scan_pairs(args.test_root, args.input_channel, args.target_channel, args.scan_log_interval)

    if args.train_first_pair_per_object:
        train_pairs = select_first_pair_per_object(train_pairs, "train")
    if args.test_first_pair_per_object:
        test_pairs = select_first_pair_per_object(test_pairs, "test")

    x_train, y_train, train_meta = pair_to_arrays(
        train_pairs,
        dataset_label="train",
        crop_size=args.crop_size,
        crops_per_image=args.train_crops_per_image,
        image_size=args.image_size,
    )
    x_test, y_test, test_meta = pair_to_arrays(
        test_pairs,
        dataset_label="test",
        crop_size=args.crop_size,
        crops_per_image=args.test_crops_per_image,
        image_size=args.image_size,
    )

    save_arrays(args.output_dir, "train", x_train, y_train, train_meta)
    save_arrays(args.output_dir, "test", x_test, y_test, test_meta)

    summary = {
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "shape": [int(v) for v in x_train.shape[1:]],
        "dtype": str(x_train.dtype),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[done] summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
