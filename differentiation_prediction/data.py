"""Dataset utilities for manifest-driven pix2pix training."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairSample:
    record_id: str
    dataset_id: str
    position_uid: str
    position_name: str
    time_index: int
    z_index: int
    width_px: int
    height_px: int
    input_path: Path
    target_path: Path
    crop_box: Tuple[int, int, int, int] | None


def read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tile_boxes(width: int, height: int, tile_size: int, tile_mode: str) -> List[Tuple[int, int, int, int]]:
    if tile_mode == "full" or width <= tile_size or height <= tile_size:
        return [(0, 0, width, height)]

    if tile_mode == "quadrants" and width >= tile_size * 2 and height >= tile_size * 2:
        return [
            (0, 0, tile_size, tile_size),
            (width - tile_size, 0, width, tile_size),
            (0, height - tile_size, tile_size, height),
            (width - tile_size, height - tile_size, width, height),
        ]

    grid_cols = max(1, int(math.ceil(width / tile_size)))
    grid_rows = max(1, int(math.ceil(height / tile_size)))
    x_positions = np.linspace(0, max(width - tile_size, 0), num=grid_cols, dtype=int).tolist()
    y_positions = np.linspace(0, max(height - tile_size, 0), num=grid_rows, dtype=int).tolist()
    return [(int(x), int(y), int(x + tile_size), int(y + tile_size)) for y in y_positions for x in x_positions]


def _subsample_rows(
    rows: Sequence[Dict[str, str]],
    time_stride: int,
    z_stride: int,
    max_pairs_per_position: int | None,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["position_uid"], []).append(row)

    selected: List[Dict[str, str]] = []
    for position_uid, items in sorted(grouped.items()):
        kept = [
            row
            for row in sorted(items, key=lambda item: (int(item["time_index"]), int(item["z_index"])))
            if int(row["time_index"]) % max(time_stride, 1) == 0 and int(row["z_index"]) % max(z_stride, 1) == 0
        ]
        if max_pairs_per_position is not None:
            kept = kept[: max_pairs_per_position]
        selected.extend(kept)
    return selected


def build_samples(
    manifest_path: Path,
    split_column: str,
    split_name: str,
    tile_size: int,
    tile_mode: str,
    time_stride: int = 1,
    z_stride: int = 1,
    max_pairs_per_position: int | None = None,
) -> List[PairSample]:
    rows = [row for row in read_manifest_rows(manifest_path) if row[split_column] == split_name]
    rows = _subsample_rows(rows, time_stride=time_stride, z_stride=z_stride, max_pairs_per_position=max_pairs_per_position)
    samples: List[PairSample] = []
    for row in rows:
        width_px = int(row["width_px"])
        height_px = int(row["height_px"])
        boxes = _tile_boxes(width_px, height_px, tile_size=tile_size, tile_mode=tile_mode)
        for tile_index, box in enumerate(boxes, start=1):
            samples.append(
                PairSample(
                    record_id=f"{row['record_id']}|tile{tile_index}",
                    dataset_id=row["dataset_id"],
                    position_uid=row["position_uid"],
                    position_name=row["position_name"],
                    time_index=int(row["time_index"]),
                    z_index=int(row["z_index"]),
                    width_px=width_px,
                    height_px=height_px,
                    input_path=Path(row["input_path"]),
                    target_path=Path(row["target_path"]),
                    crop_box=box,
                )
            )
    return samples


class ManifestPairedDataset(Dataset):
    def __init__(self, samples: Sequence[PairSample], image_size: int, random_flip: bool) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gray(self, path: Path, crop_box: Tuple[int, int, int, int] | None) -> Image.Image:
        image = Image.open(path).convert("L")
        if crop_box is not None:
            image = image.crop(crop_box)
        if self.image_size > 0 and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        input_image = self._load_gray(sample.input_path, sample.crop_box)
        target_image = self._load_gray(sample.target_path, sample.crop_box)
        if self.random_flip and random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)
        return {
            "input": self._to_tensor(input_image),
            "target": self._to_tensor(target_image),
            "key": sample.record_id,
            "position_uid": sample.position_uid,
        }
