#!/usr/bin/env python3
"""Inspect Yichao v2 Leica LIF metadata without extracting image planes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from readlif.reader import LifFile


def natural_key(text: str) -> tuple:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def infer_day_from_name(path: Path) -> str | None:
    match = re.search(r"_D(F_)?D(?P<day>\d+)", path.stem, re.IGNORECASE)
    if match:
        return f"D{match.group('day')}"
    match = re.search(r"_DF_(?P<day>\d+)$", path.stem, re.IGNORECASE)
    if match:
        return f"D{match.group('day')}"
    return None


def inspect_lif(path: Path, v2_root: Path) -> dict[str, Any]:
    lif = LifFile(str(path))
    rows: list[dict[str, Any]] = []
    fields: defaultdict[str, list[str]] = defaultdict(list)
    channel_counts: Counter[int] = Counter()
    time_counts: Counter[int] = Counter()
    z_counts: list[int] = []
    for image_index in range(len(lif.image_list)):
        image = lif.get_image(image_index)
        name = getattr(image, "name", f"image_{image_index}")
        normalized = name.replace("#", "_").replace(" ", "_")
        parts = [part for part in normalized.split("_") if part]
        field = "unknown"
        acquisition = name
        if parts:
            if parts[0].startswith("N39Rep") and len(parts) >= 2:
                field = parts[1]
                acquisition = "_".join(parts[2:]) or name
            elif parts[0].isdigit():
                field = parts[0]
                acquisition = "_".join(parts[1:]) or name
            elif parts[0].lower().startswith("series"):
                field = parts[0]
                acquisition = parts[0]
        acquisition = acquisition.replace("ALEZA", "ALEXA")
        fields[field].append(name)
        channel_counts[int(image.channels)] += 1
        time_counts[int(image.nt)] += 1
        z_counts.append(int(image.nz))
        rows.append(
            {
                "dataset_folder": str(path.parent.relative_to(v2_root)),
                "lif_path": str(path.relative_to(v2_root)),
                "day_hint": infer_day_from_name(path),
                "image_index": image_index,
                "image_name": name,
                "field_hint": field,
                "acquisition_hint": acquisition,
                "width_px": int(image.dims.x),
                "height_px": int(image.dims.y),
                "z_count": int(image.nz),
                "time_count": int(image.nt),
                "channel_count": int(image.channels),
                "scale_x_um": image.scale[0],
                "scale_y_um": image.scale[1],
                "scale_z_um": image.scale[2],
            }
        )
    return {
        "dataset_folder": str(path.parent.relative_to(v2_root)),
        "lif_path": str(path.relative_to(v2_root)),
        "day_hint": infer_day_from_name(path),
        "image_count": len(lif.image_list),
        "field_count": len(fields),
        "series_per_field_distribution": dict(sorted(Counter(len(v) for v in fields.values()).items())),
        "channel_counts_per_series": dict(sorted(channel_counts.items())),
        "time_counts_per_series": dict(sorted(time_counts.items())),
        "z_min": min(z_counts) if z_counts else None,
        "z_max": max(z_counts) if z_counts else None,
        "z_unique": sorted(set(z_counts)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=Path("DATA-Yichao-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-outputs/yichao_v2_metadata"))
    args = parser.parse_args()

    v2_root = args.v2_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [inspect_lif(path, v2_root) for path in sorted(v2_root.glob("*/*.lif"), key=lambda p: natural_key(str(p)))]
    rows = [row for summary in summaries for row in summary["rows"]]
    for summary in summaries:
        summary.pop("rows", None)

    summary_path = output_dir / "yichao_v2_lif_metadata_summary.json"
    csv_path = output_dir / "yichao_v2_lif_series_metadata.csv"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(summary_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
