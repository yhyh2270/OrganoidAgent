"""Build manifest-first views of the Yichao brightfield/fluorescence data."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PAIR_PATTERN = re.compile(
    r"^(?P<series_index>\d+)_(?P<object_name>.+)_t(?P<time_index>\d+)_z(?P<z_index>\d+)_c(?P<channel>\d+)\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)
Y3_Y4_POSITION_PATTERN = re.compile(
    r"^Experiment_1_Day_(?P<day>\d+)_Position(?P<position>\d+)$",
    re.IGNORECASE,
)
Y2_DYNAMIC_PATTERN = re.compile(
    r"^N39_TriRep_DF_D(?P<day>\d+)_Position(?P<position>\d+)$",
    re.IGNORECASE,
)
STATIC_PATTERN = re.compile(
    r"^N39_TriReP?_MUC2_mNeon_20X_(?P<sample>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    grouped_root: Path
    source_lif: Path
    width_px: int
    height_px: int
    pixel_size_um: float | None
    notes: str


@dataclass(frozen=True)
class PairRecord:
    record_id: str
    dataset_id: str
    source_lif: str
    grouped_root: str
    pair_key: str
    position_uid: str
    position_name: str
    sample_family: str
    experiment_group: str
    imaging_mode: str
    day_index: int | None
    position_index: int | None
    sample_index: int | None
    series_index: int
    time_index: int
    z_index: int
    width_px: int
    height_px: int
    pixel_size_um: float | None
    domain_group: str
    duplicate_policy: str
    baseline_split: str
    all_dynamic_split: str
    input_path: str
    target_path: str


def default_dataset_specs(repo_root: Path | None = None) -> List[DatasetSpec]:
    base = repo_root or Path("/home/lachlan/ProjectsLFS/OrganoidAgent")
    return [
        DatasetSpec(
            dataset_id="Y1",
            grouped_root=base / "Data-Yichao-v1" / "Data-Yichao-1" / "P11N&N39_Rep_DF_jpeg_all_by_object",
            source_lif=base / "Data-Yichao-v1" / "Data-Yichao-1" / "P11N&N39_Rep_DF.lif",
            width_px=1024,
            height_px=1024,
            pixel_size_um=0.303,
            notes="Static MUC2 samples duplicated in Y2; use for debug only.",
        ),
        DatasetSpec(
            dataset_id="Y2",
            grouped_root=base / "Data-Yichao-v1" / "Data-Yichao-2" / "P11N&N39_Rep_DF_jpeg_all_by_object",
            source_lif=base / "Data-Yichao-v1" / "Data-Yichao-2" / "P11N&N39_Rep_DF.lif",
            width_px=1024,
            height_px=1024,
            pixel_size_um=0.568,
            notes="Mixed file: static duplicates plus dynamic Day-2 monitoring.",
        ),
        DatasetSpec(
            dataset_id="Y3",
            grouped_root=base / "Data-Yichao-v1" / "Data-Yichao-3" / "N39_TriRep_DF_jpeg_all_by_position",
            source_lif=base / "Data-Yichao-v1" / "Data-Yichao-3" / "N39_TriRep_DF.lif",
            width_px=512,
            height_px=512,
            pixel_size_um=1.137,
            notes="Primary dynamic monitoring LIF with Day 2/3/4 positions.",
        ),
        DatasetSpec(
            dataset_id="Y4",
            grouped_root=base / "Data-Yichao-v1" / "Data-Yichao-4" / "N39_TriRep_DF_2_jpeg_all_by_position",
            source_lif=base / "Data-Yichao-v1" / "Data-Yichao-4" / "N39_TriRep_DF_2.lif",
            width_px=512,
            height_px=512,
            pixel_size_um=None,
            notes="Second dynamic monitoring LIF with Day 2/3 positions.",
        ),
    ]


def log(message: str) -> None:
    print(message, flush=True)


def _position_metadata(dataset_id: str, position_name: str) -> Dict[str, object]:
    match = Y3_Y4_POSITION_PATTERN.match(position_name)
    if match:
        day_index = int(match.group("day"))
        position_index = int(match.group("position"))
        return {
            "sample_family": f"{dataset_id}::{position_name}",
            "experiment_group": "Experiment_1",
            "imaging_mode": "dynamic",
            "day_index": day_index,
            "position_index": position_index,
            "sample_index": None,
            "domain_group": "dynamic_512",
            "duplicate_policy": "keep",
        }

    match = Y2_DYNAMIC_PATTERN.match(position_name)
    if match:
        day_index = int(match.group("day"))
        position_index = int(match.group("position"))
        return {
            "sample_family": f"{dataset_id}::{position_name}",
            "experiment_group": "N39_TriRep_DF_D2",
            "imaging_mode": "dynamic",
            "day_index": day_index,
            "position_index": position_index,
            "sample_index": None,
            "domain_group": "dynamic_1024",
            "duplicate_policy": "keep",
        }

    match = STATIC_PATTERN.match(position_name)
    if match:
        sample_index = int(match.group("sample"))
        return {
            "sample_family": f"static_muc2::{position_name.lower()}",
            "experiment_group": "MUC2_static",
            "imaging_mode": "static",
            "day_index": None,
            "position_index": None,
            "sample_index": sample_index,
            "domain_group": "static_1024",
            "duplicate_policy": "exclude_duplicate_static" if dataset_id == "Y1" else "exclude_static",
        }

    raise ValueError(f"Unrecognized position name: {dataset_id} {position_name}")


def scan_dataset(
    spec: DatasetSpec,
    input_channel: int = 0,
    target_channel: int = 1,
) -> List[PairRecord]:
    if not spec.grouped_root.exists():
        raise FileNotFoundError(f"Grouped root not found: {spec.grouped_root}")

    records: List[PairRecord] = []
    position_dirs = sorted(path for path in spec.grouped_root.iterdir() if path.is_dir())
    for position_dir in position_dirs:
        position_name = position_dir.name
        meta = _position_metadata(spec.dataset_id, position_name)
        grouped: Dict[str, Dict[int, Path]] = {}

        for path in sorted(position_dir.iterdir()):
            if not path.is_file():
                continue
            match = PAIR_PATTERN.match(path.name)
            if not match:
                continue
            channel = int(match.group("channel"))
            pair_key = (
                f"{match.group('series_index')}_{match.group('object_name')}"
                f"_t{match.group('time_index')}_z{match.group('z_index')}"
            )
            grouped.setdefault(pair_key, {})[channel] = path

        for pair_key, channels in sorted(grouped.items()):
            if input_channel not in channels or target_channel not in channels:
                continue

            match = PAIR_PATTERN.match(channels[input_channel].name)
            if match is None:
                continue
            position_uid = f"{spec.dataset_id}:{position_name}"
            record_id = f"{position_uid}:{pair_key}"
            records.append(
                PairRecord(
                    record_id=record_id,
                    dataset_id=spec.dataset_id,
                    source_lif=str(spec.source_lif),
                    grouped_root=str(spec.grouped_root),
                    pair_key=pair_key,
                    position_uid=position_uid,
                    position_name=position_name,
                    sample_family=str(meta["sample_family"]),
                    experiment_group=str(meta["experiment_group"]),
                    imaging_mode=str(meta["imaging_mode"]),
                    day_index=meta["day_index"],
                    position_index=meta["position_index"],
                    sample_index=meta["sample_index"],
                    series_index=int(match.group("series_index")),
                    time_index=int(match.group("time_index")),
                    z_index=int(match.group("z_index")),
                    width_px=spec.width_px,
                    height_px=spec.height_px,
                    pixel_size_um=spec.pixel_size_um,
                    domain_group=str(meta["domain_group"]),
                    duplicate_policy=str(meta["duplicate_policy"]),
                    baseline_split="unassigned",
                    all_dynamic_split="unassigned",
                    input_path=str(channels[input_channel]),
                    target_path=str(channels[target_channel]),
                )
            )

    log(f"[manifest:{spec.dataset_id}] positions={len(position_dirs)} pairs={len(records)}")
    return records


def _counts_for_split(total_groups: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if total_groups < 3:
        return 0, 0
    val_count = max(1, int(round(total_groups * val_ratio)))
    test_count = max(1, int(round(total_groups * test_ratio)))
    while total_groups - val_count - test_count < 1:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
    return val_count, test_count


def _split_group_ids(group_ids: Sequence[str], seed: int, val_ratio: float, test_ratio: float) -> Dict[str, str]:
    items = list(sorted(group_ids))
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count, test_count = _counts_for_split(len(items), val_ratio, test_ratio)
    assignments: Dict[str, str] = {}
    for idx, group_id in enumerate(items):
        if idx < val_count:
            assignments[group_id] = "val"
        elif idx < val_count + test_count:
            assignments[group_id] = "test"
        else:
            assignments[group_id] = "train"
    return assignments


def assign_splits(
    records: Sequence[PairRecord],
    baseline_seed: int = 17,
    all_dynamic_seed: int = 29,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> List[PairRecord]:
    baseline_assignments: Dict[str, str] = {}
    all_dynamic_assignments: Dict[str, str] = {}

    baseline_group_ids: Dict[str, List[str]] = {"Y3": [], "Y4": []}
    all_dynamic_group_ids: Dict[str, List[str]] = {"Y2": [], "Y3": [], "Y4": []}

    seen_positions: Dict[str, PairRecord] = {}
    for record in records:
        if record.position_uid not in seen_positions:
            seen_positions[record.position_uid] = record

    for position_uid, record in seen_positions.items():
        if record.dataset_id in baseline_group_ids and record.imaging_mode == "dynamic":
            baseline_group_ids[record.dataset_id].append(position_uid)
        if record.dataset_id in all_dynamic_group_ids and record.imaging_mode == "dynamic":
            all_dynamic_group_ids[record.dataset_id].append(position_uid)

    for dataset_id, group_ids in baseline_group_ids.items():
        seed = baseline_seed + sum(ord(ch) for ch in dataset_id)
        baseline_assignments.update(_split_group_ids(group_ids, seed, val_ratio, test_ratio))

    for dataset_id, group_ids in all_dynamic_group_ids.items():
        seed = all_dynamic_seed + sum(ord(ch) for ch in dataset_id)
        all_dynamic_assignments.update(_split_group_ids(group_ids, seed, val_ratio, test_ratio))

    updated: List[PairRecord] = []
    for record in records:
        if record.dataset_id == "Y1":
            baseline_split = "exclude"
            all_dynamic_split = "exclude"
        elif record.imaging_mode == "static":
            baseline_split = "exclude"
            all_dynamic_split = "exclude"
        elif record.dataset_id == "Y2":
            baseline_split = "external_test"
            all_dynamic_split = all_dynamic_assignments.get(record.position_uid, "exclude")
        elif record.dataset_id in {"Y3", "Y4"}:
            baseline_split = baseline_assignments.get(record.position_uid, "exclude")
            all_dynamic_split = all_dynamic_assignments.get(record.position_uid, "exclude")
        else:
            baseline_split = "exclude"
            all_dynamic_split = "exclude"

        updated.append(
            PairRecord(
                **{
                    **asdict(record),
                    "baseline_split": baseline_split,
                    "all_dynamic_split": all_dynamic_split,
                }
            )
        )
    return updated


def build_manifest_records(
    specs: Sequence[DatasetSpec] | None = None,
    input_channel: int = 0,
    target_channel: int = 1,
    baseline_seed: int = 17,
    all_dynamic_seed: int = 29,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> List[PairRecord]:
    dataset_specs = list(specs or default_dataset_specs())
    records: List[PairRecord] = []
    for spec in dataset_specs:
        records.extend(scan_dataset(spec, input_channel=input_channel, target_channel=target_channel))
    records = assign_splits(
        records,
        baseline_seed=baseline_seed,
        all_dynamic_seed=all_dynamic_seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return sorted(records, key=lambda item: (item.dataset_id, item.position_name, item.time_index, item.z_index))


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _group_position_rows(records: Sequence[PairRecord]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for record in records:
        row = grouped.setdefault(
            record.position_uid,
            {
                "position_uid": record.position_uid,
                "dataset_id": record.dataset_id,
                "position_name": record.position_name,
                "imaging_mode": record.imaging_mode,
                "day_index": record.day_index,
                "position_index": record.position_index,
                "sample_index": record.sample_index,
                "width_px": record.width_px,
                "height_px": record.height_px,
                "pixel_size_um": record.pixel_size_um,
                "pair_count": 0,
                "time_count": set(),
                "z_count": set(),
                "baseline_split": record.baseline_split,
                "all_dynamic_split": record.all_dynamic_split,
                "duplicate_policy": record.duplicate_policy,
                "domain_group": record.domain_group,
            },
        )
        row["pair_count"] = int(row["pair_count"]) + 1
        row["time_count"].add(record.time_index)
        row["z_count"].add(record.z_index)

    rows: List[Dict[str, object]] = []
    for row in grouped.values():
        rows.append(
            {
                **row,
                "time_count": len(row["time_count"]),
                "z_count": len(row["z_count"]),
            }
        )
    return sorted(rows, key=lambda item: (item["dataset_id"], item["position_name"]))


def _summary(records: Sequence[PairRecord]) -> Dict[str, object]:
    by_dataset: Dict[str, Dict[str, int]] = {}
    by_baseline_split: Dict[str, int] = {}
    by_all_dynamic_split: Dict[str, int] = {}
    dynamic_pairs = 0
    static_pairs = 0
    for record in records:
        dataset_summary = by_dataset.setdefault(
            record.dataset_id,
            {"pairs": 0, "dynamic_pairs": 0, "static_pairs": 0},
        )
        dataset_summary["pairs"] += 1
        by_baseline_split[record.baseline_split] = by_baseline_split.get(record.baseline_split, 0) + 1
        by_all_dynamic_split[record.all_dynamic_split] = by_all_dynamic_split.get(record.all_dynamic_split, 0) + 1
        if record.imaging_mode == "dynamic":
            dataset_summary["dynamic_pairs"] += 1
            dynamic_pairs += 1
        else:
            dataset_summary["static_pairs"] += 1
            static_pairs += 1
    return {
        "total_pairs": len(records),
        "dynamic_pairs": dynamic_pairs,
        "static_pairs": static_pairs,
        "datasets": by_dataset,
        "baseline_split_counts": by_baseline_split,
        "all_dynamic_split_counts": by_all_dynamic_split,
    }


def write_manifest_bundle(records: Sequence[PairRecord], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = output_dir / "yichao_manifest.csv"
    manifest_jsonl = output_dir / "yichao_manifest.jsonl"
    positions_csv = output_dir / "yichao_positions.csv"
    summary_json = output_dir / "yichao_summary.json"

    manifest_rows = [asdict(record) for record in records]
    position_rows = _group_position_rows(records)

    _write_csv(manifest_rows, manifest_csv)
    _write_csv(position_rows, positions_csv)
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary_json.write_text(json.dumps(_summary(records), indent=2), encoding="utf-8")
    return {
        "manifest_csv": manifest_csv,
        "manifest_jsonl": manifest_jsonl,
        "positions_csv": positions_csv,
        "summary_json": summary_json,
    }


def bundle_summary(paths: Dict[str, Path]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in paths.items())

