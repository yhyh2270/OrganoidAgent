#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_glob: str
    selection_mode: str
    stage: str
    diameters: tuple[int, int, int]
    enable_signal_recovery_fallback: bool = True


DATASET_SPECS = (
    DatasetSpec(
        name="Data-Yichao-1",
        image_glob="Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all/*_c1.jpg",
        selection_mode="single",
        stage="differentiated_irregular",
        diameters=(140, 240, 380),
    ),
    DatasetSpec(
        name="Data-Yichao-2",
        image_glob="Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object/N39_TriRep_DF_D2_Position001/*_t000_z*_c1.jpg",
        selection_mode="best_focus",
        stage="cystic_early",
        diameters=(110, 220, 360),
    ),
    DatasetSpec(
        name="Data-Yichao-3",
        image_glob="Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all/00_Experiment_1_Day_2_Position001_t000_z*_c1.jpg",
        selection_mode="best_focus",
        stage="fused_large",
        diameters=(70, 130, 220),
    ),
    DatasetSpec(
        name="Data-Yichao-4",
        image_glob="Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all/00_Experiment_1_Day_2_Position001_t000_z*_c1.jpg",
        selection_mode="best_focus",
        stage="fused_large",
        diameters=(70, 130, 220),
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "analysis-outputs/yichao_multiscale_segmentation_test"
    parser = argparse.ArgumentParser(
        description="Run the transplanted Zhengyu multiscale segmentation pipeline on one selected brightfield image from each Yichao dataset."
    )
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional subset of dataset folder names to run, for example: Data-Yichao-1 Data-Yichao-3",
    )
    parser.add_argument(
        "--gpu",
        choices=("auto", "true", "false"),
        default="auto",
        help="Use GPU if available. Default is auto.",
    )
    return parser.parse_args()


def load_multiscale_module(repo_root: Path) -> Any:
    module_path = repo_root / "analysis-tools/app80_first_replicate_multiscale_cellpose/run_multiscale_dateaware_cellpose.py"
    module_name = "organoid_multiscale_dateaware"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load multiscale pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_use_gpu(choice: str) -> bool:
    if choice == "true":
        return True
    if choice == "false":
        return False
    return bool(torch.cuda.is_available())


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"yichao_multiscale_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_rgb_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        gray = np.array(image.convert("L"))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb, gray


def paired_channel_path(path: Path, target_channel: str) -> Path:
    paired_name = re.sub(r"_c\d(\.[^.]+)$", rf"_{target_channel}\1", path.name)
    paired_path = path.with_name(paired_name)
    if paired_path == path or not paired_path.exists():
        raise FileNotFoundError(f"Could not resolve paired channel {target_channel} for {path}")
    return paired_path


def focus_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def z_index_from_name(path: Path) -> int:
    match = re.search(r"_z(\d+)_c\d\.jpg$", path.name)
    return int(match.group(1)) if match else -1


def select_image(repo_root: Path, spec: DatasetSpec) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(repo_root.glob(spec.image_glob))
    if not candidates:
        raise FileNotFoundError(f"No candidate images matched {spec.image_glob}")
    if spec.selection_mode == "single":
        chosen = candidates[0]
        return chosen, {
            "selection_mode": spec.selection_mode,
            "candidate_count": len(candidates),
            "selected_name": chosen.name,
        }

    scored: list[dict[str, Any]] = []
    for path in candidates:
        _, gray = load_rgb_gray(path)
        scored.append(
            {
                "path": path,
                "focus_score": round(focus_score(gray), 4),
                "z_index": z_index_from_name(path),
            }
        )
    scored.sort(key=lambda item: (float(item["focus_score"]), int(item["z_index"])), reverse=True)
    best = scored[0]
    chosen = Path(best["path"])
    return chosen, {
        "selection_mode": spec.selection_mode,
        "candidate_count": len(candidates),
        "selected_name": chosen.name,
        "selected_focus_score": best["focus_score"],
        "selected_z_index": best["z_index"],
        "top_focus_candidates": [
            {
                "name": Path(item["path"]).name,
                "focus_score": item["focus_score"],
                "z_index": item["z_index"],
            }
            for item in scored[:5]
        ],
    }


def to_thumb(image: np.ndarray, thumb_h: int = 240) -> np.ndarray:
    scale = thumb_h / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, thumb_h), interpolation=cv2.INTER_AREA)


def render_comparison_panel(
    brightfield_rgb: np.ndarray,
    fluorescence_rgb: np.ndarray,
    signal: np.ndarray,
    overlay_on_brightfield: np.ndarray,
    overlay_on_fluorescence: np.ndarray,
    instance_rgb: np.ndarray,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    items = [
        ("Brightfield", brightfield_rgb),
        ("Fluorescence", fluorescence_rgb),
        ("Debug Signal", cv2.cvtColor(signal, cv2.COLOR_GRAY2RGB)),
        ("Overlay On Brightfield", overlay_on_brightfield),
        ("Overlay On Fluorescence", overlay_on_fluorescence),
        ("Instance RGB", instance_rgb),
    ]
    for title, image in items:
        panel = image.copy()
        cv2.putText(
            panel,
            title,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        panels.append(panel)
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def build_gallery_row(label: str, source_rgb: np.ndarray, overlay: np.ndarray, instance_rgb: np.ndarray) -> np.ndarray:
    thumb_h = 220
    spacer = np.full((thumb_h, 10, 3), 255, dtype=np.uint8)
    row = np.concatenate([to_thumb(source_rgb, thumb_h), spacer, to_thumb(overlay, thumb_h), spacer, to_thumb(instance_rgb, thumb_h)], axis=1)
    canvas = np.full((thumb_h + 34, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    canvas[34:, :, :] = row
    return canvas


def run_one_dataset(
    repo_root: Path,
    run_dir: Path,
    spec: DatasetSpec,
    module: Any,
    model: Any,
) -> dict[str, Any]:
    selected_path, selection_info = select_image(repo_root, spec)
    dataset_out_dir = run_dir / spec.name
    dataset_out_dir.mkdir(parents=True, exist_ok=False)

    source_rgb, gray = load_rgb_gray(selected_path)
    fluorescence_path = paired_channel_path(selected_path, "c0")
    fluorescence_rgb, _ = load_rgb_gray(fluorescence_path)
    signal, support, grad_norm = module.compute_hybrid_signal(gray)

    candidates = []
    branch_summaries = []
    for branch_rank, diameter in enumerate(spec.diameters):
        try:
            masks, *_ = model.eval(gray, diameter=float(diameter), channels=[0, 0], normalize=True, do_3D=False)
            masks = masks.astype(np.uint16)
            branch_count = 0
            for label in range(1, int(masks.max()) + 1):
                candidate = module.build_candidate(masks, label, diameter, branch_rank, signal, support, grad_norm, spec.stage)
                if candidate is not None:
                    candidates.append(candidate)
                    branch_count += 1
            branch_summaries.append(
                {
                    "diameter_px": diameter,
                    "kept_candidates_before_merge": branch_count,
                    "status": "ok",
                }
            )
        except Exception as exc:
            branch_summaries.append(
                {
                    "diameter_px": diameter,
                    "kept_candidates_before_merge": 0,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )

    signal_recovery_used = False
    signal_candidates = []
    if spec.enable_signal_recovery_fallback and not candidates:
        signal_candidates = module.recover_signal_candidates(signal, support, grad_norm, spec.stage)
        candidates.extend(signal_candidates)
        signal_recovery_used = bool(signal_candidates)
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": len(signal_candidates),
                "status": "ok",
                "source": "signal_recovery_fallback",
            }
        )
    else:
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": 0,
                "status": "skipped",
                "source": "signal_recovery_fallback",
            }
        )

    kept = module.merge_candidates(candidates)
    label_mask, color_mask = module.build_outputs(source_rgb, kept)
    overlay = cv2.addWeighted(source_rgb, 0.72, color_mask, 0.55, 0)
    fluorescence_overlay = cv2.addWeighted(fluorescence_rgb, 0.72, color_mask, 0.55, 0)
    edges = cv2.Canny((label_mask > 0).astype(np.uint8) * 255, 50, 150)
    overlay[edges > 0] = (255, 0, 0)
    fluorescence_overlay[edges > 0] = (255, 0, 0)
    panel = render_comparison_panel(
        source_rgb,
        fluorescence_rgb,
        signal,
        overlay,
        fluorescence_overlay,
        color_mask,
    )

    source_png = dataset_out_dir / "source.png"
    brightfield_png = dataset_out_dir / "brightfield_input.png"
    fluorescence_png = dataset_out_dir / "fluorescence_reference.png"
    signal_png = dataset_out_dir / "signal.png"
    support_png = dataset_out_dir / "support.png"
    mask_png = dataset_out_dir / "multiscale_mask_16bit.png"
    instance_png = dataset_out_dir / "multiscale_instance_rgb.png"
    overlay_png = dataset_out_dir / "multiscale_overlay.png"
    overlay_brightfield_png = dataset_out_dir / "multiscale_overlay_on_brightfield.png"
    overlay_fluorescence_png = dataset_out_dir / "multiscale_overlay_on_fluorescence.png"
    panel_png = dataset_out_dir / "comparison_panel.png"
    stats_json = dataset_out_dir / "multiscale_stats.json"

    Image.fromarray(source_rgb).save(source_png)
    Image.fromarray(source_rgb).save(brightfield_png)
    Image.fromarray(fluorescence_rgb).save(fluorescence_png)
    Image.fromarray(signal).save(signal_png)
    Image.fromarray(support).save(support_png)
    cv2.imwrite(str(mask_png), label_mask.astype(np.uint16))
    Image.fromarray(color_mask).save(instance_png)
    Image.fromarray(overlay).save(overlay_png)
    Image.fromarray(overlay).save(overlay_brightfield_png)
    Image.fromarray(fluorescence_overlay).save(overlay_fluorescence_png)
    Image.fromarray(panel).save(panel_png)

    stats = {
        "dataset": spec.name,
        "selected_input_image": str(selected_path),
        "paired_fluorescence_image": str(fluorescence_path),
        "selection": selection_info,
        "selected_channel": "c1_brightfield",
        "segmentation_policy": "multiscale_cellpose_with_signal_recovery_fallback",
        "signal_recovery_used": signal_recovery_used,
        "stage": spec.stage,
        "diameters_px": list(spec.diameters),
        "image_shape": [int(source_rgb.shape[1]), int(source_rgb.shape[0])],
        "mask_count": int(label_mask.max()),
        "source_png": str(source_png),
        "brightfield_input_png": str(brightfield_png),
        "fluorescence_reference_png": str(fluorescence_png),
        "signal_png": str(signal_png),
        "support_png": str(support_png),
        "mask_16bit_png": str(mask_png),
        "instance_rgb_png": str(instance_png),
        "overlay_png": str(overlay_png),
        "overlay_on_brightfield_png": str(overlay_brightfield_png),
        "overlay_on_fluorescence_png": str(overlay_fluorescence_png),
        "comparison_panel_png": str(panel_png),
        "branch_summaries": branch_summaries,
        "merged_candidates": [
            {
                "area": candidate.area,
                "score": round(float(candidate.score), 4),
                "diameter_px": int(candidate.diameter),
                "support_ratio": round(float(candidate.support_ratio), 4),
                "mean_signal": round(float(candidate.mean_signal), 4),
                "edge_strength": round(float(candidate.edge_strength), 4),
                "circularity": round(float(candidate.circularity), 4),
                "source": candidate.source,
            }
            for candidate in sorted(kept, key=lambda item: item.area, reverse=True)
        ],
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["stats_json"] = str(stats_json)
    return stats


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(output_root)

    selected_names = set(args.datasets)
    specs = [spec for spec in DATASET_SPECS if not selected_names or spec.name in selected_names]
    if not specs:
        raise RuntimeError(f"No dataset specs matched --datasets={args.datasets}")

    module = load_multiscale_module(repo_root)
    model = module.models.CellposeModel(gpu=resolve_use_gpu(args.gpu))

    batch_summary = []
    gallery_rows = []
    for spec in specs:
        stats = run_one_dataset(repo_root, run_dir, spec, module, model)
        batch_summary.append(stats)
        source_rgb = np.array(Image.open(Path(stats["source_png"])).convert("RGB"))
        overlay = np.array(Image.open(Path(stats["overlay_png"])).convert("RGB"))
        instance_rgb = np.array(Image.open(Path(stats["instance_rgb_png"])).convert("RGB"))
        label = f'{spec.name} | {Path(stats["selected_input_image"]).name} | n={stats["mask_count"]}'
        gallery_rows.append(build_gallery_row(label, source_rgb, overlay, instance_rgb))

    summary_json = run_dir / "yichao_test_summary.json"
    summary_json.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if gallery_rows:
        gallery_height = sum(row.shape[0] for row in gallery_rows) + 10 * (len(gallery_rows) - 1)
        gallery_width = max(row.shape[1] for row in gallery_rows)
        gallery = np.full((gallery_height, gallery_width, 3), 255, dtype=np.uint8)
        y = 0
        for row in gallery_rows:
            gallery[y : y + row.shape[0], : row.shape[1], :] = row
            y += row.shape[0] + 10
        Image.fromarray(gallery).save(run_dir / "yichao_multiscale_gallery.png")

    print(run_dir)
    print(summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
