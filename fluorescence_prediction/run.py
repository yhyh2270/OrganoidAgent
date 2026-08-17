from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
ULTRALYTICS_DIR = MODULE_DIR / ".ultralytics"
ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

from features import extract_evidence, rounded_evidence
from inference import OrganoidPredictor
from report import EvidenceReportAgent
from segmentation import OrganoidSegmentor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def write_progress(output_dir: Path, payload: dict) -> None:
    path = output_dir / "progress.json"
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run organoid fluorescence viability prediction.")
    parser.add_argument("--input", action="append", default=[], help="Image file or directory; repeat as needed.")
    parser.add_argument("--inputs-json", help="Optional JSON file containing a list of image paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--order", choices=("asc", "desc"), default="desc")
    parser.add_argument("--instruction", default="检测类器官活性并解释图像证据")
    parser.add_argument("--max-files", type=int, default=30)
    return parser.parse_args()


def discover_inputs(raw_inputs: list[str], inputs_json: str | None, max_files: int) -> list[Path]:
    values = list(raw_inputs)
    if inputs_json:
        payload = json.loads(Path(inputs_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--inputs-json must contain a JSON list")
        values.extend(str(item) for item in payload)
    found: list[Path] = []
    for value in values:
        path = Path(value).resolve()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            raise FileNotFoundError(f"Image input not found or unsupported: {value}")
    unique = list(dict.fromkeys(found))
    if not unique:
        raise ValueError("No supported images were found")
    if len(unique) > max_files:
        raise ValueError(f"Found {len(unique)} images; maximum is {max_files}")
    return unique


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._")
    return stem[:80] or "image"


def explain_sample(sample: dict) -> str:
    evidence = sample["evidence"]
    notes = [sample["summary"]]
    circularity = evidence["circularity"]
    fill = evidence["bbox_fill_ratio"]
    contrast = evidence["contrast"]
    edge = evidence["edge_density"]
    notes.append(
        f"分割区域圆度为 {circularity:.3f}、外接框填充率为 {fill:.3f}，"
        + ("轮廓相对规则。" if circularity >= 0.70 else "轮廓存在一定不规则性。")
    )
    notes.append(
        f"区域内部对比度为 {contrast:.3f}、边缘密度为 {edge:.3f}；这些是图像证据描述，不能单独证明生物学因果。"
    )
    return " ".join(notes)


def analyze_one(index: int, path: Path, output_dir: Path, predictor: OrganoidPredictor, segmentor: OrganoidSegmentor) -> dict:
    started = time.perf_counter()
    source_bytes = path.read_bytes()
    segmented = segmentor.segment(source_bytes)
    prediction = predictor.predict(segmented.crop_jpeg)
    evidence = rounded_evidence(extract_evidence(source_bytes, segmented.mask_png))
    sample_dir = output_dir / "samples" / f"{index:03d}_{safe_stem(path.name)}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "overlay": sample_dir / "overlay.jpg",
        "crop": sample_dir / "crop.jpg",
        "mask": sample_dir / "mask.png",
    }
    assets["overlay"].write_bytes(segmented.overlay_jpeg)
    assets["crop"].write_bytes(segmented.crop_jpeg)
    assets["mask"].write_bytes(segmented.mask_png)
    sample = {
        "filename": path.name,
        "source_path": str(path),
        "viability": round(prediction.viability, 6),
        "viability_percent": round(prediction.viability * 100.0, 1),
        "level": prediction.level,
        "summary": prediction.summary,
        "segmentation": {
            "yolo_confidence": round(segmented.yolo_confidence, 4),
            "sam_score": round(segmented.sam_score, 4),
            "crop_xyxy": list(segmented.crop_xyxy),
        },
        "evidence": evidence,
        "assets": {key: str(value.relative_to(output_dir)).replace("\\", "/") for key, value in assets.items()},
        "device": prediction.device,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    sample["explanation"] = explain_sample(sample)
    (sample_dir / "result.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sample


def write_csv(samples: list[dict], path: Path) -> None:
    evidence_keys = sorted(samples[0]["evidence"]) if samples else []
    fields = [
        "rank", "filename", "source_path", "viability", "viability_percent", "level",
        "yolo_confidence", "sam_score", "elapsed_seconds", *evidence_keys,
        "overlay_path", "crop_path", "mask_path", "explanation",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, sample in enumerate(samples, start=1):
            row = {
                "rank": rank,
                "filename": sample["filename"],
                "source_path": sample["source_path"],
                "viability": sample["viability"],
                "viability_percent": sample["viability_percent"],
                "level": sample["level"],
                "yolo_confidence": sample["segmentation"]["yolo_confidence"],
                "sam_score": sample["segmentation"]["sam_score"],
                "elapsed_seconds": sample["elapsed_seconds"],
                "overlay_path": sample["assets"]["overlay"],
                "crop_path": sample["assets"]["crop"],
                "mask_path": sample["assets"]["mask"],
                "explanation": sample["explanation"],
            }
            row.update(sample["evidence"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_inputs(args.input, args.inputs_json, args.max_files)
    predictor = OrganoidPredictor()
    segmentor = OrganoidSegmentor()
    reporter = EvidenceReportAgent()
    samples: list[dict] = []
    failures: list[dict] = []
    write_progress(
        output_dir,
        {
            "status": "running",
            "total": len(paths),
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "current_index": 0,
            "current_file": None,
            "percent": 0.0,
            "elapsed_seconds": 0.0,
        },
    )
    for index, path in enumerate(paths, start=1):
        write_progress(
            output_dir,
            {
                "status": "running",
                "total": len(paths),
                "completed": index - 1,
                "succeeded": len(samples),
                "failed": len(failures),
                "current_index": index,
                "current_file": path.name,
                "percent": round((index - 1) * 100.0 / len(paths), 1),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
        )
        try:
            sample = analyze_one(index, path, output_dir, predictor, segmentor)
            samples.append(sample)
            print(f"[{index}/{len(paths)}] {path.name}: {sample['viability_percent']:.1f}%", flush=True)
        except Exception as exc:
            failures.append({"filename": path.name, "source_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(paths)}] FAILED {path.name}: {exc}", flush=True)
        write_progress(
            output_dir,
            {
                "status": "running" if index < len(paths) else "finalizing",
                "total": len(paths),
                "completed": index,
                "succeeded": len(samples),
                "failed": len(failures),
                "current_index": index,
                "current_file": path.name,
                "percent": round(index * 100.0 / len(paths), 1),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
        )
    samples.sort(key=lambda item: item["viability"], reverse=args.order == "desc")
    report = reporter.generate(samples, failures)
    result = {
        "status": "succeeded" if samples else "failed",
        "instruction": args.instruction,
        "order": args.order,
        "samples": samples,
        "failures": failures,
        "summary": {
            "total": len(paths),
            "succeeded": len(samples),
            "failed": len(failures),
            "high": sum(item["viability"] >= 0.8 for item in samples),
            "medium": sum(0.6 <= item["viability"] < 0.8 for item in samples),
            "low": sum(item["viability"] < 0.6 for item in samples),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "model": predictor.metadata(),
        "segmentation_model": segmentor.metadata(),
        "report_path": "report.md",
        "csv_path": "results.csv",
        "disclaimer": "结果仅供科研分析参考；解释基于模型输出和可计算图像特征，不构成生物学因果证明。",
    }
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    write_csv(samples, output_dir / "results.csv")
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(
        output_dir,
        {
            "status": result["status"],
            "total": len(paths),
            "completed": len(paths),
            "succeeded": len(samples),
            "failed": len(failures),
            "current_index": len(paths),
            "current_file": None,
            "percent": 100.0,
            "elapsed_seconds": result["summary"]["elapsed_seconds"],
        },
    )
    print(output_dir / "results.json", flush=True)
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
