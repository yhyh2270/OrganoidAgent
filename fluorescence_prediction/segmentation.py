from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
YOLO_SAM_DIR = ROOT
if str(YOLO_SAM_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_SAM_DIR))

from SAM import SamPredictor, sam_model_registry  # noqa: E402


@dataclass(frozen=True)
class SegmentationResult:
    crop_jpeg: bytes
    overlay_jpeg: bytes
    mask_png: bytes
    yolo_confidence: float
    sam_score: float
    crop_xyxy: tuple[int, int, int, int]


class OrganoidSegmentor:
    """YOLO box detection followed by box-prompted SAM segmentation."""

    def __init__(self) -> None:
        self.yolo_path = ROOT / "weights" / "yolo_organoid_best.pt"
        self.sam_path = ROOT / "weights" / "sam_vit_b_01ec64.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector: YOLO | None = None
        self.predictor: SamPredictor | None = None
        self._lock = Lock()

    def load(self) -> None:
        if self.detector is not None and self.predictor is not None:
            return
        if not self.yolo_path.exists():
            raise FileNotFoundError(f"找不到类器官 YOLO 权重：{self.yolo_path}")
        if not self.sam_path.exists():
            raise FileNotFoundError(f"找不到 SAM 权重：{self.sam_path}")
        detector = YOLO(self.yolo_path)
        if set(detector.names.values()) != {"organoid"}:
            raise RuntimeError(f"YOLO 权重类别不正确：{detector.names}")
        sam = sam_model_registry["vit_b"](checkpoint=str(self.sam_path))
        sam.to(device=self.device)
        self.detector = detector
        self.predictor = SamPredictor(sam)

    @staticmethod
    def _decode(data: bytes) -> tuple[np.ndarray, np.ndarray]:
        try:
            image = Image.open(io.BytesIO(data))
            image.seek(0)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("无法解析图片，请上传 PNG、JPG、JPEG 或 TIFF 文件") from exc
        rgb = np.asarray(image)
        return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _encode(image: np.ndarray, extension: str, params: list[int] | None = None) -> bytes:
        ok, encoded = cv2.imencode(extension, image, params or [])
        if not ok:
            raise RuntimeError(f"无法编码输出图像：{extension}")
        return encoded.tobytes()

    def segment(self, data: bytes) -> SegmentationResult:
        self.load()
        image_rgb, image_bgr = self._decode(data)
        with self._lock, torch.inference_mode():
            # Deliberately match yolo_SAM/predict.py: default imgsz/conf + max_det=1.
            result = self.detector(image_bgr, max_det=1, verbose=False)[0]  # type: ignore[misc]
            if result.boxes is None or len(result.boxes) == 0:
                raise ValueError("未检测到类器官，请确认图像主体清晰且完整")

            box = result.boxes[0]
            boxes = box.xyxy.to(device=self.predictor.device)  # type: ignore[union-attr]
            self.predictor.set_image(image_rgb)  # type: ignore[union-attr]
            transformed = self.predictor.transform.apply_boxes_torch(  # type: ignore[union-attr]
                boxes, image_rgb.shape[:2]
            )
            masks, scores, _ = self.predictor.predict_torch(  # type: ignore[union-attr]
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            mask = masks[0, 0].detach().cpu().numpy().astype(bool)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError("SAM 返回了空掩膜，请更换图像后重试")
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        pad = round(0.05 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(image_rgb.shape[1], x2 + pad), min(image_rgb.shape[0], y2 + pad)
        crop = image_bgr[y1:y2, x1:x2]

        overlay_rgb = image_rgb.astype(np.float32)
        color = np.array([31, 180, 123], dtype=np.float32)
        overlay_rgb[mask] = overlay_rgb[mask] * 0.55 + color * 0.45
        overlay = cv2.cvtColor(overlay_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (104, 239, 180), max(2, image_rgb.shape[1] // 900))

        return SegmentationResult(
            crop_jpeg=self._encode(crop, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
            overlay_jpeg=self._encode(overlay, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]),
            mask_png=self._encode(mask.astype(np.uint8) * 255, ".png"),
            yolo_confidence=float(box.conf[0].cpu()),
            sam_score=float(scores[0, 0].cpu()),
            crop_xyxy=(x1, y1, x2, y2),
        )

    def metadata(self) -> dict:
        return {
            "ready": self.yolo_path.exists() and self.sam_path.exists(),
            "yolo_checkpoint": str(self.yolo_path.relative_to(ROOT)),
            "sam_checkpoint": str(self.sam_path.relative_to(ROOT)),
        }
