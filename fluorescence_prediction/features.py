from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps


def extract_evidence(source_bytes: bytes, mask_png: bytes) -> dict[str, float]:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(source_bytes))).convert("RGB")
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
    if mask.shape != gray.shape:
        mask = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    area = int(mask.sum())
    if area == 0:
        raise ValueError("分割掩膜为空，无法提取形态特征")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    pixels = gray[mask].astype(np.float32)
    edges = cv2.Canny(gray, 60, 150) > 0

    return {
        "mask_area_px": float(area),
        "area_fraction": float(area / mask.size),
        "perimeter_px": perimeter,
        "circularity": float(4 * math.pi * area / max(perimeter * perimeter, 1e-6)),
        "bbox_fill_ratio": float(area / max(width * height, 1)),
        "aspect_ratio": float(width / max(height, 1)),
        "mean_brightness": float(pixels.mean() / 255.0),
        "contrast": float(pixels.std() / 255.0),
        "darkness_p90": float(np.percentile(255.0 - pixels, 90) / 255.0),
        "edge_density": float((edges & mask).sum() / area),
        "focus_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def rounded_evidence(evidence: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in evidence.items()}
