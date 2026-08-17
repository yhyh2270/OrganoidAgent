from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent

from model import MultiScaleConvNeXtTinyOrganoidTaskNet  # noqa: E402


@dataclass(frozen=True)
class Prediction:
    viability: float
    level: str
    summary: str
    device: str


class OrganoidPredictor:
    """Load the trained model once and provide thread-safe inference."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or ROOT / "weights" / "viability_best.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MultiScaleConvNeXtTinyOrganoidTaskNet | None = None
        self.input_size = 384
        self._lock = Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"找不到模型权重：{self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        args: dict[str, Any] = checkpoint.get("args", {})
        self.input_size = int(args.get("input_size", 384))
        model = MultiScaleConvNeXtTinyOrganoidTaskNet(
            proj_dim=int(args.get("proj_dim", 64)),
            hidden_dim=int(args.get("hidden_dim", 192)),
            dropout=float(args.get("dropout", 0.3)),
            pretrained=False,
            freeze_backbone=False,
            imagenet_norm=not bool(args.get("no_imagenet_norm", False)),
        )
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        # The supplied checkpoint retains an unused classification head from an
        # earlier trainer revision. It is not called by the viability forward path.
        allowed_unexpected = {"grade_head.weight", "grade_head.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed_unexpected:
            raise RuntimeError(
                "权重与模型结构不匹配："
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        model.to(self.device).eval()
        self.model = model

    @staticmethod
    def _decode(data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(data))
            image.seek(0)
            return ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("无法解析图片，请上传 PNG、JPG、JPEG 或 TIFF 文件") from exc

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        # Match training: scale the whole ROI by its longest side, then black letterbox.
        width, height = image.size
        scale = self.input_size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(new_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.input_size, self.input_size), color=(0, 0, 0))
        canvas.paste(resized, ((self.input_size - new_size[0]) // 2, (self.input_size - new_size[1]) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, data: bytes) -> Prediction:
        self.load()
        image = self._decode(data)
        tensor = self._preprocess(image).to(self.device)
        with self._lock, torch.inference_mode():
            value = float(self.model(tensor)["pred_viability"].item())  # type: ignore[misc]
        value = min(1.0, max(0.0, value))
        if value >= 0.80:
            level, summary = "高活性", "模型判断该类器官处于较高活性区间。"
        elif value >= 0.60:
            level, summary = "中等活性", "模型判断该类器官处于中等活性区间。"
        else:
            level, summary = "低活性", "模型判断该类器官处于较低活性区间。"
        return Prediction(value, level, summary, str(self.device))

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": self.checkpoint_path.exists(),
            "checkpoint": str(self.checkpoint_path.relative_to(ROOT)),
            "device": str(self.device),
            "input_size": self.input_size,
        }
