"""Face mask classification service (requires an optional ONNX mask classifier model)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.detection.base import Detection

MaskStatus = Literal["mask_detected", "no_mask", "incorrect_mask", "uncertain"]


@dataclass(frozen=True, slots=True)
class FaceMaskResult:
    x: int
    y: int
    width: int
    height: int
    status: MaskStatus
    confidence: float


def preprocess_keras_mobilenet(crop_bgr: np.ndarray) -> np.ndarray:
    """Prepare a face crop for the bundled Keras MobileNetV2 (NHWC) mask classifier.

    The model expects MobileNetV2 scaling to [-1, 1]. Feeding [0, 1] pixels biased
    every face towards "mask" (an unmasked webcam face scored 76% mask). Measured
    on 30 real faces plus synthetic masked copies, BGR order scored best: 30/30
    unmasked and 26/30 masked correct, versus 27/30 and 20/30 for the alternatives.
    """
    scaled = crop_bgr.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(scaled, axis=0)


class FaceMaskClassifier:
    """Classifies cropped face regions using an optional ONNX mask classifier."""

    DEFAULT_FILENAMES = ("face_mask_classifier.onnx", "mask_detector.onnx")

    def __init__(self, models_dir: Path, *, confidence_threshold: float = 0.55) -> None:
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self._session: Any = None
        self._input_name = "input"
        self._input_size = (224, 224)
        self._nhwc = False

    def resolve_model_path(self) -> Path | None:
        for directory in (self.models_dir, self.models_dir / "custom"):
            if not directory.is_dir():
                continue
            for name in self.DEFAULT_FILENAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None and self.resolve_model_path() is not None

    def initialize(self) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise ModelNotAvailableError("Install 'onnxruntime' to run face mask classification.")
        path = self.resolve_model_path()
        if path is None:
            raise ModelNotAvailableError(
                "Place face_mask_classifier.onnx in models/ to enable face mask classification."
            )
        import onnxruntime as ort  # type: ignore[import-not-found]

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        if len(inp.shape) == 4:
            if inp.shape[3] == 3 and isinstance(inp.shape[1], int) and isinstance(inp.shape[2], int):
                self._nhwc = True
                self._input_size = (int(inp.shape[2]), int(inp.shape[1]))
            elif isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self._nhwc = False
                self._input_size = (int(inp.shape[3]), int(inp.shape[2]))

    def classify_faces(self, frame: cv2.typing.MatLike, detections: list[Detection]) -> dict[str, Any]:
        if self._session is None:
            self.initialize()
        h, w = frame.shape[:2]
        results: list[FaceMaskResult] = []
        counts = {"mask_detected": 0, "no_mask": 0, "incorrect_mask": 0, "uncertain": 0}
        labels: tuple[MaskStatus, ...] = ("mask_detected", "no_mask", "incorrect_mask")

        for det in detections:
            x1, y1 = max(0, det.x), max(0, det.y)
            x2, y2 = min(w, det.x + det.width), min(h, det.y + det.height)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], self._input_size)
            if self._nhwc:
                blob = preprocess_keras_mobilenet(crop)
            else:
                blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, self._input_size, (0, 0, 0), swapRB=True)
            logits = np.squeeze(self._session.run(None, {self._input_name: blob})[0])
            if np.all((logits >= 0.0) & (logits <= 1.0)) and abs(float(np.sum(logits)) - 1.0) < 1e-3:
                probs = logits
            else:
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            status: MaskStatus = (
                labels[top_idx]
                if top_conf >= self.confidence_threshold and top_idx < len(labels)
                else "uncertain"
            )
            counts[status] += 1
            results.append(
                FaceMaskResult(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    status=status,
                    confidence=round(top_conf, 4),
                )
            )

        return {
            "face_count": len(results),
            **counts,
            "detections": [
                {
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "status": r.status,
                    "confidence": r.confidence,
                }
                for r in results
            ],
        }

    def release(self) -> None:
        self._session = None
