"""Optional RetinaFace ONNX face detection adapter with full ONNXRuntime inference."""

from __future__ import annotations

import importlib.util
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
    FaceDetector,
    LandmarkPoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RETINAFACE_FILENAMES = ("retinaface_resnet50.onnx", "retinaface.onnx")


class RetinaFaceDetector(FaceDetector):
    """Optional RetinaFace ONNX adapter (requires onnxruntime + model weights)."""

    model_id = "retinaface"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._session: Any = None
        self._input_name: str = "input"
        self._input_size: tuple[int, int] = (640, 640)
        self._providers: list[str] = ["CPUExecutionProvider"]

    def _has_onnxruntime(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def _resolve_model_path(self) -> Path | None:
        if self.config.model_path is not None and self.config.model_path.is_file():
            return self.config.model_path
        search_dirs: list[Path] = []
        if self.config.models_dir is not None:
            search_dirs.append(self.config.models_dir)
        else:
            search_dirs.append(PROJECT_ROOT / "models")
        for directory in search_dirs:
            for name in DEFAULT_RETINAFACE_FILENAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    def is_available(self) -> bool:
        return self._has_onnxruntime() and self._resolve_model_path() is not None

    def initialize(self) -> None:
        if not self._has_onnxruntime():
            raise ModelNotAvailableError("Optional dependency 'onnxruntime' is not installed.")
        model_path = self._resolve_model_path()
        if model_path is None:
            raise ModelNotAvailableError("RetinaFace ONNX model not found in models/.")

        import onnxruntime as ort  # type: ignore[import-not-found]

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.config.use_gpu and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._providers = self._session.get_providers()
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self._input_size = (int(inp.shape[3]), int(inp.shape[2]))
        except Exception as exc:
            raise ModelInitializationError(f"Failed to initialize RetinaFace session: {exc}") from exc
        self._initialized = True

    @staticmethod
    def _prior_boxes(height: int, width: int) -> np.ndarray:
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        anchors: list[list[float]] = []
        for k, step in enumerate(steps):
            feat_h, feat_w = ceil(height / step), ceil(width / step)
            for i, j in product(range(feat_h), range(feat_w)):
                for min_size in min_sizes[k]:
                    s_kx = min_size / width
                    s_ky = min_size / height
                    cx = (j + 0.5) * step / width
                    cy = (i + 0.5) * step / height
                    anchors.append([cx, cy, s_kx, s_ky])
        return np.array(anchors, dtype=np.float32)

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._session is None:
            raise ModelInitializationError("RetinaFace session is not initialized.")
        if frame is None or frame.size == 0:
            return []

        in_w, in_h = self._input_size
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (in_w, in_h)).astype(np.float32)
        resized -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        blob = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]

        outputs = self._session.run(None, {self._input_name: blob})
        if len(outputs) < 2:
            return []

        loc = outputs[0][0]
        conf = outputs[1][0][:, 1]
        landms = outputs[2][0] if len(outputs) >= 3 else None

        priors = self._prior_boxes(in_h, in_w)
        if priors.shape[0] != loc.shape[0]:
            return []

        variances = (0.1, 0.2)
        boxes_cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
        boxes_wh = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
        boxes = np.concatenate((boxes_cxcy - boxes_wh / 2, boxes_cxcy + boxes_wh / 2), axis=1)
        boxes *= np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)

        keep = np.where(conf >= self.config.confidence_threshold)[0]
        if len(keep) == 0:
            return []

        nms_boxes: list[list[int]] = []
        nms_scores: list[float] = []
        nms_kps: list[list[LandmarkPoint] | None] = []
        for idx in keep:
            x1 = max(0, int(boxes[idx, 0]))
            y1 = max(0, int(boxes[idx, 1]))
            x2 = min(orig_w, int(boxes[idx, 2]))
            y2 = min(orig_h, int(boxes[idx, 3]))
            bw, bh = x2 - x1, y2 - y1
            if bw < self.config.min_face_size[0] or bh < self.config.min_face_size[1]:
                continue
            kps_list: list[LandmarkPoint] | None = None
            if landms is not None:
                pts = priors[idx, :2] + landms[idx].reshape(5, 2) * variances[0] * priors[idx, 2:]
                kps_list = [LandmarkPoint(x=float(pt[0] * orig_w), y=float(pt[1] * orig_h)) for pt in pts]
            nms_boxes.append([x1, y1, bw, bh])
            nms_scores.append(float(conf[idx]))
            nms_kps.append(kps_list)

        if not nms_boxes:
            return []

        nms_indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.config.confidence_threshold, 0.4)
        return [
            Detection(
                x=nms_boxes[int(i)][0],
                y=nms_boxes[int(i)][1],
                width=nms_boxes[int(i)][2],
                height=nms_boxes[int(i)][3],
                confidence=nms_scores[int(i)],
                landmarks=nms_kps[int(i)],
            )
            for i in np.array(nms_indices).flatten()
        ]

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        reason: str | None = None
        if not self._has_onnxruntime():
            reason = "Install optional 'onnxruntime' package to enable RetinaFace."
        elif self._resolve_model_path() is None:
            reason = "Place retinaface.onnx in models/ to enable RetinaFace."
        return DetectorMetadata(
            id=self.model_id,
            name="RetinaFace (Optional ONNX)",
            version="RetinaFace-ONNX",
            description="Dense single-stage face & 5-landmark detector (optional).",
            supports_confidence=True,
            supports_landmarks=True,
            supports_gpu="CUDAExecutionProvider" in self._providers,
            gpu_active="CUDAExecutionProvider" in self._providers and self.config.use_gpu,
            is_optional=True,
            available=available,
            license="MIT (check exported weights license)",
            unavailable_reason=reason,
        )

    def release(self) -> None:
        self._session = None
        self._initialized = False
