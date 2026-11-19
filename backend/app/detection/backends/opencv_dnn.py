"""OpenCV DNN SSD (Res10 300x300) face detection backend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import Detection, DetectorConfig, DetectorMetadata, FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROTOTXT = "deploy.prototxt"
DEFAULT_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"


class OpenCVDNNDetector(FaceDetector):
    """OpenCV DNN Caffe SSD face detector (optional model weights)."""

    model_id = "opencv_dnn"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._net: cv2.dnn.Net | None = None

    def _resolve_files(self) -> tuple[Path, Path] | None:
        search_dirs: list[Path] = []
        if self.config.models_dir is not None:
            search_dirs.append(self.config.models_dir)
        search_dirs.append(PROJECT_ROOT / "models")

        for directory in search_dirs:
            proto = directory / DEFAULT_PROTOTXT
            weights = directory / DEFAULT_WEIGHTS
            if proto.is_file() and weights.is_file():
                return proto, weights
        return None

    def is_available(self) -> bool:
        return self._resolve_files() is not None

    def initialize(self) -> None:
        files = self._resolve_files()
        if files is None:
            raise ModelNotAvailableError(
                f"OpenCV DNN model files ({DEFAULT_PROTOTXT}, {DEFAULT_WEIGHTS}) not found in models/."
            )
        proto, weights = files
        try:
            net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
            if self.config.use_gpu and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except cv2.error as exc:
            raise ModelInitializationError(f"Failed to load OpenCV DNN model: {exc}") from exc
        self._net = net
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._net is None:
            raise ModelInitializationError("OpenCV DNN detector is not initialized.")
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        raw = self._net.forward()
        min_w, min_h = self.config.min_face_size
        detections: list[Detection] = []

        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self.config.confidence_threshold:
                continue
            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            bw = max(0, min(w, x2) - x1)
            bh = max(0, min(h, y2) - y1)
            if bw < min_w or bh < min_h:
                continue
            detections.append(
                Detection(x=int(x1), y=int(y1), width=int(bw), height=int(bh), confidence=confidence)
            )
        return detections

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        return DetectorMetadata(
            id=self.model_id,
            name="OpenCV DNN (Res10 SSD)",
            version="res10-300x300",
            description="Caffe SSD ResNet-10 face detector using OpenCV DNN.",
            supports_confidence=True,
            supports_landmarks=False,
            supports_gpu=cuda_ok,
            gpu_active=bool(self.config.use_gpu and cuda_ok),
            is_optional=True,
            available=available,
            license="BSD-3-Clause (OpenCV)",
            unavailable_reason=(
                None
                if available
                else f"Place {DEFAULT_PROTOTXT} and {DEFAULT_WEIGHTS} in models/."
            ),
        )

    def release(self) -> None:
        self._net = None
        self._initialized = False
