"""OpenCV YuNet ONNX face detection backend."""

from __future__ import annotations

from pathlib import Path

import cv2

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
    FaceDetector,
    LandmarkPoint,
)

DEFAULT_YUNET_FILENAMES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
    "yunet.onnx",
)
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class YuNetDetector(FaceDetector):
    """Modern DNN face detector using OpenCV FaceDetectorYN (ONNX)."""

    model_id = "yunet"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._detector: cv2.FaceDetectorYN | None = None

    def _resolve_model_path(self) -> Path | None:
        if self.config.model_path is not None and self.config.model_path.is_file():
            return self.config.model_path

        search_dirs: list[Path] = []
        if self.config.models_dir is not None:
            search_dirs.append(self.config.models_dir)
        search_dirs.append(PROJECT_ROOT / "models")
        search_dirs.append(PROJECT_ROOT)

        for directory in search_dirs:
            for filename in DEFAULT_YUNET_FILENAMES:
                candidate = directory / filename
                if candidate.is_file():
                    return candidate
        return None

    def is_available(self) -> bool:
        return hasattr(cv2, "FaceDetectorYN") and self._resolve_model_path() is not None

    def initialize(self) -> None:
        model_path = self._resolve_model_path()
        if model_path is None:
            raise ModelNotAvailableError(
                "YuNet ONNX model not found in models/ (expected face_detection_yunet_2023mar.onnx)."
            )

        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        if self.config.use_gpu and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),
                score_threshold=self.config.confidence_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=backend_id,
                target_id=target_id,
            )
        except cv2.error as exc:
            raise ModelInitializationError(f"Failed to initialize YuNet model: {exc}") from exc
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._detector is None:
            raise ModelInitializationError("YuNet detector is not initialized.")
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        min_w, min_h = self.config.min_face_size
        detections: list[Detection] = []
        for row in raw_faces:
            x, y, w, h = (int(v) for v in row[:4])
            if w < min_w or h < min_h:
                continue
            confidence = float(row[-1])
            landmarks = [LandmarkPoint(x=float(row[i]), y=float(row[i + 1])) for i in range(4, 14, 2)]
            detections.append(
                Detection(
                    x=max(0, x),
                    y=max(0, y),
                    width=max(0, w),
                    height=max(0, h),
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )
        return detections

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        return DetectorMetadata(
            id=self.model_id,
            name="YuNet (OpenCV FaceDetectorYN)",
            version="2023mar",
            description="Lightweight ONNX face & 5-landmark detector with confidence scores.",
            supports_confidence=True,
            supports_landmarks=True,
            supports_gpu=cuda_ok,
            gpu_active=bool(self.config.use_gpu and cuda_ok),
            is_optional=False,
            available=available,
            license="MIT (OpenCV Zoo)",
            unavailable_reason=(
                None if available else "Place face_detection_yunet_2023mar.onnx in the models/ directory."
            ),
        )

    def release(self) -> None:
        self._detector = None
        self._initialized = False
