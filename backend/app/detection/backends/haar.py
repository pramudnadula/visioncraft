"""Haar Cascade face detection backend."""

from __future__ import annotations

from pathlib import Path

import cv2

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import Detection, DetectorConfig, DetectorMetadata, FaceDetector

DEFAULT_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class HaarCascadeDetector(FaceDetector):
    """CPU-fast face detector using OpenCV Haar Cascades."""

    model_id = "haar"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._classifier: cv2.CascadeClassifier | None = None

    def _candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        if self.config.model_path is not None:
            candidates.append(self.config.model_path)
        if self.config.models_dir is not None:
            candidates.append(self.config.models_dir / DEFAULT_CASCADE_FILENAME)
        candidates.append(PROJECT_ROOT / DEFAULT_CASCADE_FILENAME)
        cv2_data = getattr(cv2, "data", None)
        if cv2_data is not None and hasattr(cv2_data, "haarcascades"):
            candidates.append(Path(cv2_data.haarcascades) / DEFAULT_CASCADE_FILENAME)
        return candidates

    def _resolve_cascade_path(self) -> Path | None:
        for path in self._candidate_paths():
            if path.is_file():
                return path
        return None

    def is_available(self) -> bool:
        return self._resolve_cascade_path() is not None

    def initialize(self) -> None:
        cascade_path = self._resolve_cascade_path()
        if cascade_path is None:
            raise ModelNotAvailableError(f"Haar cascade file '{DEFAULT_CASCADE_FILENAME}' not found.")
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise ModelInitializationError(f"Failed to load Haar cascade classifier from '{cascade_path}'.")
        self._classifier = classifier
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._classifier is None:
            raise ModelInitializationError("Haar cascade classifier is not initialized.")
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_face_size,
        )
        return [Detection(x=int(x), y=int(y), width=int(w), height=int(h)) for (x, y, w, h) in raw_faces]

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        return DetectorMetadata(
            id=self.model_id,
            name="Haar Cascade (Frontal Face)",
            version="OpenCV-4.x",
            description="Classic CPU-fast Viola-Jones Haar Cascade frontal face detector.",
            supports_confidence=False,
            supports_landmarks=False,
            supports_gpu=False,
            gpu_active=False,
            is_optional=False,
            available=available,
            license="BSD-3-Clause (OpenCV)",
            unavailable_reason=None if available else "Cascade XML file not found.",
        )

    def release(self) -> None:
        self._classifier = None
        self._initialized = False
