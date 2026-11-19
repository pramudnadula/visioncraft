"""Pluggable face detection backends (Haar Cascade and OpenCV YuNet)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2

from .models import Detection, DetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASCADE_FILE = "haarcascade_frontalface_default.xml"


class FaceDetectorBackend(Protocol):
    """Protocol implemented by all face detection backends."""

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Detect faces in a BGR image frame."""
        ...


class HaarCascadeBackend:
    """Fast CPU face detector using OpenCV Haar Cascades."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.classifier = self._load_classifier(self.config.cascade_path)

    @staticmethod
    def _load_classifier(cascade_path: Path | None) -> cv2.CascadeClassifier:
        candidates: list[Path] = []
        if cascade_path is not None:
            candidates.append(cascade_path)
        candidates.append(PROJECT_ROOT / DEFAULT_CASCADE_FILE)
        candidates.append(Path(cv2.data.haarcascades) / DEFAULT_CASCADE_FILE)

        for path in candidates:
            if path.is_file():
                classifier = cv2.CascadeClassifier(str(path))
                if not classifier.empty():
                    return classifier

        raise FileNotFoundError(
            f"Could not load '{DEFAULT_CASCADE_FILE}' from project root or OpenCV data."
        )

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Run multi-scale Haar cascade detection on a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
        )
        return [
            Detection(x=int(x), y=int(y), width=int(w), height=int(h))
            for (x, y, w, h) in raw_faces
        ]


class YuNetBackend:
    """Modern DNN face detector using OpenCV's FaceDetectorYN (ONNX)."""

    def __init__(self, model_path: Path, config: DetectorConfig | None = None) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet ONNX model not found: {model_path}")
        self.config = config or DetectorConfig()
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=self.config.score_threshold,
        )

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Run YuNet detection on a BGR frame."""
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        detections: list[Detection] = []
        for row in raw_faces:
            x, y, w, h = (int(v) for v in row[:4])
            confidence = float(row[-1])
            detections.append(
                Detection(
                    x=max(0, x),
                    y=max(0, y),
                    width=max(0, w),
                    height=max(0, h),
                    confidence=confidence,
                )
            )
        return detections


def create_backend(config: DetectorConfig) -> FaceDetectorBackend:
    """Factory that selects YuNet if a model path is provided, else Haar Cascade."""
    if config.yunet_model_path is not None:
        return YuNetBackend(config.yunet_model_path, config=config)
    return HaarCascadeBackend(config=config)
