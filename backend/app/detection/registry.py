"""Extensible model registry for FaceVision detection backends."""

from __future__ import annotations

from pathlib import Path

from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.detection.backends.haar import HaarCascadeDetector
from backend.app.detection.backends.opencv_dnn import OpenCVDNNDetector
from backend.app.detection.backends.retinaface import RetinaFaceDetector
from backend.app.detection.backends.scrfd import SCRFDDetector
from backend.app.detection.backends.yunet import YuNetDetector
from backend.app.detection.base import DetectorConfig, DetectorMetadata, FaceDetector


class DetectorRegistry:
    """Registry that discovers, validates, and instantiates face detectors."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self._models_dir = models_dir
        self._backends: dict[str, type[FaceDetector]] = {}

    def register(self, model_id: str, detector_cls: type[FaceDetector]) -> None:
        """Register a FaceDetector class under a unique model_id."""
        self._backends[model_id] = detector_cls

    def list_models(self) -> list[DetectorMetadata]:
        """Return metadata and availability for all registered models."""
        results: list[DetectorMetadata] = []
        for model_id, detector_cls in self._backends.items():
            cfg = DetectorConfig(model_id=model_id, models_dir=self._models_dir)
            instance = detector_cls(cfg)
            results.append(instance.get_metadata())
        return results

    def create_detector(self, config: DetectorConfig) -> FaceDetector:
        """Instantiate and initialize a detector for the requested model_id."""
        detector_cls = self._backends.get(config.model_id)
        if detector_cls is None:
            raise ModelNotAvailableError(f"Unknown detection model: '{config.model_id}'.")

        if config.models_dir is None and self._models_dir is not None:
            config.models_dir = self._models_dir

        detector = detector_cls(config)
        if not detector.is_available():
            meta = detector.get_metadata()
            reason = meta.unavailable_reason or "Required model files are missing."
            raise ModelNotAvailableError(f"Model '{config.model_id}' is unavailable: {reason}")

        detector.initialize()
        return detector


def create_default_registry(models_dir: Path | None = None) -> DetectorRegistry:
    """Create a DetectorRegistry pre-populated with all built-in and optional adapters."""
    registry = DetectorRegistry(models_dir=models_dir)
    registry.register("haar", HaarCascadeDetector)
    registry.register("yunet", YuNetDetector)
    registry.register("opencv_dnn", OpenCVDNNDetector)
    registry.register("scrfd", SCRFDDetector)
    registry.register("retinaface", RetinaFaceDetector)
    return registry
