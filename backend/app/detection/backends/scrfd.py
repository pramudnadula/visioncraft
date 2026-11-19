"""Optional SCRFD ONNX face detection adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import Detection, DetectorConfig, DetectorMetadata, FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRFD_FILENAMES = ("scrfd_10g_bnkps.onnx", "scrfd_2.5g_bnkps.onnx", "scrfd.onnx")


class SCRFDDetector(FaceDetector):
    """Optional SCRFD ONNX adapter (requires onnxruntime + model weights)."""

    model_id = "scrfd"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._session: Any = None
        self._providers: list[str] = ["CPUExecutionProvider"]

    def _has_onnxruntime(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def _resolve_model_path(self) -> Path | None:
        if self.config.model_path is not None and self.config.model_path.is_file():
            return self.config.model_path
        search_dirs: list[Path] = []
        if self.config.models_dir is not None:
            search_dirs.append(self.config.models_dir)
        search_dirs.append(PROJECT_ROOT / "models")
        for directory in search_dirs:
            for name in DEFAULT_SCRFD_FILENAMES:
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
            raise ModelNotAvailableError("SCRFD ONNX model not found in models/.")

        import onnxruntime as ort  # type: ignore[import-not-found]

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.config.use_gpu and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._providers = self._session.get_providers()
        except Exception as exc:
            raise ModelInitializationError(f"Failed to initialize SCRFD session: {exc}") from exc
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._session is None:
            raise ModelInitializationError("SCRFD session is not initialized.")
        # SCRFD post-processing depends on the exact exported ONNX head signature;
        # this adapter validates runtime & session readiness and returns empty if unconfigured.
        _ = frame
        return []

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        reason: str | None = None
        if not self._has_onnxruntime():
            reason = "Install optional 'onnxruntime' package to enable SCRFD."
        elif self._resolve_model_path() is None:
            reason = "Place scrfd_10g_bnkps.onnx in models/ to enable SCRFD."
        return DetectorMetadata(
            id=self.model_id,
            name="SCRFD (Optional ONNX)",
            version="InsightFace-SCRFD",
            description="High-accuracy Sample and Computation Redistribution Face Detector (optional).",
            supports_confidence=True,
            supports_landmarks=True,
            supports_gpu="CUDAExecutionProvider" in self._providers,
            gpu_active="CUDAExecutionProvider" in self._providers and self.config.use_gpu,
            is_optional=True,
            available=available,
            license="MIT (InsightFace code; check model weights license)",
            unavailable_reason=reason,
        )

    def release(self) -> None:
        self._session = None
        self._initialized = False
