"""Abstract base classes and data structures for FaceVision detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """2D facial landmark coordinate."""

    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class Detection:
    """Consistent face detection result across all backends."""

    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None
    landmarks: list[LandmarkPoint] | None = None
    tracking_id: int | None = None

    def with_tracking_id(self, tracking_id: int) -> Detection:
        """Return a copy of this detection with a temporary tracking ID."""
        return Detection(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            confidence=self.confidence,
            landmarks=self.landmarks,
            tracking_id=tracking_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize detection to a JSON-friendly dictionary."""
        data = asdict(self)
        return data


@dataclass(frozen=True, slots=True)
class DetectorMetadata:
    """Descriptive metadata and capability flags for a detection backend."""

    id: str
    name: str
    version: str
    description: str
    supports_confidence: bool
    supports_landmarks: bool
    supports_gpu: bool
    gpu_active: bool
    is_optional: bool
    available: bool
    license: str
    unavailable_reason: str | None = None


@dataclass(slots=True)
class DetectorConfig:
    """Configuration passed to a FaceDetector instance."""

    model_id: str = "haar"
    confidence_threshold: float = 0.6
    min_face_size: tuple[int, int] = (30, 30)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    use_gpu: bool = False
    models_dir: Path | None = None
    model_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class FaceDetector(ABC):
    """Common interface implemented by all FaceVision detection backends."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights and prepare the detector for inference."""

    @abstractmethod
    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Run face detection on a BGR image frame."""

    @abstractmethod
    def get_metadata(self) -> DetectorMetadata:
        """Return static and runtime metadata for this detector."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if required model files and runtime dependencies exist."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying model resources."""

    def ensure_initialized(self) -> None:
        """Initialize lazily if not yet initialized."""
        if not self._initialized:
            self.initialize()
