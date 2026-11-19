"""Data models and configuration types for face detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

BlurMode = Literal["none", "gaussian", "pixelate"]


@dataclass(frozen=True, slots=True)
class Detection:
    """Represents a single detected face bounding box."""

    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        """Serialize detection to a JSON-friendly dictionary."""
        return asdict(self)


@dataclass(slots=True)
class DetectorConfig:
    """Runtime configuration for detection, rendering, and exports."""

    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: tuple[int, int] = (30, 30)
    score_threshold: float = 0.6
    blur_mode: BlurMode = "none"
    box_color: tuple[int, int, int] = (0, 220, 120)
    box_thickness: int = 2
    show_hud: bool = True
    cascade_path: Path | None = None
    yunet_model_path: Path | None = None
