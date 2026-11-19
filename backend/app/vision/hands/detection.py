"""Hand detection & 21-point landmark extractor using MediaPipe Hand Landmarker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError

MODEL_FILENAME = "hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_LICENSE = "Apache-2.0 (Google MediaPipe)"


def resolve_hand_model_path(models_dir: Path | None = None) -> Path | None:
    """Locate ``hand_landmarker.task`` inside the configured models directory."""
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for directory in (base, base / "custom"):
        candidate = directory / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return None


class HandLandmarkDetector:
    """Detects left/right hands and 21 hand landmarks using MediaPipe Tasks."""

    def __init__(
        self,
        models_dir: Path | None = None,
        *,
        max_hands: int = 2,
        min_confidence: float = 0.5,
    ) -> None:
        self.models_dir = models_dir if models_dir is not None else get_settings().resolved_models_dir
        self.max_hands = max_hands
        self.min_confidence = min_confidence

    @staticmethod
    def is_available(models_dir: Path | None = None) -> bool:
        if importlib.util.find_spec("mediapipe") is None:
            return False
        return resolve_hand_model_path(models_dir) is not None

    def detect(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        if importlib.util.find_spec("mediapipe") is None:
            raise ModelNotAvailableError(
                "Optional 'mediapipe' package is required for hand landmark detection."
            )
        model_path = resolve_hand_model_path(self.models_dir)
        if model_path is None:
            raise ModelNotAvailableError(
                f"Hand model '{MODEL_FILENAME}' is missing from {self.models_dir}."
            )

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
        )
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        hands_out: list[dict[str, Any]] = []
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for idx, hand_lms in enumerate(result.hand_landmarks):
                handedness = "Unknown"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    handedness = str(result.handedness[idx][0].category_name or "Unknown")
                pts = [
                    {"x": round(lm.x * w, 2), "y": round(lm.y * h, 2), "z": round(float(lm.z), 4)}
                    for lm in hand_lms
                ]
                hands_out.append({"handedness": handedness, "landmarks": pts})

        return {
            "hand_count": len(hands_out),
            "hands": hands_out,
            "model": MODEL_FILENAME,
            "license": MODEL_LICENSE,
        }
