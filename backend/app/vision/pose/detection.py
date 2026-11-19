"""Upper/full-body human pose landmark extractor using MediaPipe Pose Landmarker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError

MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
MODEL_LICENSE = "Apache-2.0 (Google MediaPipe)"


def resolve_pose_model_path(models_dir: Path | None = None) -> Path | None:
    """Locate ``pose_landmarker_lite.task`` inside the configured models directory."""
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for directory in (base, base / "custom"):
        candidate = directory / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return None


class PoseLandmarkDetector:
    """Detects 33-point 3D human body pose landmarks using MediaPipe Tasks."""

    def __init__(self, models_dir: Path | None = None, *, min_confidence: float = 0.5) -> None:
        self.models_dir = models_dir if models_dir is not None else get_settings().resolved_models_dir
        self.min_confidence = min_confidence

    @staticmethod
    def is_available(models_dir: Path | None = None) -> bool:
        if importlib.util.find_spec("mediapipe") is None:
            return False
        return resolve_pose_model_path(models_dir) is not None

    def detect(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        if importlib.util.find_spec("mediapipe") is None:
            raise ModelNotAvailableError(
                "Optional 'mediapipe' package is required for human pose estimation."
            )
        model_path = resolve_pose_model_path(self.models_dir)
        if model_path is None:
            raise ModelNotAvailableError(
                f"Pose model '{MODEL_FILENAME}' is missing from {self.models_dir}."
            )

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
        )
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return {
                "pose_detected": False,
                "landmark_count": 0,
                "landmarks": [],
                "model": MODEL_FILENAME,
                "license": MODEL_LICENSE,
            }

        h, w = frame.shape[:2]
        pts = [
            {
                "x": round(lm.x * w, 2),
                "y": round(lm.y * h, 2),
                "z": round(float(lm.z), 4),
                "visibility": round(float(getattr(lm, "visibility", 1.0) or 0.0), 3),
                "presence": round(float(getattr(lm, "presence", 1.0) or 0.0), 3),
            }
            for lm in result.pose_landmarks[0]
        ]
        return {
            "pose_detected": True,
            "landmark_count": len(pts),
            "landmarks": pts,
            "model": MODEL_FILENAME,
            "license": MODEL_LICENSE,
        }
