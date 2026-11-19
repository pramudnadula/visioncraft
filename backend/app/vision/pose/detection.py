"""Upper/full-body human pose landmark extractor (optional MediaPipe adapter)."""

from __future__ import annotations

import importlib.util
from typing import Any

import cv2

from backend.app.core.exceptions import ModelNotAvailableError


class PoseLandmarkDetector:
    """Detects human pose landmarks when MediaPipe is installed."""

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("mediapipe") is not None

    def detect(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        if not self.is_available():
            raise ModelNotAvailableError(
                "Optional 'mediapipe' package is required for human pose estimation."
            )
        import mediapipe as mp  # type: ignore[import-not-found]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            result = pose.process(rgb)

        if not result.pose_landmarks:
            return {"pose_detected": False, "landmarks": []}

        h, w = frame.shape[:2]
        pts = [
            {
                "x": round(lm.x * w, 2),
                "y": round(lm.y * h, 2),
                "visibility": round(lm.visibility, 3),
            }
            for lm in result.pose_landmarks.landmark
        ]
        return {"pose_detected": True, "landmark_count": len(pts), "landmarks": pts}
