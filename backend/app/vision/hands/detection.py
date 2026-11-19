"""Hand detection & 21-point landmark extractor (optional MediaPipe adapter)."""

from __future__ import annotations

import importlib.util
from typing import Any

import cv2

from backend.app.core.exceptions import ModelNotAvailableError


class HandLandmarkDetector:
    """Detects left/right hands and 21 hand landmarks when MediaPipe is installed."""

    @staticmethod
    def is_available() -> bool:
        if importlib.util.find_spec("mediapipe") is None:
            return False
        import mediapipe as mp  # type: ignore[import-not-found]

        # MediaPipe >= 1.0 removed the legacy `solutions` API this adapter depends on.
        return hasattr(mp, "solutions")

    def detect(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        if not self.is_available():
            raise ModelNotAvailableError(
                "Optional 'mediapipe' package is required for hand landmark detection."
            )
        import mediapipe as mp  # type: ignore[import-not-found]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        ) as hands:
            result = hands.process(rgb)

        hands_out: list[dict[str, Any]] = []
        if result.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for idx, hand_lms in enumerate(result.multi_hand_landmarks):
                handedness = "Unknown"
                if result.multi_handedness and idx < len(result.multi_handedness):
                    handedness = result.multi_handedness[idx].classification[0].label
                pts = [
                    {"x": round(lm.x * w, 2), "y": round(lm.y * h, 2), "z": round(lm.z, 4)}
                    for lm in hand_lms.landmark
                ]
                hands_out.append({"handedness": handedness, "landmarks": pts})

        return {"hand_count": len(hands_out), "hands": hands_out}
