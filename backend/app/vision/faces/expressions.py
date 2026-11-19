"""Facial landmark & observable movement blendshape analyzer (optional MediaPipe adapter)."""

from __future__ import annotations

import importlib.util
from typing import Any

import cv2

from backend.app.core.exceptions import ModelNotAvailableError


class FacialExpressionAnalyzer:
    """Optional MediaPipe Face Landmarker / FaceMesh adapter.

    Scientific & privacy notice:
    Blendshape scores represent observable facial muscle/landmark geometry only
    and do not establish true emotional state, personality, honesty, or mental health.
    """

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("mediapipe") is not None

    def analyze(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        if not self.is_available():
            raise ModelNotAvailableError(
                "Optional 'mediapipe' package is required for 468-point facial "
                "landmark & blendshape analysis."
            )
        import mediapipe as mp  # type: ignore[import-not-found]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(rgb)

        faces_out: list[dict[str, Any]] = []
        if result.multi_face_landmarks:
            h, w = frame.shape[:2]
            for face_lms in result.multi_face_landmarks:
                pts = [
                    {"x": round(lm.x * w, 2), "y": round(lm.y * h, 2), "z": round(lm.z, 4)}
                    for lm in face_lms.landmark
                ]
                faces_out.append({"landmark_count": len(pts), "landmarks": pts[:20]})

        return {
            "face_count": len(faces_out),
            "faces": faces_out,
            "limitation_notice": (
                "Observable facial geometry only; does not infer internal emotion, intent, or mental health."
            ),
        }
