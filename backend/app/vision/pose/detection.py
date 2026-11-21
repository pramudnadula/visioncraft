"""Upper/full-body human pose landmark extractor using MediaPipe Pose Landmarker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.vision.objects.detection import YOLOv8ONNXObjectDetector

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

    def __init__(
        self,
        models_dir: Path | None = None,
        *,
        min_confidence: float = 0.5,
        person_confidence: float = 0.4,
        max_people: int = 12,
    ) -> None:
        self.models_dir = models_dir if models_dir is not None else get_settings().resolved_models_dir
        self.min_confidence = min_confidence
        self.person_confidence = person_confidence
        self.max_people = max_people

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

        h, w = frame.shape[:2]
        def make_options(conf: float) -> Any:
            return mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=conf,
                min_pose_presence_confidence=conf,
            )

        # MediaPipe Pose is a single-person model that downsamples to 256x256, so on
        # group photos it finds one person and its points drift onto neighbours.
        # Find each person with the object detector, then run pose per person crop.
        regions = self._person_regions(frame)
        poses: list[dict[str, Any]] = []
        # A person already confirmed by the object detector may still score below the
        # pose model's own cutoff (e.g. crossed arms, partial occlusion): retry lower.
        retry_conf = min(self.min_confidence, 0.25)
        with (
            mp_vision.PoseLandmarker.create_from_options(make_options(self.min_confidence)) as landmarker,
            mp_vision.PoseLandmarker.create_from_options(make_options(retry_conf)) as retry_landmarker,
        ):
            for x1, y1, x2, y2 in regions:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if not result.pose_landmarks and len(regions) > 1:
                    result = retry_landmarker.detect(mp_img)
                if not result.pose_landmarks:
                    continue
                cw, ch = x2 - x1, y2 - y1
                pts = [
                    {
                        "x": round(x1 + lm.x * cw, 2),
                        "y": round(y1 + lm.y * ch, 2),
                        "z": round(float(lm.z), 4),
                        "visibility": round(float(getattr(lm, "visibility", 1.0) or 0.0), 3),
                        "presence": round(float(getattr(lm, "presence", 1.0) or 0.0), 3),
                    }
                    for lm in result.pose_landmarks[0]
                ]
                poses.append({"bbox": {"x": x1, "y": y1, "width": cw, "height": ch}, "landmarks": pts})

        return {
            "pose_detected": bool(poses),
            "person_count": len(poses),
            "landmark_count": len(poses[0]["landmarks"]) if poses else 0,
            # First person's landmarks, kept for backwards compatibility.
            "landmarks": poses[0]["landmarks"] if poses else [],
            "poses": poses,
            "model": MODEL_FILENAME,
            "license": MODEL_LICENSE,
        }

    def _person_regions(self, frame: cv2.typing.MatLike) -> list[tuple[int, int, int, int]]:
        """Padded person boxes from the YOLO detector, or the whole frame as a fallback."""
        h, w = frame.shape[:2]
        whole = [(0, 0, w, h)]
        detector = YOLOv8ONNXObjectDetector(
            models_dir=self.models_dir, confidence_threshold=self.person_confidence
        )
        if not detector.is_available():
            return whole
        try:
            people = detector.detect(frame, allowed_classes={"person"})
        except Exception:
            return whole
        finally:
            detector.release()
        people = sorted(people, key=lambda d: d.width * d.height, reverse=True)[: self.max_people]
        regions: list[tuple[int, int, int, int]] = []
        for d in people:
            pad_x, pad_y = int(d.width * 0.15), int(d.height * 0.1)
            regions.append(
                (
                    max(0, d.x - pad_x),
                    max(0, d.y - pad_y),
                    min(w, d.x + d.width + pad_x),
                    min(h, d.y + d.height + pad_y),
                )
            )
        # Left-to-right order reads naturally in results.
        return sorted(regions, key=lambda r: r[0]) or whole
