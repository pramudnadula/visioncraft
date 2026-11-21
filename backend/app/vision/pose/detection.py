"""Upper/full-body human pose landmark extractor using MediaPipe Pose Landmarker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.detection.base import Detection
from backend.app.detection.registry import DetectorConfig, create_default_registry
from backend.app.vision.objects.detection import YOLOv8ONNXObjectDetector

# Preferred first: "full" is noticeably more accurate on small or partly hidden people
# (e.g. back rows of group photos); "lite" stays supported as a smaller fallback.
MODEL_FILENAMES = ("pose_landmarker_full.task", "pose_landmarker_lite.task")
MODEL_SHA256 = {
    "pose_landmarker_full.task": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
}
MODEL_FILENAME = MODEL_FILENAMES[-1]  # Minimum required model, kept for compatibility.
MODEL_LICENSE = "Apache-2.0 (Google MediaPipe)"


def resolve_pose_model_path(models_dir: Path | None = None) -> Path | None:
    """Locate the best available pose landmarker (full, then lite) in the models directory."""
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for name in MODEL_FILENAMES:
        for directory in (base, base / "custom"):
            candidate = directory / name
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
                f"No pose model ({' or '.join(MODEL_FILENAMES)}) found in {self.models_dir}."
            )

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

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
            for (x1, y1, x2, y2), person_box in regions:
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
                if len(regions) > 1:
                    _hide_points_outside(pts, person_box)
                px1, py1, px2, py2 = person_box
                poses.append(
                    {
                        "bbox": {"x": px1, "y": py1, "width": px2 - px1, "height": py2 - py1},
                        "landmarks": pts,
                        "face_refined": False,
                    }
                )

        if poses:
            _snap_faces_to_detector(poses, self._detect_faces(frame))

        return {
            "pose_detected": bool(poses),
            "person_count": len(poses),
            "landmark_count": len(poses[0]["landmarks"]) if poses else 0,
            # First person's landmarks, kept for backwards compatibility.
            "landmarks": poses[0]["landmarks"] if poses else [],
            "poses": poses,
            "model": model_path.name,
            "license": MODEL_LICENSE,
        }

    def _person_regions(
        self, frame: cv2.typing.MatLike
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
        """(padded crop box, person box) pairs from the YOLO detector, or the whole frame."""
        h, w = frame.shape[:2]
        whole = [((0, 0, w, h), (0, 0, w, h))]
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
        regions: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
        for d in people:
            pad_x, pad_y = int(d.width * 0.15), int(d.height * 0.1)
            padded = (
                max(0, d.x - pad_x),
                max(0, d.y - pad_y),
                min(w, d.x + d.width + pad_x),
                min(h, d.y + d.height + pad_y),
            )
            person = (max(0, d.x), max(0, d.y), min(w, d.x + d.width), min(h, d.y + d.height))
            regions.append((padded, person))
        # Left-to-right order reads naturally in results.
        return sorted(regions, key=lambda r: r[1][0]) or whole

    def _detect_faces(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Precise face boxes + 5 landmarks from YuNet, if installed (best effort)."""
        try:
            registry = create_default_registry(self.models_dir)
            detector = registry.create_detector(DetectorConfig(model_id="yunet", models_dir=self.models_dir))
        except Exception:
            return []
        try:
            return [f for f in detector.detect(frame) if f.landmarks and len(f.landmarks) >= 5]
        except Exception:
            return []
        finally:
            detector.release()


# MediaPipe Pose indices.
_NOSE = 0
_EYE_GROUPS = ((2, (1, 3, 7)), (5, (4, 6, 8)))  # eye centre -> points that move with it
_MOUTH = (9, 10)
_FACE_POINTS = range(0, 11)


def _hide_points_outside(pts: list[dict[str, Any]], box: tuple[int, int, int, int]) -> None:
    """Mark points outside the person's own box as not visible.

    In crowded photos the pose model guesses hidden body parts and often places them
    on the neighbour in front. Those guesses land outside the person's own box.
    """
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.05
    for p in pts:
        if not (x1 - mx <= p["x"] <= x2 + mx and y1 - my <= p["y"] <= y2 + my):
            p["visibility"] = 0.0
            p["outside_person"] = True


def _snap_faces_to_detector(poses: list[dict[str, Any]], faces: list[Detection]) -> None:
    """Replace each pose's coarse face points with the face detector's precise ones.

    The pose model's 11 face points are approximate (about 0.2 face-widths off on
    group photos). YuNet's eyes, nose and mouth corners are far more precise, so each
    pose is matched to the nearest unused face inside its person box and moved onto it.
    Eyes and mouth corners are paired by image x-position to avoid left/right mix-ups.
    """
    pairs: list[tuple[float, int, int]] = []
    for pi, pose in enumerate(poses):
        nose = pose["landmarks"][_NOSE]
        b = pose["bbox"]
        for fi, f in enumerate(faces):
            fcx, fcy = f.x + f.width / 2, f.y + f.height / 2
            if not (b["x"] <= fcx <= b["x"] + b["width"] and b["y"] <= fcy <= b["y"] + b["height"]):
                continue
            dist = float(np.hypot(nose["x"] - fcx, nose["y"] - fcy)) / max(f.width, 1)
            if dist <= 1.5:
                pairs.append((dist, pi, fi))
    used_p: set[int] = set()
    used_f: set[int] = set()
    for _, pi, fi in sorted(pairs):
        if pi in used_p or fi in used_f:
            continue
        used_p.add(pi)
        used_f.add(fi)
        _apply_face(poses[pi]["landmarks"], faces[fi])
        poses[pi]["face_refined"] = True


def _apply_face(pts: list[dict[str, Any]], face: Detection) -> None:
    lms = face.landmarks or []
    eyes = sorted(lms[0:2], key=lambda p: p.x)
    mouth = sorted(lms[3:5], key=lambda p: p.x)

    def move(idx: int, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - pts[idx]["x"], y - pts[idx]["y"]
        pts[idx]["x"], pts[idx]["y"] = round(x, 2), round(y, 2)
        return dx, dy

    move(_NOSE, lms[2].x, lms[2].y)
    # Pair detector eyes/mouth corners with pose ones by image x-position.
    pose_eyes = sorted(_EYE_GROUPS, key=lambda g: pts[g[0]]["x"])
    for (eye_idx, followers), target in zip(pose_eyes, eyes, strict=True):
        dx, dy = move(eye_idx, target.x, target.y)
        for j in followers:
            pts[j]["x"], pts[j]["y"] = round(pts[j]["x"] + dx, 2), round(pts[j]["y"] + dy, 2)
    for idx, target in zip(sorted(_MOUTH, key=lambda i: pts[i]["x"]), mouth, strict=True):
        move(idx, target.x, target.y)
    conf = float(face.confidence or 0.9)
    for i in _FACE_POINTS:
        pts[i]["visibility"] = max(float(pts[i]["visibility"]), conf)
        pts[i].pop("outside_person", None)
