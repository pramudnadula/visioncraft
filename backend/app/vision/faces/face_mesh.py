"""Face mesh: MediaPipe Face Landmarker service, mesh settings and overlay renderer.

Model: ``face_landmarker.task`` (Google MediaPipe, Apache-2.0). It bundles a face detector,
a 478-point 3D landmark model, a 52-score blendshape model and a geometry pipeline, and runs
fully offline on CPU. Detection, landmarks and tracking all run locally.

Scientific / privacy notice: blendshape scores describe *observable facial movement* only.
They do not establish a person's emotional state, intent, honesty or health.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.core.logging import get_logger
from backend.app.detection.base import Detection

logger = get_logger("vision.face_mesh")

MODEL_FILENAME = "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
MODEL_LICENSE = "Apache-2.0 (Google MediaPipe)"
LANDMARK_COUNT = 478
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

MeshPreset = Literal["full_mesh", "contours", "features", "dots"]
DISCLAIMER = (
    "Blendshape scores describe observable facial movement only; they do not reliably indicate "
    "emotion, intent, honesty or health and must not be used for high-impact decisions."
)


class FaceMeshSettings(BaseModel):
    """User-configurable face mesh options, shared by jobs, webcam and API requests."""

    enabled: bool = False
    max_faces: int = Field(default=3, ge=1, le=10)
    min_confidence: float = Field(default=0.5, ge=0.1, le=0.95)
    color: str = "#00E5FF"
    feature_color: str = "#FFD400"
    opacity: float = Field(default=0.7, ge=0.0, le=1.0)
    thickness: int = Field(default=1, ge=1, le=4)
    show_landmarks: bool = False
    landmark_size: int = Field(default=1, ge=1, le=5)
    show_tessellation: bool = True
    show_contours: bool = False
    show_face_oval: bool = True
    show_eyes: bool = True
    show_eyebrows: bool = True
    show_lips: bool = True
    show_nose: bool = True
    show_irises: bool = True
    include_landmarks_in_report: bool = False

    @field_validator("color", "feature_color")
    @classmethod
    def _valid_hex(cls, value: str) -> str:
        if not _HEX.match(value):
            raise ValueError("Colors must be hex values like #00E5FF.")
        return value

    @classmethod
    def from_preset(cls, preset: MeshPreset, **overrides: Any) -> FaceMeshSettings:
        base: dict[str, Any] = {"enabled": True}
        if preset == "full_mesh":
            base.update(show_tessellation=True, show_contours=False)
        elif preset == "contours":
            base.update(show_tessellation=False, show_contours=True, show_face_oval=True)
        elif preset == "features":
            base.update(show_tessellation=False, show_contours=False)
        elif preset == "dots":
            base.update(
                show_tessellation=False,
                show_contours=False,
                show_face_oval=False,
                show_eyes=False,
                show_eyebrows=False,
                show_lips=False,
                show_nose=False,
                show_irises=False,
                show_landmarks=True,
            )
        base.update(overrides)
        return cls(**base)


@dataclass(frozen=True, slots=True)
class FaceMeshFace:
    """One detected face: 3D landmarks, tracking id, pose and expression indicators."""

    index: int
    tracking_id: int | None
    bbox: tuple[int, int, int, int]
    landmarks_px: np.ndarray  # shape (478, 2) pixel coordinates
    landmarks_z: np.ndarray  # shape (478,) relative depth (same scale as normalized x)
    blendshapes: dict[str, float]
    head_pose: dict[str, float] | None

    def summary(self, *, include_landmarks: bool = False) -> dict[str, Any]:
        top = sorted(self.blendshapes.items(), key=lambda kv: kv[1], reverse=True)
        top = [(k, v) for k, v in top if k != "_neutral" and v >= 0.05][:6]
        data: dict[str, Any] = {
            "face_index": self.index,
            "tracking_id": self.tracking_id,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "landmark_count": int(self.landmarks_px.shape[0]),
            "head_pose": self.head_pose,
            "expression_indicators": [{"name": k, "score": round(v, 3)} for k, v in top],
        }
        if include_landmarks:
            data["landmarks"] = [
                [round(float(x), 1), round(float(y), 1), round(float(z), 4)]
                for (x, y), z in zip(self.landmarks_px, self.landmarks_z, strict=True)
            ]
        return data


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    r, g, b = int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
    return b, g, r


def _connections() -> dict[str, list[tuple[int, int]]]:
    from mediapipe.tasks.python import vision

    c = vision.FaceLandmarksConnections

    def pairs(items: Any) -> list[tuple[int, int]]:
        return [(int(i.start), int(i.end)) for i in items]

    return {
        "tessellation": pairs(c.FACE_LANDMARKS_TESSELATION),
        "contours": pairs(c.FACE_LANDMARKS_CONTOURS),
        "face_oval": pairs(c.FACE_LANDMARKS_FACE_OVAL),
        "eyes": pairs(c.FACE_LANDMARKS_LEFT_EYE) + pairs(c.FACE_LANDMARKS_RIGHT_EYE),
        "eyebrows": pairs(c.FACE_LANDMARKS_LEFT_EYEBROW) + pairs(c.FACE_LANDMARKS_RIGHT_EYEBROW),
        "lips": pairs(c.FACE_LANDMARKS_LIPS),
        "nose": pairs(c.FACE_LANDMARKS_NOSE),
        "irises": pairs(c.FACE_LANDMARKS_LEFT_IRIS) + pairs(c.FACE_LANDMARKS_RIGHT_IRIS),
    }


_CONNECTION_CACHE: dict[str, list[tuple[int, int]]] = {}


def _get_connections() -> dict[str, list[tuple[int, int]]]:
    if not _CONNECTION_CACHE:
        _CONNECTION_CACHE.update(_connections())
    return _CONNECTION_CACHE


def _segments(points: np.ndarray, pairs: list[tuple[int, int]]) -> list[np.ndarray]:
    n = points.shape[0]
    return [np.array([points[a], points[b]], dtype=np.int32) for a, b in pairs if a < n and b < n]


def render_face_mesh(
    frame: cv2.typing.MatLike,
    faces: list[FaceMeshFace],
    settings: FaceMeshSettings,
    connections: dict[str, list[tuple[int, int]]] | None = None,
) -> cv2.typing.MatLike:
    """Draw the configured mesh onto ``frame`` in place and return it."""
    if not settings.enabled or not faces or settings.opacity <= 0.0:
        return frame

    conns = connections if connections is not None else _get_connections()
    mesh_bgr = _hex_to_bgr(settings.color)
    feat_bgr = _hex_to_bgr(settings.feature_color)
    overlay = frame.copy()

    for face in faces:
        pts = face.landmarks_px
        if settings.show_tessellation:
            cv2.polylines(
                overlay,
                _segments(pts, conns["tessellation"]),
                False,
                mesh_bgr,
                settings.thickness,
                cv2.LINE_AA,
            )
        if settings.show_contours:
            cv2.polylines(
                overlay,
                _segments(pts, conns["contours"]),
                False,
                feat_bgr,
                settings.thickness,
                cv2.LINE_AA,
            )
        feature_regions = (
            ("face_oval", settings.show_face_oval),
            ("eyes", settings.show_eyes),
            ("eyebrows", settings.show_eyebrows),
            ("lips", settings.show_lips),
            ("nose", settings.show_nose),
            ("irises", settings.show_irises),
        )
        for region, visible in feature_regions:
            if visible:
                cv2.polylines(
                    overlay,
                    _segments(pts, conns[region]),
                    False,
                    feat_bgr,
                    settings.thickness + 1 if region != "nose" else settings.thickness,
                    cv2.LINE_AA,
                )
        if settings.show_landmarks:
            for x, y in pts.astype(np.int32):
                cv2.circle(overlay, (int(x), int(y)), settings.landmark_size, feat_bgr, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, settings.opacity, frame, 1.0 - settings.opacity, 0.0, dst=frame)

    # Tracking labels stay fully opaque so they remain readable.
    for face in faces:
        if face.tracking_id is not None:
            x, y, _w, _h = face.bbox
            cv2.putText(
                frame,
                f"Face #{face.tracking_id}",
                (x, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                feat_bgr,
                2,
                cv2.LINE_AA,
            )
    return frame


def _head_pose(matrix: np.ndarray) -> dict[str, float] | None:
    """Approximate yaw/pitch/roll in degrees from the model's facial transformation matrix."""
    try:
        rot = np.array(matrix, dtype=np.float64)[:3, :3]
        scale = np.linalg.norm(rot, axis=0)
        if np.any(scale == 0):
            return None
        rot = rot / scale
        angles, *_ = cv2.RQDecomp3x3(rot)
        return {
            "pitch_deg": round(float(angles[0]), 1),
            "yaw_deg": round(float(angles[1]), 1),
            "roll_deg": round(float(angles[2]), 1),
        }
    except (cv2.error, ValueError):
        return None


def resolve_model_path(models_dir: Path) -> Path | None:
    for directory in (models_dir, models_dir / "custom"):
        candidate = directory / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return None


def mesh_status(models_dir: Path) -> dict[str, Any]:
    """Describe availability, licence and integrity of the face mesh model for the UI/API."""
    has_pkg = importlib.util.find_spec("mediapipe") is not None
    path = resolve_model_path(models_dir)
    checksum_ok: bool | None = None
    if path is not None:
        checksum_ok = hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256
    reason = None
    if not has_pkg:
        reason = "Install the optional 'mediapipe' package (run scripts/install_face_mesh.sh)."
    elif path is None:
        reason = f"Place {MODEL_FILENAME} in models/ (download: {MODEL_URL})."
    return {
        "available": has_pkg and path is not None,
        "model": MODEL_FILENAME,
        "license": MODEL_LICENSE,
        "landmarks": LANDMARK_COUNT,
        "offline": True,
        "checksum_verified": checksum_ok,
        "download_url": MODEL_URL,
        "unavailable_reason": reason,
        "disclaimer": DISCLAIMER,
    }


class FaceMeshService:
    """Runs MediaPipe Face Landmarker for stills (IMAGE) or streams (VIDEO, with tracking)."""

    def __init__(
        self,
        models_dir: Path,
        *,
        max_faces: int = 3,
        min_confidence: float = 0.5,
        stream: bool = False,
    ) -> None:
        if importlib.util.find_spec("mediapipe") is None:
            raise ModelNotAvailableError(
                "Face mesh needs the optional 'mediapipe' package (run scripts/install_face_mesh.sh)."
            )
        path = resolve_model_path(models_dir)
        if path is None:
            raise ModelNotAvailableError(
                f"Face mesh model '{MODEL_FILENAME}' not found in models/. Download it from {MODEL_URL}."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            logger.warning("%s does not match the verified checksum; using it anyway.", MODEL_FILENAME)

        from mediapipe.tasks.python import BaseOptions, vision

        # Imported lazily: backend.app.processing imports this module (avoids a circular import).
        from backend.app.processing.tracking import FaceTracker

        self._stream = stream
        self._last_ts_ms = -1
        self._t0 = time.monotonic()
        self._tracker = FaceTracker(iou_threshold=0.25, max_missed_frames=10)
        self._connections = _get_connections()
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO if stream else vision.RunningMode.IMAGE,
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # MediaPipe raises generic RuntimeError/ValueError
            raise ModelInitializationError(f"Failed to start face mesh model: {exc}") from exc

    def _next_timestamp(self, timestamp_ms: int | None) -> int:
        ts = timestamp_ms if timestamp_ms is not None else int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts_ms + 1)  # MediaPipe requires strictly increasing timestamps
        self._last_ts_ms = ts
        return ts

    def detect(self, frame: cv2.typing.MatLike, *, timestamp_ms: int | None = None) -> list[FaceMeshFace]:
        import mediapipe as mp

        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._stream:
            result = self._landmarker.detect_for_video(image, self._next_timestamp(timestamp_ms))
        else:
            result = self._landmarker.detect(image)

        raw: list[tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float] | None]] = []
        for i, lms in enumerate(result.face_landmarks):
            arr = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
            px = np.stack([arr[:, 0] * w, arr[:, 1] * h], axis=1)
            blend = (
                {c.category_name: float(c.score) for c in result.face_blendshapes[i]}
                if i < len(result.face_blendshapes)
                else {}
            )
            pose = (
                _head_pose(result.facial_transformation_matrixes[i])
                if i < len(result.facial_transformation_matrixes)
                else None
            )
            raw.append((px, arr[:, 2], blend, pose))

        boxes: list[Detection] = []
        for px, _z, _b, _p in raw:
            x0, y0 = np.clip(px.min(axis=0), 0, [w - 1, h - 1])
            x1, y1 = np.clip(px.max(axis=0), 0, [w - 1, h - 1])
            boxes.append(Detection(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0)))

        tracked = (
            self._tracker.update(boxes)
            if self._stream
            else [b.with_tracking_id(i + 1) for i, b in enumerate(boxes)]
        )
        return [
            FaceMeshFace(
                index=i,
                tracking_id=tracked[i].tracking_id,
                bbox=(boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height),
                landmarks_px=raw[i][0],
                landmarks_z=raw[i][1],
                blendshapes=raw[i][2],
                head_pose=raw[i][3],
            )
            for i in range(len(raw))
        ]

    def render(
        self,
        frame: cv2.typing.MatLike,
        faces: list[FaceMeshFace],
        settings: FaceMeshSettings,
    ) -> cv2.typing.MatLike:
        return render_face_mesh(frame, faces, settings, self._connections)

    def close(self) -> None:
        self._landmarker.close()


def summarize_faces(faces: list[FaceMeshFace], settings: FaceMeshSettings) -> dict[str, Any]:
    """JSON-serializable per-frame/per-image summary used by reports and API responses."""
    return {
        "face_count": len(faces),
        "faces": [f.summary(include_landmarks=settings.include_landmarks_in_report) for f in faces],
        "model": {"name": MODEL_FILENAME, "license": MODEL_LICENSE, "landmarks_per_face": LANDMARK_COUNT},
        "settings": settings.model_dump(),
        "disclaimer": DISCLAIMER,
    }
