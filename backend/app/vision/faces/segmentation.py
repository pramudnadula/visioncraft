"""Real-world face/head segmentation & contour masking using MediaPipe models."""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import get_settings
from backend.app.detection.base import Detection
from backend.app.vision.faces.face_mesh import (
    FaceMeshFace,
    FaceMeshService,
    _get_connections,
)
from backend.app.vision.faces.face_mesh import (
    resolve_model_path as resolve_face_mesh_model_path,
)

SEGMENTER_FILENAME = "selfie_multiclass_256x256.tflite"
SEGMENTER_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
SEGMENTER_LICENSE = "Apache-2.0 (Google MediaPipe)"

# Multiclass selfie segmenter labels: 0=bg, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=accessories
_FACE_CLASSES = (1, 3, 5)

_CACHE_LOCK = threading.Lock()
_SEGMENTER_CACHE: dict[Path, Any] = {}
_MESH_CACHE: dict[Path, FaceMeshService] = {}


def resolve_segmenter_model_path(models_dir: Path | None = None) -> Path | None:
    """Locate ``selfie_multiclass_256x256.tflite`` inside the configured models directory."""
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for directory in (base, base / "custom"):
        candidate = directory / SEGMENTER_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _get_segmenter(models_dir: Path) -> Any | None:
    if importlib.util.find_spec("mediapipe") is None:
        return None
    model_path = resolve_segmenter_model_path(models_dir)
    if model_path is None:
        return None
    with _CACHE_LOCK:
        cached = _SEGMENTER_CACHE.get(model_path)
        if cached is not None:
            return cached
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        _SEGMENTER_CACHE[model_path] = segmenter
        return segmenter


def _get_mesh_service(models_dir: Path) -> FaceMeshService | None:
    if importlib.util.find_spec("mediapipe") is None:
        return None
    model_path = resolve_face_mesh_model_path(models_dir)
    if model_path is None:
        return None
    with _CACHE_LOCK:
        cached = _MESH_CACHE.get(models_dir)
        if cached is not None:
            return cached
        service = FaceMeshService(models_dir, max_faces=10, min_confidence=0.35, stream=False)
        _MESH_CACHE[models_dir] = service
        return service


def _ordered_oval_polygon(face: FaceMeshFace) -> np.ndarray | None:
    """Order MediaPipe FACE_OVAL edge pairs into a closed polygon."""
    pairs = _get_connections().get("face_oval", [])
    if not pairs:
        return None
    adj: dict[int, list[int]] = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = pairs[0][0]
    order = [start]
    prev = -1
    cur = start
    for _ in range(len(pairs) + 1):
        nxt = next((n for n in adj.get(cur, ()) if n != prev), None)
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) < 3:
        return None
    pts = face.landmarks_px[order]
    return np.round(pts).astype(np.int32)


def _match_face(det: Detection, mesh_faces: list[FaceMeshFace]) -> FaceMeshFace | None:
    if not mesh_faces:
        return None
    dcx = det.x + det.width * 0.5
    dcy = det.y + det.height * 0.5
    best: FaceMeshFace | None = None
    best_dist = float("inf")
    max_tol = max(det.width, det.height) * 0.85
    for mf in mesh_faces:
        mx, my, mw, mh = mf.bbox
        mcx = mx + mw * 0.5
        mcy = my + mh * 0.5
        dist = float(np.hypot(dcx - mcx, dcy - mcy))
        if dist < best_dist and dist <= max_tol:
            best_dist = dist
            best = mf
    return best


@dataclass(slots=True)
class FacePrivacyContext:
    """Cached per-frame segmentation + 478-landmark metadata for privacy rendering."""

    head_mask: np.ndarray | None  # float32 (H, W) in [0, 1]
    mesh_faces: list[FaceMeshFace]

    def match_mesh(self, det: Detection) -> FaceMeshFace | None:
        return _match_face(det, self.mesh_faces)

    def build_roi_alpha(
        self,
        frame_shape: tuple[int, ...],
        det: Detection,
        bounds: tuple[int, int, int, int],
        *,
        mesh_face: FaceMeshFace | None = None,
    ) -> np.ndarray:
        """Return a soft-feathered float32 (roi_h, roi_w, 1) mask in [0, 1]."""
        x1, y1, x2, y2 = bounds
        roi_w, roi_h = x2 - x1, y2 - y1
        mask: np.ndarray = np.zeros((roi_h, roi_w), dtype=np.float32)

        if mesh_face is not None:
            poly = _ordered_oval_polygon(mesh_face)
            if poly is not None:
                shifted = poly - np.array([[x1, y1]], dtype=np.int32)
                cv2.fillPoly(mask, [shifted], 1.0)
                dilate_k = max(5, (min(roi_w, roi_h) // 14) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
                mask = cv2.dilate(mask, kernel, iterations=1)

        if self.head_mask is not None:
            seg_roi = self.head_mask[y1:y2, x1:x2]
            if seg_roi.ndim == 3:
                seg_roi = seg_roi[..., 0]
            if mesh_face is not None:
                # Combine face oval with segmented face/accessories inside ROI
                mask = np.maximum(mask, seg_roi * 0.95)
            else:
                mask = seg_roi.copy()

        if float(mask.max()) < 0.05:
            # Graceful anatomical ellipse fallback when no face model matches
            cx, cy = roi_w // 2, roi_h // 2
            ax, ay = max(4, int(roi_w * 0.44)), max(4, int(roi_h * 0.48))
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

        feather = max(5, (min(roi_w, roi_h) // 10) | 1)
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        return np.clip(mask, 0.0, 1.0)[:, :, None]


def build_face_privacy_context(
    frame: cv2.typing.MatLike,
    *,
    models_dir: Path | None = None,
    need_segmentation: bool = True,
    need_mesh: bool = True,
) -> FacePrivacyContext:
    """Run MediaPipe segmentation & face landmarker once per frame for all detections."""
    resolved = models_dir if models_dir is not None else get_settings().resolved_models_dir
    head_mask: np.ndarray | None = None
    mesh_faces: list[FaceMeshFace] = []

    if need_segmentation:
        segmenter = _get_segmenter(resolved)
        if segmenter is not None:
            try:
                import mediapipe as mp

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                with _CACHE_LOCK:
                    seg_res = segmenter.segment(mp_img)
                if seg_res.category_mask is not None:
                    cat = np.asarray(seg_res.category_mask.numpy_view())
                    # Newer MediaPipe releases return (H, W, 1); normalise to (H, W).
                    if cat.ndim == 3:
                        cat = cat[..., 0]
                    head_mask = np.isin(cat, _FACE_CLASSES).astype(np.float32)
            except Exception:
                head_mask = None

    if need_mesh:
        mesh_svc = _get_mesh_service(resolved)
        if mesh_svc is not None:
            try:
                with _CACHE_LOCK:
                    mesh_faces = mesh_svc.detect(frame)
            except Exception:
                mesh_faces = []

    return FacePrivacyContext(head_mask=head_mask, mesh_faces=mesh_faces)
