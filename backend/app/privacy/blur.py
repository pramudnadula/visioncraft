"""Privacy anonymization and frame annotation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, get_args

import cv2
import numpy as np

from backend.app.detection.base import Detection

if TYPE_CHECKING:
    from backend.app.vision.faces.segmentation import FacePrivacyContext

BlurMode = Literal["none", "gaussian", "pixelate", "mask", "mesh", "geometric", "mesh_3d"]
PRIVACY_MODES: tuple[str, ...] = get_args(BlurMode)


def validate_blur_mode(mode: str) -> BlurMode:
    """Return a typed blur mode or raise ValueError for unsupported values."""
    if mode not in PRIVACY_MODES:
        raise ValueError(f"Unsupported privacy mode '{mode}'. Supported: {', '.join(PRIVACY_MODES)}")
    return mode  # type: ignore[return-value]


def _clip_roi(frame: cv2.typing.MatLike, detection: Detection) -> tuple[int, int, int, int] | None:
    """Clamp detection coordinates to frame bounds; return None if empty."""
    img_h, img_w = frame.shape[:2]
    x1 = max(0, detection.x)
    y1 = max(0, detection.y)
    x2 = min(img_w, detection.x + detection.width)
    y2 = min(img_h, detection.y + detection.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _blend_roi(
    frame: cv2.typing.MatLike,
    bounds: tuple[int, int, int, int],
    effect_roi: cv2.typing.MatLike,
    alpha: np.ndarray,
) -> None:
    """Alpha-blend ``effect_roi`` into ``frame`` inside ``bounds`` using soft mask ``alpha``."""
    x1, y1, x2, y2 = bounds
    base = frame[y1:y2, x1:x2].astype(np.float32)
    blended = effect_roi.astype(np.float32) * alpha + base * (1.0 - alpha)
    frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def blur_face_region(
    frame: cv2.typing.MatLike,
    detection: Detection,
    mode: BlurMode,
    *,
    context: FacePrivacyContext | None = None,
) -> None:
    """Apply real-world contoured privacy anonymization or 3D mesh to a detected face."""
    if mode == "none":
        return

    bounds = _clip_roi(frame, detection)
    if bounds is None:
        return

    x1, y1, x2, y2 = bounds
    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    if context is None and mode in {"gaussian", "pixelate", "mask", "mesh", "geometric", "mesh_3d"}:
        from backend.app.vision.faces.segmentation import build_face_privacy_context

        context = build_face_privacy_context(
            frame,
            need_segmentation=mode in {"gaussian", "pixelate", "mask", "mesh"},
            need_mesh=True,
        )

    mesh_face = context.match_mesh(detection) if context is not None else None

    if mode == "gaussian":
        kernel_w = max(15, (roi_w // 3) | 1)
        kernel_h = max(15, (roi_h // 3) | 1)
        blurred = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)
        assert context is not None
        alpha = context.build_roi_alpha(frame.shape, detection, bounds, mesh_face=mesh_face)
        _blend_roi(frame, bounds, blurred, alpha)
    elif mode == "pixelate":
        small = cv2.resize(
            roi,
            (max(1, roi_w // 12), max(1, roi_h // 12)),
            interpolation=cv2.INTER_LINEAR,
        )
        pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        assert context is not None
        alpha = context.build_roi_alpha(frame.shape, detection, bounds, mesh_face=mesh_face)
        _blend_roi(frame, bounds, pixelated, alpha)
    elif mode == "mask":
        # Real-world anatomical privacy shield contoured to the face silhouette
        assert context is not None
        alpha = context.build_roi_alpha(frame.shape, detection, bounds, mesh_face=mesh_face)
        shield = np.full_like(roi, (24, 26, 34))
        # Subtle gradient + visor highlight across upper-face eye line
        visor_y1 = int(roi_h * 0.28)
        visor_y2 = int(roi_h * 0.48)
        if visor_y2 > visor_y1:
            shield[visor_y1:visor_y2, :] = (36, 42, 58)
        _blend_roi(frame, bounds, shield, alpha * 0.96)
        # Draw crisp contoured rim around the mask
        mask_u8 = (alpha[:, :, 0] > 0.35).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted = [c + np.array([[[x1, y1]]], dtype=np.int32) for c in contours]
        cv2.drawContours(frame, shifted, -1, (190, 165, 95), 2, cv2.LINE_AA)
    elif mode == "mesh":
        # Real-world 3D-contoured privacy net: blur identity underneath + 3D mesh overlay
        kernel_w = max(15, (roi_w // 3) | 1)
        kernel_h = max(15, (roi_h // 3) | 1)
        blurred = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)
        assert context is not None
        alpha = context.build_roi_alpha(frame.shape, detection, bounds, mesh_face=mesh_face)
        _blend_roi(frame, bounds, blurred, alpha)
        if mesh_face is not None:
            from backend.app.vision.faces.face_mesh import FaceMeshSettings, render_face_mesh

            render_face_mesh(
                frame,
                [mesh_face],
                FaceMeshSettings.from_preset(
                    "full_mesh",
                    color="#E2E8F0",
                    feature_color="#38BDF8",
                    opacity=0.82,
                    thickness=1,
                ),
            )
        else:
            # Contoured geodesic net fallback clipped to the face silhouette
            net = roi.copy()
            step = max(10, min(roi_w, roi_h) // 8)
            for gx in range(0, roi_w, step):
                cv2.line(net, (gx, 0), (gx, roi_h), (226, 232, 240), 1, cv2.LINE_AA)
            for gy in range(0, roi_h, step):
                cv2.line(net, (0, gy), (roi_w, gy), (226, 232, 240), 1, cv2.LINE_AA)
            _blend_roi(frame, bounds, net, alpha * 0.75)
    elif mode == "geometric":
        if mesh_face is not None:
            from backend.app.vision.faces.face_mesh import FaceMeshSettings, render_face_mesh

            render_face_mesh(
                frame,
                [mesh_face],
                FaceMeshSettings.from_preset(
                    "contours",
                    color="#00E5FF",
                    feature_color="#FFD400",
                    opacity=0.9,
                    thickness=1,
                    show_landmarks=True,
                ),
            )
        else:
            _draw_geometric_face_mesh(frame, detection, bounds)
    elif mode == "mesh_3d":
        from backend.app.vision.faces.mesh_3d import draw_3d_face_mesh

        draw_3d_face_mesh(frame, detection, mesh_face=mesh_face)


def _draw_geometric_face_mesh(
    frame: cv2.typing.MatLike,
    detection: Detection,
    bounds: tuple[int, int, int, int],
) -> None:
    """Draw geometric facial landmarks and Delaunay triangulation mesh over a face."""
    x1, y1, x2, y2 = bounds
    w = x2 - x1
    h = y2 - y1
    if w < 16 or h < 16:
        return

    img_h, img_w = frame.shape[:2]
    # Base facial geometry control points (normalized to face box)
    rel_points = [
        (0.50, 0.05),
        (0.30, 0.08),
        (0.70, 0.08),  # forehead
        (0.12, 0.28),
        (0.88, 0.28),  # temples
        (0.24, 0.25),
        (0.38, 0.24),
        (0.62, 0.24),
        (0.76, 0.25),  # eyebrows
        (0.50, 0.35),  # nose bridge
        (0.08, 0.50),
        (0.92, 0.50),  # cheeks
        (0.14, 0.72),
        (0.86, 0.72),  # jaw
        (0.32, 0.88),
        (0.50, 0.96),
        (0.68, 0.88),  # chin
    ]
    pts: list[tuple[float, float]] = [
        (
            min(float(img_w - 2), max(1.0, x1 + px * w)),
            min(float(img_h - 2), max(1.0, y1 + py * h)),
        )
        for (px, py) in rel_points
    ]

    if detection.landmarks:
        for lm in detection.landmarks:
            lx = min(float(img_w - 2), max(1.0, lm.x))
            ly = min(float(img_h - 2), max(1.0, lm.y))
            pts.append((lx, ly))
    else:
        # Fallback eye/nose/mouth anchors when backend doesn't emit landmarks
        for px, py in [(0.33, 0.36), (0.67, 0.36), (0.50, 0.56), (0.36, 0.74), (0.64, 0.74)]:
            pts.append((x1 + px * w, y1 + py * h))

    subdiv = cv2.Subdiv2D((0, 0, img_w, img_h))
    for pt in pts:
        subdiv.insert(pt)

    yellow = (0, 230, 255)
    for tri in subdiv.getTriangleList():
        p1 = (int(tri[0]), int(tri[1]))
        p2 = (int(tri[2]), int(tri[3]))
        p3 = (int(tri[4]), int(tri[5]))
        if all(0 <= p[0] < img_w and 0 <= p[1] < img_h for p in (p1, p2, p3)):
            cv2.line(frame, p1, p2, yellow, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p3, yellow, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p1, yellow, 1, cv2.LINE_AA)

    for px, py in pts:
        cv2.circle(frame, (int(px), int(py)), 2, yellow, -1, cv2.LINE_AA)


def apply_privacy_blur(
    frame: cv2.typing.MatLike,
    detections: Sequence[Detection],
    mode: BlurMode,
    *,
    selected_indices: Iterable[int] | None = None,
    selected_tracking_ids: Iterable[int] | None = None,
) -> cv2.typing.MatLike:
    """Apply privacy blur to all or selected detected faces, returning a new frame."""
    result = frame.copy()
    if mode == "none":
        return result

    idx_set = set(selected_indices) if selected_indices is not None else None
    tid_set = set(selected_tracking_ids) if selected_tracking_ids is not None else None

    from backend.app.vision.faces.segmentation import build_face_privacy_context

    context = (
        build_face_privacy_context(
            frame,
            need_segmentation=mode in {"gaussian", "pixelate", "mask", "mesh"},
            need_mesh=True,
        )
        if detections
        else None
    )

    for idx, det in enumerate(detections):
        if idx_set is not None or tid_set is not None:
            match_idx = idx_set is not None and idx in idx_set
            match_tid = tid_set is not None and det.tracking_id is not None and det.tracking_id in tid_set
            if not (match_idx or match_tid):
                continue
        blur_face_region(result, det, mode, context=context)

    return result


def draw_detections(
    frame: cv2.typing.MatLike,
    detections: Sequence[Detection],
    *,
    blur_mode: BlurMode = "none",
    selected_indices: Iterable[int] | None = None,
    box_color: tuple[int, int, int] = (0, 220, 120),
    box_thickness: int = 2,
    show_hud: bool = True,
    fps: float | None = None,
    draw_boxes: bool = True,
) -> cv2.typing.MatLike:
    """Render privacy blur, bounding boxes, landmarks, tracking IDs, and HUD."""
    annotated = apply_privacy_blur(
        frame,
        detections,
        blur_mode,
        selected_indices=selected_indices,
    )

    if draw_boxes:
        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

            label_parts: list[str] = []
            if det.tracking_id is not None:
                label_parts.append(f"#{det.tracking_id}")
            if det.confidence is not None:
                label_parts.append(f"{det.confidence:.0%}")

            if label_parts:
                cv2.putText(
                    annotated,
                    " ".join(label_parts),
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

            if det.landmarks:
                for lm in det.landmarks:
                    cv2.circle(annotated, (int(lm.x), int(lm.y)), 2, (255, 200, 0), -1)

    if show_hud:
        draw_hud(annotated, len(detections), fps=fps)

    return annotated


def draw_hud(
    frame: cv2.typing.MatLike,
    face_count: int,
    fps: float | None = None,
) -> None:
    """Draw a compact status bar in the top-left corner of the frame."""
    text = f"Faces: {face_count}"
    if fps is not None:
        text = f"{text} | FPS: {fps:.1f}"

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 10
    cv2.rectangle(
        frame,
        (10, 10),
        (10 + text_w + pad * 2, 10 + text_h + baseline + pad),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (10 + pad, 10 + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
