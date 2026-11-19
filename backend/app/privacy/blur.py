"""Privacy anonymization and frame annotation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import cv2

from backend.app.detection.base import Detection

BlurMode = Literal["none", "gaussian", "pixelate", "mask", "mesh", "geometric"]


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


def blur_face_region(
    frame: cv2.typing.MatLike,
    detection: Detection,
    mode: BlurMode,
) -> None:
    """Blur or pixelate a single detected face region in-place."""
    if mode == "none":
        return

    bounds = _clip_roi(frame, detection)
    if bounds is None:
        return

    x1, y1, x2, y2 = bounds
    roi = frame[y1:y2, x1:x2]

    if mode == "gaussian":
        kernel_w = max(15, ((x2 - x1) // 3) | 1)
        kernel_h = max(15, ((y2 - y1) // 3) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)
    elif mode == "pixelate":
        roi_h, roi_w = roi.shape[:2]
        small = cv2.resize(
            roi,
            (max(1, roi_w // 12), max(1, roi_h // 12)),
            interpolation=cv2.INTER_LINEAR,
        )
        frame[y1:y2, x1:x2] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    elif mode == "mask":
        frame[y1:y2, x1:x2] = (15, 15, 15)
    elif mode == "mesh":
        roi_h, roi_w = roi.shape[:2]
        small = cv2.resize(
            roi,
            (max(1, roi_w // 10), max(1, roi_h // 10)),
            interpolation=cv2.INTER_LINEAR,
        )
        covered = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        step = max(8, min(roi_w, roi_h) // 8)
        for gx in range(0, roi_w, step):
            cv2.line(covered, (gx, 0), (gx, roi_h), (25, 25, 25), 1)
        for gy in range(0, roi_h, step):
            cv2.line(covered, (0, gy), (roi_w, gy), (25, 25, 25), 1)
        frame[y1:y2, x1:x2] = covered
    elif mode == "geometric":
        _draw_geometric_face_mesh(frame, detection, bounds)


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
        (0.50, 0.05), (0.30, 0.08), (0.70, 0.08),  # forehead
        (0.12, 0.28), (0.88, 0.28),                # temples
        (0.24, 0.25), (0.38, 0.24), (0.62, 0.24), (0.76, 0.25),  # eyebrows
        (0.50, 0.35),                              # nose bridge
        (0.08, 0.50), (0.92, 0.50),                # cheeks
        (0.14, 0.72), (0.86, 0.72),                # jaw
        (0.32, 0.88), (0.50, 0.96), (0.68, 0.88),  # chin
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

    for idx, det in enumerate(detections):
        if idx_set is not None or tid_set is not None:
            match_idx = idx_set is not None and idx in idx_set
            match_tid = tid_set is not None and det.tracking_id is not None and det.tracking_id in tid_set
            if not (match_idx or match_tid):
                continue
        blur_face_region(result, det, mode)

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
