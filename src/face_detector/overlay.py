"""Frame rendering utilities: bounding boxes, privacy blur, and HUD overlays."""

from __future__ import annotations

from collections.abc import Sequence

import cv2

from .models import BlurMode, Detection, DetectorConfig


def _clip_roi(
    frame: cv2.typing.MatLike, detection: Detection
) -> tuple[int, int, int, int] | None:
    """Clamp detection coordinates to image bounds; return None if empty."""
    img_h, img_w = frame.shape[:2]
    x1 = max(0, detection.x)
    y1 = max(0, detection.y)
    x2 = min(img_w, detection.x + detection.width)
    y2 = min(img_h, detection.y + detection.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_privacy_blur(
    frame: cv2.typing.MatLike,
    detection: Detection,
    mode: BlurMode,
) -> None:
    """Blur or pixelate a detected face region in-place."""
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
        frame[y1:y2, x1:x2] = cv2.resize(
            small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
        )


def draw_detections(
    frame: cv2.typing.MatLike,
    detections: Sequence[Detection],
    config: DetectorConfig,
    fps: float | None = None,
) -> cv2.typing.MatLike:
    """Render bounding boxes, optional privacy blur, and HUD onto a frame copy."""
    annotated = frame.copy()

    for det in detections:
        if config.blur_mode != "none":
            apply_privacy_blur(annotated, det, config.blur_mode)

        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            config.box_color,
            config.box_thickness,
        )

        if det.confidence is not None:
            label = f"{det.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.box_color,
                2,
                cv2.LINE_AA,
            )

    if config.show_hud:
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

    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
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
