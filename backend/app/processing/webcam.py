"""Shared webcam frame processing helper for browser & desktop streams."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.detection.base import Detection, FaceDetector
from backend.app.privacy.blur import BlurMode, draw_detections


@dataclass(slots=True)
class WebcamFrameResult:
    """Result for a single processed webcam frame."""

    detections: list[Detection]
    latency_ms: float
    annotated_jpeg: bytes | None = None


class WebcamProcessor:
    """Processes live frames from browser MediaDevices or local OpenCV capture."""

    def process_frame(
        self,
        frame: cv2.typing.MatLike,
        detector: FaceDetector,
        *,
        blur_mode: BlurMode = "none",
        draw_boxes: bool = True,
        encode_jpeg: bool = True,
    ) -> WebcamFrameResult:
        start = time.perf_counter()
        detections = detector.detect(frame)
        latency_ms = (time.perf_counter() - start) * 1000.0

        annotated_jpeg: bytes | None = None
        if encode_jpeg:
            rendered = draw_detections(
                frame,
                detections,
                blur_mode=blur_mode,
                draw_boxes=draw_boxes,
                show_hud=False,
            )
            ok, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                annotated_jpeg = buf.tobytes()

        return WebcamFrameResult(
            detections=detections,
            latency_ms=latency_ms,
            annotated_jpeg=annotated_jpeg,
        )

    @staticmethod
    def decode_base64_frame(data_url_or_b64: str) -> cv2.typing.MatLike | None:
        """Decode a base64 JPEG/PNG frame sent over WebSocket."""
        payload = data_url_or_b64.split(",", 1)[1] if "," in data_url_or_b64 else data_url_or_b64
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
