"""Shared webcam frame processing helper for browser & desktop streams."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.app.detection.base import Detection, FaceDetector
from backend.app.privacy.blur import BlurMode, draw_detections
from backend.app.vision.faces.emotion import (
    EmotionMode,
    analyze_face_emotions,
    draw_emotion_overlays,
)
from backend.app.vision.faces.face_mesh import FaceMeshService, FaceMeshSettings, summarize_faces


@dataclass(slots=True)
class WebcamFrameResult:
    """Result for a single processed webcam frame."""

    detections: list[Detection]
    latency_ms: float
    annotated_jpeg: bytes | None = None
    face_mesh: dict[str, object] | None = None
    emotions: dict[str, object] | None = None


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
        mesh_service: FaceMeshService | None = None,
        mesh_settings: FaceMeshSettings | None = None,
        emotion_mode: EmotionMode = "off",
        models_dir: Path | None = None,
    ) -> WebcamFrameResult:
        start = time.perf_counter()
        detections = detector.detect(frame)
        latency_ms = (time.perf_counter() - start) * 1000.0

        annotated_jpeg: bytes | None = None
        mesh_summary: dict[str, object] | None = None
        emotion_summary: dict[str, object] | None = None
        if encode_jpeg:
            rendered = draw_detections(
                frame,
                detections,
                blur_mode=blur_mode,
                draw_boxes=draw_boxes,
                show_hud=False,
            )
            mesh_faces = None
            if mesh_service is not None and mesh_settings is not None and mesh_settings.enabled:
                mesh_faces = mesh_service.detect(frame)
                mesh_service.render(rendered, mesh_faces, mesh_settings)
                mesh_summary = summarize_faces(
                    mesh_faces, mesh_settings.model_copy(update={"include_landmarks_in_report": False})
                )
            if emotion_mode != "off" and detections:
                emotion_summary = analyze_face_emotions(
                    frame,
                    detections,
                    mode=emotion_mode,
                    models_dir=models_dir,
                    mesh_faces=mesh_faces,
                )
                draw_emotion_overlays(rendered, detections, emotion_summary)
            ok, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                annotated_jpeg = buf.tobytes()

        return WebcamFrameResult(
            detections=detections,
            latency_ms=latency_ms,
            annotated_jpeg=annotated_jpeg,
            face_mesh=mesh_summary if encode_jpeg else None,
            emotions=emotion_summary if encode_jpeg else None,
        )

    @staticmethod
    def decode_base64_frame(data_url_or_b64: str) -> cv2.typing.MatLike | None:
        """Decode a base64 JPEG/PNG frame sent over WebSocket."""
        payload = data_url_or_b64.split(",", 1)[1] if "," in data_url_or_b64 else data_url_or_b64
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
