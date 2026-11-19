"""Streaming frame-based video processing service for FaceVision."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.exceptions import JobCancelledError, MediaValidationError, StorageError
from backend.app.core.logging import get_logger
from backend.app.detection.base import Detection, DetectorMetadata, FaceDetector
from backend.app.privacy.blur import BlurMode, apply_privacy_blur, draw_detections
from backend.app.processing.tracking import FaceTracker
from backend.app.vision.faces.face_mesh import (
    DISCLAIMER,
    MODEL_FILENAME,
    MODEL_LICENSE,
    FaceMeshService,
    FaceMeshSettings,
    summarize_faces,
)

logger = get_logger("processing.video")

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Extracted video stream properties."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float


@dataclass(slots=True)
class VideoProcessingConfig:
    """Configuration for frame-by-frame video processing."""

    blur_mode: BlurMode = "none"
    enable_tracking: bool = True
    target_fps: float | None = None
    target_width: int | None = None
    target_height: int | None = None
    show_hud: bool = True
    draw_boxes: bool = True
    save_privacy_video: bool = False
    max_frames: int | None = None
    max_duration_seconds: int = 3600
    face_mesh: FaceMeshSettings | None = None
    models_dir: Path | None = None


@dataclass(slots=True)
class VideoTimelineEvent:
    """Per-frame detection summary for timeline visualization."""

    frame_number: int
    timestamp_seconds: float
    face_count: int
    detections: list[dict[str, Any]]


@dataclass(slots=True)
class VideoProcessingResult:
    """Summary and artifact paths for a processed video."""

    metadata: VideoMetadata
    frames_processed: int
    total_detections: int
    unique_track_count: int
    duration_ms: float
    model_metadata: DetectorMetadata
    output_video_path: Path
    privacy_video_path: Path | None = None
    report_json_path: Path | None = None
    report_csv_path: Path | None = None
    timeline: list[VideoTimelineEvent] = field(default_factory=list)
    face_mesh: dict[str, Any] | None = None


class VideoProcessingService:
    """Processes video streams frame-by-frame without loading full videos into memory."""

    @staticmethod
    def extract_metadata(video_path: Path) -> VideoMetadata:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MediaValidationError("Unable to open video file; codec or container may be unsupported.")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise MediaValidationError("Video has invalid dimensions or zero frames.")
            duration_seconds = (total_frames / fps) if total_frames > 0 else 0.0
            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=max(0, total_frames),
                duration_seconds=duration_seconds,
            )
        finally:
            cap.release()

    @staticmethod
    def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
        fourcc_fn = cv2.VideoWriter_fourcc  # type: ignore[attr-defined]
        for codec in ("avc1", "mp4v"):
            fourcc = fourcc_fn(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, size)
            if writer.isOpened():
                return writer
            writer.release()
        raise StorageError(f"Failed to initialize video writer for '{path.name}'.")

    def process(
        self,
        input_path: Path,
        detector: FaceDetector,
        output_dir: Path,
        config: VideoProcessingConfig | None = None,
        *,
        job_id: str | None = None,
        input_name: str = "",
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> VideoProcessingResult:
        cfg = config or VideoProcessingConfig()
        meta = self.extract_metadata(input_path)
        if meta.duration_seconds > cfg.max_duration_seconds:
            raise MediaValidationError(
                f"Video duration ({meta.duration_seconds:.1f}s) exceeds limit ({cfg.max_duration_seconds}s)."
            )

        mesh_cfg = cfg.face_mesh
        mesh_service: FaceMeshService | None = None
        if mesh_cfg is not None and mesh_cfg.enabled:
            if cfg.models_dir is None:
                raise MediaValidationError("Face mesh requires a models directory.")
            mesh_service = FaceMeshService(
                cfg.models_dir,
                max_faces=mesh_cfg.max_faces,
                min_confidence=mesh_cfg.min_confidence,
                stream=True,
            )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            if mesh_service is not None:
                mesh_service.close()
            raise MediaValidationError("Unable to open video file for processing.")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_w = cfg.target_width or meta.width
        out_h = cfg.target_height or meta.height
        out_fps = cfg.target_fps if (cfg.target_fps and 0 < cfg.target_fps <= meta.fps) else meta.fps
        frame_step = max(1, round(meta.fps / out_fps))

        output_video_path = output_dir / "processed.mp4"
        writer = self._open_writer(output_video_path, out_fps, (out_w, out_h))

        privacy_video_path: Path | None = None
        privacy_writer: cv2.VideoWriter | None = None
        if cfg.save_privacy_video:
            privacy_video_path = output_dir / "privacy_blurred.mp4"
            privacy_writer = self._open_writer(privacy_video_path, out_fps, (out_w, out_h))

        tracker = FaceTracker() if cfg.enable_tracking else None
        unique_track_ids: set[int] = set()
        timeline: list[VideoTimelineEvent] = []
        frames_processed = 0
        raw_frame_idx = 0
        total_detections = 0
        mesh_frames: list[dict[str, Any]] = []
        mesh_track_ids: set[int] = set()
        mesh_frames_with_faces = 0
        start = time.perf_counter()

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError("Video processing cancelled by user.")

                if pause_event is not None:
                    while pause_event.is_set():
                        if cancel_event is not None and cancel_event.is_set():
                            raise JobCancelledError("Video processing cancelled while paused.")
                        time.sleep(0.1)

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                raw_frame_idx += 1
                if (raw_frame_idx - 1) % frame_step != 0:
                    continue

                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                detections: list[Detection] = detector.detect(frame)
                if tracker is not None:
                    detections = tracker.update(detections)
                    for det in detections:
                        if det.tracking_id is not None:
                            unique_track_ids.add(det.tracking_id)

                annotated = draw_detections(
                    frame,
                    detections,
                    blur_mode=cfg.blur_mode,
                    show_hud=cfg.show_hud,
                    draw_boxes=cfg.draw_boxes,
                )
                if mesh_service is not None and mesh_cfg is not None:
                    mesh_faces = mesh_service.detect(
                        frame, timestamp_ms=int((raw_frame_idx - 1) / meta.fps * 1000)
                    )
                    mesh_service.render(annotated, mesh_faces, mesh_cfg)
                    if mesh_faces:
                        mesh_frames_with_faces += 1
                    for mf in mesh_faces:
                        if mf.tracking_id is not None:
                            mesh_track_ids.add(mf.tracking_id)
                    mesh_frames.append(
                        {
                            "frame_number": frames_processed + 1,
                            "timestamp_seconds": round((raw_frame_idx - 1) / meta.fps, 3),
                            **summarize_faces(mesh_faces, mesh_cfg),
                        }
                    )
                writer.write(annotated)

                if privacy_writer is not None:
                    privacy_mode: BlurMode = cfg.blur_mode if cfg.blur_mode != "none" else "gaussian"
                    blurred = apply_privacy_blur(frame, detections, privacy_mode)
                    privacy_writer.write(blurred)

                frames_processed += 1
                total_detections += len(detections)
                timestamp_sec = round((raw_frame_idx - 1) / meta.fps, 3)
                timeline.append(
                    VideoTimelineEvent(
                        frame_number=frames_processed,
                        timestamp_seconds=timestamp_sec,
                        face_count=len(detections),
                        detections=[d.to_dict() for d in detections],
                    )
                )

                if progress_callback is not None:
                    estimated_total = (
                        max(1, meta.total_frames // frame_step) if meta.total_frames > 0 else frames_processed
                    )
                    progress_callback(frames_processed, estimated_total, len(detections))

                if cfg.max_frames is not None and frames_processed >= cfg.max_frames:
                    break
        finally:
            cap.release()
            writer.release()
            if privacy_writer is not None:
                privacy_writer.release()
            if mesh_service is not None:
                mesh_service.close()

        if frames_processed == 0:
            raise MediaValidationError("Video contained zero readable frames.")

        duration_ms = (time.perf_counter() - start) * 1000.0
        model_meta = detector.get_metadata()

        mesh_summary: dict[str, Any] | None = None
        if mesh_cfg is not None and mesh_cfg.enabled:
            for frame_entry in mesh_frames:
                frame_entry.pop("model", None)
                frame_entry.pop("settings", None)
                frame_entry.pop("disclaimer", None)
            mesh_summary = {
                "frames_analyzed": len(mesh_frames),
                "frames_with_faces": mesh_frames_with_faces,
                "unique_face_tracks": len(mesh_track_ids),
                "max_faces_in_frame": max((f["face_count"] for f in mesh_frames), default=0),
                "model": {"name": MODEL_FILENAME, "license": MODEL_LICENSE},
                "settings": mesh_cfg.model_dump(),
                "disclaimer": DISCLAIMER,
                "note": "Tracking ids are temporary per-video markers, not verified identities.",
            }
            (output_dir / "face_mesh.json").write_text(
                json.dumps({**mesh_summary, "frames": mesh_frames}, indent=2), encoding="utf-8"
            )

        report_json_path = output_dir / "report.json"
        report_payload = {
            "job_id": job_id,
            "input": {
                "filename": input_name or input_path.name,
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "total_frames": meta.total_frames,
                "duration_seconds": round(meta.duration_seconds, 2),
            },
            "model": {
                "id": model_meta.id,
                "name": model_meta.name,
                "version": model_meta.version,
            },
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "unique_track_count": len(unique_track_ids),
            "processing_duration_ms": round(duration_ms, 2),
            "face_mesh": mesh_summary,
            "timeline": [
                {
                    "frame_number": ev.frame_number,
                    "timestamp_seconds": ev.timestamp_seconds,
                    "face_count": ev.face_count,
                    "detections": ev.detections,
                }
                for ev in timeline
            ],
        }
        report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        report_csv_path = output_dir / "summary.csv"
        with report_csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv_writer = csv.writer(fh)
            csv_writer.writerow(["frame_number", "timestamp_seconds", "face_count"])
            for ev in timeline:
                csv_writer.writerow([ev.frame_number, ev.timestamp_seconds, ev.face_count])

        logger.info(
            "Processed video (%d frames, %d detections, %.1fms)",
            frames_processed,
            total_detections,
            duration_ms,
        )
        return VideoProcessingResult(
            metadata=meta,
            frames_processed=frames_processed,
            total_detections=total_detections,
            unique_track_count=len(unique_track_ids),
            duration_ms=duration_ms,
            model_metadata=model_meta,
            output_video_path=output_video_path,
            privacy_video_path=privacy_video_path,
            report_json_path=report_json_path,
            report_csv_path=report_csv_path,
            timeline=timeline,
            face_mesh=mesh_summary,
        )
