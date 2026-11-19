"""Unified background job manager with bounded concurrency, cancellation, and recovery."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    CloudConsentRequiredError,
    JobCancelledError,
    JobNotFoundError,
    JobStateError,
    MediaNotFoundError,
)
from backend.app.core.logging import get_logger
from backend.app.database.models import MediaAsset, ProcessingJob
from backend.app.database.session import DatabaseManager
from backend.app.detection.base import DetectorConfig
from backend.app.detection.registry import DetectorRegistry
from backend.app.jobs.events import JobEvent, JobEventBus
from backend.app.processing.batch import BatchProcessingService
from backend.app.processing.image import ImageProcessingConfig, ImageProcessingService
from backend.app.processing.video import VideoProcessingConfig, VideoProcessingService
from backend.app.storage.base import StorageAdapter

logger = get_logger("jobs.manager")

JobStatus = Literal[
    "QUEUED",
    "VALIDATING",
    "PROCESSING",
    "FINALIZING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
JobType = Literal["image", "video", "batch"]

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}
ACTIVE_STATES = {"QUEUED", "VALIDATING", "PROCESSING", "FINALIZING"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobManager:
    """Coordinates persistent job lifecycle, bounded worker execution, and cancellation."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: DatabaseManager,
        storage: StorageAdapter,
        registry: DetectorRegistry,
        event_bus: JobEventBus,
    ) -> None:
        self.settings = settings
        self.db = db
        self.storage = storage
        self.registry = registry
        self.event_bus = event_bus
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="facevision-worker",
        )
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._pause_events: dict[str, threading.Event] = {}

        self._image_service = ImageProcessingService()
        self._video_service = VideoProcessingService()
        self._batch_service = BatchProcessingService()

    def recover_on_startup(self) -> None:
        """Recover interrupted jobs after an unexpected process restart."""
        requeue_ids: list[str] = []
        with self.db.session_scope() as session:
            stale = (
                session.query(ProcessingJob)
                .filter(ProcessingJob.status.in_(["VALIDATING", "PROCESSING", "FINALIZING"]))
                .all()
            )
            for job in stale:
                job.status = "FAILED"
                job.error_message = "Job interrupted by server restart. You can retry this job."
                job.completed_at = _utcnow()

            queued = session.query(ProcessingJob).filter(ProcessingJob.status == "QUEUED").all()
            requeue_ids = [j.id for j in queued]

        for job_id in requeue_ids:
            self._submit_job(job_id)

    def shutdown(self) -> None:
        """Signal active jobs to cancel and shut down worker threads."""
        with self._lock:
            for ev in self._cancel_events.values():
                ev.set()
        self._executor.shutdown(wait=False)

    def create_job(
        self,
        *,
        job_type: JobType,
        media_ids: list[str],
        model_id: str,
        processing_mode: str,
        cloud_consent: bool,
        config: dict[str, Any],
    ) -> ProcessingJob:
        if processing_mode == "cloud" and not (cloud_consent and self.settings.allow_cloud_processing):
            raise CloudConsentRequiredError(
                "Cloud processing requires explicit user consent and cloud mode enabled in settings."
            )
        if not media_ids:
            raise MediaNotFoundError("At least one media_id is required to create a job.")

        with self.db.session_scope() as session:
            assets = session.query(MediaAsset).filter(MediaAsset.id.in_(media_ids)).all()
            found_ids = {a.id for a in assets}
            missing = [mid for mid in media_ids if mid not in found_ids]
            if missing:
                raise MediaNotFoundError(f"Media asset(s) not found: {', '.join(missing)}")

            job_id = str(uuid.uuid4())
            job = ProcessingJob(
                id=job_id,
                job_type=job_type,
                status="QUEUED",
                progress=0.0,
                model_id=model_id,
                processing_mode=processing_mode,
                config_json=json.dumps(config),
                input_media_ids_json=json.dumps(media_ids),
                faces_detected=0,
            )
            session.add(job)

        self._publish(job_id, "QUEUED", 0.0, 0, "Job queued")
        self._submit_job(job_id)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> ProcessingJob:
        with self.db.session_scope() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            session.expunge(job)
            return job

    def list_jobs(self, *, status: str | None = None, limit: int = 50) -> list[ProcessingJob]:
        with self.db.session_scope() as session:
            query = session.query(ProcessingJob)
            if status:
                query = query.filter(ProcessingJob.status == status)
            jobs = query.order_by(ProcessingJob.created_at.desc()).limit(limit).all()
            for j in jobs:
                session.expunge(j)
            return jobs

    def cancel_job(self, job_id: str) -> ProcessingJob:
        with self.db.session_scope() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            if job.status in TERMINAL_STATES:
                raise JobStateError(f"Cannot cancel job in '{job.status}' state.")

            with self._lock:
                if job_id in self._cancel_events:
                    self._cancel_events[job_id].set()

            if job.status == "QUEUED":
                job.status = "CANCELLED"
                job.completed_at = _utcnow()
                self._publish(job_id, "CANCELLED", job.progress, job.faces_detected, "Cancelled before start")

        return self.get_job(job_id)

    def retry_job(self, job_id: str) -> ProcessingJob:
        with self.db.session_scope() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise JobNotFoundError(f"Job '{job_id}' not found.")
            if job.status not in {"FAILED", "CANCELLED"}:
                raise JobStateError(f"Only FAILED or CANCELLED jobs can be retried (current: {job.status}).")

            job.status = "QUEUED"
            job.progress = 0.0
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            job.output_manifest_json = None
            job.faces_detected = 0

        self._publish(job_id, "QUEUED", 0.0, 0, "Job re-queued")
        self._submit_job(job_id)
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job.status in ACTIVE_STATES:
            raise JobStateError("Cancel active job before deleting.")
        self.storage.delete(f"outputs/{job_id}")
        with self.db.session_scope() as session:
            db_job = session.get(ProcessingJob, job_id)
            if db_job is not None:
                session.delete(db_job)

    def _submit_job(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events[job_id] = threading.Event()
            self._pause_events[job_id] = threading.Event()
        self._executor.submit(self._run_job, job_id)

    def _update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        faces_detected: int | None = None,
        error_message: str | None = None,
        output_manifest: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        message: str | None = None,
    ) -> None:
        with self.db.session_scope() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = round(min(1.0, max(0.0, progress)), 4)
            if faces_detected is not None:
                job.faces_detected = faces_detected
            if error_message is not None:
                job.error_message = error_message
            if output_manifest is not None:
                job.output_manifest_json = json.dumps(output_manifest)
            if started_at is not None:
                job.started_at = started_at
            if completed_at is not None:
                job.completed_at = completed_at

            curr_status = job.status
            curr_progress = job.progress
            curr_faces = job.faces_detected

        self._publish(job_id, curr_status, curr_progress, curr_faces, message)

    def _publish(
        self,
        job_id: str,
        status: str,
        progress: float,
        faces_detected: int,
        message: str | None = None,
    ) -> None:
        self.event_bus.publish(
            JobEvent(
                job_id=job_id,
                status=status,
                progress=progress,
                faces_detected=faces_detected,
                message=message,
            )
        )

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            cancel_event = self._cancel_events.get(job_id, threading.Event())
            pause_event = self._pause_events.get(job_id, threading.Event())

        detector = None
        try:
            if cancel_event.is_set():
                raise JobCancelledError("Job cancelled before execution.")

            job = self.get_job(job_id)
            if job.status == "CANCELLED":
                return

            self._update_job(
                job_id,
                status="VALIDATING",
                progress=0.05,
                started_at=_utcnow(),
                message="Validating inputs and initializing model",
            )

            cfg_dict = json.loads(job.config_json)
            media_ids: list[str] = json.loads(job.input_media_ids_json)

            with self.db.session_scope() as session:
                assets = session.query(MediaAsset).filter(MediaAsset.id.in_(media_ids)).all()
                asset_map = {a.id: a for a in assets}
                for a in assets:
                    session.expunge(a)

            detector_cfg = DetectorConfig(
                model_id=job.model_id,
                confidence_threshold=float(cfg_dict.get("confidence_threshold", 0.6)),
                min_face_size=(
                    int(cfg_dict.get("min_face_size", 30)),
                    int(cfg_dict.get("min_face_size", 30)),
                ),
                scale_factor=float(cfg_dict.get("scale_factor", 1.1)),
                min_neighbors=int(cfg_dict.get("min_neighbors", 5)),
                use_gpu=bool(cfg_dict.get("use_gpu", False)),
                models_dir=self.settings.resolved_models_dir,
            )
            detector = self.registry.create_detector(detector_cfg)

            self._update_job(job_id, status="PROCESSING", progress=0.1, message="Processing media")
            job_output_dir = self.settings.output_dir / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            manifest: dict[str, Any] = {}
            total_faces = 0

            if job.job_type == "image":
                asset = asset_map[media_ids[0]]
                input_path = self.storage.resolve_local_path(asset.storage_key)
                img_cfg = ImageProcessingConfig(
                    blur_mode=cfg_dict.get("blur_mode", "none"),
                    selected_face_indices=cfg_dict.get("selected_face_indices"),
                    save_crops=bool(cfg_dict.get("save_crops", True)),
                    save_privacy_image=bool(cfg_dict.get("save_privacy_image", True)),
                    show_hud=bool(cfg_dict.get("show_hud", True)),
                    draw_boxes=bool(cfg_dict.get("draw_boxes", True)),
                    max_dimension=self.settings.max_image_dimension,
                )
                img_res = self._image_service.process(
                    input_path,
                    detector,
                    job_output_dir,
                    config=img_cfg,
                    job_id=job_id,
                    input_name=asset.original_filename,
                )
                total_faces = len(img_res.detections)
                manifest = self._build_image_manifest(job_id, job_output_dir, img_res)

            elif job.job_type == "video":
                asset = asset_map[media_ids[0]]
                input_path = self.storage.resolve_local_path(asset.storage_key)
                vid_cfg = VideoProcessingConfig(
                    blur_mode=cfg_dict.get("blur_mode", "none"),
                    enable_tracking=bool(cfg_dict.get("enable_tracking", True)),
                    target_fps=cfg_dict.get("target_fps"),
                    show_hud=bool(cfg_dict.get("show_hud", True)),
                    draw_boxes=bool(cfg_dict.get("draw_boxes", True)),
                    save_privacy_video=bool(cfg_dict.get("save_privacy_video", False)),
                    max_frames=cfg_dict.get("max_frames"),
                    max_duration_seconds=self.settings.max_video_duration_seconds,
                )

                def _on_video_progress(done: int, total: int, faces: int) -> None:
                    frac = 0.1 + 0.8 * (done / max(1, total))
                    self._update_job(
                        job_id,
                        progress=frac,
                        faces_detected=faces,
                        message=f"Frame {done}/{total}",
                    )

                vid_res = self._video_service.process(
                    input_path,
                    detector,
                    job_output_dir,
                    config=vid_cfg,
                    job_id=job_id,
                    input_name=asset.original_filename,
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                    progress_callback=_on_video_progress,
                )
                total_faces = vid_res.total_detections
                manifest = self._build_video_manifest(job_id, job_output_dir, vid_res)

            elif job.job_type == "batch":
                inputs: list[tuple[str, str, Path]] = []
                for mid in media_ids:
                    asset = asset_map[mid]
                    inputs.append(
                        (mid, asset.original_filename, self.storage.resolve_local_path(asset.storage_key))
                    )

                batch_cfg = ImageProcessingConfig(
                    blur_mode=cfg_dict.get("blur_mode", "none"),
                    save_crops=bool(cfg_dict.get("save_crops", False)),
                    save_privacy_image=bool(cfg_dict.get("save_privacy_image", True)),
                    show_hud=bool(cfg_dict.get("show_hud", True)),
                    draw_boxes=bool(cfg_dict.get("draw_boxes", True)),
                    max_dimension=self.settings.max_image_dimension,
                )

                def _on_batch_progress(done: int, total: int, faces: int) -> None:
                    frac = 0.1 + 0.8 * (done / max(1, total))
                    self._update_job(
                        job_id,
                        progress=frac,
                        faces_detected=faces,
                        message=f"Item {done}/{total}",
                    )

                batch_res = self._batch_service.process(
                    inputs,
                    detector,
                    job_output_dir,
                    config=batch_cfg,
                    job_id=job_id,
                    cancel_event=cancel_event,
                    progress_callback=_on_batch_progress,
                )
                total_faces = batch_res.total_faces
                manifest = {
                    "type": "batch",
                    "total_items": len(batch_res.items),
                    "succeeded": batch_res.succeeded,
                    "failed": batch_res.failed,
                    "total_faces": batch_res.total_faces,
                    "items": [
                        {
                            "media_id": it.media_id,
                            "filename": it.filename,
                            "status": it.status,
                            "face_count": it.face_count,
                            "error": it.error,
                        }
                        for it in batch_res.items
                    ],
                    "files": {
                        "zip": "batch_outputs.zip",
                        "report_json": "batch_report.json",
                        "report_csv": "batch_summary.csv",
                    },
                }

            self._update_job(job_id, status="FINALIZING", progress=0.95, message="Finalizing outputs")
            self._update_job(
                job_id,
                status="COMPLETED",
                progress=1.0,
                faces_detected=total_faces,
                output_manifest=manifest,
                completed_at=_utcnow(),
                message="Completed",
            )
        except JobCancelledError as exc:
            logger.info("Job %s cancelled: %s", job_id, exc)
            self._update_job(
                job_id,
                status="CANCELLED",
                error_message=str(exc),
                completed_at=_utcnow(),
                message="Cancelled",
            )
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            self._update_job(
                job_id,
                status="FAILED",
                error_message=str(exc),
                completed_at=_utcnow(),
                message=f"Failed: {exc}",
            )
        finally:
            if detector is not None:
                detector.release()
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._pause_events.pop(job_id, None)

    @staticmethod
    def _build_image_manifest(job_id: str, output_dir: Path, res: Any) -> dict[str, Any]:
        files: dict[str, Any] = {
            "annotated": "annotated.png",
            "report_json": "report.json",
            "report_csv": "summary.csv",
        }
        if res.blurred_path and res.blurred_path.is_file():
            files["privacy_blurred"] = res.blurred_path.name
        if res.crop_paths:
            files["crops"] = [str(p.relative_to(output_dir)) for p in res.crop_paths]
        return {
            "type": "image",
            "job_id": job_id,
            "width": res.width,
            "height": res.height,
            "face_count": len(res.detections),
            "duration_ms": round(res.duration_ms, 2),
            "detections": [d.to_dict() for d in res.detections],
            "files": files,
        }

    @staticmethod
    def _build_video_manifest(job_id: str, output_dir: Path, res: Any) -> dict[str, Any]:
        files: dict[str, Any] = {
            "processed_video": "processed.mp4",
            "report_json": "report.json",
            "report_csv": "summary.csv",
        }
        if res.privacy_video_path and res.privacy_video_path.is_file():
            files["privacy_video"] = res.privacy_video_path.name
        return {
            "type": "video",
            "job_id": job_id,
            "width": res.metadata.width,
            "height": res.metadata.height,
            "fps": res.metadata.fps,
            "frames_processed": res.frames_processed,
            "total_detections": res.total_detections,
            "unique_track_count": res.unique_track_count,
            "duration_ms": round(res.duration_ms, 2),
            "files": files,
        }
