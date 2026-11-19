"""Tests for image, video, tracking, and batch processing services."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.app.core.exceptions import JobCancelledError, MediaValidationError
from backend.app.detection.backends.haar import HaarCascadeDetector
from backend.app.detection.base import Detection, DetectorConfig
from backend.app.processing.batch import BatchProcessingService
from backend.app.processing.image import ImageProcessingConfig, ImageProcessingService
from backend.app.processing.tracking import FaceTracker
from backend.app.processing.video import VideoProcessingConfig, VideoProcessingService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_image_processing_service_outputs_and_reports(tmp_path: Path) -> None:
    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    detector.initialize()
    service = ImageProcessingService()

    res = service.process(
        PROJECT_ROOT / "RDJ.png",
        detector,
        tmp_path / "out",
        config=ImageProcessingConfig(blur_mode="gaussian", save_crops=True, save_privacy_image=True),
        job_id="job-1",
        input_name="RDJ.png",
    )
    assert len(res.detections) >= 1
    assert res.annotated_path and res.annotated_path.is_file()
    assert res.blurred_path and res.blurred_path.is_file()
    assert res.report_json_path and res.report_json_path.is_file()
    assert res.report_csv_path and res.report_csv_path.is_file()
    assert len(res.crop_paths) == len(res.detections)

    report = json.loads(res.report_json_path.read_text(encoding="utf-8"))
    assert report["job_id"] == "job-1"
    assert report["detection_count"] == len(res.detections)


def test_face_tracker_assigns_consistent_temporary_ids() -> None:
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=5)
    f1 = tracker.update([Detection(x=10, y=10, width=50, height=50)])
    f2 = tracker.update([Detection(x=12, y=12, width=50, height=50)])
    assert f1[0].tracking_id == 1
    assert f2[0].tracking_id == 1


def test_video_processing_and_cancellation(tmp_path: Path) -> None:
    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    detector.initialize()
    service = VideoProcessingService()

    res = service.process(
        PROJECT_ROOT / "clip.mp4",
        detector,
        tmp_path / "vid_out",
        config=VideoProcessingConfig(blur_mode="pixelate", max_frames=8, save_privacy_video=True),
        job_id="vid-1",
    )
    assert res.frames_processed == 8
    assert res.output_video_path.is_file()
    assert res.privacy_video_path and res.privacy_video_path.is_file()
    assert res.report_json_path and res.report_json_path.is_file()

    # Verify cancellation raises JobCancelledError
    cancel_ev = threading.Event()
    cancel_ev.set()
    with pytest.raises(JobCancelledError):
        service.process(
            PROJECT_ROOT / "clip.mp4",
            detector,
            tmp_path / "vid_cancel",
            cancel_event=cancel_ev,
        )


def test_corrupt_video_raises_validation_error(tmp_path: Path) -> None:
    bad_video = tmp_path / "corrupt.mp4"
    bad_video.write_bytes(b"not a real video")
    with pytest.raises(MediaValidationError):
        VideoProcessingService.extract_metadata(bad_video)


def test_batch_processing_isolates_failures_and_creates_zip(tmp_path: Path) -> None:
    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    detector.initialize()
    service = BatchProcessingService()

    missing = tmp_path / "missing.png"
    res = service.process(
        [
            ("id1", "RDJ.png", PROJECT_ROOT / "RDJ.png"),
            ("id2", "missing.png", missing),
        ],
        detector,
        tmp_path / "batch_out",
        job_id="batch-1",
    )
    assert res.succeeded == 1
    assert res.failed == 1
    assert res.zip_path and res.zip_path.is_file()
