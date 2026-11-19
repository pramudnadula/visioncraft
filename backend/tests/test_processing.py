"""Tests for image, video, tracking, and batch processing services."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.core.exceptions import JobCancelledError, MediaValidationError
from backend.app.detection.base import Detection, DetectorConfig, DetectorMetadata, FaceDetector
from backend.app.processing.batch import BatchProcessingService
from backend.app.processing.image import ImageProcessingConfig, ImageProcessingService
from backend.app.processing.tracking import FaceTracker
from backend.app.processing.video import VideoProcessingConfig, VideoProcessingService


class _SyntheticDetector(FaceDetector):
    """Deterministic in-memory detector for processing pipeline tests."""

    def initialize(self) -> None:
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection(x=w // 4, y=h // 4, width=w // 2, height=h // 2, confidence=0.92)]

    def get_metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            id="synthetic",
            name="Synthetic Test Detector",
            version="1.0",
            description="Deterministic test detector",
            supports_confidence=True,
            supports_landmarks=False,
            supports_gpu=False,
            gpu_active=False,
            is_optional=False,
            available=True,
            license="MIT",
        )

    def is_available(self) -> bool:
        return True

    def release(self) -> None:
        self._initialized = False


def _create_image(path: Path) -> Path:
    rng = np.random.default_rng(7)
    img = rng.integers(20, 230, size=(160, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _create_video(path: Path, frames: int = 10) -> Path:
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (160, 120))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((120, 160, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_image_processing_service_outputs_and_reports(tmp_path: Path) -> None:
    detector = _SyntheticDetector(DetectorConfig(model_id="synthetic"))
    detector.initialize()
    service = ImageProcessingService()
    src_img = _create_image(tmp_path / "input.png")

    res = service.process(
        src_img,
        detector,
        tmp_path / "out",
        config=ImageProcessingConfig(blur_mode="gaussian", save_crops=True, save_privacy_image=True),
        job_id="job-1",
        input_name="input.png",
    )
    assert len(res.detections) == 1
    assert res.annotated_path and res.annotated_path.is_file()
    assert res.blurred_path and res.blurred_path.is_file()
    assert res.report_json_path and res.report_json_path.is_file()
    assert res.report_csv_path and res.report_csv_path.is_file()
    assert len(res.crop_paths) == 1

    report = json.loads(res.report_json_path.read_text(encoding="utf-8"))
    assert report["job_id"] == "job-1"
    assert report["detection_count"] == 1


def test_face_tracker_assigns_consistent_temporary_ids() -> None:
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=5)
    f1 = tracker.update([Detection(x=10, y=10, width=50, height=50)])
    f2 = tracker.update([Detection(x=12, y=12, width=50, height=50)])
    assert f1[0].tracking_id == 1
    assert f2[0].tracking_id == 1


def test_video_processing_and_cancellation(tmp_path: Path) -> None:
    detector = _SyntheticDetector(DetectorConfig(model_id="synthetic"))
    detector.initialize()
    service = VideoProcessingService()
    src_video = _create_video(tmp_path / "input.mp4", frames=12)

    res = service.process(
        src_video,
        detector,
        tmp_path / "vid_out",
        config=VideoProcessingConfig(blur_mode="pixelate", max_frames=8, save_privacy_video=True),
        job_id="vid-1",
    )
    assert res.frames_processed == 8
    assert res.output_video_path.is_file()
    assert res.privacy_video_path and res.privacy_video_path.is_file()
    assert res.report_json_path and res.report_json_path.is_file()

    cancel_ev = threading.Event()
    cancel_ev.set()
    with pytest.raises(JobCancelledError):
        service.process(
            src_video,
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
    detector = _SyntheticDetector(DetectorConfig(model_id="synthetic"))
    detector.initialize()
    service = BatchProcessingService()
    src_img = _create_image(tmp_path / "valid.png")

    missing = tmp_path / "missing.png"
    res = service.process(
        [
            ("id1", "valid.png", src_img),
            ("id2", "missing.png", missing),
        ],
        detector,
        tmp_path / "batch_out",
        job_id="batch-1",
    )
    assert res.succeeded == 1
    assert res.failed == 1
    assert res.zip_path and res.zip_path.is_file()
