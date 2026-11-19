"""Unit and integration tests for face_detector using synthetic media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_detector import (
    Detection,
    DetectorConfig,
    FacePipeline,
    HaarCascadeBackend,
    draw_detections,
)


class _StubBackend:
    """Deterministic detector backend for pipeline unit tests."""

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection(x=w // 4, y=h // 4, width=w // 2, height=h // 2, confidence=0.95)]


def _write_synthetic_image(path: Path, width: int = 240, height: int = 240) -> Path:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 60, (120, 140, 160), -1)
    cv2.imwrite(str(path), img)
    return path


def _write_synthetic_video(path: Path, frames: int = 12, width: int = 160, height: int = 120) -> Path:
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_haar_backend_initializes_and_runs_on_synthetic_frame() -> None:
    backend = HaarCascadeBackend()
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    detections = backend.detect(img)
    assert isinstance(detections, list)


def test_process_image_writes_output_and_crops(tmp_path: Path) -> None:
    src_img = _write_synthetic_image(tmp_path / "sample.png")
    pipeline = FacePipeline(backend=_StubBackend())
    out_img = tmp_path / "annotated.png"
    crops_dir = tmp_path / "crops"

    result = pipeline.process_image(
        src_img,
        output_path=out_img,
        crops_dir=crops_dir,
        show=False,
    )

    assert len(result.detections) == 1
    assert out_img.is_file()
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == 1


def test_privacy_blur_modes_modify_face_region() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)
    det = Detection(x=20, y=20, width=80, height=80)

    for mode in ("gaussian", "pixelate"):
        config = DetectorConfig(blur_mode=mode, show_hud=False)
        rendered = draw_detections(img, [det], config)
        assert rendered.shape == img.shape
        assert not (rendered[20:100, 20:100] == img[20:100, 20:100]).all()


def test_process_video_with_max_frames(tmp_path: Path) -> None:
    src_video = _write_synthetic_video(tmp_path / "sample.mp4", frames=15)
    pipeline = FacePipeline(backend=_StubBackend())
    out_video = tmp_path / "out.mp4"
    summary = pipeline.process_video(
        src_video,
        output_path=out_video,
        show=False,
        max_frames=10,
    )
    assert summary["frames_processed"] == 10
    assert out_video.is_file()


def test_cli_json_export(tmp_path: Path) -> None:
    from face_detector.cli import main

    src_img = _write_synthetic_image(tmp_path / "sample.jpg")
    report = tmp_path / "report.json"
    exit_code = main(
        [
            "--image",
            str(src_img),
            "--no-show",
            "--export-json",
            str(report),
        ]
    )
    assert exit_code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "face_count" in data
    assert len(data["detections"]) == data["face_count"]


def test_missing_image_raises_file_not_found(tmp_path: Path) -> None:
    pipeline = FacePipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_image(tmp_path / "does_not_exist.png", show=False)
