"""Unit and integration tests for face_detector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from face_detector import (
    Detection,
    DetectorConfig,
    FacePipeline,
    HaarCascadeBackend,
    draw_detections,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_haar_backend_detects_faces_in_sample_image() -> None:
    backend = HaarCascadeBackend()
    img = cv2.imread(str(PROJECT_ROOT / "RDJ.png"))
    assert img is not None
    detections = backend.detect(img)
    assert len(detections) >= 1
    assert all(isinstance(d, Detection) and d.width > 0 and d.height > 0 for d in detections)


def test_process_image_writes_output_and_crops(tmp_path: Path) -> None:
    pipeline = FacePipeline()
    out_img = tmp_path / "annotated.png"
    crops_dir = tmp_path / "crops"

    result = pipeline.process_image(
        PROJECT_ROOT / "me.png",
        output_path=out_img,
        crops_dir=crops_dir,
        show=False,
    )

    assert len(result.detections) >= 1
    assert out_img.is_file()
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == len(result.detections)


def test_privacy_blur_modes_modify_face_region() -> None:
    img = cv2.imread(str(PROJECT_ROOT / "RDJ.png"))
    assert img is not None
    det = Detection(x=20, y=20, width=80, height=80)

    for mode in ("gaussian", "pixelate"):
        config = DetectorConfig(blur_mode=mode, show_hud=False)
        rendered = draw_detections(img, [det], config)
        assert rendered.shape == img.shape
        assert not (rendered[20:100, 20:100] == img[20:100, 20:100]).all()


def test_process_video_with_max_frames(tmp_path: Path) -> None:
    pipeline = FacePipeline()
    out_video = tmp_path / "out.mp4"
    summary = pipeline.process_video(
        PROJECT_ROOT / "clip.mp4",
        output_path=out_video,
        show=False,
        max_frames=10,
    )
    assert summary["frames_processed"] == 10
    assert out_video.is_file()


def test_cli_json_export(tmp_path: Path) -> None:
    from face_detector.cli import main

    report = tmp_path / "report.json"
    exit_code = main(
        [
            "--image",
            str(PROJECT_ROOT / "rdi.jpg"),
            "--no-show",
            "--export-json",
            str(report),
        ]
    )
    assert exit_code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["face_count"] >= 1
    assert len(data["detections"]) == data["face_count"]


def test_missing_image_raises_file_not_found() -> None:
    pipeline = FacePipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_image(PROJECT_ROOT / "does_not_exist.png", show=False)
