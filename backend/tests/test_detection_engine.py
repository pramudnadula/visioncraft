"""Tests for the shared FaceVision detection engine and privacy blur."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.detection.backends.haar import HaarCascadeDetector
from backend.app.detection.base import Detection, DetectorConfig
from backend.app.detection.registry import create_default_registry
from backend.app.privacy.blur import apply_privacy_blur

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_haar_detector_lifecycle_and_detection() -> None:
    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    assert detector.is_available() is True
    meta = detector.get_metadata()
    assert meta.id == "haar"
    assert meta.supports_gpu is False
    assert meta.gpu_active is False

    detector.initialize()
    img = cv2.imread(str(PROJECT_ROOT / "RDJ.png"))
    assert img is not None
    detections = detector.detect(img)
    assert len(detections) >= 1
    assert all(d.width > 0 and d.height > 0 for d in detections)
    detector.release()


def test_registry_lists_models_and_rejects_missing_weights(tmp_path: Path) -> None:
    registry = create_default_registry(models_dir=tmp_path)
    models = registry.list_models()
    ids = {m.id for m in models}
    assert {"haar", "yunet", "opencv_dnn", "scrfd", "retinaface"}.issubset(ids)

    with pytest.raises(ModelNotAvailableError):
        registry.create_detector(DetectorConfig(model_id="yunet", models_dir=tmp_path))

    with pytest.raises(ModelNotAvailableError):
        registry.create_detector(DetectorConfig(model_id="nonexistent", models_dir=tmp_path))


def test_selective_privacy_blur() -> None:
    frame = np.full((120, 200, 3), 180, dtype=np.uint8)
    # Add high-contrast pattern in two face regions
    frame[10:50, 10:50] = 20
    frame[10:30, 10:30] = 240
    frame[10:50, 110:150] = 20
    frame[10:30, 110:130] = 240

    dets = [
        Detection(x=10, y=10, width=40, height=40),
        Detection(x=110, y=10, width=40, height=40),
    ]
    blurred = apply_privacy_blur(frame, dets, "pixelate", selected_indices=[0])
    # Face 0 is modified, Face 1 remains untouched
    assert not np.array_equal(blurred[10:50, 10:50], frame[10:50, 10:50])
    assert np.array_equal(blurred[10:50, 110:150], frame[10:50, 110:150])
