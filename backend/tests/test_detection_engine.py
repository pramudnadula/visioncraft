"""Tests for the shared FaceVision detection engine and privacy blur."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.detection.backends.haar import HaarCascadeDetector
from backend.app.detection.base import Detection, DetectorConfig
from backend.app.detection.registry import create_default_registry
from backend.app.privacy.blur import apply_privacy_blur


def test_haar_detector_lifecycle_and_detection() -> None:
    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    assert detector.is_available() is True
    meta = detector.get_metadata()
    assert meta.id == "haar"
    assert meta.supports_gpu is False
    assert meta.gpu_active is False

    detector.initialize()
    assert detector._classifier is not None

    img = np.zeros((160, 160, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert isinstance(detections, list)
    detector.release()
    assert detector._classifier is None


def test_haar_detector_rejects_document_like_image_without_faces() -> None:
    import cv2

    detector = HaarCascadeDetector(DetectorConfig(model_id="haar", confidence_threshold=0.6))
    detector.initialize()
    # Create an invoice-like white page with dark borders, table lines, and text blocks
    doc = np.full((600, 450, 3), 255, dtype=np.uint8)
    cv2.rectangle(doc, (30, 30), (420, 570), (20, 20, 20), 2)
    for y in range(80, 520, 35):
        cv2.line(doc, (45, y), (405, y), (60, 60, 60), 1)
        cv2.putText(
            doc, "INVOICE ITEM #12345  $99.00", (55, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1
        )

    detections = detector.detect(doc)
    assert detections == []
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
    frame[10:50, 10:50] = 20
    frame[10:30, 10:30] = 240
    frame[10:50, 110:150] = 20
    frame[10:30, 110:130] = 240

    dets = [
        Detection(x=10, y=10, width=40, height=40),
        Detection(x=110, y=10, width=40, height=40),
    ]
    for mode in ("pixelate", "mask", "mesh", "geometric"):
        blurred = apply_privacy_blur(frame, dets, mode, selected_indices=[0])
        assert not np.array_equal(blurred[10:50, 10:50], frame[10:50, 10:50])
        assert np.array_equal(blurred[10:50, 110:150], frame[10:50, 110:150])


def test_every_privacy_mode_renders_a_distinct_result() -> None:
    from backend.app.detection.base import LandmarkPoint
    from backend.app.privacy.blur import PRIVACY_MODES

    rng = np.random.default_rng(3)
    frame = rng.integers(40, 220, size=(360, 360, 3), dtype=np.uint8)
    det = Detection(
        x=100,
        y=70,
        width=160,
        height=210,
        confidence=0.9,
        landmarks=[
            LandmarkPoint(150, 140),
            LandmarkPoint(210, 140),
            LandmarkPoint(180, 185),
            LandmarkPoint(155, 230),
            LandmarkPoint(205, 230),
        ],
    )
    renders = {mode: apply_privacy_blur(frame, [det], mode) for mode in PRIVACY_MODES}
    assert np.array_equal(renders["none"], frame)
    modes = list(renders)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            assert not np.array_equal(renders[a], renders[b]), f"{a} and {b} render identically"


def test_unknown_privacy_mode_is_rejected() -> None:
    import pytest

    from backend.app.privacy.blur import validate_blur_mode

    with pytest.raises(ValueError):
        validate_blur_mode("sparkles")
