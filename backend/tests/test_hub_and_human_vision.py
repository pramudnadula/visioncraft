"""Tests for Phase D (Face Mask), Phase E (Sign Language), and Phase F (Model Hub)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.main import create_app
from backend.app.vision.faces.mask_detection import FaceMaskClassifier
from backend.app.vision.sign_language.recognition import SignLanguageSequenceBuffer, list_sign_languages


def test_face_mask_classifier_requires_model_and_reports_unavailable(tmp_path: Path) -> None:
    clf = FaceMaskClassifier(models_dir=tmp_path)
    assert clf.is_available() is False
    try:
        clf.initialize()
        raise AssertionError("Expected ModelNotAvailableError")
    except ModelNotAvailableError:
        pass


def test_sign_language_sequence_buffer_and_honest_language_status() -> None:
    buf = SignLanguageSequenceBuffer(window_size=3)
    assert buf.is_ready() is False
    buf.push([0.1, 0.2])
    buf.push([0.2, 0.3])
    buf.push([0.3, 0.4])
    assert buf.is_ready() is True
    assert len(buf.to_sequence()) == 3

    langs = list_sign_languages()
    assert any(lang["code"] == "FinSL" and lang["model_available"] is False for lang in langs)
    assert any(lang["code"] == "BSL" and lang["model_available"] is False for lang in langs)


def test_model_hub_compatibility_benchmark_and_custom_delete(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    with TestClient(create_app(custom_settings=settings)) as client:
        comp = client.get("/api/v1/models/compatibility")
        assert comp.status_code == 200
        assert "python_version" in comp.json()
        assert "opencv_version" in comp.json()

        bench = client.post(
            "/api/v1/models/benchmark",
            json={"model_id": "haar", "iterations": 3},
        )
        assert bench.status_code == 200
        assert bench.json()["avg_inference_ms"] >= 0
        assert bench.json()["estimated_fps"] > 0

        imp = client.post(
            "/api/v1/models/import",
            files={"file": ("temp_model.onnx", b"onnx-bytes", "application/octet-stream")},
        )
        assert imp.status_code == 201
        assert client.delete("/api/v1/models/custom/temp_model.onnx").status_code == 204
