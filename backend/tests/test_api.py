"""FastAPI REST API integration tests."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_client(tmp_path: Path, *, api_token: str | None = None) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=PROJECT_ROOT / "models",
        api_token=api_token,
    )
    app = create_app(custom_settings=settings)
    return TestClient(app)


def test_health_capabilities_models_and_stats(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        caps = client.get("/api/v1/capabilities")
        assert caps.status_code == 200
        assert caps.json()["local_processing"] is True

        models = client.get("/api/v1/models")
        assert models.status_code == 200
        haar = next(m for m in models.json() if m["id"] == "haar")
        assert haar["available"] is True

        stats = client.get("/api/v1/stats")
        assert stats.status_code == 200
        assert stats.json()["total_jobs"] == 0


def test_image_upload_job_execution_and_artifacts(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        img_bytes = (PROJECT_ROOT / "RDJ.png").read_bytes()
        up = client.post(
            "/api/v1/media/upload",
            files={"file": ("RDJ.png", img_bytes, "image/png")},
        )
        assert up.status_code == 201
        media_id = up.json()["id"]

        job_resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "image",
                "media_ids": [media_id],
                "model_id": "haar",
                "blur_mode": "gaussian",
            },
        )
        assert job_resp.status_code == 201
        job_id = job_resp.json()["id"]

        # Poll until completed
        status = ""
        for _ in range(40):
            status_resp = client.get(f"/api/v1/jobs/{job_id}")
            assert status_resp.status_code == 200
            status = status_resp.json()["status"]
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.1)

        assert status == "COMPLETED"
        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 200
        assert result_resp.json()["face_count"] >= 1

        art = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.png")
        assert art.status_code == 200
        assert len(art.content) > 100

        # Cleanup
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
        assert client.delete(f"/api/v1/media/{media_id}").status_code == 204


def test_cloud_consent_and_spoofed_upload_and_auth(tmp_path: Path) -> None:
    with _make_client(tmp_path, api_token="secret-token") as client:
        # Missing token rejected
        assert client.get("/api/v1/models").status_code == 401

        headers = {"X-FaceVision-Token": "secret-token"}
        assert client.get("/api/v1/models", headers=headers).status_code == 200

        # Spoofed file rejected
        bad_up = client.post(
            "/api/v1/media/upload",
            headers=headers,
            files={"file": ("evil.png", b"not-a-png", "image/png")},
        )
        assert bad_up.status_code == 400

        # Valid upload, but cloud job without consent rejected
        img_bytes = (PROJECT_ROOT / "RDJ.png").read_bytes()
        up = client.post(
            "/api/v1/media/upload",
            headers=headers,
            files={"file": ("RDJ.png", img_bytes, "image/png")},
        )
        assert up.status_code == 201
        media_id = up.json()["id"]

        cloud_job = client.post(
            "/api/v1/jobs",
            headers=headers,
            json={
                "job_type": "image",
                "media_ids": [media_id],
                "model_id": "haar",
                "processing_mode": "cloud",
                "cloud_consent": False,
            },
        )
        assert cloud_job.status_code == 403
        assert cloud_job.json()["error"] == "cloud_consent_required"


def test_settings_persistence(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        r1 = client.get("/api/v1/settings")
        assert r1.status_code == 200
        assert r1.json()["default_blur_mode"] == "none"

        r2 = client.put(
            "/api/v1/settings",
            json={"default_blur_mode": "pixelate", "retention_days": 14},
        )
        assert r2.status_code == 200
        assert r2.json()["default_blur_mode"] == "pixelate"
        assert r2.json()["retention_days"] == 14
