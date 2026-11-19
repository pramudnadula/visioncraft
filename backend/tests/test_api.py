"""FastAPI REST API integration tests using synthetic media."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
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


def _synthetic_png_bytes() -> bytes:
    img = np.full((160, 160, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


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
        img_bytes = _synthetic_png_bytes()
        up = client.post(
            "/api/v1/media/upload",
            files={"file": ("sample.png", img_bytes, "image/png")},
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
        assert "face_count" in result_resp.json()

        art = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.png")
        assert art.status_code == 200
        assert len(art.content) > 100

        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
        assert client.delete(f"/api/v1/media/{media_id}").status_code == 204


def test_cloud_consent_and_spoofed_upload_and_auth(tmp_path: Path) -> None:
    with _make_client(tmp_path, api_token="secret-token") as client:
        assert client.get("/api/v1/models").status_code == 401

        headers = {"X-FaceVision-Token": "secret-token"}
        assert client.get("/api/v1/models", headers=headers).status_code == 200

        bad_up = client.post(
            "/api/v1/media/upload",
            headers=headers,
            files={"file": ("evil.png", b"not-a-png", "image/png")},
        )
        assert bad_up.status_code == 400

        img_bytes = _synthetic_png_bytes()
        up = client.post(
            "/api/v1/media/upload",
            headers=headers,
            files={"file": ("sample.png", img_bytes, "image/png")},
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


def _synthetic_mp4_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (160, 120))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((120, 160, 3), (i * 30) % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_video_and_batch_jobs_and_websocket_webcam(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        # 1. Video job
        vid_bytes = _synthetic_mp4_bytes(tmp_path)
        v_up = client.post(
            "/api/v1/media/upload",
            files={"file": ("clip.mp4", vid_bytes, "video/mp4")},
        )
        assert v_up.status_code == 201
        vid_id = v_up.json()["id"]

        v_job = client.post(
            "/api/v1/jobs",
            json={"job_type": "video", "media_ids": [vid_id], "model_id": "haar"},
        )
        assert v_job.status_code == 201
        v_job_id = v_job.json()["id"]

        status = ""
        for _ in range(40):
            status = client.get(f"/api/v1/jobs/{v_job_id}").json()["status"]
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.1)
        assert status == "COMPLETED"
        assert client.get(f"/api/v1/jobs/{v_job_id}/artifacts/processed.mp4").status_code == 200

        # 2. Batch job + ZIP download
        img_bytes = _synthetic_png_bytes()
        i_up = client.post(
            "/api/v1/media/upload",
            files={"file": ("img1.png", img_bytes, "image/png")},
        )
        assert i_up.status_code == 201
        b_job = client.post(
            "/api/v1/jobs",
            json={"job_type": "batch", "media_ids": [i_up.json()["id"]], "model_id": "haar"},
        )
        assert b_job.status_code == 201
        b_job_id = b_job.json()["id"]

        for _ in range(40):
            status = client.get(f"/api/v1/jobs/{b_job_id}").json()["status"]
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.1)
        assert status == "COMPLETED"
        assert client.get(f"/api/v1/jobs/{b_job_id}/artifacts/batch_outputs.zip").status_code == 200

        # 3. WebSocket webcam frame processing
        import base64

        b64_img = base64.b64encode(img_bytes).decode("ascii")
        with client.websocket_connect("/ws/v1/webcam") as ws:
            ws.send_json({"frame": f"data:image/png;base64,{b64_img}", "model_id": "haar"})
            msg = ws.receive_json()
            assert "face_count" in msg
            assert "latency_ms" in msg
            assert msg["annotated_frame"].startswith("data:image/jpeg;base64,")


def test_custom_model_import_and_listing(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    app = create_app(custom_settings=settings)
    with TestClient(app) as client:
        imp = client.post(
            "/api/v1/models/import",
            files={"file": ("custom_face.onnx", b"fake-onnx-weights", "application/octet-stream")},
        )
        assert imp.status_code == 201
        assert imp.json()["filename"] == "custom_face.onnx"

        listed = client.get("/api/v1/models/custom")
        assert listed.status_code == 200
        assert any(m["filename"] == "custom_face.onnx" for m in listed.json())

        bad = client.post(
            "/api/v1/models/import",
            files={"file": ("bad.exe", b"nope", "application/octet-stream")},
        )
        assert bad.status_code == 400
