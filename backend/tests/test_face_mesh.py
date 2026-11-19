"""Tests for the MediaPipe-based face mesh feature."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.main import create_app
from backend.app.reports.generator import generate_report_file
from backend.app.vision.faces.face_mesh import (
    LANDMARK_COUNT,
    FaceMeshFace,
    FaceMeshService,
    FaceMeshSettings,
    _head_pose,
    mesh_status,
    render_face_mesh,
    resolve_model_path,
)

pytest.importorskip("mediapipe")

PROJECT_MODELS = Path(__file__).resolve().parents[2] / "models"
needs_model = pytest.mark.skipif(
    resolve_model_path(PROJECT_MODELS) is None, reason="face_landmarker.task not installed"
)


def _synthetic_face(width: int = 400, height: int = 400) -> FaceMeshFace:
    """478 deterministic landmark points laid out on an ellipse-filled grid."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, LANDMARK_COUNT)
    radii = np.sqrt(rng.uniform(0, 1, LANDMARK_COUNT))
    pts = np.stack(
        [width / 2 + np.cos(angles) * radii * 90, height / 2 + np.sin(angles) * radii * 120], axis=1
    ).astype(np.float32)
    return FaceMeshFace(
        index=0,
        tracking_id=1,
        bbox=(110, 80, 180, 240),
        landmarks_px=pts,
        landmarks_z=np.zeros(LANDMARK_COUNT, dtype=np.float32),
        blendshapes={"jawOpen": 0.4, "_neutral": 0.1},
        head_pose=None,
    )


def _blank(width: int = 400, height: int = 400) -> np.ndarray:
    return np.full((height, width, 3), 120, dtype=np.uint8)


def test_settings_reject_invalid_color_and_opacity() -> None:
    with pytest.raises(ValidationError):
        FaceMeshSettings(color="red")
    with pytest.raises(ValidationError):
        FaceMeshSettings(opacity=1.5)
    with pytest.raises(ValidationError):
        FaceMeshSettings(max_faces=0)


def test_renderer_respects_enabled_flag_and_style_options() -> None:
    face = _synthetic_face()
    base = _blank()

    disabled = render_face_mesh(base.copy(), [face], FaceMeshSettings(enabled=False))
    assert np.array_equal(disabled, base)

    def render(**kw: object) -> np.ndarray:
        return render_face_mesh(base.copy(), [face], FaceMeshSettings(enabled=True, **kw))  # type: ignore[arg-type]

    full = render()
    contours = render(show_tessellation=False, show_contours=True)
    dots = render(
        show_tessellation=False,
        show_face_oval=False,
        show_eyes=False,
        show_eyebrows=False,
        show_lips=False,
        show_nose=False,
        show_irises=False,
        show_landmarks=True,
    )
    red = render(color="#FF0000")
    faint = render(opacity=0.2)

    assert not np.array_equal(full, base)
    assert not np.array_equal(full, contours)
    assert not np.array_equal(full, dots)
    assert not np.array_equal(full, red)
    # lower opacity must change the frame less than full opacity
    assert np.abs(faint.astype(int) - base).sum() < np.abs(full.astype(int) - base).sum()


def test_landmark_visibility_toggle_adds_points() -> None:
    face = _synthetic_face()
    without = render_face_mesh(_blank(), [face], FaceMeshSettings(enabled=True, show_landmarks=False))
    with_pts = render_face_mesh(_blank(), [face], FaceMeshSettings(enabled=True, show_landmarks=True))
    assert not np.array_equal(without, with_pts)


def test_head_pose_identity_matrix_is_zero() -> None:
    pose = _head_pose(np.eye(4))
    assert pose == {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}


def test_status_and_service_report_missing_model(tmp_path: Path) -> None:
    status = mesh_status(tmp_path)
    assert status["available"] is False
    assert "face_landmarker.task" in str(status["unavailable_reason"])
    with pytest.raises(ModelNotAvailableError):
        FaceMeshService(tmp_path)


@needs_model
def test_real_model_reports_no_faces_on_blank_image() -> None:
    status = mesh_status(PROJECT_MODELS)
    assert status["available"] is True
    assert status["checksum_verified"] is True
    service = FaceMeshService(PROJECT_MODELS)
    try:
        assert service.detect(_blank(320, 320)) == []
    finally:
        service.close()


@needs_model
@pytest.mark.skipif(not os.environ.get("FACEVISION_TEST_FACE_IMAGE"), reason="set FACEVISION_TEST_FACE_IMAGE")
def test_real_face_has_478_landmarks_and_stable_tracking() -> None:
    img = cv2.imread(os.environ["FACEVISION_TEST_FACE_IMAGE"])
    assert img is not None
    service = FaceMeshService(PROJECT_MODELS, stream=True)
    try:
        ids = []
        for i in range(5):
            faces = service.detect(img, timestamp_ms=i * 33)
            assert len(faces) == 1
            assert faces[0].landmarks_px.shape == (LANDMARK_COUNT, 2)
            assert faces[0].head_pose is not None
            assert len(faces[0].blendshapes) == 52
            ids.append(faces[0].tracking_id)
        assert len(set(ids)) == 1
    finally:
        service.close()


def _client(tmp_path: Path, models_dir: Path) -> TestClient:
    return TestClient(create_app(custom_settings=Settings(data_dir=tmp_path / "data", models_dir=models_dir)))


def _png(width: int = 200, height: int = 200) -> bytes:
    ok, buf = cv2.imencode(".png", _blank(width, height))
    assert ok
    return buf.tobytes()


def _wait(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@needs_model
def test_image_job_with_mesh_writes_summary_and_artifact(tmp_path: Path) -> None:
    with _client(tmp_path, PROJECT_MODELS) as client:
        mid = client.post("/api/v1/media/upload", files={"file": ("a.png", _png(), "image/png")}).json()["id"]
        job = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "image",
                "media_ids": [mid],
                "model_id": "haar",
                "face_mesh": FaceMeshSettings(enabled=True).model_dump(),
            },
        ).json()
        done = _wait(client, job["id"])
        assert done["status"] == "COMPLETED"
        mesh = done["output_manifest"]["face_mesh"]
        assert mesh["face_count"] == 0
        assert mesh["model"]["license"].startswith("Apache-2.0")
        art = client.get(f"/api/v1/jobs/{job['id']}/artifacts/face_mesh.json")
        assert art.status_code == 200
        assert json.loads(art.content)["face_count"] == 0


def test_image_job_fails_clearly_when_mesh_model_missing(tmp_path: Path) -> None:
    with _client(tmp_path, tmp_path / "empty_models") as client:
        mid = client.post("/api/v1/media/upload", files={"file": ("a.png", _png(), "image/png")}).json()["id"]
        job = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "image",
                "media_ids": [mid],
                "model_id": "haar",
                "face_mesh": FaceMeshSettings(enabled=True).model_dump(),
            },
        ).json()
        done = _wait(client, job["id"])
        # Missing the mesh model must be an explicit failure, never a silent skip.
        assert done["status"] == "FAILED"
        assert "face_landmarker.task" in str(done["error_message"])


@needs_model
def test_video_job_with_mesh_reports_frames(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))  # type: ignore[attr-defined]
    for i in range(6):
        writer.write(np.full((120, 160, 3), i * 20, dtype=np.uint8))
    writer.release()
    with _client(tmp_path, PROJECT_MODELS) as client:
        mid = client.post(
            "/api/v1/media/upload", files={"file": ("v.mp4", video.read_bytes(), "video/mp4")}
        ).json()["id"]
        job = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "video",
                "media_ids": [mid],
                "model_id": "haar",
                "face_mesh": FaceMeshSettings(enabled=True).model_dump(),
            },
        ).json()
        done = _wait(client, job["id"])
        assert done["status"] == "COMPLETED"
        mesh = done["output_manifest"]["face_mesh"]
        assert mesh["frames_analyzed"] == 6
        assert mesh["unique_face_tracks"] == 0
        frames = json.loads(client.get(f"/api/v1/jobs/{job['id']}/artifacts/face_mesh.json").content)[
            "frames"
        ]
        assert len(frames) == 6


@needs_model
def test_webcam_websocket_mesh_and_validation(tmp_path: Path) -> None:
    frame = "data:image/png;base64," + base64.b64encode(_png(320, 240)).decode()
    with _client(tmp_path, PROJECT_MODELS) as client, client.websocket_connect("/ws/v1/webcam") as ws:
        ws.send_json(
            {"frame": frame, "model_id": "haar", "mesh": FaceMeshSettings(enabled=True).model_dump()}
        )
        msg = ws.receive_json()
        assert msg["face_mesh"]["face_count"] == 0
        assert msg["annotated_frame"].startswith("data:image/jpeg")

        ws.send_json({"frame": frame, "model_id": "haar", "mesh": {"enabled": True, "color": "nope"}})
        assert "Invalid face mesh settings" in ws.receive_json()["error"]


def test_reports_include_face_mesh(tmp_path: Path) -> None:
    manifest = {
        "face_mesh": {
            "face_count": 1,
            "faces": [
                {
                    "tracking_id": 1,
                    "head_pose": {"yaw_deg": 5.0, "pitch_deg": 1.0, "roll_deg": 0.0},
                    "expression_indicators": [{"name": "jawOpen", "score": 0.4}],
                }
            ],
            "model": {"name": "face_landmarker.task", "license": "Apache-2.0"},
            "disclaimer": "Observable movement only.",
        }
    }
    common = dict(
        output_dir=tmp_path,
        title="T",
        template="simple_summary",
        job_id="j",
        job_type="image",
        model_id="haar",
        faces_detected=1,
        manifest=manifest,
    )
    js = json.loads(generate_report_file(report_id="a", fmt="json", **common).read_text())  # type: ignore[arg-type]
    assert js["face_mesh"]["face_count"] == 1
    csv_text = generate_report_file(report_id="b", fmt="csv", **common).read_text()  # type: ignore[arg-type]
    assert "face_mesh_enabled,yes" in csv_text
    pdf = generate_report_file(report_id="c", fmt="pdf", **common)  # type: ignore[arg-type]
    assert pdf.read_bytes()[:4] == b"%PDF"
