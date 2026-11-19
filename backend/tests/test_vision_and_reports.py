"""Tests for Phase C (Object Tracking, Vehicle Counting) and Phase G (PDF/CSV/JSON Reports)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.reports.generator import generate_report_file
from backend.app.vision.objects.detection import ObjectDetection
from backend.app.vision.objects.tracking import ObjectTracker
from backend.app.vision.vehicles.counting import CountingLine, CountingZone, VehicleCounter


def test_object_tracker_and_vehicle_line_zone_counting() -> None:
    tracker = ObjectTracker(iou_threshold=0.2, max_missed=5)
    line = CountingLine(0, 100, 300, 100)
    zone = CountingZone(zone_id="z1", name="Lane 1", polygon=[(0, 80), (300, 80), (300, 200), (0, 200)])
    counter = VehicleCounter(line=line, zones=[zone])

    # Frame 1: car above line (y=65..105 -> center y = 85)
    f1 = tracker.update([ObjectDetection("1", 2, "car", 0.9, 50, 65, 40, 40)])
    # Frame 2: car crosses below line (y=85..125 -> center y = 105)
    f2 = tracker.update([ObjectDetection("2", 2, "car", 0.9, 52, 85, 40, 40)])
    # Frame 3: same car stays below line (y=100..140 -> center y = 120, should not double count!)
    f3 = tracker.update([ObjectDetection("3", 2, "car", 0.9, 54, 100, 40, 40)])

    counter.update(f1)
    counter.update(f2)
    summary = counter.update(f3)

    assert f1[0].tracking_id == f2[0].tracking_id == f3[0].tracking_id == 1
    assert summary.total_vehicles == 1
    assert summary.cars == 1
    assert summary.entry_count == 1
    assert summary.exit_count == 0
    assert summary.zone_counts["z1"] == 1


def test_report_generator_pdf_csv_json(tmp_path: Path) -> None:
    manifest = {"face_count": 2, "duration_ms": 18.5}
    pdf_path = generate_report_file(
        output_dir=tmp_path,
        report_id="rep1",
        title="Test Report",
        template="simple_summary",
        fmt="pdf",
        job_id="job-12345678",
        job_type="image",
        model_id="yunet",
        faces_detected=2,
        manifest=manifest,
    )
    assert pdf_path.is_file()
    assert pdf_path.read_bytes()[:4] == b"%PDF"

    csv_path = generate_report_file(
        output_dir=tmp_path,
        report_id="rep2",
        title="Test CSV",
        template="business_report",
        fmt="csv",
        job_id="job-12345678",
        job_type="image",
        model_id="haar",
        faces_detected=1,
        manifest=manifest,
    )
    assert "total_detections,1" in csv_path.read_text(encoding="utf-8")

    json_path = generate_report_file(
        output_dir=tmp_path,
        report_id="rep3",
        title="Test JSON",
        template="detailed_technical",
        fmt="json",
        job_id="job-12345678",
        job_type="image",
        model_id="haar",
        faces_detected=1,
        manifest=manifest,
    )
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["total_detections"] == 1


def test_vision_capabilities_and_reports_api(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok

    with TestClient(create_app(custom_settings=settings)) as client:
        caps = client.get("/api/v1/vision/capabilities")
        assert caps.status_code == 200
        assert caps.json()["face_detection"] is True

        up = client.post("/api/v1/media/upload", files={"file": ("a.png", buf.tobytes(), "image/png")})
        mid = up.json()["id"]
        job_id = client.post(
            "/api/v1/jobs",
            json={"job_type": "image", "media_ids": [mid], "model_id": "haar"},
        ).json()["id"]

        for _ in range(40):
            if client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "COMPLETED":
                break
            time.sleep(0.1)

        rep = client.post(
            "/api/v1/reports/generate",
            json={"job_id": job_id, "title": "API PDF Report", "format": "pdf"},
        )
        assert rep.status_code == 201
        dl = client.get(rep.json()["download_url"])
        assert dl.status_code == 200
        assert dl.content[:4] == b"%PDF"


def test_3d_face_mesh_and_head_pose() -> None:
    from backend.app.detection.base import Detection
    from backend.app.vision.faces.mesh_3d import analyze_faces_3d, draw_3d_face_mesh

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    det = Detection(x=80, y=60, width=160, height=200, confidence=0.95)
    pose = draw_3d_face_mesh(frame, det)
    assert isinstance(pose.yaw_deg, float)
    assert frame.sum() > 0

    res = analyze_faces_3d(frame, [det])
    assert res["face_count"] == 1
    assert len(res["faces"][0]["landmarks_3d"]) == 6
