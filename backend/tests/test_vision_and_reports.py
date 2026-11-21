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


def test_pose_landmark_detector_with_tasks_model() -> None:
    from pathlib import Path

    from backend.app.vision.pose.detection import PoseLandmarkDetector

    models_dir = Path("models")
    if not PoseLandmarkDetector.is_available(models_dir):
        return
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = PoseLandmarkDetector(models_dir=models_dir).detect(frame)
    assert res["pose_detected"] is False
    assert res["landmarks"] == []
    assert res["model"] in ("pose_landmarker_full.task", "pose_landmarker_lite.task")


def test_privacy_modes_produce_distinct_outputs() -> None:
    from backend.app.detection.base import Detection
    from backend.app.privacy.blur import apply_privacy_blur

    rng = np.random.default_rng(42)
    frame = rng.integers(30, 220, size=(240, 240, 3), dtype=np.uint8)
    det = Detection(x=40, y=30, width=140, height=160, confidence=0.95)

    outputs = {}
    for mode in ("none", "gaussian", "pixelate", "mask", "mesh", "geometric", "mesh_3d"):
        out = apply_privacy_blur(frame, [det], mode)  # type: ignore[arg-type]
        assert out.shape == frame.shape
        outputs[mode] = out.tobytes()

    assert len(set(outputs.values())) == 7


def test_hand_object_and_mask_models_initialize_and_run() -> None:
    from pathlib import Path

    from backend.app.detection.base import Detection
    from backend.app.vision.faces.mask_detection import FaceMaskClassifier
    from backend.app.vision.hands.detection import HandLandmarkDetector
    from backend.app.vision.objects.detection import YOLOv8ONNXObjectDetector

    models_dir = Path("models")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    if HandLandmarkDetector.is_available(models_dir):
        hand_res = HandLandmarkDetector(models_dir=models_dir).detect(frame)
        assert hand_res["hand_count"] == 0
        assert hand_res["model"] == "hand_landmarker.task"

    obj_det = YOLOv8ONNXObjectDetector(models_dir=models_dir)
    if obj_det.is_available():
        assert isinstance(obj_det.detect(frame), list)
        obj_det.release()

    mask_clf = FaceMaskClassifier(models_dir=models_dir)
    if mask_clf.is_available():
        det = Detection(x=40, y=40, width=100, height=100, confidence=0.9)
        res = mask_clf.classify_faces(frame, [det])
        assert res["face_count"] == 1
        mask_clf.release()


def test_face_emotion_ferplus_and_hybrid() -> None:
    from pathlib import Path

    from backend.app.detection.base import Detection
    from backend.app.vision.faces.emotion import (
        analyze_face_emotions,
        draw_emotion_overlays,
        is_ferplus_available,
    )

    models_dir = Path("models")
    if not is_ferplus_available(models_dir):
        return

    rng = np.random.default_rng(7)
    frame = rng.integers(40, 200, size=(240, 240, 3), dtype=np.uint8)
    det = Detection(x=40, y=40, width=140, height=140, confidence=0.92)
    res = analyze_face_emotions(frame, [det], mode="ferplus", models_dir=models_dir)
    assert res is not None
    assert res["face_count"] == 1
    assert res["faces"][0]["dominant_emotion"] in {
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    }
    draw_emotion_overlays(frame, [det], res)


def test_heic_convert_and_u2net_background_removal(tmp_path: Path) -> None:
    from PIL import Image

    from backend.app.media.image.tools import convert_image
    from backend.app.vision.objects.background_remove import (
        remove_background_to_file,
        resolve_u2net_path,
    )

    src = tmp_path / "sample.png"
    Image.new("RGB", (96, 96), (40, 120, 200)).save(src)

    heic_out = tmp_path / "out.heic"
    conv_res = convert_image(src, heic_out, target_format="heic", quality=80)
    assert conv_res.output_format == "HEIF"
    assert heic_out.is_file() and heic_out.stat().st_size > 0

    models_dir = Path("models")
    if resolve_u2net_path(models_dir) is not None:
        bg_out = tmp_path / "nobg.png"
        bg_res = remove_background_to_file(
            src,
            bg_out,
            models_dir=models_dir,
            model_type="universal",
            background_mode="transparent",
        )
        assert bg_res.output_format == "PNG"
        assert bg_out.is_file() and bg_out.stat().st_size > 0


def test_asl_sign_language_registry_and_detector() -> None:
    from pathlib import Path

    from backend.app.vision.sign_language.recognition import ASLSignDetector, list_sign_languages

    models_dir = Path("models")
    langs = {item["code"]: item for item in list_sign_languages(models_dir)}
    assert langs["ASL"]["model_available"] is True
    assert langs["FinSL"]["model_available"] is False
    assert langs["BSL"]["model_available"] is False

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = ASLSignDetector(models_dir=models_dir).detect(frame)
    assert res["language"] == "ASL"
    assert isinstance(res["asl_letters"], list)
    assert isinstance(res["gestures"], list)


def test_privacy_alpha_accepts_three_dim_segmentation_mask() -> None:
    """MediaPipe >=1.x returns (H, W, 1) category masks; privacy blur must not crash."""
    from backend.app.detection.base import Detection
    from backend.app.vision.faces.segmentation import FacePrivacyContext

    head = np.zeros((120, 120, 1), dtype=np.float32)
    head[30:90, 30:90] = 1.0
    from backend.app.vision.faces.face_mesh import FaceMeshFace

    mesh_face = FaceMeshFace(
        index=0,
        tracking_id=None,
        bbox=(30, 30, 60, 60),
        landmarks_px=np.full((478, 2), 60.0, dtype=np.float32),
        landmarks_z=np.zeros(478, dtype=np.float32),
        blendshapes={},
        head_pose=None,
    )
    ctx = FacePrivacyContext(head_mask=head, mesh_faces=[mesh_face])
    det = Detection(x=20, y=10, width=60, height=100, confidence=0.9)
    alpha = ctx.build_roi_alpha((120, 120, 3), det, (20, 10, 80, 110), mesh_face=mesh_face)
    assert alpha.shape[:2] == (100, 60)
    assert float(alpha.max()) > 0.5


def test_mask_classifier_uses_mobilenet_scaling() -> None:
    """The Keras MobileNetV2 mask model needs [-1, 1] input; [0, 1] mislabels bare faces as masked."""
    from backend.app.vision.faces.mask_detection import preprocess_keras_mobilenet

    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[..., 0] = 255  # blue channel in BGR
    blob = preprocess_keras_mobilenet(crop)
    assert blob.shape == (1, 224, 224, 3)
    assert blob.dtype == np.float32
    assert float(blob.min()) == -1.0
    assert float(blob.max()) == 1.0
    # Channel order is preserved (BGR), not swapped to RGB.
    assert float(blob[0, 0, 0, 0]) == 1.0
    assert float(blob[0, 0, 0, 2]) == -1.0


def test_pose_person_regions_fall_back_to_whole_frame(tmp_path: Path) -> None:
    """Without an object detector, pose runs on the whole frame (single-person mode)."""
    from backend.app.vision.pose.detection import PoseLandmarkDetector

    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    regions = PoseLandmarkDetector(models_dir=tmp_path)._person_regions(frame)
    assert regions == [((0, 0, 200, 120), (0, 0, 200, 120))]



def test_pose_prefers_full_model_over_lite(tmp_path: Path) -> None:
    from backend.app.vision.pose.detection import resolve_pose_model_path

    (tmp_path / "pose_landmarker_lite.task").write_bytes(b"lite")
    assert resolve_pose_model_path(tmp_path) == tmp_path / "pose_landmarker_lite.task"
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "pose_landmarker_full.task").write_bytes(b"full")
    assert resolve_pose_model_path(tmp_path) == tmp_path / "custom" / "pose_landmarker_full.task"


def _pose_points(nose_x: float, nose_y: float) -> list[dict[str, float]]:
    pts = [{"x": nose_x, "y": nose_y, "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    # Rough left/right face layout in image space (pose 2/5 eyes, 9/10 mouth).
    for idx, dx in ((1, 6), (2, 8), (3, 10), (7, 14), (4, -6), (5, -8), (6, -10), (8, -14), (9, 5), (10, -5)):
        pts[idx] = {**pts[idx], "x": nose_x + dx}
    return pts


def test_pose_hides_points_outside_person_box() -> None:
    from backend.app.vision.pose.detection import _hide_points_outside

    pts = [{"x": 50.0, "y": 50.0, "visibility": 0.9}, {"x": 300.0, "y": 50.0, "visibility": 0.9}]
    _hide_points_outside(pts, (0, 0, 100, 200))
    assert pts[0]["visibility"] == 0.9
    assert pts[1]["visibility"] == 0.0 and pts[1]["outside_person"] is True


def test_pose_face_points_snap_to_face_detector() -> None:
    from backend.app.detection.base import Detection, LandmarkPoint
    from backend.app.vision.pose.detection import _snap_faces_to_detector

    pose = {"bbox": {"x": 0, "y": 0, "width": 200, "height": 400}, "landmarks": _pose_points(90, 60)}
    face = Detection(
        x=80, y=30, width=40, height=50, confidence=0.95,
        landmarks=[
            LandmarkPoint(110, 50), LandmarkPoint(90, 50),  # eyes (unordered on purpose)
            LandmarkPoint(100, 62),  # nose
            LandmarkPoint(92, 72), LandmarkPoint(108, 72),  # mouth corners
        ],
    )
    _snap_faces_to_detector([pose], [face])
    lm = pose["landmarks"]
    assert pose["face_refined"] is True
    assert (lm[0]["x"], lm[0]["y"]) == (100, 62)
    # Pose eye 2 sits right of eye 5 in this layout, so it gets the right-hand detector eye.
    assert (lm[2]["x"], lm[5]["x"]) == (110, 90)
    assert {lm[9]["x"], lm[10]["x"]} == {92, 108}


def test_pose_face_snap_ignores_faces_outside_person() -> None:
    from backend.app.detection.base import Detection, LandmarkPoint
    from backend.app.vision.pose.detection import _snap_faces_to_detector

    pose = {
        "bbox": {"x": 0, "y": 0, "width": 100, "height": 300},
        "landmarks": _pose_points(50, 40),
        "face_refined": False,
    }
    far_face = Detection(
        x=400, y=30, width=40, height=50, confidence=0.9,
        landmarks=[LandmarkPoint(410, 50), LandmarkPoint(430, 50), LandmarkPoint(420, 60),
                   LandmarkPoint(412, 70), LandmarkPoint(428, 70)],
    )
    _snap_faces_to_detector([pose], [far_face])
    assert pose["face_refined"] is False
    assert pose["landmarks"][0]["x"] == 50


def test_face_mesh_crop_results_map_back_to_full_frame() -> None:
    """Per-face crops must place landmarks in full-image pixels with frame-scaled depth."""
    from types import SimpleNamespace

    from backend.app.vision.faces.face_mesh import _parse_landmarker_result

    lm = [SimpleNamespace(x=0.5, y=0.25, z=0.1)] * 478
    result = SimpleNamespace(face_landmarks=[lm], face_blendshapes=[], facial_transformation_matrixes=[])
    (px, z, blend, pose), = _parse_landmarker_result(result, 100, 40, 80, 80, 400)
    assert tuple(px[0]) == (140.0, 60.0)
    assert abs(float(z[0]) - 0.1 * 80 / 400) < 1e-6
    assert blend == {} and pose is None
