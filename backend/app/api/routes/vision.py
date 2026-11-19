"""General AI Vision and Vehicle Analytics endpoints."""

from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import MediaNotFoundError
from backend.app.database.models import MediaAsset
from backend.app.detection.base import DetectorConfig
from backend.app.vision.faces.expressions import FacialExpressionAnalyzer
from backend.app.vision.faces.mask_detection import FaceMaskClassifier
from backend.app.vision.faces.mesh_3d import analyze_faces_3d
from backend.app.vision.hands.detection import HandLandmarkDetector
from backend.app.vision.objects.detection import VEHICLE_CLASSES, YOLOv8ONNXObjectDetector
from backend.app.vision.objects.tracking import ObjectTracker
from backend.app.vision.pose.detection import PoseLandmarkDetector
from backend.app.vision.sign_language.recognition import list_sign_languages
from backend.app.vision.vehicles.counting import CountingLine, VehicleCounter

router = APIRouter(
    prefix="/vision",
    tags=["vision"],
    dependencies=[Depends(require_api_token)],
)


class ObjectDetectRequest(BaseModel):
    media_id: str
    confidence_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    vehicles_only: bool = False


class VehicleCountRequest(BaseModel):
    media_id: str
    confidence_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    line_y_ratio: float = Field(default=0.5, ge=0.05, le=0.95)
    max_frames: int = Field(default=150, ge=1, le=3600)


@router.get("/capabilities")
def get_vision_capabilities(container: AppStateContainer = Depends(get_container)) -> dict[str, object]:
    models_dir = container.settings.resolved_models_dir
    obj_det = YOLOv8ONNXObjectDetector(models_dir=models_dir)
    mask_clf = FaceMaskClassifier(models_dir=models_dir)
    return {
        "face_detection": True,
        "face_mesh_3d_ready": True,
        "face_mask_ready": mask_clf.is_available(),
        "facial_analysis_ready": FacialExpressionAnalyzer.is_available(),
        "hand_landmarks_ready": HandLandmarkDetector.is_available(),
        "pose_estimation_ready": PoseLandmarkDetector.is_available(),
        "object_detection_ready": obj_det.is_available(),
        "vehicle_analytics_ready": obj_det.is_available(),
        "sign_languages": list_sign_languages(),
        "object_model_hint": (
            "Place yolov8n.onnx in models/ (or download via Model Hub) to enable object & vehicle detection."
        ),
    }


@router.get("/sign-language/languages")
def get_sign_languages() -> list[dict[str, object]]:
    return list_sign_languages()


@router.post("/faces/mask-detect")
def detect_face_masks(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, payload.media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{payload.media_id}' not found.")
        storage_key = asset.storage_key

    local_path = container.storage.resolve_local_path(storage_key)
    frame = cv2.imread(str(local_path))
    if frame is None:
        raise MediaNotFoundError("Could not decode image.")

    face_det = container.registry.create_detector(
        DetectorConfig(model_id="yunet", models_dir=container.settings.resolved_models_dir)
        if (container.settings.resolved_models_dir / "face_detection_yunet_2023mar.onnx").is_file()
        else DetectorConfig(model_id="haar", models_dir=container.settings.resolved_models_dir)
    )
    mask_clf = FaceMaskClassifier(
        models_dir=container.settings.resolved_models_dir,
        confidence_threshold=payload.confidence_threshold,
    )
    try:
        faces = face_det.detect(frame)
        return mask_clf.classify_faces(frame, faces)
    finally:
        face_det.release()
        mask_clf.release()


def _load_media_frame(container: AppStateContainer, media_id: str) -> cv2.typing.MatLike:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{media_id}' not found.")
        storage_key = asset.storage_key
    frame = cv2.imread(str(container.storage.resolve_local_path(storage_key)))
    if frame is None:
        raise MediaNotFoundError("Could not decode image.")
    return frame


@router.post("/faces/analyze")
def analyze_facial_landmarks(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    frame = _load_media_frame(container, payload.media_id)
    return FacialExpressionAnalyzer().analyze(frame)


@router.post("/hands/detect")
def detect_hand_landmarks(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    frame = _load_media_frame(container, payload.media_id)
    return HandLandmarkDetector().detect(frame)


@router.post("/pose/detect")
def detect_pose_landmarks(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    frame = _load_media_frame(container, payload.media_id)
    return PoseLandmarkDetector().detect(frame)


@router.post("/faces/mesh-3d")
def analyze_3d_face_mesh(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    frame = _load_media_frame(container, payload.media_id)
    face_det = container.registry.create_detector(
        DetectorConfig(model_id="yunet", models_dir=container.settings.resolved_models_dir)
        if (container.settings.resolved_models_dir / "face_detection_yunet_2023mar.onnx").is_file()
        else DetectorConfig(model_id="haar", models_dir=container.settings.resolved_models_dir)
    )
    try:
        faces = face_det.detect(frame)
        return analyze_faces_3d(frame, faces)
    finally:
        face_det.release()


@router.post("/objects/detect")
def detect_objects(
    payload: ObjectDetectRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, payload.media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{payload.media_id}' not found.")
        storage_key = asset.storage_key

    local_path = container.storage.resolve_local_path(storage_key)
    frame = cv2.imread(str(local_path))
    if frame is None:
        raise MediaNotFoundError("Could not decode image.")

    detector = YOLOv8ONNXObjectDetector(
        models_dir=container.settings.resolved_models_dir,
        confidence_threshold=payload.confidence_threshold,
    )
    try:
        allowed = VEHICLE_CLASSES if payload.vehicles_only else None
        detections = detector.detect(frame, allowed_classes=allowed)
    finally:
        detector.release()

    return {
        "media_id": payload.media_id,
        "count": len(detections),
        "detections": [d.to_dict() for d in detections],
    }


@router.post("/vehicles/count")
def count_vehicles_in_video(
    payload: VehicleCountRequest,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, payload.media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{payload.media_id}' not found.")
        storage_key = asset.storage_key

    local_path = container.storage.resolve_local_path(storage_key)
    cap = cv2.VideoCapture(str(local_path))
    if not cap.isOpened():
        raise MediaNotFoundError("Could not open video.")

    detector = YOLOv8ONNXObjectDetector(
        models_dir=container.settings.resolved_models_dir,
        confidence_threshold=payload.confidence_threshold,
    )
    detector.initialize()
    tracker = ObjectTracker()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    line_y = int(height * payload.line_y_ratio)
    counter = VehicleCounter(line=CountingLine(0, line_y, width, line_y))

    frames_processed = 0
    try:
        while frames_processed < payload.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            dets = detector.detect(frame, allowed_classes=VEHICLE_CLASSES, frame_number=frames_processed + 1)
            tracked = tracker.update(dets)
            counter.update(tracked)
            frames_processed += 1
    finally:
        cap.release()
        detector.release()

    s = counter.summary
    return {
        "media_id": payload.media_id,
        "frames_processed": frames_processed,
        "total_vehicles": s.total_vehicles,
        "cars": s.cars,
        "buses": s.buses,
        "trucks": s.trucks,
        "motorcycles": s.motorcycles,
        "bicycles": s.bicycles,
        "entry_count": s.entry_count,
        "exit_count": s.exit_count,
    }
