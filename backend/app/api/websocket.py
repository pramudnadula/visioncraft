"""WebSocket endpoints for real-time job progress and live webcam detection."""

from __future__ import annotations

import asyncio
import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.app.core.dependencies import AppStateContainer
from backend.app.core.exceptions import FaceVisionError, JobNotFoundError
from backend.app.detection.base import DetectorConfig
from backend.app.privacy.blur import validate_blur_mode
from backend.app.processing.webcam import WebcamProcessor
from backend.app.vision.faces.emotion import validate_emotion_mode
from backend.app.vision.faces.face_mesh import FaceMeshService, FaceMeshSettings

ws_router = APIRouter(prefix="/ws/v1", tags=["websocket"])


def _check_ws_token(websocket: WebSocket, container: AppStateContainer) -> bool:
    expected = container.settings.api_token
    if not expected:
        return True
    token = websocket.query_params.get("token") or websocket.headers.get("x-facevision-token")
    return token == expected


@ws_router.websocket("/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    container: AppStateContainer = websocket.app.state.container
    if not _check_ws_token(websocket, container):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    try:
        job = container.job_manager.get_job(job_id)
    except JobNotFoundError:
        await websocket.send_json({"error": "job_not_found", "job_id": job_id})
        await websocket.close(code=4404)
        return

    await websocket.send_json(
        {
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "faces_detected": job.faces_detected,
        }
    )
    if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        await websocket.close()
        return

    queue = container.event_bus.subscribe(job_id)
    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
            await websocket.send_json(event.to_dict())
            if event.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
    except (TimeoutError, WebSocketDisconnect):
        pass
    finally:
        container.event_bus.unsubscribe(job_id, queue)


@ws_router.websocket("/webcam")
async def webcam_stream_ws(websocket: WebSocket) -> None:
    """Process live frames from browser MediaDevices over WebSocket."""
    container: AppStateContainer = websocket.app.state.container
    if not _check_ws_token(websocket, container):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    processor = WebcamProcessor()
    current_model_id = "haar"
    mesh_service: FaceMeshService | None = None
    mesh_key: tuple[int, float] | None = None
    detector = container.registry.create_detector(
        DetectorConfig(model_id=current_model_id, models_dir=container.settings.resolved_models_dir)
    )

    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)
            model_id = str(payload.get("model_id", "haar"))
            draw_boxes = bool(payload.get("draw_boxes", True))
            try:
                blur_mode = validate_blur_mode(str(payload.get("blur_mode", "none")))
                emotion_mode = validate_emotion_mode(str(payload.get("emotion_mode", "off")))
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            mesh_settings: FaceMeshSettings | None = None
            mesh_payload = payload.get("mesh")
            if mesh_payload:
                try:
                    mesh_settings = FaceMeshSettings.model_validate(mesh_payload)
                except ValidationError as exc:
                    await websocket.send_json(
                        {"error": f"Invalid face mesh settings: {exc.errors()[0]['msg']}"}
                    )
                    continue
            if mesh_settings is not None and mesh_settings.enabled:
                key = (mesh_settings.max_faces, mesh_settings.min_confidence)
                if mesh_service is None or key != mesh_key:
                    if mesh_service is not None:
                        mesh_service.close()
                        mesh_service = None
                    try:
                        mesh_service = FaceMeshService(
                            container.settings.resolved_models_dir,
                            max_faces=mesh_settings.max_faces,
                            min_confidence=mesh_settings.min_confidence,
                            stream=True,
                        )
                        mesh_key = key
                    except FaceVisionError as exc:
                        await websocket.send_json({"error": exc.message})
                        continue

            if model_id != current_model_id:
                try:
                    new_detector = container.registry.create_detector(
                        DetectorConfig(model_id=model_id, models_dir=container.settings.resolved_models_dir)
                    )
                except FaceVisionError as exc:
                    await websocket.send_json({"error": exc.message})
                    continue
                detector.release()
                detector = new_detector
                current_model_id = model_id

            frame_b64 = payload.get("frame")
            frame = processor.decode_base64_frame(str(frame_b64)) if frame_b64 else None
            if frame is None:
                await websocket.send_json({"error": "Invalid or empty camera frame."})
                continue

            res = processor.process_frame(
                frame,
                detector,
                blur_mode=blur_mode,
                draw_boxes=draw_boxes,
                encode_jpeg=True,
                mesh_service=mesh_service,
                mesh_settings=mesh_settings,
                emotion_mode=emotion_mode,
                models_dir=container.settings.resolved_models_dir,
            )
            encoded_out = (
                f"data:image/jpeg;base64,{base64.b64encode(res.annotated_jpeg).decode('ascii')}"
                if res.annotated_jpeg
                else None
            )
            await websocket.send_json(
                {
                    "face_count": len(res.detections),
                    "latency_ms": round(res.latency_ms, 2),
                    "detections": [d.to_dict() for d in res.detections],
                    "annotated_frame": encoded_out,
                    "face_mesh": res.face_mesh,
                    "emotions": res.emotions,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        detector.release()
        if mesh_service is not None:
            mesh_service.close()
