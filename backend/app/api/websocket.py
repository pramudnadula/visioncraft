"""WebSocket endpoints for real-time job progress and live webcam detection."""

from __future__ import annotations

import asyncio
import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.core.dependencies import AppStateContainer
from backend.app.core.exceptions import JobNotFoundError
from backend.app.detection.base import DetectorConfig
from backend.app.processing.webcam import WebcamProcessor

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
    except (WebSocketDisconnect, asyncio.TimeoutError):
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
    detector = container.registry.create_detector(
        DetectorConfig(model_id=current_model_id, models_dir=container.settings.resolved_models_dir)
    )

    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)
            model_id = str(payload.get("model_id", "haar"))
            blur_mode = payload.get("blur_mode", "none")
            draw_boxes = bool(payload.get("draw_boxes", True))

            if model_id != current_model_id:
                detector.release()
                detector = container.registry.create_detector(
                    DetectorConfig(model_id=model_id, models_dir=container.settings.resolved_models_dir)
                )
                current_model_id = model_id

            frame_b64 = payload.get("frame")
            if not frame_b64:
                continue
            frame = processor.decode_base64_frame(str(frame_b64))
            if frame is None:
                continue

            res = processor.process_frame(
                frame,
                detector,
                blur_mode=blur_mode,
                draw_boxes=draw_boxes,
                encode_jpeg=True,
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
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        detector.release()
