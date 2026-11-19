"""System health, capabilities, and dashboard statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.database.models import MediaAsset, ProcessingJob
from backend.app.schemas.api import (
    CapabilitiesResponse,
    DashboardStatsResponse,
    HealthResponse,
)
from backend.app.storage.local import IMAGE_SIGNATURES, VIDEO_SIGNATURES

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def get_health(container: AppStateContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=container.settings.app_name,
        version=container.settings.app_version,
        execution_mode=container.settings.execution_mode,
    )


@router.get("/capabilities", response_model=CapabilitiesResponse, dependencies=[Depends(require_api_token)])
def get_capabilities(container: AppStateContainer = Depends(get_container)) -> CapabilitiesResponse:
    s = container.settings
    return CapabilitiesResponse(
        execution_mode=s.execution_mode,
        local_processing=True,
        cloud_processing_enabled=s.allow_cloud_processing,
        storage_backend=s.storage_backend,
        max_upload_size_bytes=s.max_upload_size_bytes,
        max_image_dimension=s.max_image_dimension,
        max_video_duration_seconds=s.max_video_duration_seconds,
        supported_image_extensions=sorted(IMAGE_SIGNATURES.keys()),
        supported_video_extensions=sorted(VIDEO_SIGNATURES.keys()),
    )


@router.get("/stats", response_model=DashboardStatsResponse, dependencies=[Depends(require_api_token)])
def get_dashboard_stats(container: AppStateContainer = Depends(get_container)) -> DashboardStatsResponse:
    with container.db.session_scope() as session:
        total_media = session.query(func.count(MediaAsset.id)).scalar() or 0
        total_jobs = session.query(func.count(ProcessingJob.id)).scalar() or 0
        completed_jobs = (
            session.query(func.count(ProcessingJob.id))
            .filter(ProcessingJob.status == "COMPLETED")
            .scalar()
            or 0
        )
        active_jobs = (
            session.query(func.count(ProcessingJob.id))
            .filter(ProcessingJob.status.in_(["QUEUED", "VALIDATING", "PROCESSING", "FINALIZING"]))
            .scalar()
            or 0
        )
        failed_jobs = (
            session.query(func.count(ProcessingJob.id))
            .filter(ProcessingJob.status == "FAILED")
            .scalar()
            or 0
        )
        total_faces = session.query(func.coalesce(func.sum(ProcessingJob.faces_detected), 0)).scalar() or 0

    return DashboardStatsResponse(
        total_media=int(total_media),
        total_jobs=int(total_jobs),
        completed_jobs=int(completed_jobs),
        active_jobs=int(active_jobs),
        failed_jobs=int(failed_jobs),
        total_faces_detected=int(total_faces),
        execution_mode=container.settings.execution_mode,
        cloud_processing_enabled=container.settings.allow_cloud_processing,
    )
