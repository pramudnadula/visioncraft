"""Pydantic v2 request and response schemas for FaceVision API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.privacy.blur import BlurMode


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    execution_mode: str


class CapabilitiesResponse(BaseModel):
    execution_mode: str
    local_processing: bool = True
    cloud_processing_enabled: bool
    storage_backend: str
    max_upload_size_bytes: int
    max_image_dimension: int
    max_video_duration_seconds: int
    supported_image_extensions: list[str]
    supported_video_extensions: list[str]


class MediaAssetResponse(BaseModel):
    id: str
    original_filename: str
    media_type: str
    mime_type: str
    extension: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    created_at: datetime
    duplicate_of: str | None = None


class JobCreateRequest(BaseModel):
    job_type: Literal["image", "video", "batch"]
    media_ids: list[str] = Field(min_length=1)
    model_id: str = "haar"
    processing_mode: Literal["local_desktop", "web_local_agent", "cloud"] = "web_local_agent"
    cloud_consent: bool = False
    blur_mode: BlurMode = "none"
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_face_size: int = Field(default=30, ge=10, le=1000)
    scale_factor: float = Field(default=1.1, gt=1.0, le=2.0)
    min_neighbors: int = Field(default=5, ge=1, le=20)
    use_gpu: bool = False
    enable_tracking: bool = True
    save_crops: bool = True
    save_privacy_image: bool = True
    save_privacy_video: bool = False
    selected_face_indices: list[int] | None = None
    target_fps: float | None = Field(default=None, gt=0, le=120)
    show_hud: bool = True
    draw_boxes: bool = True
    max_frames: int | None = Field(default=None, ge=1)


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: float
    model_id: str
    processing_mode: str
    input_media_ids: list[str]
    faces_detected: int
    error_message: str | None = None
    output_manifest: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    version: str
    description: str
    supports_confidence: bool
    supports_landmarks: bool
    supports_gpu: bool
    gpu_active: bool
    is_optional: bool
    available: bool
    license: str
    unavailable_reason: str | None = None


class SettingsResponse(BaseModel):
    default_model_id: str
    execution_mode: str
    default_blur_mode: BlurMode
    default_confidence_threshold: float
    allow_cloud_processing: bool
    retention_days: int


class SettingsUpdateRequest(BaseModel):
    default_model_id: str | None = None
    default_blur_mode: BlurMode | None = None
    default_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_cloud_processing: bool | None = None
    retention_days: int | None = Field(default=None, ge=1, le=3650)


class DashboardStatsResponse(BaseModel):
    total_media: int
    total_jobs: int
    completed_jobs: int
    active_jobs: int
    failed_jobs: int
    total_faces_detected: int
    execution_mode: str
    cloud_processing_enabled: bool
