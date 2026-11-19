"""SQLAlchemy ORM models for media assets, processing jobs, and settings."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MediaAsset(Base):
    """Uploaded image or video metadata."""

    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)  # image | video
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ProcessingJob(Base):
    """Background face detection / privacy processing job."""

    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(16), nullable=False)  # image | video | batch
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_media_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_manifest_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    faces_detected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppSetting(Base):
    """Persisted key-value application settings."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class MediaOperation(Base):
    """Recorded image/video editing operation (crop, resize, convert, compress)."""

    __tablename__ = "media_operations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_media_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    operation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    settings_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    output_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="COMPLETED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
