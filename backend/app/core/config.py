"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ExecutionMode = Literal["local_desktop", "web_local_agent", "cloud"]


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings for FaceVision backend."""

    model_config = SettingsConfigDict(
        env_prefix="FACEVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FaceVision"
    app_version: str = "2.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Paths
    project_root: Path = Field(default_factory=_default_project_root)
    data_dir: Path | None = None
    models_dir: Path | None = None
    database_url: str | None = None

    # Server & security
    host: str = "127.0.0.1"
    port: int = 8000
    api_token: str | None = None
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:1420",
            "tauri://localhost",
            "https://tauri.localhost",
        ]
    )

    # Hybrid execution & cloud consent
    execution_mode: ExecutionMode = "web_local_agent"
    allow_cloud_processing: bool = False
    cloud_api_url: str | None = None
    storage_backend: Literal["local", "s3"] = "local"

    # Optional S3-compatible storage
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Media & processing limits
    max_upload_size_bytes: int = 250 * 1024 * 1024  # 250 MB
    max_image_dimension: int = 16384
    max_video_duration_seconds: int = 3600
    max_concurrent_jobs: int = 2
    retention_days: int = 30

    # Default detection settings
    default_model_id: str = "haar"
    default_blur_mode: Literal["none", "gaussian", "pixelate"] = "none"
    default_confidence_threshold: float = 0.6

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir if self.data_dir is not None else self.project_root / ".facevision_data"

    @property
    def resolved_models_dir(self) -> Path:
        return self.models_dir if self.models_dir is not None else self.project_root / "models"

    @property
    def upload_dir(self) -> Path:
        return self.resolved_data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.resolved_data_dir / "outputs"

    @property
    def temp_dir(self) -> Path:
        return self.resolved_data_dir / "tmp"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = self.resolved_data_dir / "facevision.db"
        return f"sqlite:///{db_path}"

    def ensure_directories(self) -> None:
        """Create required local runtime directories."""
        for directory in (
            self.resolved_data_dir,
            self.resolved_models_dir,
            self.upload_dir,
            self.output_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
