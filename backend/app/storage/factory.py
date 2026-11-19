"""Storage adapter factory supporting local, S3-compatible (R2/MinIO/B2/AWS), and Firebase/GCS."""

from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.core.exceptions import StorageError
from backend.app.storage.base import StorageAdapter
from backend.app.storage.cloud import S3StorageAdapter
from backend.app.storage.firebase_gcs import FirebaseGCSStorageAdapter
from backend.app.storage.local import LocalStorageAdapter


def create_storage_adapter(settings: Settings) -> StorageAdapter:
    """Instantiate the configured storage backend."""
    backend = settings.storage_backend
    if backend == "local":
        return LocalStorageAdapter(settings.resolved_data_dir)

    if backend == "s3":
        if not settings.s3_bucket:
            raise StorageError("FACEVISION_S3_BUCKET is required when storage_backend='s3'.")
        return S3StorageAdapter(
            bucket=settings.s3_bucket,
            cache_dir=settings.resolved_data_dir / "s3_cache",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )

    if backend in {"firebase", "gcs"}:
        if not settings.gcs_bucket:
            raise StorageError(f"FACEVISION_GCS_BUCKET is required when storage_backend='{backend}'.")
        return FirebaseGCSStorageAdapter(
            bucket=settings.gcs_bucket,
            cache_dir=settings.resolved_data_dir / "gcs_cache",
            credentials_path=settings.firebase_credentials_path,
        )

    raise StorageError(f"Unsupported storage_backend: '{backend}'.")
