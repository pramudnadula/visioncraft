"""Optional S3-compatible object storage adapter."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from backend.app.core.exceptions import StorageError
from backend.app.storage.base import StorageAdapter, StoredObject


class S3StorageAdapter(StorageAdapter):
    """S3-compatible storage adapter backed by a local cache directory."""

    def __init__(
        self,
        *,
        bucket: str,
        cache_dir: Path,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        if importlib.util.find_spec("boto3") is None:
            raise StorageError("S3 storage requires optional 'boto3' dependency.")
        import boto3  # type: ignore[import-not-found]

        self.bucket = bucket
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def _safe_cache_path(self, storage_key: str) -> Path:
        candidate = (self.cache_dir / storage_key.lstrip("/")).resolve()
        if not candidate.is_relative_to(self.cache_dir):
            raise StorageError("Invalid storage key: path traversal detected.")
        return candidate

    def save_bytes(self, data: bytes, storage_key: str) -> StoredObject:
        self._client.put_object(Bucket=self.bucket, Key=storage_key, Body=data)
        cache_path = self._safe_cache_path(storage_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return StoredObject(
            storage_key=storage_key,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def resolve_local_path(self, storage_key: str) -> Path:
        cache_path = self._safe_cache_path(storage_key)
        if not cache_path.is_file():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, storage_key, str(cache_path))
        return cache_path

    def exists(self, storage_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=storage_key)
            return True
        except Exception:
            return False

    def delete(self, storage_key: str) -> bool:
        self._client.delete_object(Bucket=self.bucket, Key=storage_key)
        cache_path = self._safe_cache_path(storage_key)
        if cache_path.is_file():
            cache_path.unlink(missing_ok=True)
        return True
