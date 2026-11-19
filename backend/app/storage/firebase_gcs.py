"""Optional Firebase Storage / Google Cloud Storage adapter."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from backend.app.core.exceptions import StorageError
from backend.app.storage.base import StorageAdapter, StoredObject


class FirebaseGCSStorageAdapter(StorageAdapter):
    """Storage adapter for Firebase Storage and Google Cloud Storage buckets."""

    def __init__(
        self,
        *,
        bucket: str,
        cache_dir: Path,
        credentials_path: Path | None = None,
    ) -> None:
        if importlib.util.find_spec("google.cloud.storage") is None:
            raise StorageError("Firebase/GCS storage requires optional 'google-cloud-storage' dependency.")
        from google.cloud import storage as gcs  # type: ignore[import-not-found]

        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if credentials_path is not None:
            self._client: Any = gcs.Client.from_service_account_json(str(credentials_path))
        else:
            self._client = gcs.Client()
        self._bucket: Any = self._client.bucket(bucket)

    def _safe_cache_path(self, storage_key: str) -> Path:
        candidate = (self.cache_dir / storage_key.lstrip("/")).resolve()
        if not candidate.is_relative_to(self.cache_dir):
            raise StorageError("Invalid storage key: path traversal detected.")
        return candidate

    def save_bytes(self, data: bytes, storage_key: str) -> StoredObject:
        blob = self._bucket.blob(storage_key)
        blob.upload_from_string(data)
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
            blob = self._bucket.blob(storage_key)
            blob.download_to_filename(str(cache_path))
        return cache_path

    def exists(self, storage_key: str) -> bool:
        return bool(self._bucket.blob(storage_key).exists())

    def delete(self, storage_key: str) -> bool:
        blob = self._bucket.blob(storage_key)
        if blob.exists():
            blob.delete()
        cache_path = self._safe_cache_path(storage_key)
        if cache_path.is_file():
            cache_path.unlink(missing_ok=True)
        return True
