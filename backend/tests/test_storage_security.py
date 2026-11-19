"""Security and storage validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import MediaValidationError, StorageError
from backend.app.storage.factory import create_storage_adapter
from backend.app.storage.local import LocalStorageAdapter, validate_media_signature


def test_validate_media_signature_accepts_valid_headers() -> None:
    assert validate_media_signature(b"\xff\xd8\xff\xe0rest", ".jpg") == ("image", "image/jpeg")
    assert validate_media_signature(b"\x89PNG\r\n\x1a\nrest", ".png") == ("image", "image/png")
    assert validate_media_signature(b"\x00\x00\x00\x18ftypmp42", ".mp4") == ("video", "video/mp4")


def test_validate_media_signature_rejects_spoofed_files() -> None:
    with pytest.raises(MediaValidationError):
        validate_media_signature(b"#!/bin/sh\nrm -rf /", ".jpg")

    with pytest.raises(MediaValidationError):
        validate_media_signature(b"MZ\x90\x00", ".exe")


def test_local_storage_blocks_path_traversal(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    with pytest.raises(StorageError):
        storage.save_bytes(b"hello", "../../etc/passwd")

    with pytest.raises(StorageError):
        storage.resolve_local_path("../outside.txt")


def test_storage_factory_local_and_missing_cloud_config(tmp_path: Path) -> None:
    local_adapter = create_storage_adapter(Settings(data_dir=tmp_path, storage_backend="local"))
    assert isinstance(local_adapter, LocalStorageAdapter)

    with pytest.raises(StorageError):
        create_storage_adapter(Settings(data_dir=tmp_path, storage_backend="s3", s3_bucket=None))

    with pytest.raises(StorageError):
        create_storage_adapter(Settings(data_dir=tmp_path, storage_backend="firebase", gcs_bucket=None))
