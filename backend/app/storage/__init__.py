"""Storage adapters and media signature validation."""

from .base import StorageAdapter, StoredObject
from .cloud import S3StorageAdapter
from .local import LocalStorageAdapter, validate_media_signature

__all__ = [
    "LocalStorageAdapter",
    "S3StorageAdapter",
    "StorageAdapter",
    "StoredObject",
    "validate_media_signature",
]
