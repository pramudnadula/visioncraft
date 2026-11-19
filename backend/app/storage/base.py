"""Abstract storage adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Metadata for a stored media object."""

    storage_key: str
    size_bytes: int
    sha256: str


class StorageAdapter(ABC):
    """Interface for local filesystem and optional S3-compatible object storage."""

    @abstractmethod
    def save_bytes(self, data: bytes, storage_key: str) -> StoredObject:
        """Persist raw bytes under an internal UUID-based storage key."""

    @abstractmethod
    def resolve_local_path(self, storage_key: str) -> Path:
        """Return a safe local Path for reading/processing."""

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Check whether an object exists."""

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Delete an object if it exists."""
