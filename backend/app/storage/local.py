"""Local filesystem storage adapter with signature & path-traversal security."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Literal

from backend.app.core.exceptions import MediaValidationError, StorageError
from backend.app.storage.base import StorageAdapter, StoredObject

MediaKind = Literal["image", "video"]

IMAGE_SIGNATURES: dict[str, tuple[str, ...]] = {
    ".jpg": ("image/jpeg",),
    ".jpeg": ("image/jpeg",),
    ".png": ("image/png",),
    ".bmp": ("image/bmp",),
    ".webp": ("image/webp",),
    ".gif": ("image/gif",),
    ".tiff": ("image/tiff",),
    ".tif": ("image/tiff",),
    ".ico": ("image/x-icon",),
    ".heic": ("image/heic",),
    ".heif": ("image/heif",),
}

VIDEO_SIGNATURES: dict[str, tuple[str, ...]] = {
    ".mp4": ("video/mp4",),
    ".mov": ("video/quicktime",),
    ".avi": ("video/x-msvideo",),
    ".mkv": ("video/x-matroska",),
    ".webm": ("video/webm",),
}


def _detect_image_mime(header: bytes) -> str | None:
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:2] == b"BM":
        return "image/bmp"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if header[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs"):
            return "image/heic"
        if brand in (b"mif1", b"msf1"):
            return "image/heif"
    return None


def _detect_video_mime(header: bytes, ext: str) -> str | None:
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "video/quicktime" if ext == ".mov" else "video/mp4"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"AVI ":
        return "video/x-msvideo"
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm" if ext == ".webm" else "video/x-matroska"
    return None


def validate_media_signature(header: bytes, extension: str) -> tuple[MediaKind, str]:
    """Validate file magic bytes against supported image/video signatures."""
    ext = extension.lower()
    if len(header) < 4:
        raise MediaValidationError("Uploaded file is empty or too small to be valid media.")

    if ext in IMAGE_SIGNATURES:
        detected = _detect_image_mime(header)
        if detected is not None:
            return "image", detected
        raise MediaValidationError(f"File signature is not a valid image format (declared '{ext}').")

    if ext in VIDEO_SIGNATURES:
        detected = _detect_video_mime(header, ext)
        if detected is not None:
            return "video", detected
        raise MediaValidationError(f"File signature is not a valid video format (declared '{ext}').")

    raise MediaValidationError(f"Unsupported media file extension: '{ext}'.")


class LocalStorageAdapter(StorageAdapter):
    """Stores uploads and outputs under a confined root directory using UUID keys."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, storage_key: str) -> Path:
        clean_key = storage_key.lstrip("/")
        candidate = (self.root_dir / clean_key).resolve()
        if not candidate.is_relative_to(self.root_dir):
            raise StorageError("Invalid storage key: path traversal detected.")
        return candidate

    def save_bytes(self, data: bytes, storage_key: str) -> StoredObject:
        target = self._safe_path(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        return StoredObject(storage_key=storage_key, size_bytes=len(data), sha256=digest)

    def resolve_local_path(self, storage_key: str) -> Path:
        return self._safe_path(storage_key)

    def exists(self, storage_key: str) -> bool:
        return self._safe_path(storage_key).exists()

    def delete(self, storage_key: str) -> bool:
        target = self._safe_path(storage_key)
        if target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            shutil.rmtree(target)
            return True
        return False

    def cleanup_expired(self, retention_days: int) -> int:
        """Delete files older than retention_days; return count deleted."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                deleted += 1
        return deleted
