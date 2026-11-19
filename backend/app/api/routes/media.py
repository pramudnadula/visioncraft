"""Media upload, listing, inspection, and deletion endpoints."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import MediaNotFoundError, MediaValidationError
from backend.app.database.models import MediaAsset
from backend.app.processing.video import VideoProcessingService
from backend.app.schemas.api import MediaAssetResponse
from backend.app.storage.local import validate_media_signature

router = APIRouter(prefix="/media", tags=["media"], dependencies=[Depends(require_api_token)])


def _sanitize_display_name(filename: str | None) -> str:
    if not filename:
        return "upload"
    return Path(filename).name[:200] or "upload"


def _to_response(asset: MediaAsset, duplicate_of: str | None = None) -> MediaAssetResponse:
    return MediaAssetResponse(
        id=asset.id,
        original_filename=asset.original_filename,
        media_type=asset.media_type,
        mime_type=asset.mime_type,
        extension=asset.extension,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
        width=asset.width,
        height=asset.height,
        duration_seconds=asset.duration_seconds,
        created_at=asset.created_at,
        duplicate_of=duplicate_of,
    )


@router.post("/upload", response_model=MediaAssetResponse, status_code=201)
async def upload_media(
    file: UploadFile = File(...),
    container: AppStateContainer = Depends(get_container),
) -> MediaAssetResponse:
    display_name = _sanitize_display_name(file.filename)
    ext = Path(display_name).suffix.lower()
    if not ext:
        raise MediaValidationError("Uploaded file must have a valid media extension.")

    data = await file.read()
    max_bytes = container.settings.max_upload_size_bytes
    if len(data) == 0:
        raise MediaValidationError("Uploaded file is empty.")
    if len(data) > max_bytes:
        raise MediaValidationError(f"File exceeds maximum upload size of {max_bytes} bytes.")

    media_kind, mime_type = validate_media_signature(data[:32], ext)
    media_id = str(uuid.uuid4())
    storage_key = f"uploads/{media_id}{ext}"
    stored = container.storage.save_bytes(data, storage_key)
    local_path = container.storage.resolve_local_path(storage_key)

    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None

    try:
        if media_kind == "image":
            img = cv2.imread(str(local_path))
            if img is None or img.size == 0:
                raise MediaValidationError("Uploaded image could not be decoded.")
            height, width = img.shape[:2]
            max_dim = container.settings.max_image_dimension
            if width > max_dim or height > max_dim:
                raise MediaValidationError(
                    f"Image dimensions ({width}x{height}) exceed limit ({max_dim})."
                )
        else:
            vmeta = VideoProcessingService.extract_metadata(local_path)
            if vmeta.duration_seconds > container.settings.max_video_duration_seconds:
                raise MediaValidationError(
                    f"Video duration ({vmeta.duration_seconds:.1f}s) exceeds limit."
                )
            width = vmeta.width
            height = vmeta.height
            duration_seconds = round(vmeta.duration_seconds, 2)
    except Exception:
        container.storage.delete(storage_key)
        raise

    sha256 = hashlib.sha256(data).hexdigest()
    duplicate_of: str | None = None
    with container.db.session_scope() as session:
        existing = session.query(MediaAsset).filter(MediaAsset.sha256 == sha256).first()
        if existing is not None:
            duplicate_of = existing.id

        asset = MediaAsset(
            id=media_id,
            original_filename=display_name,
            media_type=media_kind,
            mime_type=mime_type,
            extension=ext,
            size_bytes=stored.size_bytes,
            sha256=sha256,
            storage_key=storage_key,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )
        session.add(asset)
        session.flush()
        session.refresh(asset)
        session.expunge(asset)

    return _to_response(asset, duplicate_of=duplicate_of)


@router.get("", response_model=list[MediaAssetResponse])
def list_media(
    limit: int = 100,
    container: AppStateContainer = Depends(get_container),
) -> list[MediaAssetResponse]:
    with container.db.session_scope() as session:
        assets = (
            session.query(MediaAsset)
            .order_by(MediaAsset.created_at.desc())
            .limit(min(limit, 500))
            .all()
        )
        for a in assets:
            session.expunge(a)
    return [_to_response(a) for a in assets]


@router.get("/{media_id}", response_model=MediaAssetResponse)
def get_media(
    media_id: str,
    container: AppStateContainer = Depends(get_container),
) -> MediaAssetResponse:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{media_id}' not found.")
        session.expunge(asset)
    return _to_response(asset)


@router.get("/{media_id}/file")
def download_media_file(
    media_id: str,
    container: AppStateContainer = Depends(get_container),
) -> FileResponse:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{media_id}' not found.")
        session.expunge(asset)

    local_path = container.storage.resolve_local_path(asset.storage_key)
    if not local_path.is_file():
        raise MediaNotFoundError("Media file is no longer present in storage.")
    return FileResponse(path=str(local_path), media_type=asset.mime_type, filename=asset.original_filename)


@router.delete("/{media_id}", status_code=204)
def delete_media(
    media_id: str,
    container: AppStateContainer = Depends(get_container),
) -> None:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{media_id}' not found.")
        storage_key = asset.storage_key
        session.delete(asset)
    container.storage.delete(storage_key)
