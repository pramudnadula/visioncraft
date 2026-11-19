"""Universal image editing endpoints: crop, resize, convert, compress, and metadata."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import MediaNotFoundError, StorageError
from backend.app.database.models import MediaAsset, MediaOperation
from backend.app.media.image.tools import (
    ImageOperationResult,
    compress_image,
    convert_image,
    crop_image,
    inspect_image_metadata,
    resize_image,
)

router = APIRouter(
    prefix="/media/image",
    tags=["media-tools"],
    dependencies=[Depends(require_api_token)],
)


class CropRequest(BaseModel):
    media_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    rotate_deg: int = 0
    flip_h: bool = False
    flip_v: bool = False


class ResizeRequest(BaseModel):
    media_id: str
    width: int | None = Field(default=None, ge=1, le=16384)
    height: int | None = Field(default=None, ge=1, le=16384)
    max_width: int | None = Field(default=None, ge=1, le=16384)
    max_height: int | None = Field(default=None, ge=1, le=16384)
    scale_percent: float | None = Field(default=None, gt=1, le=500)
    preserve_aspect: bool = True


class ConvertRequest(BaseModel):
    media_id: str
    target_format: str = "webp"
    quality: int = Field(default=90, ge=1, le=100)
    background_color: str = "#ffffff"
    preserve_metadata: bool = True
    preserve_animation: bool = True
    width: int | None = Field(default=None, ge=1, le=16384)
    height: int | None = Field(default=None, ge=1, le=16384)


class CompressRequest(BaseModel):
    media_id: str
    quality: int = Field(default=75, ge=1, le=100)
    target_kb: int | None = Field(default=None, ge=5, le=50000)
    target_format: str | None = "jpg"
    max_width: int | None = Field(default=None, ge=1, le=16384)
    max_height: int | None = Field(default=None, ge=1, le=16384)


class ImageOperationResponse(BaseModel):
    operation_id: str
    operation_type: str
    output_filename: str
    output_format: str
    width: int
    height: int
    original_size_bytes: int
    output_size_bytes: int
    reduction_percent: float
    target_achieved: bool
    animation_preserved: bool
    note: str | None = None
    download_url: str


def _resolve_source_asset(container: AppStateContainer, media_id: str) -> tuple[MediaAsset, Path]:
    with container.db.session_scope() as session:
        asset = session.get(MediaAsset, media_id)
        if asset is None:
            raise MediaNotFoundError(f"Media '{media_id}' not found.")
        session.expunge(asset)
    local_path = container.storage.resolve_local_path(asset.storage_key)
    if not local_path.is_file():
        raise MediaNotFoundError("Media file is missing from storage.")
    return asset, local_path


def _record_operation(
    container: AppStateContainer,
    *,
    op_id: str,
    media_id: str,
    op_type: str,
    settings_dict: dict[str, object],
    res: ImageOperationResult,
) -> ImageOperationResponse:
    with container.db.session_scope() as session:
        session.add(
            MediaOperation(
                id=op_id,
                source_media_id=media_id,
                operation_type=op_type,
                settings_json=json.dumps(settings_dict),
                output_filename=res.output_path.name,
                output_path=str(res.output_path),
                status="COMPLETED",
            )
        )
    return ImageOperationResponse(
        operation_id=op_id,
        operation_type=op_type,
        output_filename=res.output_path.name,
        output_format=res.output_format,
        width=res.width,
        height=res.height,
        original_size_bytes=res.original_size_bytes,
        output_size_bytes=res.output_size_bytes,
        reduction_percent=res.reduction_percent,
        target_achieved=res.target_achieved,
        animation_preserved=res.animation_preserved,
        note=res.note,
        download_url=f"{container.settings.api_prefix}/media/image/operations/{op_id}/download",
    )


@router.get("/{media_id}/metadata")
def get_image_metadata(
    media_id: str,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, object]:
    _, src_path = _resolve_source_asset(container, media_id)
    return asdict(inspect_image_metadata(src_path))


@router.post("/crop", response_model=ImageOperationResponse)
def api_crop_image(
    payload: CropRequest,
    container: AppStateContainer = Depends(get_container),
) -> ImageOperationResponse:
    asset, src_path = _resolve_source_asset(container, payload.media_id)
    op_id = str(uuid.uuid4())
    ext = asset.extension or ".png"
    out_path = container.settings.output_dir / "media_ops" / f"{op_id}_cropped{ext}"
    res = crop_image(
        src_path,
        out_path,
        x=payload.x,
        y=payload.y,
        width=payload.width,
        height=payload.height,
        rotate_deg=payload.rotate_deg,
        flip_h=payload.flip_h,
        flip_v=payload.flip_v,
    )
    return _record_operation(
        container,
        op_id=op_id,
        media_id=payload.media_id,
        op_type="crop",
        settings_dict=payload.model_dump(),
        res=res,
    )


@router.post("/resize", response_model=ImageOperationResponse)
def api_resize_image(
    payload: ResizeRequest,
    container: AppStateContainer = Depends(get_container),
) -> ImageOperationResponse:
    asset, src_path = _resolve_source_asset(container, payload.media_id)
    op_id = str(uuid.uuid4())
    ext = asset.extension or ".png"
    out_path = container.settings.output_dir / "media_ops" / f"{op_id}_resized{ext}"
    res = resize_image(
        src_path,
        out_path,
        width=payload.width,
        height=payload.height,
        max_width=payload.max_width,
        max_height=payload.max_height,
        scale_percent=payload.scale_percent,
        preserve_aspect=payload.preserve_aspect,
    )
    return _record_operation(
        container,
        op_id=op_id,
        media_id=payload.media_id,
        op_type="resize",
        settings_dict=payload.model_dump(),
        res=res,
    )


@router.post("/convert", response_model=ImageOperationResponse)
def api_convert_image(
    payload: ConvertRequest,
    container: AppStateContainer = Depends(get_container),
) -> ImageOperationResponse:
    _, src_path = _resolve_source_asset(container, payload.media_id)
    op_id = str(uuid.uuid4())
    clean_fmt = payload.target_format.lower().lstrip(".")
    out_path = container.settings.output_dir / "media_ops" / f"{op_id}_converted.{clean_fmt}"
    res = convert_image(
        src_path,
        out_path,
        target_format=clean_fmt,
        quality=payload.quality,
        background_color=payload.background_color,
        preserve_metadata=payload.preserve_metadata,
        preserve_animation=payload.preserve_animation,
        width=payload.width,
        height=payload.height,
    )
    return _record_operation(
        container,
        op_id=op_id,
        media_id=payload.media_id,
        op_type="convert",
        settings_dict=payload.model_dump(),
        res=res,
    )


@router.post("/compress", response_model=ImageOperationResponse)
def api_compress_image(
    payload: CompressRequest,
    container: AppStateContainer = Depends(get_container),
) -> ImageOperationResponse:
    _, src_path = _resolve_source_asset(container, payload.media_id)
    op_id = str(uuid.uuid4())
    clean_fmt = (payload.target_format or "jpg").lower().lstrip(".")
    out_path = container.settings.output_dir / "media_ops" / f"{op_id}_compressed.{clean_fmt}"
    res = compress_image(
        src_path,
        out_path,
        quality=payload.quality,
        target_kb=payload.target_kb,
        target_format=clean_fmt,
        max_width=payload.max_width,
        max_height=payload.max_height,
    )
    return _record_operation(
        container,
        op_id=op_id,
        media_id=payload.media_id,
        op_type="compress",
        settings_dict=payload.model_dump(),
        res=res,
    )


@router.get("/operations/{operation_id}/download")
def download_operation_output(
    operation_id: str,
    container: AppStateContainer = Depends(get_container),
) -> FileResponse:
    with container.db.session_scope() as session:
        op = session.get(MediaOperation, operation_id)
        if op is None:
            raise MediaNotFoundError(f"Operation '{operation_id}' not found.")
        out_path = Path(op.output_path).resolve()
        filename = op.output_filename

    ops_root = (container.settings.output_dir / "media_ops").resolve()
    if not out_path.is_relative_to(ops_root):
        raise StorageError("Invalid operation output path.")
    if not out_path.is_file():
        raise MediaNotFoundError("Operation output file not found.")
    return FileResponse(path=str(out_path), filename=filename)
