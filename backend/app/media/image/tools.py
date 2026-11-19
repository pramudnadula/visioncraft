"""Universal Pillow-based image crop, resize, convert, compress, and metadata service."""

from __future__ import annotations

import base64
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageOps

from backend.app.core.exceptions import MediaValidationError

if importlib.util.find_spec("pillow_heif") is not None:
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()

SUPPORTED_FORMATS: dict[str, str] = {
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tiff": "TIFF",
    "tif": "TIFF",
    "gif": "GIF",
    "ico": "ICO",
    "heic": "HEIF",
    "heif": "HEIF",
}


@dataclass(frozen=True, slots=True)
class ImageMetadataInfo:
    format: str
    mode: str
    width: int
    height: int
    size_bytes: int
    is_animated: bool
    frame_count: int
    has_alpha: bool
    exif_keys: list[str]


@dataclass(frozen=True, slots=True)
class ImageOperationResult:
    output_path: Path
    output_format: str
    width: int
    height: int
    original_size_bytes: int
    output_size_bytes: int
    reduction_percent: float
    target_achieved: bool = True
    animation_preserved: bool = False
    note: str | None = None


def _open_image(input_path: Path) -> Image.Image:
    if not input_path.is_file():
        raise MediaValidationError("Source image file not found.")
    try:
        img = Image.open(input_path)
        img.load()
        return ImageOps.exif_transpose(img)
    except Exception as exc:
        raise MediaValidationError(f"Failed to open image: {exc}") from exc


def _flatten_alpha(img: Image.Image, bg_hex: str) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        rgb_bg = ImageColor.getrgb(bg_hex)
        canvas = Image.new("RGB", rgba.size, rgb_bg[:3])
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def inspect_image_metadata(input_path: Path) -> ImageMetadataInfo:
    img = _open_image(input_path)
    exif = img.getexif()
    exif_keys = [str(k) for k in exif.keys()] if exif else []
    n_frames = int(getattr(img, "n_frames", 1) or 1)
    return ImageMetadataInfo(
        format=(img.format or input_path.suffix.lstrip(".")).upper(),
        mode=img.mode,
        width=img.width,
        height=img.height,
        size_bytes=input_path.stat().st_size,
        is_animated=n_frames > 1,
        frame_count=n_frames,
        has_alpha=img.mode in ("RGBA", "LA") or "transparency" in img.info,
        exif_keys=exif_keys,
    )


def crop_image(
    input_path: Path,
    output_path: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    rotate_deg: int = 0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ImageOperationResult:
    orig_size = input_path.stat().st_size
    img = _open_image(input_path)
    if rotate_deg % 360 != 0:
        img = img.rotate(-rotate_deg, expand=True)
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)

    x1 = max(0, min(img.width - 1, x))
    y1 = max(0, min(img.height - 1, y))
    x2 = max(x1 + 1, min(img.width, x1 + max(1, width)))
    y2 = max(y1 + 1, min(img.height, y1 + max(1, height)))
    cropped = img.crop((x1, y1, x2, y2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = SUPPORTED_FORMATS.get(output_path.suffix.lstrip(".").lower(), img.format or "PNG")
    save_img = _flatten_alpha(cropped, "#ffffff") if fmt in ("JPEG", "BMP") else cropped
    save_img.save(output_path, format=fmt, quality=95)
    out_size = output_path.stat().st_size
    reduction = round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size > 0 else 0.0
    return ImageOperationResult(
        output_path=output_path,
        output_format=fmt,
        width=cropped.width,
        height=cropped.height,
        original_size_bytes=orig_size,
        output_size_bytes=out_size,
        reduction_percent=reduction,
    )


def resize_image(
    input_path: Path,
    output_path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    scale_percent: float | None = None,
    preserve_aspect: bool = True,
) -> ImageOperationResult:
    orig_size = input_path.stat().st_size
    img = _open_image(input_path)
    target_w, target_h = img.width, img.height

    if scale_percent is not None and scale_percent > 0:
        target_w = max(1, round(img.width * (scale_percent / 100.0)))
        target_h = max(1, round(img.height * (scale_percent / 100.0)))
    elif width is not None and height is not None:
        if preserve_aspect:
            ratio = min(width / img.width, height / img.height)
            target_w = max(1, round(img.width * ratio))
            target_h = max(1, round(img.height * ratio))
        else:
            target_w, target_h = max(1, width), max(1, height)
    elif width is not None:
        target_w = max(1, width)
        target_h = max(1, round(img.height * (target_w / img.width))) if preserve_aspect else img.height
    elif height is not None:
        target_h = max(1, height)
        target_w = max(1, round(img.width * (target_h / img.height))) if preserve_aspect else img.width

    if max_width is not None and target_w > max_width:
        ratio = max_width / target_w
        target_w = max_width
        target_h = max(1, round(target_h * ratio))
    if max_height is not None and target_h > max_height:
        ratio = max_height / target_h
        target_h = max_height
        target_w = max(1, round(target_w * ratio))

    resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = SUPPORTED_FORMATS.get(output_path.suffix.lstrip(".").lower(), img.format or "PNG")
    save_img = _flatten_alpha(resized, "#ffffff") if fmt in ("JPEG", "BMP") else resized
    save_img.save(output_path, format=fmt, quality=92)
    out_size = output_path.stat().st_size
    reduction = round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size > 0 else 0.0
    return ImageOperationResult(
        output_path=output_path,
        output_format=fmt,
        width=resized.width,
        height=resized.height,
        original_size_bytes=orig_size,
        output_size_bytes=out_size,
        reduction_percent=reduction,
    )


def convert_image(
    input_path: Path,
    output_path: Path,
    *,
    target_format: str,
    quality: int = 90,
    background_color: str = "#ffffff",
    preserve_metadata: bool = True,
    preserve_animation: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> ImageOperationResult:
    fmt = SUPPORTED_FORMATS.get(target_format.lower())
    if fmt is None:
        raise MediaValidationError(f"Unsupported target format: '{target_format}'.")

    orig_size = input_path.stat().st_size
    raw_img = Image.open(input_path)
    n_frames = int(getattr(raw_img, "n_frames", 1) or 1)
    is_animated = n_frames > 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_animated and preserve_animation and fmt in ("GIF", "WEBP"):
        frames: list[Image.Image] = []
        durations: list[int] = []
        for idx in range(n_frames):
            raw_img.seek(idx)
            frame = raw_img.copy()
            if width and height:
                frame = frame.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(frame)
            durations.append(int(raw_img.info.get("duration", 100)))
        frames[0].save(
            output_path,
            format=fmt,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=int(raw_img.info.get("loop", 0)),
        )
        out_size = output_path.stat().st_size
        return ImageOperationResult(
            output_path=output_path,
            output_format=fmt,
            width=frames[0].width,
            height=frames[0].height,
            original_size_bytes=orig_size,
            output_size_bytes=out_size,
            reduction_percent=round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size else 0.0,
            animation_preserved=True,
        )

    img = _open_image(input_path)
    if width and height:
        img = img.resize((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
    if fmt == "ICO" and (img.width > 256 or img.height > 256):
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)

    save_img = _flatten_alpha(img, background_color) if fmt in ("JPEG", "BMP") else img
    save_kwargs: dict[str, Any] = {"format": fmt}
    if fmt in ("JPEG", "WEBP", "HEIF"):
        save_kwargs["quality"] = max(1, min(100, quality))
    if preserve_metadata and "exif" in raw_img.info and fmt in ("JPEG", "WEBP", "TIFF", "HEIF"):
        save_kwargs["exif"] = raw_img.info["exif"]

    save_img.save(output_path, **save_kwargs)
    out_size = output_path.stat().st_size
    note = (
        "Source image was animated; converted to a static frame."
        if is_animated and fmt not in ("GIF", "WEBP")
        else None
    )
    return ImageOperationResult(
        output_path=output_path,
        output_format=fmt,
        width=save_img.width,
        height=save_img.height,
        original_size_bytes=orig_size,
        output_size_bytes=out_size,
        reduction_percent=round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size else 0.0,
        animation_preserved=False,
        note=note,
    )


def _encode_to_bytes(img: Image.Image, fmt: str, quality: int, background_color: str = "#ffffff") -> bytes:
    buf = io.BytesIO()
    working = img
    if fmt == "ICO" and (working.width > 256 or working.height > 256):
        working = working.copy()
        working.thumbnail((256, 256), Image.Resampling.LANCZOS)
    save_img = _flatten_alpha(working, background_color) if fmt in ("JPEG", "BMP") else working
    kwargs: dict[str, Any] = {"format": fmt}
    if fmt in ("JPEG", "PNG", "WEBP"):
        kwargs["optimize"] = True
    if fmt in ("JPEG", "WEBP", "HEIF"):
        kwargs["quality"] = quality
    save_img.save(buf, **kwargs)
    return buf.getvalue()


def encode_preview_data_url(img: Image.Image, fmt: str, encoded_bytes: bytes) -> str:
    """Return a browser-renderable data URL for live preview (using PNG/JPEG fallback for HEIF/TIFF/BMP)."""
    if fmt in ("JPEG", "PNG", "WEBP", "GIF"):
        mime = f"image/{fmt.lower()}"
        return f"data:{mime};base64,{base64.b64encode(encoded_bytes).decode('ascii')}"
    # Re-open or render browser-safe preview for formats browsers may not decode natively (HEIC/TIFF/BMP/ICO)
    preview_img = Image.open(io.BytesIO(encoded_bytes))
    preview_img.load()
    preview_img.thumbnail((960, 960), Image.Resampling.LANCZOS)
    pbuf = io.BytesIO()
    preview_fmt = "PNG" if preview_img.mode in ("RGBA", "LA", "PA") else "JPEG"
    preview_img.save(pbuf, format=preview_fmt, quality=88)
    mime = f"image/{preview_fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(pbuf.getvalue()).decode('ascii')}"


def compress_image(
    input_path: Path,
    output_path: Path,
    *,
    quality: int = 75,
    target_kb: int | None = None,
    target_format: str | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
) -> ImageOperationResult:
    orig_size = input_path.stat().st_size
    img = _open_image(input_path)
    if max_width or max_height:
        mw = max_width or img.width
        mh = max_height or img.height
        img.thumbnail((mw, mh), Image.Resampling.LANCZOS)

    ext_fmt = SUPPORTED_FORMATS.get(
        (target_format or output_path.suffix.lstrip(".") or "jpg").lower(),
        "JPEG",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target_kb is None:
        encoded = _encode_to_bytes(img, ext_fmt, max(1, min(100, quality)))
        output_path.write_bytes(encoded)
        out_size = len(encoded)
        return ImageOperationResult(
            output_path=output_path,
            output_format=ext_fmt,
            width=img.width,
            height=img.height,
            original_size_bytes=orig_size,
            output_size_bytes=out_size,
            reduction_percent=round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size else 0.0,
        )

    target_bytes = max(1024, target_kb * 1024)
    best_bytes = _encode_to_bytes(img, ext_fmt, 85)
    working = img.copy()
    achieved = len(best_bytes) <= target_bytes

    for _scale_attempt in range(5):
        lo, hi = 15, 95
        candidate_best: bytes | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = _encode_to_bytes(working, ext_fmt, mid)
            if len(trial) <= target_bytes:
                candidate_best = trial
                lo = mid + 1
            else:
                hi = mid - 1
        if candidate_best is not None:
            best_bytes = candidate_best
            achieved = True
            break
        # Downscale by 80% if quality reduction alone wasn't enough
        best_bytes = _encode_to_bytes(working, ext_fmt, 15)
        if working.width < 120 or working.height < 120:
            break
        working = working.resize(
            (max(1, int(working.width * 0.8)), max(1, int(working.height * 0.8))),
            Image.Resampling.LANCZOS,
        )

    output_path.write_bytes(best_bytes)
    out_size = len(best_bytes)
    note = (
        None
        if achieved
        else (
            f"Target {target_kb} KB could not be reached without excessive quality loss; "
            f"achieved {round(out_size / 1024, 1)} KB."
        )
    )
    return ImageOperationResult(
        output_path=output_path,
        output_format=ext_fmt,
        width=working.width,
        height=working.height,
        original_size_bytes=orig_size,
        output_size_bytes=out_size,
        reduction_percent=round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size else 0.0,
        target_achieved=achieved,
        note=note,
    )
