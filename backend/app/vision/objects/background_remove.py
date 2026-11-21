"""AI-powered image background removal using U2-Net-P ONNX and MediaPipe segmentation."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageColor

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.media.image.tools import ImageOperationResult

BgModelType = Literal["universal", "portrait"]
BgOutputMode = Literal["transparent", "color", "blur"]

U2NETP_FILENAME = "u2netp.onnx"
U2NETP_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"

_LOCK = threading.Lock()
_U2NET_CACHE: dict[Path, Any] = {}


def resolve_u2net_path(models_dir: Path | None = None) -> Path | None:
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for directory in (base, base / "custom"):
        candidate = directory / U2NETP_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _get_u2net_session(models_dir: Path) -> Any:
    if importlib.util.find_spec("onnxruntime") is None:
        raise ModelNotAvailableError("Install 'onnxruntime' to run U2-Net-P background removal.")
    path = resolve_u2net_path(models_dir)
    if path is None:
        raise ModelNotAvailableError(
            f"Place {U2NETP_FILENAME} in models/ to enable universal AI background removal."
        )
    with _LOCK:
        cached = _U2NET_CACHE.get(path)
        if cached is not None:
            return cached
        import onnxruntime as ort  # type: ignore[import-not-found]

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _U2NET_CACHE[path] = session
        return session


def _predict_u2net_alpha(rgb: np.ndarray, models_dir: Path) -> np.ndarray:
    """Run U2-Net-P (320x320 ImageNet-normalized) and return float32 alpha matte in [0, 1]."""
    session = _get_u2net_session(models_dir)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (resized - mean) / std
    blob = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    pred = np.squeeze(session.run(None, {input_name: blob})[0]).astype(np.float32)
    p_min, p_max = float(pred.min()), float(pred.max())
    if p_max - p_min > 1e-6:
        pred = (pred - p_min) / (p_max - p_min)
    else:
        pred = np.zeros_like(pred)
    alpha = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    # Soft edge refinement to clean halo artifacts
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.asarray(np.clip(alpha, 0.0, 1.0), dtype=np.float32)


def _predict_portrait_alpha(rgb: np.ndarray, models_dir: Path) -> np.ndarray:
    """Run MediaPipe Selfie Multiclass Segmenter for foreground person vs background."""
    from backend.app.vision.faces.segmentation import _get_segmenter

    segmenter = _get_segmenter(models_dir)
    if segmenter is None:
        raise ModelNotAvailableError(
            "MediaPipe portrait segmenter (selfie_multiclass_256x256.tflite) is unavailable."
        )
    import mediapipe as mp

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _LOCK:
        res = segmenter.segment(mp_img)
    if res.category_mask is None:
        return np.ones(rgb.shape[:2], dtype=np.float32)
    cat = np.asarray(res.category_mask.numpy_view())
    if cat.ndim == 3:
        cat = cat[..., 0]
    fg: np.ndarray = (cat > 0).astype(np.float32)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    return np.asarray(np.clip(fg, 0.0, 1.0), dtype=np.float32)


def remove_background_pil(
    input_path: Path,
    *,
    models_dir: Path | None = None,
    model_type: BgModelType = "universal",
    background_mode: BgOutputMode = "transparent",
    background_color: str = "#ffffff",
) -> Image.Image:
    """Remove or replace image background and return a PIL Image."""
    resolved = models_dir if models_dir is not None else get_settings().resolved_models_dir
    with Image.open(input_path) as raw:
        raw.load()
        rgb_img = raw.convert("RGB")
    rgb = np.array(rgb_img)

    if model_type == "portrait":
        alpha = _predict_portrait_alpha(rgb, resolved)
    else:
        alpha = _predict_u2net_alpha(rgb, resolved)

    if background_mode == "transparent":
        rgba = np.dstack([rgb, np.round(alpha * 255.0).astype(np.uint8)])
        return Image.fromarray(rgba)

    if background_mode == "blur":
        k = max(21, (min(rgb.shape[0], rgb.shape[1]) // 18) | 1)
        bg = cv2.GaussianBlur(rgb, (k, k), 0).astype(np.float32)
    else:
        r, g, b = ImageColor.getrgb(background_color)[:3]
        bg = np.full_like(rgb, (r, g, b), dtype=np.float32)

    a3 = alpha[:, :, None]
    comp = np.clip(rgb.astype(np.float32) * a3 + bg * (1.0 - a3), 0, 255).astype(np.uint8)
    return Image.fromarray(comp)


def remove_background_to_file(
    input_path: Path,
    output_path: Path,
    *,
    models_dir: Path | None = None,
    model_type: BgModelType = "universal",
    background_mode: BgOutputMode = "transparent",
    background_color: str = "#ffffff",
) -> ImageOperationResult:
    """Remove background and save the result to ``output_path``."""
    orig_size = input_path.stat().st_size
    out_img = remove_background_pil(
        input_path,
        models_dir=models_dir,
        model_type=model_type,
        background_mode=background_mode,
        background_color=background_color,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "PNG" if background_mode == "transparent" else "PNG"
    out_img.save(output_path, format=fmt, optimize=True)
    out_size = output_path.stat().st_size
    return ImageOperationResult(
        output_path=output_path,
        output_format=fmt,
        width=out_img.width,
        height=out_img.height,
        original_size_bytes=orig_size,
        output_size_bytes=out_size,
        reduction_percent=round(((orig_size - out_size) / orig_size) * 100.0, 2) if orig_size else 0.0,
        note=f"Background removed ({model_type}, {background_mode}).",
    )
