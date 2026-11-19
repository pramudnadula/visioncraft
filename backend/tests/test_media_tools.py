"""Tests for Phase B universal image editing, conversion, resizing, and compression."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.media.image.tools import (
    compress_image,
    convert_image,
    crop_image,
    inspect_image_metadata,
    resize_image,
)


def _create_rgba_png(path: Path, width: int = 400, height: int = 300) -> Path:
    img = Image.new("RGBA", (width, height), (30, 120, 220, 180))
    img.save(path, format="PNG")
    return path


def test_crop_resize_convert_and_compress_services(tmp_path: Path) -> None:
    src = _create_rgba_png(tmp_path / "src.png", 640, 480)
    meta = inspect_image_metadata(src)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.has_alpha is True

    cropped = crop_image(
        src,
        tmp_path / "cropped.png",
        x=50,
        y=50,
        width=200,
        height=150,
        rotate_deg=90,
        flip_h=True,
    )
    assert cropped.width == 200
    assert cropped.height == 150

    resized = resize_image(src, tmp_path / "resized.png", width=320, preserve_aspect=True)
    assert resized.width == 320
    assert resized.height == 240

    converted = convert_image(
        src,
        tmp_path / "converted.jpg",
        target_format="jpeg",
        quality=85,
        background_color="#ffffff",
    )
    assert converted.output_format == "JPEG"
    assert converted.output_path.is_file()

    compressed = compress_image(
        src,
        tmp_path / "compressed.jpg",
        target_kb=25,
        target_format="jpg",
    )
    assert compressed.output_path.is_file()
    assert compressed.output_size_bytes == compressed.output_path.stat().st_size


def test_media_tools_api_endpoints(tmp_path: Path) -> None:
    src = _create_rgba_png(tmp_path / "upload.png", 500, 400)
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    with TestClient(create_app(custom_settings=settings)) as client:
        up = client.post(
            "/api/v1/media/upload",
            files={"file": ("upload.png", src.read_bytes(), "image/png")},
        )
        assert up.status_code == 201
        media_id = up.json()["id"]

        meta_resp = client.get(f"/api/v1/media/image/{media_id}/metadata")
        assert meta_resp.status_code == 200
        assert meta_resp.json()["width"] == 500

        crop_resp = client.post(
            "/api/v1/media/image/crop",
            json={"media_id": media_id, "x": 10, "y": 10, "width": 120, "height": 120},
        )
        assert crop_resp.status_code == 200
        dl = client.get(crop_resp.json()["download_url"])
        assert dl.status_code == 200

        conv_resp = client.post(
            "/api/v1/media/image/convert",
            json={"media_id": media_id, "target_format": "webp", "quality": 80},
        )
        assert conv_resp.status_code == 200
        assert conv_resp.json()["output_format"] == "WEBP"

        comp_resp = client.post(
            "/api/v1/media/image/compress",
            json={"media_id": media_id, "target_kb": 50, "target_format": "jpg"},
        )
        assert comp_resp.status_code == 200
        assert comp_resp.json()["output_size_bytes"] > 0
