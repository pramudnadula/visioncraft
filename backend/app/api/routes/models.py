"""Model discovery, HuggingFace ONNX search, download, and local import endpoints."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import MediaValidationError, StorageError
from backend.app.schemas.api import ModelInfoResponse

router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(require_api_token)])

ALLOWED_MODEL_EXTENSIONS = {".onnx", ".xml", ".prototxt", ".caffemodel"}
MAX_MODEL_DOWNLOAD_BYTES = 250 * 1024 * 1024


class ModelSearchResult(BaseModel):
    id: str
    downloads: int
    likes: int
    pipeline_tag: str | None = None
    url: str


class ModelDownloadRequest(BaseModel):
    url: str
    filename: str | None = None


class CustomModelFile(BaseModel):
    filename: str
    size_bytes: int


def _sanitize_model_filename(name: str) -> str:
    base = Path(name).name
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not clean or Path(clean).suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
        raise MediaValidationError(
            f"Unsupported model file extension. Allowed: {', '.join(sorted(ALLOWED_MODEL_EXTENSIONS))}"
        )
    return clean


def _validate_public_https_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise MediaValidationError("Model download URL must use https://.")
    try:
        ip_str = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast:
            raise MediaValidationError("Model download URL cannot point to internal network addresses.")
    except socket.gaierror as exc:
        raise MediaValidationError(f"Could not resolve host '{parsed.hostname}'.") from exc
    return url.strip()


@router.get("", response_model=list[ModelInfoResponse])
def list_models(container: AppStateContainer = Depends(get_container)) -> list[ModelInfoResponse]:
    models = container.registry.list_models()
    return [ModelInfoResponse(**asdict(m)) for m in models]


@router.get("/custom", response_model=list[CustomModelFile])
def list_custom_models(container: AppStateContainer = Depends(get_container)) -> list[CustomModelFile]:
    custom_dir = container.settings.resolved_models_dir / "custom"
    if not custom_dir.is_dir():
        return []
    return [
        CustomModelFile(filename=p.name, size_bytes=p.stat().st_size)
        for p in sorted(custom_dir.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_MODEL_EXTENSIONS
    ]


@router.get("/search", response_model=list[ModelSearchResult])
def search_huggingface_models(
    q: str = Query(default="face detection", min_length=2, max_length=100),
) -> list[ModelSearchResult]:
    """Search HuggingFace Hub for ONNX deep learning models."""
    encoded = urllib.parse.urlencode({"search": q, "filter": "onnx", "limit": 12})
    api_url = f"https://huggingface.co/api/models?{encoded}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "FaceVision/2.1"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise StorageError(f"Model search failed: {exc}") from exc

    results: list[ModelSearchResult] = []
    for item in items:
        repo_id = str(item.get("id", ""))
        if not repo_id:
            continue
        results.append(
            ModelSearchResult(
                id=repo_id,
                downloads=int(item.get("downloads", 0) or 0),
                likes=int(item.get("likes", 0) or 0),
                pipeline_tag=item.get("pipeline_tag"),
                url=f"https://huggingface.co/{repo_id}",
            )
        )
    return results


@router.post("/import", response_model=CustomModelFile, status_code=201)
async def import_model_file(
    file: UploadFile = File(...),
    container: AppStateContainer = Depends(get_container),
) -> CustomModelFile:
    """Import a local .onnx/.xml/.caffemodel file into models/custom/."""
    filename = _sanitize_model_filename(file.filename or "model.onnx")
    data = await file.read()
    if not data or len(data) > MAX_MODEL_DOWNLOAD_BYTES:
        raise MediaValidationError("Model file is empty or exceeds 250 MB limit.")

    custom_dir = container.settings.resolved_models_dir / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    target = (custom_dir / filename).resolve()
    if not target.is_relative_to(custom_dir.resolve()):
        raise StorageError("Invalid model filename.")
    target.write_bytes(data)
    return CustomModelFile(filename=filename, size_bytes=len(data))


@router.post("/download", response_model=CustomModelFile, status_code=201)
def download_model_from_url(
    payload: ModelDownloadRequest,
    container: AppStateContainer = Depends(get_container),
) -> CustomModelFile:
    """Download an ONNX/model file from a verified HTTPS URL into models/custom/."""
    safe_url = _validate_public_https_url(payload.url)
    inferred_name = payload.filename or Path(urllib.parse.urlparse(safe_url).path).name or "model.onnx"
    filename = _sanitize_model_filename(inferred_name)

    custom_dir = container.settings.resolved_models_dir / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    target = (custom_dir / filename).resolve()
    if not target.is_relative_to(custom_dir.resolve()):
        raise StorageError("Invalid model filename.")

    req = urllib.request.Request(safe_url, headers={"User-Agent": "FaceVision/2.1"})
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, target.open("wb") as out_fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MODEL_DOWNLOAD_BYTES:
                    out_fh.close()
                    target.unlink(missing_ok=True)
                    raise MediaValidationError("Remote model exceeds 250 MB download limit.")
                out_fh.write(chunk)
    except MediaValidationError:
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise StorageError(f"Failed to download model: {exc}") from exc

    return CustomModelFile(filename=filename, size_bytes=total)
