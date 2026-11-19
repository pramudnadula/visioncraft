"""Model discovery and metadata endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.schemas.api import ModelInfoResponse

router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(require_api_token)])


@router.get("", response_model=list[ModelInfoResponse])
def list_models(container: AppStateContainer = Depends(get_container)) -> list[ModelInfoResponse]:
    models = container.registry.list_models()
    return [ModelInfoResponse(**asdict(m)) for m in models]
