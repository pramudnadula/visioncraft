"""FastAPI dependency injection and container."""

from __future__ import annotations

from dataclasses import dataclass
from fastapi import Depends, Header, Request

from backend.app.core.config import Settings
from backend.app.core.exceptions import AuthenticationError
from backend.app.database.session import DatabaseManager
from backend.app.detection.registry import DetectorRegistry
from backend.app.jobs.events import JobEventBus
from backend.app.jobs.manager import JobManager
from backend.app.storage.base import StorageAdapter


@dataclass(slots=True)
class AppStateContainer:
    """Holds singleton services bound to FastAPI app lifespan."""

    settings: Settings
    db: DatabaseManager
    storage: StorageAdapter
    registry: DetectorRegistry
    event_bus: JobEventBus
    job_manager: JobManager


def get_container(request: Request) -> AppStateContainer:
    return request.app.state.container  # type: ignore[no-any-return]


def require_api_token(
    container: AppStateContainer = Depends(get_container),
    authorization: str | None = Header(default=None),
    x_facevision_token: str | None = Header(default=None),
) -> None:
    """Enforce local/cloud API token when `FACEVISION_API_TOKEN` is configured."""
    expected = container.settings.api_token
    if not expected:
        return
    token = x_facevision_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != expected:
        raise AuthenticationError("Invalid or missing FaceVision API token.")
