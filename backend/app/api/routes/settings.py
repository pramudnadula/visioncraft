"""Application settings and cloud connectivity endpoints."""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from datetime import UTC, datetime
from urllib.parse import urlparse

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import MediaValidationError
from backend.app.database.models import AppSetting
from backend.app.schemas.api import (
    CloudTestRequest,
    CloudTestResponse,
    SettingsResponse,
    SettingsUpdateRequest,
)

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[Depends(require_api_token)])

SETTINGS_KEY = "user_preferences"


def _validate_cloud_url(url: str) -> str:
    """Validate cloud URL scheme and block SSRF to link-local metadata endpoints."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MediaValidationError("Cloud API URL must be a valid http:// or https:// URL.")
    try:
        resolved_ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(resolved_ip)
        if ip_obj.is_link_local or ip_obj.is_multicast:
            raise MediaValidationError("Cloud API URL cannot point to link-local or multicast addresses.")
    except socket.gaierror:
        pass
    return url.strip().rstrip("/")


def _current_settings(container: AppStateContainer) -> SettingsResponse:
    s = container.settings
    base = {
        "default_model_id": s.default_model_id,
        "execution_mode": s.execution_mode,
        "default_blur_mode": s.default_blur_mode,
        "default_confidence_threshold": s.default_confidence_threshold,
        "allow_cloud_processing": s.allow_cloud_processing,
        "cloud_api_url": s.cloud_api_url,
        "retention_days": s.retention_days,
    }
    with container.db.session_scope() as session:
        row = session.get(AppSetting, SETTINGS_KEY)
        if row is not None:
            saved = json.loads(row.value_json)
            base.update(saved)
    return SettingsResponse.model_validate(base)


@router.get("", response_model=SettingsResponse)
def get_user_settings(container: AppStateContainer = Depends(get_container)) -> SettingsResponse:
    return _current_settings(container)


@router.put("", response_model=SettingsResponse)
def update_user_settings(
    payload: SettingsUpdateRequest,
    container: AppStateContainer = Depends(get_container),
) -> SettingsResponse:
    current = _current_settings(container).model_dump()
    updates = payload.model_dump(exclude_none=True)
    if "cloud_api_url" in updates and updates["cloud_api_url"]:
        updates["cloud_api_url"] = _validate_cloud_url(str(updates["cloud_api_url"]))
    current.update(updates)

    with container.db.session_scope() as session:
        row = session.get(AppSetting, SETTINGS_KEY)
        if row is None:
            row = AppSetting(key=SETTINGS_KEY, value_json=json.dumps(current))
            session.add(row)
        else:
            row.value_json = json.dumps(current)
            row.updated_at = datetime.now(UTC)

    if "allow_cloud_processing" in updates:
        container.settings.allow_cloud_processing = bool(updates["allow_cloud_processing"])
    if "cloud_api_url" in updates:
        container.settings.cloud_api_url = updates["cloud_api_url"]
    if "retention_days" in updates:
        container.settings.retention_days = int(updates["retention_days"])

    return SettingsResponse.model_validate(current)


@router.post("/test-cloud", response_model=CloudTestResponse)
def test_cloud_connection(payload: CloudTestRequest) -> CloudTestResponse:
    """Verify connectivity to a configured remote FaceVision server."""
    base_url = _validate_cloud_url(payload.cloud_api_url)
    health_url = f"{base_url}/api/v1/health"
    req = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.getcode()
            return CloudTestResponse(
                reachable=status == 200,
                status_code=status,
                message=f"Connected to remote FaceVision server (HTTP {status}).",
            )
    except urllib.error.HTTPError as exc:
        return CloudTestResponse(
            reachable=False,
            status_code=exc.code,
            message=f"Remote server returned HTTP {exc.code}.",
        )
    except Exception as exc:
        return CloudTestResponse(
            reachable=False,
            status_code=None,
            message=f"Connection failed: {exc}",
        )
