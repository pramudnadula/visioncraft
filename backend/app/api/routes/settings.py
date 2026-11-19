"""Application settings endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.database.models import AppSetting
from backend.app.schemas.api import SettingsResponse, SettingsUpdateRequest

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[Depends(require_api_token)])

SETTINGS_KEY = "user_preferences"


def _current_settings(container: AppStateContainer) -> SettingsResponse:
    s = container.settings
    base = {
        "default_model_id": s.default_model_id,
        "execution_mode": s.execution_mode,
        "default_blur_mode": s.default_blur_mode,
        "default_confidence_threshold": s.default_confidence_threshold,
        "allow_cloud_processing": s.allow_cloud_processing,
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
    current.update(updates)

    with container.db.session_scope() as session:
        row = session.get(AppSetting, SETTINGS_KEY)
        if row is None:
            row = AppSetting(key=SETTINGS_KEY, value_json=json.dumps(current))
            session.add(row)
        else:
            row.value_json = json.dumps(current)
            row.updated_at = datetime.now(UTC)

    # Reflect mutable runtime toggles
    if "allow_cloud_processing" in updates:
        container.settings.allow_cloud_processing = bool(updates["allow_cloud_processing"])
    if "retention_days" in updates:
        container.settings.retention_days = int(updates["retention_days"])

    return SettingsResponse(**current)
