"""Processing job submission, status, cancellation, retry, and artifact download endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import JobNotFoundError, JobStateError, StorageError
from backend.app.database.models import ProcessingJob
from backend.app.schemas.api import JobCreateRequest, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_api_token)])


def _job_to_response(job: ProcessingJob) -> JobResponse:
    return JobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        model_id=job.model_id,
        processing_mode=job.processing_mode,
        input_media_ids=json.loads(job.input_media_ids_json),
        faces_detected=job.faces_detected,
        error_message=job.error_message,
        output_manifest=json.loads(job.output_manifest_json) if job.output_manifest_json else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("", response_model=JobResponse, status_code=201)
def create_job(
    payload: JobCreateRequest,
    container: AppStateContainer = Depends(get_container),
) -> JobResponse:
    config_dict = payload.model_dump(
        exclude={"job_type", "media_ids", "model_id", "processing_mode", "cloud_consent"}
    )
    job = container.job_manager.create_job(
        job_type=payload.job_type,
        media_ids=payload.media_ids,
        model_id=payload.model_id,
        processing_mode=payload.processing_mode,
        cloud_consent=payload.cloud_consent,
        config=config_dict,
    )
    return _job_to_response(job)


@router.get("", response_model=list[JobResponse])
def list_jobs(
    status: str | None = None,
    limit: int = 50,
    container: AppStateContainer = Depends(get_container),
) -> list[JobResponse]:
    jobs = container.job_manager.list_jobs(status=status, limit=min(limit, 200))
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    container: AppStateContainer = Depends(get_container),
) -> JobResponse:
    job = container.job_manager.get_job(job_id)
    return _job_to_response(job)


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    container: AppStateContainer = Depends(get_container),
) -> JobResponse:
    job = container.job_manager.cancel_job(job_id)
    return _job_to_response(job)


@router.post("/{job_id}/retry", response_model=JobResponse)
def retry_job(
    job_id: str,
    container: AppStateContainer = Depends(get_container),
) -> JobResponse:
    job = container.job_manager.retry_job(job_id)
    return _job_to_response(job)


@router.delete("/{job_id}", status_code=204)
def delete_job(
    job_id: str,
    container: AppStateContainer = Depends(get_container),
) -> None:
    container.job_manager.delete_job(job_id)


@router.get("/{job_id}/result")
def get_job_result(
    job_id: str,
    container: AppStateContainer = Depends(get_container),
) -> dict[str, Any]:
    job = container.job_manager.get_job(job_id)
    if job.status != "COMPLETED" or not job.output_manifest_json:
        raise JobStateError(f"Job '{job_id}' has no completed result (status: {job.status}).")
    manifest: dict[str, Any] = json.loads(job.output_manifest_json)
    return manifest


@router.get("/{job_id}/artifacts/{artifact_path:path}")
def download_job_artifact(
    job_id: str,
    artifact_path: str,
    container: AppStateContainer = Depends(get_container),
) -> FileResponse:
    container.job_manager.get_job(job_id)
    job_output_root = (container.settings.output_dir / job_id).resolve()
    target = (job_output_root / artifact_path).resolve()
    if not target.is_relative_to(job_output_root):
        raise StorageError("Invalid artifact path.")
    if not target.is_file():
        raise JobNotFoundError(f"Artifact '{artifact_path}' not found for job '{job_id}'.")
    return FileResponse(path=str(target), filename=target.name)
