"""Report generation, listing, and download endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.core.dependencies import AppStateContainer, get_container, require_api_token
from backend.app.core.exceptions import JobStateError, MediaNotFoundError, StorageError
from backend.app.database.models import GeneratedReport
from backend.app.reports.generator import ReportFormat, ReportTemplate, generate_report_file

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
    dependencies=[Depends(require_api_token)],
)


class GenerateReportRequest(BaseModel):
    job_id: str
    title: str = "FaceVision Analysis Report"
    template: ReportTemplate = "simple_summary"
    format: ReportFormat = "pdf"


class ReportResponse(BaseModel):
    id: str
    job_id: str
    title: str
    report_type: str
    format: str
    created_at: datetime
    download_url: str


@router.post("/generate", response_model=ReportResponse, status_code=201)
def create_report(
    payload: GenerateReportRequest,
    container: AppStateContainer = Depends(get_container),
) -> ReportResponse:
    job = container.job_manager.get_job(payload.job_id)
    if job.status != "COMPLETED" or not job.output_manifest_json:
        raise JobStateError("Reports can only be generated for COMPLETED jobs.")

    manifest = json.loads(job.output_manifest_json)
    report_id = str(uuid.uuid4())
    reports_dir = container.settings.output_dir / "reports"
    out_path = generate_report_file(
        output_dir=reports_dir,
        report_id=report_id,
        title=payload.title,
        template=payload.template,
        fmt=payload.format,
        job_id=job.id,
        job_type=job.job_type,
        model_id=job.model_id,
        faces_detected=job.faces_detected,
        manifest=manifest,
    )

    with container.db.session_scope() as session:
        row = GeneratedReport(
            id=report_id,
            job_id=job.id,
            title=payload.title,
            report_type=payload.template,
            format=payload.format,
            output_path=str(out_path),
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        created_at = row.created_at

    return ReportResponse(
        id=report_id,
        job_id=job.id,
        title=payload.title,
        report_type=payload.template,
        format=payload.format,
        created_at=created_at,
        download_url=f"{container.settings.api_prefix}/reports/{report_id}/download",
    )


@router.get("", response_model=list[ReportResponse])
def list_reports(container: AppStateContainer = Depends(get_container)) -> list[ReportResponse]:
    with container.db.session_scope() as session:
        rows = session.query(GeneratedReport).order_by(GeneratedReport.created_at.desc()).limit(100).all()
        return [
            ReportResponse(
                id=r.id,
                job_id=r.job_id,
                title=r.title,
                report_type=r.report_type,
                format=r.format,
                created_at=r.created_at,
                download_url=f"{container.settings.api_prefix}/reports/{r.id}/download",
            )
            for r in rows
        ]


@router.get("/{report_id}/download")
def download_report(
    report_id: str,
    container: AppStateContainer = Depends(get_container),
) -> FileResponse:
    with container.db.session_scope() as session:
        row = session.get(GeneratedReport, report_id)
        if row is None:
            raise MediaNotFoundError(f"Report '{report_id}' not found.")
        target = Path(row.output_path).resolve()

    reports_root = (container.settings.output_dir / "reports").resolve()
    if not target.is_relative_to(reports_root):
        raise StorageError("Invalid report path.")
    if not target.is_file():
        raise MediaNotFoundError("Report file not found.")
    return FileResponse(path=str(target), filename=target.name)
