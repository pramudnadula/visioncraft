"""Batch image processing service with failure isolation and ZIP export."""

from __future__ import annotations

import csv
import json
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.core.exceptions import JobCancelledError
from backend.app.core.logging import get_logger
from backend.app.detection.base import FaceDetector
from backend.app.processing.image import ImageProcessingConfig, ImageProcessingService

logger = get_logger("processing.batch")


@dataclass(slots=True)
class BatchItemResult:
    """Outcome for a single item in a batch job."""

    media_id: str
    filename: str
    status: str  # "completed" | "failed"
    face_count: int = 0
    error: str | None = None
    output_dir: Path | None = None


@dataclass(slots=True)
class BatchProcessingResult:
    """Aggregated outcome for a batch processing job."""

    items: list[BatchItemResult] = field(default_factory=list)
    total_faces: int = 0
    succeeded: int = 0
    failed: int = 0
    zip_path: Path | None = None
    report_json_path: Path | None = None
    report_csv_path: Path | None = None


class BatchProcessingService:
    """Processes multiple images, isolating per-file errors and packaging a ZIP export."""

    def __init__(self) -> None:
        self._image_service = ImageProcessingService()

    def process(
        self,
        inputs: list[tuple[str, str, Path]],  # (media_id, display_name, file_path)
        detector: FaceDetector,
        output_dir: Path,
        config: ImageProcessingConfig | None = None,
        *,
        job_id: str | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> BatchProcessingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = BatchProcessingResult()
        total = len(inputs)

        for idx, (media_id, filename, file_path) in enumerate(inputs, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError("Batch job cancelled by user.")

            item_dir = output_dir / f"item_{idx}_{media_id[:8]}"
            try:
                img_res = self._image_service.process(
                    file_path,
                    detector,
                    item_dir,
                    config=config,
                    job_id=job_id,
                    input_name=filename,
                )
                face_count = len(img_res.detections)
                result.items.append(
                    BatchItemResult(
                        media_id=media_id,
                        filename=filename,
                        status="completed",
                        face_count=face_count,
                        output_dir=item_dir,
                    )
                )
                result.total_faces += face_count
                result.succeeded += 1
            except Exception as exc:
                logger.warning("Batch item %s failed: %s", media_id, exc)
                result.items.append(
                    BatchItemResult(
                        media_id=media_id,
                        filename=filename,
                        status="failed",
                        error=str(exc),
                    )
                )
                result.failed += 1

            if progress_callback is not None:
                progress_callback(idx, total, result.total_faces)

        report_json_path = output_dir / "batch_report.json"
        payload: dict[str, Any] = {
            "job_id": job_id,
            "total_items": total,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "total_faces": result.total_faces,
            "items": [
                {
                    "media_id": item.media_id,
                    "filename": item.filename,
                    "status": item.status,
                    "face_count": item.face_count,
                    "error": item.error,
                }
                for item in result.items
            ],
        }
        report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result.report_json_path = report_json_path

        report_csv_path = output_dir / "batch_summary.csv"
        with report_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["media_id", "filename", "status", "face_count", "error"])
            for item in result.items:
                writer.writerow(
                    [item.media_id, item.filename, item.status, item.face_count, item.error or ""]
                )
        result.report_csv_path = report_csv_path

        zip_path = output_dir / "batch_outputs.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(report_json_path, arcname="batch_report.json")
            zf.write(report_csv_path, arcname="batch_summary.csv")
            for item in result.items:
                if item.output_dir and item.output_dir.is_dir():
                    for file in item.output_dir.rglob("*"):
                        if file.is_file():
                            arcname = f"{item.output_dir.name}/{file.relative_to(item.output_dir)}"
                            zf.write(file, arcname=arcname)
        result.zip_path = zip_path
        return result
