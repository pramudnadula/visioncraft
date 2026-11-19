"""Plain-language PDF, CSV, and JSON report generator for FaceVision jobs."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fpdf import FPDF

ReportFormat = Literal["pdf", "csv", "json"]
ReportTemplate = Literal[
    "simple_summary",
    "detailed_technical",
    "business_report",
    "vehicle_analytics",
    "media_optimization",
]


def _build_plain_summary(job_type: str, model_id: str, faces_detected: int, manifest: dict[str, Any]) -> str:
    if job_type == "image":
        return (
            f"FaceVision analyzed the uploaded image using the '{model_id}' model "
            f"and detected {faces_detected} face(s)."
        )
    if job_type == "video":
        frames = manifest.get("frames_processed", 0)
        tracks = manifest.get("unique_track_count", 0)
        return (
            f"FaceVision processed {frames} video frame(s) using '{model_id}', recording "
            f"{faces_detected} total frame-level detections across {tracks} temporary track(s)."
        )
    if job_type == "batch":
        succeeded = manifest.get("succeeded", 0)
        total = manifest.get("total_items", 0)
        return (
            f"FaceVision batch-processed {succeeded} of {total} image(s) using '{model_id}', "
            f"detecting {faces_detected} face(s) in total."
        )
    return f"Completed {job_type} job using '{model_id}' with {faces_detected} detection(s)."


def generate_report_file(
    *,
    output_dir: Path,
    report_id: str,
    title: str,
    template: ReportTemplate,
    fmt: ReportFormat,
    job_id: str,
    job_type: str,
    model_id: str,
    faces_detected: int,
    manifest: dict[str, Any],
) -> Path:
    """Generate a PDF, CSV, or JSON report and return its output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    summary_text = _build_plain_summary(job_type, model_id, faces_detected, manifest)

    if fmt == "json":
        out_path = output_dir / f"report_{report_id}.json"
        payload = {
            "report_id": report_id,
            "title": title,
            "template": template,
            "generated_at": generated_at,
            "job_id": job_id,
            "job_type": job_type,
            "model_id": model_id,
            "summary": summary_text,
            "total_detections": faces_detected,
            "limitations": (
                "Temporary tracking IDs are session-only visual continuity markers and do not "
                "represent verified biometric identities."
            ),
            "details": manifest,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    if fmt == "csv":
        out_path = output_dir / f"report_{report_id}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["field", "value"])
            writer.writerow(["report_id", report_id])
            writer.writerow(["title", title])
            writer.writerow(["template", template])
            writer.writerow(["generated_at", generated_at])
            writer.writerow(["job_id", job_id])
            writer.writerow(["job_type", job_type])
            writer.writerow(["model_id", model_id])
            writer.writerow(["total_detections", faces_detected])
            writer.writerow(["summary", summary_text])
        return out_path

    # PDF generation via fpdf2
    out_path = output_dir / f"report_{report_id}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 110, 120)
    pdf.cell(
        0,
        7,
        f"Template: {template.replace('_', ' ').title()} | Generated: {generated_at} | Job: {job_id[:8]}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Processing Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Key Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for label, val in [
        ("Job Type", job_type),
        ("Detection Model", model_id),
        ("Total Detections", str(faces_detected)),
        ("Processing Duration (ms)", str(manifest.get("duration_ms", "N/A"))),
    ]:
        pdf.cell(0, 6, f"- {label}: {val}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Limitations & Privacy Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        "Automated detections depend on lighting, resolution, and model thresholds. "
        "Temporary tracking IDs are non-biometric session markers only.",
    )
    pdf.output(str(out_path))
    return out_path
