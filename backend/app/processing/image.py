"""Image processing service for FaceVision."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from backend.app.core.exceptions import MediaValidationError
from backend.app.core.logging import get_logger
from backend.app.detection.base import Detection, DetectorMetadata, FaceDetector
from backend.app.privacy.blur import BlurMode, apply_privacy_blur, draw_detections

logger = get_logger("processing.image")


@dataclass(slots=True)
class ImageProcessingConfig:
    """Options controlling image face detection, privacy blur, and exports."""

    blur_mode: BlurMode = "none"
    selected_face_indices: list[int] | None = None
    save_crops: bool = False
    save_privacy_image: bool = False
    show_hud: bool = True
    draw_boxes: bool = True
    box_color: tuple[int, int, int] = (0, 220, 120)
    box_thickness: int = 2
    max_dimension: int = 16384


@dataclass(slots=True)
class ImageProcessingResult:
    """Outputs and metadata produced by processing an image."""

    detections: list[Detection]
    width: int
    height: int
    duration_ms: float
    model_metadata: DetectorMetadata
    annotated_path: Path | None = None
    blurred_path: Path | None = None
    crop_paths: list[Path] = field(default_factory=list)
    report_json_path: Path | None = None
    report_csv_path: Path | None = None

    def to_report_dict(self, *, job_id: str | None = None, input_name: str = "") -> dict[str, Any]:
        return {
            "job_id": job_id,
            "input": {"filename": input_name, "width": self.width, "height": self.height},
            "model": {
                "id": self.model_metadata.id,
                "name": self.model_metadata.name,
                "version": self.model_metadata.version,
            },
            "detection_count": len(self.detections),
            "processing_duration_ms": round(self.duration_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
        }


class ImageProcessingService:
    """Stateless service for single-image face detection and privacy export."""

    def process(
        self,
        input_path: Path,
        detector: FaceDetector,
        output_dir: Path,
        config: ImageProcessingConfig | None = None,
        *,
        job_id: str | None = None,
        input_name: str = "",
    ) -> ImageProcessingResult:
        cfg = config or ImageProcessingConfig()
        if not input_path.is_file():
            raise MediaValidationError("Input image file does not exist.")

        frame = cv2.imread(str(input_path))
        if frame is None or frame.size == 0:
            raise MediaValidationError("Failed to decode image file.")

        height, width = frame.shape[:2]
        if width > cfg.max_dimension or height > cfg.max_dimension:
            raise MediaValidationError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed ({cfg.max_dimension})."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        detections = detector.detect(frame)
        duration_ms = (time.perf_counter() - start) * 1000.0
        meta = detector.get_metadata()

        annotated = draw_detections(
            frame,
            detections,
            blur_mode=cfg.blur_mode,
            selected_indices=cfg.selected_face_indices,
            box_color=cfg.box_color,
            box_thickness=cfg.box_thickness,
            show_hud=cfg.show_hud,
            draw_boxes=cfg.draw_boxes,
        )
        annotated_path = output_dir / "annotated.png"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise MediaValidationError("Failed to write annotated output image.")

        blurred_path: Path | None = None
        if cfg.save_privacy_image or cfg.blur_mode != "none":
            privacy_mode: BlurMode = cfg.blur_mode if cfg.blur_mode != "none" else "gaussian"
            blurred = apply_privacy_blur(
                frame,
                detections,
                privacy_mode,
                selected_indices=cfg.selected_face_indices,
            )
            blurred_path = output_dir / "privacy_blurred.png"
            cv2.imwrite(str(blurred_path), blurred)

        crop_paths: list[Path] = []
        if cfg.save_crops and detections:
            crops_dir = output_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            padding = 10
            for idx, det in enumerate(detections, start=1):
                x1 = max(0, det.x - padding)
                y1 = max(0, det.y - padding)
                x2 = min(width, det.x + det.width + padding)
                y2 = min(height, det.y + det.height + padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_file = crops_dir / f"face_{idx}.png"
                if cv2.imwrite(str(crop_file), frame[y1:y2, x1:x2]):
                    crop_paths.append(crop_file)

        result = ImageProcessingResult(
            detections=detections,
            width=width,
            height=height,
            duration_ms=duration_ms,
            model_metadata=meta,
            annotated_path=annotated_path,
            blurred_path=blurred_path,
            crop_paths=crop_paths,
        )

        report_json_path = output_dir / "report.json"
        report_payload = result.to_report_dict(job_id=job_id, input_name=input_name or input_path.name)
        report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        result.report_json_path = report_json_path

        report_csv_path = output_dir / "summary.csv"
        self._write_csv(report_csv_path, detections)
        result.report_csv_path = report_csv_path

        logger.info(
            "Processed image (%dx%d): %d face(s) in %.1fms", width, height, len(detections), duration_ms
        )
        return result

    @staticmethod
    def _write_csv(csv_path: Path, detections: list[Detection]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["face_index", "x", "y", "width", "height", "confidence"])
            for idx, det in enumerate(detections, start=1):
                writer.writerow(
                    [
                        idx,
                        det.x,
                        det.y,
                        det.width,
                        det.height,
                        f"{det.confidence:.4f}" if det.confidence is not None else "",
                    ]
                )
