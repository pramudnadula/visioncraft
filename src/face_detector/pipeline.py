"""High-level processing pipeline for images, directories, videos, and streams."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .backends import FaceDetectorBackend, create_backend
from .models import Detection, DetectorConfig
from .overlay import draw_detections

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_TITLE = "Face Detector"


@dataclass(slots=True)
class ImageResult:
    """Result summary for a single processed image."""

    source: str
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        """Serialize image result for JSON export."""
        return {
            "source": self.source,
            "face_count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


class FacePipeline:
    """Coordinates face detection, annotation, crop saving, and JSON reporting."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        backend: FaceDetectorBackend | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.backend = backend or create_backend(self.config)

    @staticmethod
    def save_crops(
        frame: cv2.typing.MatLike,
        detections: list[Detection],
        crops_dir: Path,
        prefix: str,
        padding: int = 10,
    ) -> list[Path]:
        """Save cropped face regions to disk and return written file paths."""
        crops_dir.mkdir(parents=True, exist_ok=True)
        img_h, img_w = frame.shape[:2]
        saved: list[Path] = []

        for idx, det in enumerate(detections, start=1):
            x1 = max(0, det.x - padding)
            y1 = max(0, det.y - padding)
            x2 = min(img_w, det.x + det.width + padding)
            y2 = min(img_h, det.y + det.height + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_path = crops_dir / f"{prefix}_face_{idx}.png"
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            saved.append(crop_path)

        return saved

    def process_image(
        self,
        image_path: Path,
        output_path: Path | None = None,
        crops_dir: Path | None = None,
        show: bool = True,
    ) -> ImageResult:
        """Detect faces in a single image file."""
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        detections = self.backend.detect(frame)
        annotated = draw_detections(frame, detections, self.config)
        print(f"[{image_path.name}] Detected {len(detections)} face(s)")

        if crops_dir is not None and detections:
            self.save_crops(frame, detections, crops_dir, prefix=image_path.stem)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

        if show:
            cv2.imshow(WINDOW_TITLE, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ImageResult(source=str(image_path), detections=detections)

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        crops_dir: Path | None = None,
    ) -> list[ImageResult]:
        """Batch-process all supported images inside a directory."""
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        image_paths = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {input_dir}")

        results: list[ImageResult] = []
        for img_path in image_paths:
            out_path = (output_dir / f"{img_path.stem}_detected{img_path.suffix}") if output_dir else None
            result = self.process_image(
                img_path,
                output_path=out_path,
                crops_dir=crops_dir,
                show=False,
            )
            results.append(result)
        return results

    def process_video(
        self,
        source: int | Path,
        output_path: Path | None = None,
        show: bool = True,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Detect faces across frames from a video file or webcam stream."""
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        writer: cv2.VideoWriter | None = None
        frame_count = 0
        total_detections = 0
        prev_time = time.perf_counter()

        try:
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                now = time.perf_counter()
                elapsed = now - prev_time
                live_fps = (1.0 / elapsed) if elapsed > 0 else 0.0
                prev_time = now

                detections = self.backend.detect(frame)
                annotated = draw_detections(frame, detections, self.config, fps=live_fps)

                frame_count += 1
                total_detections += len(detections)

                if writer is not None:
                    writer.write(annotated)

                if show:
                    cv2.imshow(WINDOW_TITLE, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        label = f"webcam({source})" if isinstance(source, int) else source.name
        print(f"[{label}] Processed {frame_count} frame(s); total detections: {total_detections}")
        return {
            "source": str(source),
            "frames_processed": frame_count,
            "total_detections": total_detections,
        }


def export_json_report(payload: Any, report_path: Path) -> None:
    """Write detection metadata to a formatted JSON file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved JSON report to {report_path}")
