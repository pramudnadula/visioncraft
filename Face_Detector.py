"""Face detection on images, video files, or webcam using OpenCV Haar Cascades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2

DEFAULT_CASCADE = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
WINDOW_TITLE = "Face Detector"


def load_cascade(cascade_path: Path | None = None) -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier from disk or OpenCV's built-in data."""
    candidates: list[Path] = []
    if cascade_path is not None:
        candidates.append(cascade_path)
    candidates.append(DEFAULT_CASCADE)
    candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    for path in candidates:
        if path.is_file():
            classifier = cv2.CascadeClassifier(str(path))
            if not classifier.empty():
                return classifier

    raise FileNotFoundError(
        "Could not load 'haarcascade_frontalface_default.xml' from project or OpenCV data."
    )


def detect_and_annotate(
    frame: cv2.typing.MatLike,
    classifier: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> tuple[cv2.typing.MatLike, Sequence[Sequence[int]]]:
    """Detect faces in a BGR frame and draw bounding boxes with labels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    annotated = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return annotated, faces


def process_image(
    image_path: Path,
    classifier: cv2.CascadeClassifier,
    output_path: Path | None = None,
    show: bool = True,
) -> int:
    """Run face detection on a single image file and return detected face count."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    annotated, faces = detect_and_annotate(img, classifier)
    print(f"Detected {len(faces)} face(s) in {image_path.name}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to {output_path}")

    if show:
        cv2.imshow(WINDOW_TITLE, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(faces)


def process_video(
    source: int | Path,
    classifier: cv2.CascadeClassifier,
    output_path: Path | None = None,
    show: bool = True,
    max_frames: int | None = None,
) -> int:
    """Run face detection on a video file or webcam stream."""
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    total_detections = 0

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

            annotated, faces = detect_and_annotate(frame, classifier)
            frame_count += 1
            total_detections += len(faces)

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
    print(f"Processed {frame_count} frame(s) from {label}; total face detections: {total_detections}")
    if output_path is not None:
        print(f"Saved annotated video to {output_path}")

    return frame_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="OpenCV Haar Cascade Face Detector")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("-i", "--image", type=Path, help="Path to an input image.")
    source_group.add_argument("-v", "--video", type=Path, help="Path to an input video file.")
    source_group.add_argument(
        "-w", "--webcam", type=int, nargs="?", const=0, help="Webcam index (default: 0)."
    )
    parser.add_argument("-o", "--output", type=Path, help="Optional path to save annotated output.")
    parser.add_argument(
        "-c", "--cascade", type=Path, help="Optional path to a Haar cascade XML file."
    )
    parser.add_argument("--no-show", action="store_true", help="Run without opening a GUI window.")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames (useful for testing)."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for face detection CLI."""
    args = parse_args(argv)
    classifier = load_cascade(args.cascade)
    show = not args.no_show

    if args.image is not None:
        process_image(args.image, classifier, output_path=args.output, show=show)
    elif args.video is not None:
        process_video(
            args.video,
            classifier,
            output_path=args.output,
            show=show,
            max_frames=args.max_frames,
        )
    else:
        cam_index = 0 if args.webcam is None else args.webcam
        process_video(
            cam_index,
            classifier,
            output_path=args.output,
            show=show,
            max_frames=args.max_frames,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
