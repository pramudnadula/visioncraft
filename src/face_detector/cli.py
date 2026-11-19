"""Command-line interface for Face Detector."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .models import DetectorConfig
from .pipeline import FacePipeline, export_json_report


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="face-detect",
        description="Modular OpenCV face detection for images, directories, videos, and webcams.",
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("-i", "--image", type=Path, help="Input image path.")
    source_group.add_argument("-d", "--input-dir", type=Path, help="Directory of images to batch process.")
    source_group.add_argument("-v", "--video", type=Path, help="Input video file path.")
    source_group.add_argument(
        "-w", "--webcam", type=int, nargs="?", const=0, help="Webcam device index (default: 0)."
    )

    parser.add_argument("-o", "--output", type=Path, help="Output file or directory for annotated media.")
    parser.add_argument("--save-crops", type=Path, help="Directory to save cropped face images.")
    parser.add_argument("--export-json", type=Path, help="Path to write JSON detection metadata.")

    parser.add_argument(
        "--blur",
        choices=("none", "gaussian", "pixelate"),
        default="none",
        help="Anonymize detected faces with blur or pixelation (default: none).",
    )
    parser.add_argument("--no-hud", action="store_true", help="Disable the face count / FPS overlay.")
    parser.add_argument("--no-show", action="store_true", help="Run headless without opening a GUI window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop video/webcam after N frames.")

    parser.add_argument("-c", "--cascade", type=Path, help="Custom Haar cascade XML file path.")
    parser.add_argument("--yunet-model", type=Path, help="Path to YuNet ONNX model (uses DNN backend).")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor (default: 1.1).")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar minNeighbors (default: 5).")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    config = DetectorConfig(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        blur_mode=args.blur,
        show_hud=not args.no_hud,
        cascade_path=args.cascade,
        yunet_model_path=args.yunet_model,
    )
    pipeline = FacePipeline(config=config)
    show = not args.no_show

    if args.image is not None:
        result = pipeline.process_image(
            args.image,
            output_path=args.output,
            crops_dir=args.save_crops,
            show=show,
        )
        if args.export_json is not None:
            export_json_report(result.to_dict(), args.export_json)

    elif args.input_dir is not None:
        results = pipeline.process_directory(
            args.input_dir,
            output_dir=args.output,
            crops_dir=args.save_crops,
        )
        if args.export_json is not None:
            export_json_report([r.to_dict() for r in results], args.export_json)

    elif args.video is not None:
        summary = pipeline.process_video(
            args.video,
            output_path=args.output,
            show=show,
            max_frames=args.max_frames,
        )
        if args.export_json is not None:
            export_json_report(summary, args.export_json)

    else:
        cam_index = 0 if args.webcam is None else args.webcam
        summary = pipeline.process_video(
            cam_index,
            output_path=args.output,
            show=show,
            max_frames=args.max_frames,
        )
        if args.export_json is not None:
            export_json_report(summary, args.export_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
