"""Modular OpenCV face detection package."""

from .backends import FaceDetectorBackend, HaarCascadeBackend, YuNetBackend, create_backend
from .models import BlurMode, Detection, DetectorConfig
from .overlay import apply_privacy_blur, draw_detections, draw_hud
from .pipeline import FacePipeline, ImageResult, export_json_report

__all__ = [
    "BlurMode",
    "Detection",
    "DetectorConfig",
    "FaceDetectorBackend",
    "FacePipeline",
    "HaarCascadeBackend",
    "ImageResult",
    "YuNetBackend",
    "apply_privacy_blur",
    "create_backend",
    "draw_detections",
    "draw_hud",
    "export_json_report",
]
