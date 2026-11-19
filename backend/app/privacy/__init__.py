"""Privacy protection and frame annotation utilities."""

from .blur import BlurMode, apply_privacy_blur, blur_face_region, draw_detections, draw_hud

__all__ = [
    "BlurMode",
    "apply_privacy_blur",
    "blur_face_region",
    "draw_detections",
    "draw_hud",
]
