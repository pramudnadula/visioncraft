"""Real-time facial expression / emotion analysis (FER+ ONNX + MediaPipe blendshapes)."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any, Literal, get_args

import cv2
import numpy as np

from backend.app.core.config import get_settings
from backend.app.detection.base import Detection
from backend.app.vision.faces.face_mesh import FaceMeshFace
from backend.app.vision.faces.segmentation import build_face_privacy_context

EmotionMode = Literal["off", "hybrid", "ferplus", "blendshapes"]
EMOTION_MODES: tuple[str, ...] = get_args(EmotionMode)

FERPLUS_FILENAME = "emotion-ferplus-8.onnx"
FERPLUS_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
FERPLUS_LABELS: tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

EMOTION_DISCLAIMER = (
    "Facial expression scores reflect observable facial muscle patterns only; they do not "
    "establish internal emotional state, intent, or honesty."
)

_LOCK = threading.Lock()
_SESSION_CACHE: dict[Path, Any] = {}

# BGR pill colors per dominant emotion
_EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happiness": (80, 210, 110),
    "surprise": (0, 215, 255),
    "neutral": (200, 200, 210),
    "sadness": (220, 140, 60),
    "anger": (65, 75, 240),
    "disgust": (90, 190, 160),
    "fear": (210, 100, 220),
    "contempt": (150, 160, 230),
}


def validate_emotion_mode(mode: str) -> EmotionMode:
    if mode not in EMOTION_MODES:
        raise ValueError(f"Unsupported emotion mode '{mode}'. Supported: {', '.join(EMOTION_MODES)}")
    return mode  # type: ignore[return-value]


def resolve_ferplus_model_path(models_dir: Path | None = None) -> Path | None:
    base = models_dir if models_dir is not None else get_settings().resolved_models_dir
    for directory in (base, base / "custom"):
        candidate = directory / FERPLUS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def is_ferplus_available(models_dir: Path | None = None) -> bool:
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and resolve_ferplus_model_path(models_dir) is not None
    )


def _get_ferplus_session(models_dir: Path) -> Any | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    path = resolve_ferplus_model_path(models_dir)
    if path is None:
        return None
    with _LOCK:
        cached = _SESSION_CACHE.get(path)
        if cached is not None:
            return cached
        import onnxruntime as ort  # type: ignore[import-not-found]

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _SESSION_CACHE[path] = session
        return session


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp), dtype=np.float64)


def _infer_ferplus(session: Any, frame: cv2.typing.MatLike, det: Detection) -> dict[str, float] | None:
    h, w = frame.shape[:2]
    x1 = max(0, det.x)
    y1 = max(0, det.y)
    x2 = min(w, det.x + det.width)
    y2 = min(h, det.y + det.height)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    blob = resized.reshape(1, 1, 64, 64)
    input_name = session.get_inputs()[0].name
    logits = np.squeeze(session.run(None, {input_name: blob})[0]).astype(np.float64)
    probs = _softmax(logits)
    return {label: float(probs[i]) for i, label in enumerate(FERPLUS_LABELS)}


def _infer_blendshapes(mesh_face: FaceMeshFace) -> dict[str, float]:
    bs = mesh_face.blendshapes
    if not bs:
        return {}

    def avg(a: str, b: str) -> float:
        return 0.5 * (bs.get(a, 0.0) + bs.get(b, 0.0))

    smile = avg("mouthSmileLeft", "mouthSmileRight")
    frown = avg("mouthFrownLeft", "mouthFrownRight")
    brow_down = avg("browDownLeft", "browDownRight")
    brow_up = bs.get("browInnerUp", 0.0)
    jaw_open = bs.get("jawOpen", 0.0)
    eye_wide = avg("eyeWideLeft", "eyeWideRight")
    sneer = avg("noseSneerLeft", "noseSneerRight")
    squint = avg("eyeSquintLeft", "eyeSquintRight")

    raw = {
        "happiness": min(1.0, smile * 1.15 + squint * 0.2),
        "surprise": min(1.0, jaw_open * 0.65 + brow_up * 0.6 + eye_wide * 0.55),
        "sadness": min(1.0, frown * 0.8 + brow_up * 0.45),
        "anger": min(1.0, brow_down * 0.95 + bs.get("mouthPressLeft", 0.0) * 0.35),
        "disgust": min(1.0, sneer * 1.1 + bs.get("mouthUpperUpLeft", 0.0) * 0.3),
        "fear": min(1.0, eye_wide * 0.7 + brow_up * 0.45 + jaw_open * 0.35),
        "contempt": min(1.0, abs(bs.get("mouthSmileLeft", 0.0) - bs.get("mouthSmileRight", 0.0)) * 0.9),
    }
    peak_expr = max(raw.values())
    raw["neutral"] = max(0.05, 1.0 - peak_expr * 1.1)
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def analyze_face_emotions(
    frame: cv2.typing.MatLike,
    detections: list[Detection],
    *,
    mode: EmotionMode = "hybrid",
    models_dir: Path | None = None,
    mesh_faces: list[FaceMeshFace] | None = None,
) -> dict[str, Any] | None:
    """Estimate facial expression scores for detected faces."""
    if mode == "off":
        return None

    resolved = models_dir if models_dir is not None else get_settings().resolved_models_dir
    session = _get_ferplus_session(resolved) if mode in {"hybrid", "ferplus"} else None

    ctx = None
    if mode in {"hybrid", "blendshapes"}:
        if mesh_faces is not None:
            from backend.app.vision.faces.segmentation import FacePrivacyContext

            ctx = FacePrivacyContext(head_mask=None, mesh_faces=mesh_faces)
        else:
            ctx = build_face_privacy_context(
                frame,
                models_dir=resolved,
                need_segmentation=False,
                need_mesh=True,
            )

    faces_out: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        fer_scores = _infer_ferplus(session, frame, det) if session is not None else None
        matched_mesh = ctx.match_mesh(det) if ctx is not None else None
        bs_scores = _infer_blendshapes(matched_mesh) if matched_mesh is not None else None

        combined: dict[str, float] = {}
        source = "none"
        if mode == "ferplus" and fer_scores:
            combined = fer_scores
            source = "ferplus_onnx"
        elif mode == "blendshapes" and bs_scores:
            combined = bs_scores
            source = "mediapipe_blendshapes"
        elif fer_scores and bs_scores:
            for label in FERPLUS_LABELS:
                combined[label] = 0.65 * fer_scores.get(label, 0.0) + 0.35 * bs_scores.get(label, 0.0)
            source = "hybrid"
        elif fer_scores:
            combined = fer_scores
            source = "ferplus_onnx"
        elif bs_scores:
            combined = bs_scores
            source = "mediapipe_blendshapes"

        if not combined:
            continue

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        dominant, dom_score = ranked[0]
        faces_out.append(
            {
                "face_index": idx,
                "tracking_id": det.tracking_id,
                "dominant_emotion": dominant,
                "confidence": round(float(dom_score), 3),
                "source": source,
                "scores": [{"emotion": k, "score": round(float(v), 3)} for k, v in ranked[:5]],
            }
        )

    return {
        "mode": mode,
        "face_count": len(faces_out),
        "faces": faces_out,
        "disclaimer": EMOTION_DISCLAIMER,
    }


def draw_emotion_overlays(
    frame: cv2.typing.MatLike,
    detections: list[Detection],
    emotion_result: dict[str, Any] | None,
) -> None:
    """Draw compact emotion pill badges onto ``frame`` in place."""
    if not emotion_result or not emotion_result.get("faces"):
        return
    for face in emotion_result["faces"]:
        idx = int(face["face_index"])
        if idx >= len(detections):
            continue
        det = detections[idx]
        label = str(face["dominant_emotion"]).capitalize()
        conf = float(face["confidence"])
        text = f"{label} {conf:.0%}"
        color = _EMOTION_COLORS.get(str(face["dominant_emotion"]), (200, 200, 210))

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        px = max(6, det.x)
        py = min(frame.shape[0] - 8, det.y + det.height + th + 12)
        cv2.rectangle(frame, (px, py - th - 8), (px + tw + 12, py + 4), (20, 22, 30), -1)
        cv2.rectangle(frame, (px, py - th - 8), (px + tw + 12, py + 4), color, 1)
        cv2.putText(frame, text, (px + 6, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
