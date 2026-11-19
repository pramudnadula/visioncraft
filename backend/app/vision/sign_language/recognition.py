"""Sign language sequence buffer, language-specific model registry, and ASL inference."""

from __future__ import annotations

import ast
import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError

ASL_ONNX_FILENAME = "asl_yolo11n.onnx"
GESTURE_TASK_FILENAME = "gesture_recognizer.task"
_ASL_DEFAULT_NAMES = tuple(chr(c) for c in range(ord("A"), ord("Z")))  # A-Y (25 classes)


@dataclass(frozen=True, slots=True)
class SignLanguageSpec:
    code: str
    name: str
    model_available: bool
    pipeline_ready: bool
    status_label: str
    note: str


def _resolve_model(models_dir: Path, filename: str) -> Path | None:
    for directory in (models_dir, models_dir / "custom"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def list_sign_languages(models_dir: Path | None = None) -> list[dict[str, Any]]:
    resolved = models_dir if models_dir is not None else get_settings().resolved_models_dir
    landmarks_ready = (
        _resolve_model(resolved, "hand_landmarker.task") is not None
        and _resolve_model(resolved, "pose_landmarker_lite.task") is not None
    )
    asl_ready = (
        _resolve_model(resolved, ASL_ONNX_FILENAME) is not None
        or _resolve_model(resolved, GESTURE_TASK_FILENAME) is not None
    )
    finsl_ready = _resolve_model(resolved, "finsl_sign.onnx") is not None
    bsl_ready = _resolve_model(resolved, "bsl_sign.onnx") is not None

    specs = (
        SignLanguageSpec(
            code="ASL",
            name="American Sign Language",
            model_available=asl_ready,
            pipeline_ready=landmarks_ready or asl_ready,
            status_label=(
                "Ready"
                if asl_ready
                else ("Landmarks Ready" if landmarks_ready else "Model Required")
            ),
            note=(
                "Enabled via asl_yolo11n.onnx (A-Y letters) & gesture_recognizer.task."
                if asl_ready
                else "Place asl_yolo11n.onnx or gesture_recognizer.task in models/."
            ),
        ),
        SignLanguageSpec(
            code="FinSL",
            name="Finnish Sign Language",
            model_available=finsl_ready,
            pipeline_ready=landmarks_ready,
            status_label=(
                "Ready"
                if finsl_ready
                else ("Landmarks Ready" if landmarks_ready else "Model Required")
            ),
            note=(
                "Enabled via finsl_sign.onnx."
                if finsl_ready
                else (
                    "Hand/pose landmark pipeline ready; add finsl_sign.onnx in models/ "
                    "for vocabulary classification."
                )
            ),
        ),
        SignLanguageSpec(
            code="BSL",
            name="British Sign Language",
            model_available=bsl_ready,
            pipeline_ready=landmarks_ready,
            status_label=(
                "Ready"
                if bsl_ready
                else ("Landmarks Ready" if landmarks_ready else "Model Required")
            ),
            note=(
                "Enabled via bsl_sign.onnx."
                if bsl_ready
                else (
                    "Hand/pose landmark pipeline ready; add bsl_sign.onnx in models/ "
                    "for vocabulary classification."
                )
            ),
        ),
    )
    return [
        {
            "code": s.code,
            "name": s.name,
            "model_available": s.model_available,
            "pipeline_ready": s.pipeline_ready,
            "status_label": s.status_label,
            "note": s.note,
        }
        for s in specs
    ]


class SignLanguageSequenceBuffer:
    """Sliding temporal window storing normalized hand/pose landmark frames."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = max(1, window_size)
        self._frames: deque[list[float]] = deque(maxlen=self.window_size)

    def push(self, feature_vector: list[float]) -> None:
        self._frames.append(feature_vector)

    def is_ready(self) -> bool:
        return len(self._frames) == self.window_size

    def to_sequence(self) -> list[list[float]]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()


class ASLSignDetector:
    """Detects ASL fingerspelling letters (YOLO11n ONNX) and MediaPipe hand gestures."""

    def __init__(self, models_dir: Path | None = None, *, confidence_threshold: float = 0.35) -> None:
        self.models_dir = models_dir if models_dir is not None else get_settings().resolved_models_dir
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: cv2.typing.MatLike) -> dict[str, Any]:
        asl_path = _resolve_model(self.models_dir, ASL_ONNX_FILENAME)
        gesture_path = _resolve_model(self.models_dir, GESTURE_TASK_FILENAME)
        if asl_path is None and gesture_path is None:
            raise ModelNotAvailableError(
                "Place asl_yolo11n.onnx or gesture_recognizer.task in models/ for ASL recognition."
            )

        letters: list[dict[str, Any]] = []
        gestures: list[dict[str, Any]] = []

        if asl_path is not None and importlib.util.find_spec("onnxruntime") is not None:
            import onnxruntime as ort  # type: ignore[import-not-found]

            session = ort.InferenceSession(str(asl_path), providers=["CPUExecutionProvider"])
            meta = session.get_modelmeta().custom_metadata_map
            names_map: dict[int, str] = {i: n for i, n in enumerate(_ASL_DEFAULT_NAMES)}
            if "names" in meta:
                try:
                    parsed = ast.literal_eval(meta["names"])
                    if isinstance(parsed, dict):
                        names_map = {int(k): str(v) for k, v in parsed.items()}
                except Exception:
                    pass

            inp = session.get_inputs()[0]
            in_w, in_h = 640, 640
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                in_w, in_h = int(inp.shape[3]), int(inp.shape[2])

            orig_h, orig_w = frame.shape[:2]
            resized = cv2.resize(frame, (in_w, in_h))
            blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (in_w, in_h), (0, 0, 0), swapRB=True)
            raw = session.run(None, {inp.name: blob})[0]
            preds = np.squeeze(raw, axis=0).T
            if preds.shape[1] >= 5:
                boxes = preds[:, :4]
                scores = preds[:, 4:]
                cids = np.argmax(scores, axis=1)
                confs = scores[np.arange(len(cids)), cids]
                keep = confs >= self.confidence_threshold
                sx, sy = orig_w / in_w, orig_h / in_h
                nms_boxes, nms_confs, nms_cids = [], [], []
                for (cx, cy, bw, bh), cid, conf in zip(boxes[keep], cids[keep], confs[keep], strict=True):
                    x1 = max(0, int((cx - bw / 2) * sx))
                    y1 = max(0, int((cy - bh / 2) * sy))
                    nms_boxes.append([x1, y1, max(1, int(bw * sx)), max(1, int(bh * sy))])
                    nms_confs.append(float(conf))
                    nms_cids.append(int(cid))
                if nms_boxes:
                    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confs, self.confidence_threshold, 0.45)
                    for idx in np.array(indices).flatten():
                        bx, by, bw, bh = nms_boxes[int(idx)]
                        letters.append(
                            {
                                "letter": names_map.get(nms_cids[int(idx)], f"class_{nms_cids[int(idx)]}"),
                                "confidence": round(nms_confs[int(idx)], 4),
                                "bbox": {"x": bx, "y": by, "width": bw, "height": bh},
                            }
                        )

        if gesture_path is not None and importlib.util.find_spec("mediapipe") is not None:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            opts = mp_vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(gesture_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
            )
            with mp_vision.GestureRecognizer.create_from_options(opts) as recognizer:
                gres = recognizer.recognize(mp_img)
            for idx, hand_gestures in enumerate(gres.gestures or ()):
                if not hand_gestures:
                    continue
                top = hand_gestures[0]
                handedness = "Unknown"
                if gres.handedness and idx < len(gres.handedness) and gres.handedness[idx]:
                    handedness = str(gres.handedness[idx][0].category_name or "Unknown")
                gestures.append(
                    {
                        "gesture": str(top.category_name),
                        "confidence": round(float(top.score), 4),
                        "handedness": handedness,
                    }
                )

        return {
            "language": "ASL",
            "asl_letters": letters,
            "gestures": gestures,
        }
