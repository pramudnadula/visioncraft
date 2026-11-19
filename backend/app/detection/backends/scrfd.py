"""Optional SCRFD ONNX face detection adapter with full ONNXRuntime inference."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError
from backend.app.detection.base import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
    FaceDetector,
    LandmarkPoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRFD_FILENAMES = ("scrfd_10g_bnkps.onnx", "scrfd_2.5g_bnkps.onnx", "det_10g.onnx", "scrfd.onnx")


class SCRFDDetector(FaceDetector):
    """Optional SCRFD ONNX adapter (requires onnxruntime + model weights)."""

    model_id = "scrfd"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._session: Any = None
        self._input_name: str = "input.1"
        self._input_size: tuple[int, int] = (640, 640)
        self._providers: list[str] = ["CPUExecutionProvider"]

    def _has_onnxruntime(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def _resolve_model_path(self) -> Path | None:
        if self.config.model_path is not None and self.config.model_path.is_file():
            return self.config.model_path
        search_dirs: list[Path] = []
        if self.config.models_dir is not None:
            search_dirs.append(self.config.models_dir)
        else:
            search_dirs.append(PROJECT_ROOT / "models")
        for directory in search_dirs:
            for name in DEFAULT_SCRFD_FILENAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    def is_available(self) -> bool:
        return self._has_onnxruntime() and self._resolve_model_path() is not None

    def initialize(self) -> None:
        if not self._has_onnxruntime():
            raise ModelNotAvailableError("Optional dependency 'onnxruntime' is not installed.")
        model_path = self._resolve_model_path()
        if model_path is None:
            raise ModelNotAvailableError("SCRFD ONNX model not found in models/.")

        import onnxruntime as ort  # type: ignore[import-not-found]

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.config.use_gpu and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._providers = self._session.get_providers()
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self._input_size = (int(inp.shape[3]), int(inp.shape[2]))
        except Exception as exc:
            raise ModelInitializationError(f"Failed to initialize SCRFD session: {exc}") from exc
        self._initialized = True

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        self.ensure_initialized()
        if self._session is None:
            raise ModelInitializationError("SCRFD session is not initialized.")
        if frame is None or frame.size == 0:
            return []

        in_w, in_h = self._input_size
        orig_h, orig_w = frame.shape[:2]
        scale = min(in_w / orig_w, in_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        det_img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / 128.0,
            (in_w, in_h),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )

        outputs = self._session.run(None, {self._input_name: blob})
        fmc = 3
        strides = [8, 16, 32]
        num_anchors = 2
        if len(outputs) < fmc * 2:
            return []

        all_boxes: list[list[int]] = []
        all_scores: list[float] = []
        all_kps: list[list[LandmarkPoint] | None] = []
        has_kps = len(outputs) >= fmc * 3

        for idx, stride in enumerate(strides):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outputs[idx + fmc * 2].reshape(-1, 10) * stride if has_kps else None

            feat_h, feat_w = in_h // stride, in_w // stride
            anchor_centers = np.stack(list(np.mgrid[:feat_h, :feat_w][::-1]), axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(-1, 2)

            keep = np.where(scores >= self.config.confidence_threshold)[0]
            for k in keep:
                cx, cy = anchor_centers[k]
                l_off, t_off, r_off, b_off = bbox_preds[k]
                x1 = max(0, int((cx - l_off) / scale))
                y1 = max(0, int((cy - t_off) / scale))
                x2 = min(orig_w, int((cx + r_off) / scale))
                y2 = min(orig_h, int((cy + b_off) / scale))
                bw, bh = x2 - x1, y2 - y1
                if bw < self.config.min_face_size[0] or bh < self.config.min_face_size[1]:
                    continue

                kps_list: list[LandmarkPoint] | None = None
                if kps_preds is not None:
                    kp_row = kps_preds[k]
                    kps_list = [
                        LandmarkPoint(
                            x=float((cx + kp_row[i]) / scale), y=float((cy + kp_row[i + 1]) / scale)
                        )
                        for i in range(0, 10, 2)
                    ]

                all_boxes.append([x1, y1, bw, bh])
                all_scores.append(float(scores[k]))
                all_kps.append(kps_list)

        if not all_boxes:
            return []

        nms_indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.config.confidence_threshold, 0.4)
        results: list[Detection] = []
        for i in np.array(nms_indices).flatten():
            x, y, w, h = all_boxes[int(i)]
            results.append(
                Detection(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=all_scores[int(i)],
                    landmarks=all_kps[int(i)],
                )
            )
        return results

    def get_metadata(self) -> DetectorMetadata:
        available = self.is_available()
        reason: str | None = None
        if not self._has_onnxruntime():
            reason = "Install optional 'onnxruntime' package to enable SCRFD."
        elif self._resolve_model_path() is None:
            reason = "Place scrfd_10g_bnkps.onnx in models/ to enable SCRFD."
        return DetectorMetadata(
            id=self.model_id,
            name="SCRFD (Optional ONNX)",
            version="InsightFace-SCRFD",
            description="High-accuracy Sample and Computation Redistribution Face Detector (optional).",
            supports_confidence=True,
            supports_landmarks=True,
            supports_gpu="CUDAExecutionProvider" in self._providers,
            gpu_active="CUDAExecutionProvider" in self._providers and self.config.use_gpu,
            is_optional=True,
            available=available,
            license="MIT (InsightFace code; check model weights license)",
            unavailable_reason=reason,
        )

    def release(self) -> None:
        self._session = None
        self._initialized = False
