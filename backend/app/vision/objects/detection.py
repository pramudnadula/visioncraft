"""Universal ONNX Object Detection Engine (COCO 80-class YOLOv8/ONNX compatible)."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.exceptions import ModelInitializationError, ModelNotAvailableError

COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    """Unified detection result for general objects and vehicles."""

    id: str
    class_id: int
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    frame_number: int = 0
    timestamp: float = 0.0
    tracking_id: int | None = None

    def with_tracking_id(self, tracking_id: int) -> ObjectDetection:
        return ObjectDetection(
            id=self.id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            tracking_id=tracking_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YOLOv8ONNXObjectDetector:
    """General-purpose ONNX object detector supporting COCO classes."""

    DEFAULT_FILENAMES = ("yolov8n.onnx", "yolov8s.onnx", "yolo11n.onnx", "object_detector.onnx")

    def __init__(
        self,
        *,
        models_dir: Path,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        model_path: Path | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.explicit_path = model_path
        self._session: Any = None
        self._input_name = "images"
        self._input_size = (640, 640)

    def resolve_model_path(self) -> Path | None:
        if self.explicit_path is not None and self.explicit_path.is_file():
            return self.explicit_path
        for directory in (self.models_dir, self.models_dir / "custom"):
            if not directory.is_dir():
                continue
            for name in self.DEFAULT_FILENAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None and self.resolve_model_path() is not None

    def initialize(self) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise ModelNotAvailableError("Install 'onnxruntime' to run ONNX object detection.")
        path = self.resolve_model_path()
        if path is None:
            raise ModelNotAvailableError(
                "No ONNX object detection model found in models/ (e.g., yolov8n.onnx)."
            )
        import onnxruntime as ort  # type: ignore[import-not-found]

        try:
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self._input_size = (int(inp.shape[3]), int(inp.shape[2]))
        except Exception as exc:
            raise ModelInitializationError(f"Failed to load object detector: {exc}") from exc

    def detect(
        self,
        frame: cv2.typing.MatLike,
        *,
        allowed_classes: set[str] | None = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[ObjectDetection]:
        if self._session is None:
            self.initialize()
        if frame is None or frame.size == 0:
            return []

        in_w, in_h = self._input_size
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (in_w, in_h))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (in_w, in_h), (0, 0, 0), swapRB=True)
        raw = self._session.run(None, {self._input_name: blob})[0]
        # YOLOv8 output shape: (1, 84, 8400) -> transpose to (8400, 84)
        preds = np.squeeze(raw, axis=0).T
        if preds.shape[1] < 5:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        if len(confidences) == 0:
            return []

        sx, sy = orig_w / in_w, orig_h / in_h
        nms_boxes: list[list[int]] = []
        nms_scores: list[float] = []
        nms_cids: list[int] = []
        for (cx, cy, bw, bh), cid, conf in zip(boxes_xywh, class_ids, confidences, strict=True):
            cname = COCO_CLASSES[int(cid)] if int(cid) < len(COCO_CLASSES) else f"class_{int(cid)}"
            if allowed_classes is not None and cname not in allowed_classes:
                continue
            x1 = max(0, int((cx - bw / 2) * sx))
            y1 = max(0, int((cy - bh / 2) * sy))
            w_px = max(1, int(bw * sx))
            h_px = max(1, int(bh * sy))
            nms_boxes.append([x1, y1, w_px, h_px])
            nms_scores.append(float(conf))
            nms_cids.append(int(cid))

        if not nms_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.confidence_threshold, self.iou_threshold)
        results: list[ObjectDetection] = []
        for seq_idx, i in enumerate(np.array(indices).flatten()):
            idx = int(i)
            x, y, w, h = nms_boxes[idx]
            cid = nms_cids[idx]
            cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
            results.append(
                ObjectDetection(
                    id=f"f{frame_number}_o{seq_idx}",
                    class_id=cid,
                    class_name=cname,
                    confidence=round(nms_scores[idx], 4),
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
            )
        return results

    def release(self) -> None:
        self._session = None
