"""Multi-class IoU object tracker for video & vehicle analytics."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.vision.objects.detection import ObjectDetection


def _iou(a: ObjectDetection, b: ObjectDetection) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class _ObjTrack:
    tracking_id: int
    last: ObjectDetection
    missed: int = 0


class ObjectTracker:
    """Assigns temporary tracking IDs per object class across frames."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _ObjTrack] = {}

    def update(self, detections: list[ObjectDetection]) -> list[ObjectDetection]:
        if not detections:
            self._age(set())
            return []

        pairs: list[tuple[float, int, int]] = []
        for d_idx, det in enumerate(detections):
            for tid, tr in self._tracks.items():
                if tr.last.class_id != det.class_id:
                    continue
                score = _iou(det, tr.last)
                if score >= self.iou_threshold:
                    pairs.append((score, d_idx, tid))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[int] = set()
        det_to_tid: dict[int, int] = {}
        for _, d_idx, tid in pairs:
            if d_idx in det_to_tid or tid in matched_tracks:
                continue
            det_to_tid[d_idx] = tid
            matched_tracks.add(tid)

        out: list[ObjectDetection] = []
        for d_idx, det in enumerate(detections):
            if d_idx in det_to_tid:
                tid = det_to_tid[d_idx]
            else:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = _ObjTrack(tracking_id=tid, last=det, missed=0)
            out.append(det.with_tracking_id(tid))

        self._age(matched_tracks)
        return out

    def _age(self, matched: set[int]) -> None:
        expired = [
            tid for tid, tr in self._tracks.items() if tid not in matched and tr.missed >= self.max_missed
        ]
        for tid, tr in self._tracks.items():
            if tid not in matched:
                tr.missed += 1
        for tid in expired:
            del self._tracks[tid]
