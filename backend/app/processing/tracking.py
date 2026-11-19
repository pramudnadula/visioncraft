"""Temporary IoU-based face tracking across video frames.

Tracking IDs are ephemeral session identifiers for visual continuity only
and must never be presented as verified biometric identities.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.detection.base import Detection


def _compute_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass(slots=True)
class _TrackState:
    tracking_id: int
    last_detection: Detection
    missed_frames: int = 0


class FaceTracker:
    """Greedy IoU tracker assigning temporary tracking IDs to face detections."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign temporary tracking IDs to current frame detections."""
        if not detections:
            self._age_tracks(matched_ids=set())
            return []

        matched_track_ids: set[int] = set()
        assigned: list[Detection] = []

        # Build candidate (iou, det_idx, track_id) pairs sorted descending
        pairs: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for tid, state in self._tracks.items():
                iou = _compute_iou(det, state.last_detection)
                if iou >= self.iou_threshold:
                    pairs.append((iou, det_idx, tid))
        pairs.sort(key=lambda item: item[0], reverse=True)

        det_to_tid: dict[int, int] = {}
        for _, det_idx, tid in pairs:
            if det_idx in det_to_tid or tid in matched_track_ids:
                continue
            det_to_tid[det_idx] = tid
            matched_track_ids.add(tid)

        for det_idx, det in enumerate(detections):
            if det_idx in det_to_tid:
                tid = det_to_tid[det_idx]
            else:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = _TrackState(tracking_id=tid, last_detection=det, missed_frames=0)
            assigned.append(det.with_tracking_id(tid))

        self._age_tracks(matched_track_ids)
        return assigned

    def _age_tracks(self, matched_ids: set[int]) -> None:
        expired: list[int] = []
        for tid, state in self._tracks.items():
            if tid not in matched_ids:
                state.missed_frames += 1
                if state.missed_frames > self.max_missed_frames:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]
