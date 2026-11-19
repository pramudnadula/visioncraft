"""Vehicle line-crossing and polygon-zone counting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.app.vision.objects.detection import ObjectDetection


@dataclass(frozen=True, slots=True)
class CountingLine:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class CountingZone:
    zone_id: str
    name: str
    polygon: list[tuple[int, int]]


@dataclass(slots=True)
class VehicleCountingSummary:
    total_vehicles: int = 0
    cars: int = 0
    buses: int = 0
    trucks: int = 0
    motorcycles: int = 0
    bicycles: int = 0
    entry_count: int = 0
    exit_count: int = 0
    zone_counts: dict[str, int] = field(default_factory=dict)


class VehicleCounter:
    """Prevents duplicate vehicle counts using tracking IDs, line-crossing, and zones."""

    def __init__(
        self,
        *,
        line: CountingLine | None = None,
        zones: list[CountingZone] | None = None,
    ) -> None:
        self.line = line
        self.zones = zones or []
        self._seen_track_ids: set[int] = set()
        self._last_side: dict[int, float] = {}
        self._counted_line_ids: set[int] = set()
        self._zone_seen_ids: dict[str, set[int]] = {z.zone_id: set() for z in self.zones}
        self.summary = VehicleCountingSummary(zone_counts={z.zone_id: 0 for z in self.zones})

    @staticmethod
    def _center(det: ObjectDetection) -> tuple[float, float]:
        return det.x + det.width / 2.0, det.y + det.height / 2.0

    def _side_of_line(self, px: float, py: float) -> float:
        assert self.line is not None
        return float(
            (self.line.x2 - self.line.x1) * (py - self.line.y1)
            - (self.line.y2 - self.line.y1) * (px - self.line.x1)
        )

    def update(self, tracked_vehicles: list[ObjectDetection]) -> VehicleCountingSummary:
        for det in tracked_vehicles:
            if det.tracking_id is None:
                continue
            tid = det.tracking_id
            if tid not in self._seen_track_ids:
                self._seen_track_ids.add(tid)
                self.summary.total_vehicles += 1
                if det.class_name == "car":
                    self.summary.cars += 1
                elif det.class_name == "bus":
                    self.summary.buses += 1
                elif det.class_name == "truck":
                    self.summary.trucks += 1
                elif det.class_name == "motorcycle":
                    self.summary.motorcycles += 1
                elif det.class_name == "bicycle":
                    self.summary.bicycles += 1

            cx, cy = self._center(det)
            if self.line is not None:
                side = self._side_of_line(cx, cy)
                prev = self._last_side.get(tid)
                if prev is not None and tid not in self._counted_line_ids:
                    if prev < 0 <= side:
                        self.summary.entry_count += 1
                        self._counted_line_ids.add(tid)
                    elif prev > 0 >= side:
                        self.summary.exit_count += 1
                        self._counted_line_ids.add(tid)
                self._last_side[tid] = side

            for zone in self.zones:
                if len(zone.polygon) < 3 or tid in self._zone_seen_ids[zone.zone_id]:
                    continue
                poly = np.array(zone.polygon, dtype=np.int32)
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    self._zone_seen_ids[zone.zone_id].add(tid)
                    self.summary.zone_counts[zone.zone_id] = len(self._zone_seen_ids[zone.zone_id])

        return self.summary
