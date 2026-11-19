"""Thread-safe event bus for broadcasting job progress to WebSocket clients."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class JobEvent:
    """Real-time job status/progress payload."""

    job_id: str
    status: str
    progress: float
    faces_detected: int = 0
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobEventBus:
    """Manages per-job asyncio queues for WebSocket subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[JobEvent]]]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue[JobEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[JobEvent] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobEvent]) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if not subs:
                return
            self._subscribers[job_id] = [(lp, q) for (lp, q) in subs if q is not queue]
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    def publish(self, event: JobEvent) -> None:
        with self._lock:
            subs = list(self._subscribers.get(event.job_id, []))
        for loop, queue in subs:
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._put_nowait_drop_oldest, queue, event)

    @staticmethod
    def _put_nowait_drop_oldest(queue: asyncio.Queue[JobEvent], event: JobEvent) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
