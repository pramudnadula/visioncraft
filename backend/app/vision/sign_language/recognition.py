"""Sign language sequence buffer and language-specific model registry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SignLanguageSpec:
    code: str
    name: str
    model_available: bool
    note: str


SUPPORTED_SIGN_LANGUAGES: tuple[SignLanguageSpec, ...] = (
    SignLanguageSpec(
        code="ASL",
        name="American Sign Language",
        model_available=False,
        note="Requires an ASL landmark sequence ONNX model and evaluation dataset.",
    ),
    SignLanguageSpec(
        code="FinSL",
        name="Finnish Sign Language",
        model_available=False,
        note="Requires a FinSL landmark sequence ONNX model and evaluation dataset.",
    ),
    SignLanguageSpec(
        code="BSL",
        name="British Sign Language",
        model_available=False,
        note="Requires a BSL landmark sequence ONNX model and evaluation dataset.",
    ),
)


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


def list_sign_languages() -> list[dict[str, Any]]:
    return [
        {
            "code": s.code,
            "name": s.name,
            "model_available": s.model_available,
            "note": s.note,
        }
        for s in SUPPORTED_SIGN_LANGUAGES
    ]
