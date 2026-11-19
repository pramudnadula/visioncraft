"""Environment compatibility and model benchmarking services for AI Model Hub."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.detection.base import DetectorConfig
from backend.app.detection.registry import DetectorRegistry


@dataclass(frozen=True, slots=True)
class EnvironmentCompatibility:
    python_version: str
    os_name: str
    architecture: str
    cpu_count: int
    opencv_version: str
    onnxruntime_available: bool
    onnx_providers: list[str]
    cuda_devices: int
    mediapipe_available: bool
    free_disk_gb: float


def check_environment_compatibility(data_dir: Path) -> dict[str, Any]:
    """Inspect host Python, OS, OpenCV, ONNXRuntime, GPU, and disk capabilities."""
    ort_ok = importlib.util.find_spec("onnxruntime") is not None
    providers: list[str] = []
    if ort_ok:
        import onnxruntime as ort  # type: ignore[import-not-found]

        providers = list(ort.get_available_providers())

    cuda_count = 0
    if hasattr(cv2, "cuda"):
        try:
            cuda_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            cuda_count = 0

    usage = shutil.disk_usage(data_dir if data_dir.exists() else Path.cwd())
    info = EnvironmentCompatibility(
        python_version=platform.python_version(),
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        opencv_version=cv2.__version__,
        onnxruntime_available=ort_ok,
        onnx_providers=providers,
        cuda_devices=cuda_count,
        mediapipe_available=importlib.util.find_spec("mediapipe") is not None,
        free_disk_gb=round(usage.free / (1024**3), 2),
    )
    return asdict(info)


def benchmark_detector(
    registry: DetectorRegistry,
    model_id: str,
    *,
    iterations: int = 10,
) -> dict[str, Any]:
    """Run a real timed benchmark of a registered detector on a 640x480 test frame."""
    t_init_start = time.perf_counter()
    detector = registry.create_detector(DetectorConfig(model_id=model_id))
    init_ms = (time.perf_counter() - t_init_start) * 1000.0

    frame = np.full((480, 640, 3), 140, dtype=np.uint8)
    try:
        # Warmup pass
        detector.detect(frame)

        times_ms: list[float] = []
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            detector.detect(frame)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(times_ms) / len(times_ms)
        fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        meta = detector.get_metadata()
        return {
            "model_id": model_id,
            "model_name": meta.name,
            "iterations": len(times_ms),
            "resolution": "640x480",
            "initialization_ms": round(init_ms, 2),
            "avg_inference_ms": round(avg_ms, 2),
            "estimated_fps": round(fps, 1),
            "execution_provider": "CUDA" if meta.gpu_active else "CPU",
        }
    finally:
        detector.release()
