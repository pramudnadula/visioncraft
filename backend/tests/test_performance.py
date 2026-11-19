"""Performance and concurrency tests for FaceVision."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.detection.backends.haar import HaarCascadeDetector
from backend.app.detection.base import DetectorConfig
from backend.app.main import create_app
from backend.app.processing.image import ImageProcessingConfig, ImageProcessingService


def test_large_4k_image_processing_completes_cleanly(tmp_path: Path) -> None:
    large_path = tmp_path / "uhd_4k.png"
    img = np.full((2160, 3840, 3), 160, dtype=np.uint8)
    cv2.imwrite(str(large_path), img)

    detector = HaarCascadeDetector(DetectorConfig(model_id="haar"))
    detector.initialize()
    service = ImageProcessingService()

    start = time.perf_counter()
    res = service.process(
        large_path,
        detector,
        tmp_path / "out",
        config=ImageProcessingConfig(blur_mode="none"),
    )
    elapsed = time.perf_counter() - start
    detector.release()

    assert res.width == 3840
    assert res.height == 2160
    assert elapsed < 15.0


def test_concurrent_jobs_complete_without_errors(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        max_concurrent_jobs=2,
    )
    img = np.full((240, 240, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    png_bytes = buf.tobytes()

    with TestClient(create_app(custom_settings=settings)) as client:
        media_ids: list[str] = []
        for idx in range(3):
            up = client.post(
                "/api/v1/media/upload",
                files={"file": (f"img_{idx}.png", png_bytes, "image/png")},
            )
            assert up.status_code == 201
            media_ids.append(up.json()["id"])

        job_ids = [
            client.post(
                "/api/v1/jobs",
                json={"job_type": "image", "media_ids": [mid], "model_id": "haar"},
            ).json()["id"]
            for mid in media_ids
        ]

        completed = set()
        for _ in range(60):
            for jid in job_ids:
                if jid not in completed:
                    st = client.get(f"/api/v1/jobs/{jid}").json()["status"]
                    if st == "COMPLETED":
                        completed.add(jid)
            if len(completed) == len(job_ids):
                break
            time.sleep(0.1)

        assert len(completed) == 3
