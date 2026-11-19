# FaceVision REST & WebSocket API Reference

Base URL: `/api/v1`

## Authentication
If `FACEVISION_API_TOKEN` is set on the server, include one of:
- `X-FaceVision-Token: <token>`
- `Authorization: Bearer <token>`

---

## System Endpoints
- `GET /api/v1/health` — Returns service status, app version, and active `execution_mode`.
- `GET /api/v1/capabilities` — Returns upload limits, supported extensions, storage backend, and cloud opt-in state.
- `GET /api/v1/stats` — Returns live counts from SQLite/PostgreSQL (`total_media`, `total_jobs`, `completed_jobs`, `active_jobs`, `failed_jobs`, `total_faces_detected`).

## Media Endpoints
- `POST /api/v1/media/upload` — Multipart upload with magic-byte signature validation, UUID storage key, and duplicate SHA-256 detection.
- `GET /api/v1/media` — List uploaded media metadata.
- `GET /api/v1/media/{media_id}` — Get metadata for a single asset.
- `GET /api/v1/media/{media_id}/file` — Stream/download the original uploaded media safely.
- `DELETE /api/v1/media/{media_id}` — Delete media metadata and stored file.

## Job Endpoints
- `POST /api/v1/jobs` — Enqueue an `image`, `video`, or `batch` job. Cloud jobs require `processing_mode="cloud"` and `cloud_consent=true` (returns `403 cloud_consent_required` otherwise).
- `GET /api/v1/jobs` — List jobs (optional `?status=` filter).
- `GET /api/v1/jobs/{job_id}` — Get job status, progress (`0.0–1.0`), and manifest.
- `POST /api/v1/jobs/{job_id}/cancel` — Cancel a queued or running job.
- `POST /api/v1/jobs/{job_id}/retry` — Retry a `FAILED` or `CANCELLED` job.
- `GET /api/v1/jobs/{job_id}/result` — Return completed job manifest.
- `GET /api/v1/jobs/{job_id}/artifacts/{artifact_path}` — Download `annotated.png`, `privacy_blurred.png`, `processed.mp4`, `report.json`, `summary.csv`, `batch_outputs.zip`, or face crops.
- `DELETE /api/v1/jobs/{job_id}` — Delete job metadata and output artifacts.

## Model Endpoints
- `GET /api/v1/models` — List registered detectors and runtime availability.
- `GET /api/v1/models/custom` — List custom imported models in `models/custom/`.
- `GET /api/v1/models/search?q=<query>` — Search HuggingFace Hub for ONNX models.
- `POST /api/v1/models/download` — Download a public `https://` model into `models/custom/` (SSRF-protected).
- `POST /api/v1/models/import` — Upload a local `.onnx`/`.xml`/`.prototxt`/`.caffemodel` file.

## Settings Endpoints
- `GET /api/v1/settings` — Get persisted user preferences.
- `PUT /api/v1/settings` — Update default model, blur mode, confidence threshold, cloud opt-in, cloud URL, and retention days.
- `POST /api/v1/settings/test-cloud` — Test reachability of a remote FaceVision server's `/api/v1/health`.

## WebSocket Endpoints
- `WS /ws/v1/jobs/{job_id}` — Real-time job progress events.
- `WS /ws/v1/webcam` — Real-time base64 frame detection and privacy blur stream.
