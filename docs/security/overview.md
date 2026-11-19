# FaceVision API & Security Overview

## REST Endpoints (`/api/v1`)
- `GET /health` — Service health & execution mode.
- `GET /capabilities` — Upload limits, supported extensions, cloud opt-in status.
- `GET /stats` — Real-time dashboard statistics from SQLite/PostgreSQL.
- `POST /media/upload`, `GET /media`, `GET /media/{id}`, `GET /media/{id}/file`, `DELETE /media/{id}`
- `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `POST /jobs/{id}/retry`, `DELETE /jobs/{id}`, `GET /jobs/{id}/result`, `GET /jobs/{id}/artifacts/{path}`
- `GET /models` — Available detection backends and metadata.
- `GET /settings`, `PUT /settings` — Persisted user preferences.

## WebSocket Endpoints (`/ws/v1`)
- `/ws/v1/jobs/{job_id}` — Real-time job state & progress stream.
- `/ws/v1/webcam` — Live frame detection & privacy blur stream.

## Security Controls
- **Magic-Byte Signature Validation**: Uploaded media headers are verified against supported image/video signatures.
- **UUID Storage & Path Traversal Prevention**: User filenames are never used as storage paths; all reads/writes are confined to `data_dir`.
- **Cloud Consent Enforcement**: `processing_mode="cloud"` is rejected with `403 cloud_consent_required` unless explicitly consented and enabled.
- **Optional Local API Token**: Set `FACEVISION_API_TOKEN` to require `X-FaceVision-Token` / `Authorization: Bearer` on all protected endpoints and WebSockets.
