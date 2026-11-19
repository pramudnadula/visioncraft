# Migration Plan — Legacy CLI to FaceVision

## Strategy

Adopt an **incremental, strangler-fig migration**:
1. Preserve `src/face_detector/` and `Face_Detector.py` so all existing CLI commands and tests remain green at every step.
2. Build the modernized shared engine under `backend/app/detection/`, `backend/app/privacy/`, and `backend/app/processing/` using the proven OpenCV/NumPy logic from `src/face_detector/` while adding lifecycle management, tracking, cancellation, structured logging, and security validation.
3. Layer FastAPI (`/api/v1` + `/ws/v1`), SQLAlchemy persistence, local/cloud storage adapters, and the background job worker on top of the engine.
4. Build the React + TypeScript + Tailwind frontend (`apps/web/`) and connect every control to real API/WebSocket endpoints.
5. Package the desktop application with Tauri 2 (`apps/desktop/`) reusing `apps/web/` and managing the local Python backend lifecycle.

## Ordered Implementation Phases

- **Phase 1: Repository Audit & Planning** (Current) — Audit repo, create architecture docs, baseline tests.
- **Phase 2: Backend Foundation & Shared Detection Engine** — Config, logging, exceptions, `FaceDetector` ABC, `DetectorRegistry`, Haar, YuNet, OpenCV DNN, optional SCRFD/RetinaFace adapters, privacy blur/pixelate/selective blur.
- **Phase 3: Image, Video, Tracking & Batch Processing** — Streaming frame-based video processing, IoU face tracker, cancellation, timeline events, JSON/CSV/ZIP exports, corrupt-video handling.
- **Phase 4: Storage, Database & Background Job System** — SQLite/PostgreSQL via SQLAlchemy, local UUID storage with path-traversal/signature/size/dimension validation, optional S3 adapter, async job queue & worker with WebSocket progress events.
- **Phase 5: FastAPI REST & WebSocket API** — Versioned `/api/v1` routes (`/health`, `/capabilities`, `/media`, `/jobs`, `/models`, `/settings`) and `/ws/v1/jobs/{job_id}` + webcam WS.
- **Phase 6: Modern Web Application (`apps/web/`)** — Dashboard, Image Studio (before/after, selective blur), Video Studio (player, timeline, tracking), Live Webcam (MediaDevices API), Batch Processing, History, Model Management, Settings, Hybrid mode consent UI.
- **Phase 7: Desktop Application (`apps/desktop/`)** — Tauri 2 configuration, Rust sidecar lifecycle manager for Python FastAPI, native file/folder dialogs, scoped capabilities, cross-platform build docs.
- **Phase 8: Security Hardening, Comprehensive Testing & Documentation** — Backend + API + security tests, frontend Vitest tests, Docker Compose, complete `docs/` suite.
