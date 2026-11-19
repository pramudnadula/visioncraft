# Target Architecture — FaceVision

## 1. High-Level Overview

FaceVision is a hybrid, cross-platform AI Face Detection & Privacy platform built around a **single shared Python computer vision engine** powering:
1. **CLI & Python Library** (`src/face_detector/` — backward-compatible).
2. **FastAPI Backend** (`backend/app/` — REST + WebSocket API, background job worker, SQLite/PostgreSQL metadata, local/S3 storage).
3. **Modern Web Application** (`apps/web/` — React + TypeScript + Vite + Tailwind CSS).
4. **Desktop Application** (`apps/desktop/` — Tauri 2 wrapping the React frontend and managing the local Python backend lifecycle).

---

## 2. Repository Layout

```text
Face_Detection_Python/
├── apps/
│   ├── web/                          # React + TypeScript + Vite + Tailwind web app
│   └── desktop/                      # Tauri 2 desktop shell (src-tauri/)
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI application factory & lifespan
│   │   ├── api/
│   │   │   ├── routes/               # health, capabilities, media, jobs, models, settings, webcam
│   │   │   └── websocket.py          # /ws/v1/jobs/{job_id} & /ws/v1/webcam
│   │   ├── core/                     # config, logging, exceptions, security, dependencies
│   │   ├── detection/                # Shared FaceDetector ABC, registry, backends (haar, yunet, opencv_dnn, scrfd, retinaface)
│   │   ├── processing/               # image, video, webcam, batch, tracking
│   │   ├── privacy/                  # blur (gaussian, pixelate, selective)
│   │   ├── jobs/                     # manager, worker, events, state machine
│   │   ├── storage/                  # base, local (UUID + retention), cloud (S3-compatible)
│   │   ├── database/                 # SQLAlchemy models & session management (SQLite/PostgreSQL)
│   │   └── schemas/                  # Pydantic v2 request/response schemas
│   └── tests/                        # Backend unit, integration, security, and API tests
├── src/face_detector/                # Backward-compatible CLI & Python API
├── models/                           # Local model storage & manifest (Haar XML, optional ONNX)
├── docs/                             # Architecture, API, models, security, deployment, user guide
├── docker/                           # Dockerfile & server deployment configs
├── docker-compose.yml                # Optional server/cloud deployment
├── pyproject.toml                    # Unified Python project & dependencies
└── README.md
```

---

## 3. Execution Modes

| Mode | Where Processing Runs | Media Storage | Offline Support |
| :--- | :--- | :--- | :--- |
| **LOCAL DESKTOP** | Embedded/local Python FastAPI sidecar managed by Tauri 2 | Local filesystem + SQLite | Full offline after install |
| **WEB LOCAL AGENT** | Local Python FastAPI service on user's machine (`127.0.0.1`) | Local filesystem + SQLite | Full offline |
| **CLOUD (Optional)** | Remote FastAPI + worker + PostgreSQL + S3-compatible storage | S3-compatible object storage | Requires explicit user consent per transfer |

**Non-negotiable invariant:** Never silently fall back from local processing to cloud processing.
