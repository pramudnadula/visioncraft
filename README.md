# FaceVision — AI Face Detection & Privacy Platform

**FaceVision** is a modern, cross-platform AI face detection, tracking, and privacy anonymization platform. Evolved from the legacy 2022 Python face-detection script, it preserves full backward compatibility with the `face-detect` CLI while providing:

- **Modern Web App** (`apps/web/`): React, TypeScript, Vite, Tailwind CSS, TanStack Query, React Router.
- **Desktop App** (`apps/desktop/`): Tauri 2 shell managing the local Python backend lifecycle.
- **Shared Python CV Engine & FastAPI Backend** (`backend/app/`):
  - Pluggable detectors: Haar Cascade (built-in), YuNet ONNX, OpenCV DNN, and optional SCRFD/RetinaFace adapters.
  - Image Studio: dynamic uploads, selective privacy blur (Gaussian / pixelate), face crops, JSON/CSV reports.
  - Video Studio: streaming frame-by-frame processing, temporary IoU face tracking, cancellation, MP4 + timeline JSON/CSV export.
  - Live Webcam: real-time WebSocket detection & privacy blur via browser `MediaDevices`.
  - Batch Processing: multi-file & folder import with ZIP export.
  - Hybrid Local-First Privacy: local processing by default; cloud processing strictly opt-in with explicit consent.

---

## Quick Start

### 1. Setup & Run Python Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[backend,dev]"
facevision-api
```

### 2. Run Web Application

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`.

### 3. Legacy CLI (Preserved)

```bash
face-detect --image RDJ.png --blur gaussian --output output/RDJ_blurred.png --no-show
```

---

## Testing & Verification

```bash
# Python backend + legacy CLI tests & type check
.venv/bin/pytest -v
.venv/bin/mypy backend/app

# Frontend unit tests & production build
cd apps/web && npm test && npm run build
```

---

## Documentation
- [Installation Guide](docs/installation.md)
- [Development Guide](docs/development.md)
- [User Guide](docs/user-guide.md)
- [Security Overview](docs/security/overview.md)
- [Architecture Audit & Migration Plan](docs/architecture/legacy-audit.md)
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md)

Licensed under the [MIT License](LICENSE).
