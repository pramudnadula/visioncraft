# FaceVision — Cross-Platform AI Vision & Media Platform

**FaceVision** is a modern, local-first AI Computer Vision & Media Processing Platform evolved from a 2022 Python Haar Cascade script. It preserves full backward compatibility with the `face-detect` CLI while providing a React web app, FastAPI backend, Tauri 2 desktop shell, and modular CV/media studios:

- **AI Face Detection & Privacy Studio:** Haar Cascade (`detectMultiScale3`), YuNet, OpenCV DNN, SCRFD, and RetinaFace; privacy modes (`gaussian`, `pixelate`, `mask`, `mesh`, and `geometric` Delaunay facial landmark triangulation); selective face blurring; face crops; streaming video tracking; live webcam WebSocket detection.
- **Media Editing Studio (`apps/web/` + `backend/app/media/`):** Interactive Crop & Transform modal, Resize, Format Conversion (JPEG/PNG/WEBP/BMP/TIFF/GIF with alpha & animation handling), and iterative Target-KB Image Compression.
- **Object & Vehicle AI Studio (`backend/app/vision/`):** COCO 80-class YOLOv8 ONNX adapter, multi-class IoU tracking, line-crossing & polygon-zone vehicle counting, plus optional adapters for Face Mask classification, Facial Landmarks, Hands, Pose, and Sign Language sequence buffering (`ASL`, `FinSL`, `BSL`).
- **AI Model Hub (`backend/app/models/`):** HuggingFace ONNX search, SSRF-protected HTTPS download, local `.onnx` import/removal, environment compatibility checks, and live inference benchmarking.
- **Reports & Storage:** Plain-language **PDF**, **CSV**, and **JSON** reports; pluggable **local disk**, **S3-compatible** (Cloudflare R2, MinIO, Backblaze B2, AWS S3), and **Firebase / GCS** storage; **SQLite** (default) or **PostgreSQL/MySQL** via `.env`.

---

## Quick Start

### 1. Setup & Run Python Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[backend,dev,onnx]"
# or without editable install: pip install -r requirements.txt
cp .env.example .env
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
face-detect --image path/to/image.jpg --blur gaussian --output output/blurred.png --no-show
```

---

## Testing & Verification

```bash
.venv/bin/ruff check backend/ src/ tests/
.venv/bin/mypy backend/app
.venv/bin/pytest -v

cd apps/web && npm test && npm run build
```

---

## Documentation
- [Installation Guide](docs/installation.md)
- [Development Guide](docs/development.md)
- [User Guide](docs/user-guide.md)
- [API Overview](docs/api/overview.md)
- [Models Overview](docs/models/overview.md)
- [Storage & Databases](docs/deployment/storage-and-databases.md)
- [Desktop & Cloud Deployment](docs/deployment/desktop-and-cloud.md)
- [Security Overview](docs/security/overview.md)
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md)

Licensed under the [MIT License](LICENSE).
