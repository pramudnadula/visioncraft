# Installation & Quick Start

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Node.js 20+ (for the React web UI)
- Rust & Cargo (optional, only required to build the Tauri 2 desktop binary)
- FFmpeg (optional, for Docker/server transcoding)

## 1. Python Backend & CLI Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[backend,dev]"
```

Start the local FastAPI backend:

```bash
facevision-api
# or: uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## 2. Web Application Setup

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` and `/ws` to `http://127.0.0.1:8000`.

## 3. Desktop Application (Tauri 2)

```bash
cd apps/desktop
npm install
npm run dev
```
