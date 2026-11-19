# Desktop & Cloud Deployment Guide

## 1. Desktop Application (Tauri 2)

### Prerequisites
- Rust & Cargo (`https://rustup.rs`)
- Node.js 20+
- Python 3.10+ with FaceVision backend dependencies installed

### Development
```bash
cd apps/desktop
npm install
npm run dev
```

### Production Builds (macOS / Windows / Linux)
```bash
cd apps/desktop
npm run build
```
Output installers are generated under `apps/desktop/src-tauri/target/release/bundle/`.

---

## 2. Server / Cloud Deployment (Docker Compose)

1. Copy `.env.example` to `.env` and set `FACEVISION_API_TOKEN` and any optional S3/R2/PostgreSQL settings.
2. Start the backend and PostgreSQL:
   ```bash
   docker compose up -d --build
   ```
3. Verify health:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```
