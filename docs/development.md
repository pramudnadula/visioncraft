# Development & Testing Guide

## Backend Tests, Lint & Type Checking

```bash
.venv/bin/pytest -v
.venv/bin/mypy backend/app
.venv/bin/ruff check backend/ src/ tests/
```

## Frontend Tests & Production Build

```bash
cd apps/web
npm test
npm run build
```

## Backward-Compatible CLI

```bash
face-detect --image path/to/image.jpg --blur gaussian --output output/blurred.png --no-show
```
