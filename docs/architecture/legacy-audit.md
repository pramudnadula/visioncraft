# Legacy Repository Audit — Face_Detection_Python

**Audit Date:** 2026-11-19  
**Target:** Transform `Face_Detection_Python` (v2.0.0) into **FaceVision** (Web + Desktop + Hybrid AI Face Detection & Privacy Platform).

---

## 1. Existing Project Structure

```text
Face_Detection_Python/
├── Face_Detector.py                  # Backward-compatible CLI wrapper
├── haarcascade_frontalface_default.xml # Bundled OpenCV Haar Cascade (~930 KB)
├── Makefile                          # setup, test, demo, clean targets
├── pyproject.toml                    # Package metadata (face-detector 2.0.0), pytest config
├── requirements.txt                  # Pinned numpy==2.4.6, opencv-python==4.14.0.94
├── README.md                         # CLI & Python API documentation
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── RDJ.png, me.png, rdi.jpg          # Sample test images at repo root
├── clip.mp4, clips.mp4               # Sample test videos at repo root
├── output/                           # Generated demo outputs (me_blurred.png, clip_pixelated.mp4)
├── src/face_detector/
│   ├── __init__.py                   # Public package exports
│   ├── __main__.py                   # `python -m face_detector` entrypoint
│   ├── models.py                     # BlurMode, Detection, DetectorConfig dataclasses
│   ├── backends.py                   # FaceDetectorBackend Protocol, HaarCascadeBackend, YuNetBackend, create_backend
│   ├── overlay.py                    # _clip_roi, apply_privacy_blur, draw_detections, draw_hud
│   ├── pipeline.py                   # ImageResult, FacePipeline, export_json_report
│   └── cli.py                        # argparse CLI (`face-detect`)
└── tests/
    └── test_face_detector.py         # 6 pytest unit/integration tests
```

---

## 2. Current Execution Flow

```text
Entrypoint (`face-detect` | `python -m face_detector` | `python Face_Detector.py`)
  └─▶ cli.py: `build_parser()` + `main(argv)`
        └─▶ models.py: `DetectorConfig(...)`
              └─▶ pipeline.py: `FacePipeline(config)`
                    └─▶ backends.py: `create_backend(config)`
                          ├─▶ `YuNetBackend` (if `yunet_model_path` set)
                          └─▶ `HaarCascadeBackend` (default, loads `haarcascade_frontalface_default.xml`)
                    └─▶ Input routing (`--image` | `--input-dir` | `--video` | `--webcam`)
                          └─▶ `backend.detect(frame)` -> `list[Detection]`
                          └─▶ `overlay.draw_detections(frame, detections, config, fps)`
                                └─▶ `apply_privacy_blur()` + bounding box + confidence + `draw_hud()`
                          └─▶ Output management (`cv2.imwrite`, `cv2.VideoWriter`, `save_crops`, `export_json_report`, `cv2.imshow`)
```

---

## 3. Existing File & Module Responsibilities

| Module | Responsibility | Quality & Status |
| :--- | :--- | :--- |
| `src/face_detector/models.py` | `BlurMode`, `Detection`, `DetectorConfig` | Clean, typed, immutable `Detection` (`frozen=True, slots=True`). |
| `src/face_detector/backends.py` | `FaceDetectorBackend` protocol, `HaarCascadeBackend`, `YuNetBackend`, `create_backend` | Clean, but lacks lifecycle (`initialize`/`release`/`is_available`/`get_metadata`), landmarks, tracking IDs, and registry. |
| `src/face_detector/overlay.py` | ROI clipping, Gaussian/pixelate blur, bounding boxes, HUD | Solid ROI-clamped blur/pixelate logic. Reusable directly or via privacy module. |
| `src/face_detector/pipeline.py` | Image, batch dir, video, webcam orchestration, crop saving, JSON export | Uses `print()` instead of `logging`; synchronous; no cancellation, progress callbacks, tracking, or corrupt-video guards. |
| `src/face_detector/cli.py` | `argparse` CLI | Clean and well-tested; must remain backward-compatible. |
| `Face_Detector.py` | Root wrapper adding `src/` to `sys.path` | Backward-compatible entrypoint; preserve. |
| `tests/test_face_detector.py` | 6 unit/integration tests | All 6 pass (`pytest -v` in 1.52s). Relies on sample media at repo root. |

---

## 4. Reusable Components

1. **Detection Core (`HaarCascadeBackend`, `YuNetBackend`)**: Working OpenCV Haar and YuNet ONNX implementations.
2. **Privacy & Overlay (`apply_privacy_blur`, `draw_detections`, `draw_hud`, `_clip_roi`)**: Safe ROI clamping, Gaussian kernel sizing, and pixelation logic.
3. **CLI & Public Package (`src/face_detector/`)**: Can be preserved as the backward-compatible CLI & library, delegating to or wrapping the modernized shared engine.
4. **Bundled Model (`haarcascade_frontalface_default.xml`)**: Validated Haar model at repo root (can also be copied/referenced in `models/`).

---

## 5. Technical Debt & Gaps

1. **No Web/API Layer**: No FastAPI server, no REST endpoints, no WebSocket progress streaming.
2. **No Background Job System**: Video/batch processing blocks the caller; no job queue, state machine, cancellation, retry, or persistence.
3. **No Persistence or Storage Abstraction**: Outputs are written directly to arbitrary paths; no UUID media storage, no SQLite/PostgreSQL metadata, no retention/cleanup.
4. **No Security Controls**: No file signature validation, upload size/dimension/duration limits, path traversal protection, or local API auth.
5. **No Face Tracking or Selective Blur**: Video processing treats every frame independently without tracking IDs, timeline events, or selective face blurring.
6. **Logging & Error Handling**: `pipeline.py` uses `print()` statements; video writer failures or zero-frame/corrupt videos are not explicitly handled.
7. **Hardcoded Paths**: `src/face_detector/` itself has **no** hardcoded media paths (only `DEFAULT_CASCADE_FILE = "haarcascade_frontalface_default.xml"`). However, `Makefile` (`demo` target) and `README.md` reference root sample files (`RDJ.png`, `me.png`, `rdi.jpg`, `clip.mp4`).
8. **Environment Tooling Notes**:
   - Python 3.11.7 (`.venv`), Node.js v22.23.4, and npm are available.
   - `cargo`/`rustc` (needed to compile Tauri 2 Rust binary) and `ffmpeg` are **not** currently installed on PATH in this environment.

---

## 6. Dependencies Assessment

- **Retain**: `numpy>=2.0,<3`, `opencv-python>=4.10,<5`, `pytest>=8.0`.
- **Add (Backend)**: `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pydantic-settings`, `sqlalchemy>=2.0`, `python-multipart`, `websockets`, `httpx` (dev/test), `ruff` / `mypy` (dev). Optional: `boto3` (S3 cloud storage), `onnxruntime` (for optional SCRFD/RetinaFace adapters if installed).
- **Add (Frontend)**: React 18/19, TypeScript, Vite, Tailwind CSS, Lucide icons, React Router, TanStack Query, Vitest, React Testing Library.
- **Add (Desktop)**: Tauri 2 (`@tauri-apps/api`, `@tauri-apps/cli`, `src-tauri/`).

---

## 7. Breaking-Change Risks

1. **CLI Compatibility**: Existing `face-detect`, `python -m face_detector`, and `python Face_Detector.py` invocations and `from face_detector import ...` imports must continue working without breaking `tests/test_face_detector.py`.
2. **OpenCV Version**: `opencv-python==4.14.0.94` with `numpy==2.4.6` is installed in `.venv`. Any new dependencies must remain compatible with NumPy 2.x.
3. **Headless Video Codecs**: OpenCV `VideoWriter` with `mp4v` works in tests, while browser `<video>` playback prefers H.264 (`avc1`) or `mp4v`/WebM fallback; we should handle codec selection cleanly and report if `ffmpeg` transcoding is unavailable.
