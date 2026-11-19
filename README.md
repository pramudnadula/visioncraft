# Face Detection Python

A modular, open-source face detection CLI and Python library built with OpenCV. Originally started in 2022 as a single-script Haar Cascade experiment, it is structured into clean, testable components for real-world image, batch directory, video, and live webcam workflows.

## Features

- **Multiple Input Sources**: Live webcam, single image, batch image directory, or video file.
- **Pluggable Detection Backends**:
  - **Haar Cascade** (default, zero extra downloads, CPU-fast)
  - **YuNet DNN** (`--yunet-model <model.onnx>` for higher accuracy with confidence scores)
- **Privacy Anonymization**: Blur (`--blur gaussian`) or pixelate (`--blur pixelate`) detected faces for GDPR/media workflows.
- **Face Crop Extraction**: Automatically extract and save cropped faces (`--save-crops <dir>`).
- **Structured JSON Reports**: Export bounding boxes, confidence scores, and frame counts (`--export-json <file.json>`).
- **Live HUD**: Real-time face counter and FPS overlay on video/webcam streams (`--no-hud` to disable).

---

## Architecture

```text
Face_Detection_Python/
├── src/face_detector/
│   ├── models.py      # Detection & DetectorConfig dataclasses
│   ├── backends.py    # Pluggable detectors (HaarCascadeBackend, YuNetBackend)
│   ├── overlay.py     # Bounding boxes, Gaussian/pixelate privacy blur, HUD
│   ├── pipeline.py    # Image, batch directory, video/webcam, crops, JSON export
│   └── cli.py         # Argparse CLI (`face-detect` / `python -m face_detector`)
├── tests/             # Pytest unit & integration tests
├── Face_Detector.py   # Backward-compatible CLI wrapper
├── pyproject.toml     # Package metadata & console script entrypoint
├── Makefile           # One-command setup, test, and demo tasks
└── requirements.txt   # Pinned runtime dependencies
```

| Component | Responsibility |
| :--- | :--- |
| `models.py` | Immutable `Detection` objects and `DetectorConfig` settings. |
| `backends.py` | `FaceDetectorBackend` protocol, `HaarCascadeBackend`, `YuNetBackend`, and factory. |
| `overlay.py` | Frame annotation, ROI-safe Gaussian/pixelate anonymization, and FPS/count HUD. |
| `pipeline.py` | End-to-end orchestration for images, folders, video streams, crop saving, and JSON output. |
| `cli.py` | Unified CLI entrypoint exposed as `face-detect`, `python -m face_detector`, and `Face_Detector.py`. |

---

## Quick Setup

### Option 1: One-Command Setup (Recommended)

```bash
make setup
source .venv/bin/activate
```

### Option 2: Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

*(If you only want runtime dependencies without editable install: `pip install -r requirements.txt`)*

---

## CLI Usage

Once `.venv` is activated, you can use `face-detect` (or `python Face_Detector.py`):

### 1. Live Webcam
Press `q` or `Esc` to quit.
```bash
face-detect
```

### 2. Single Image
```bash
face-detect --image RDJ.png
```

### 3. Privacy Blur / Pixelation
```bash
face-detect --image me.png --blur gaussian --output output/me_blurred.png --no-show
face-detect --video clip.mp4 --blur pixelate --output output/clip_pixelated.mp4 --no-show
```

### 4. Extract Face Crops + Export JSON Metadata
```bash
face-detect --image rdi.jpg --save-crops output/crops --export-json output/rdi.json --no-show
```

### 5. Batch Process a Directory of Images
```bash
face-detect --input-dir . --output output/batch --export-json output/batch.json
```

### 6. Use Modern YuNet DNN Backend (Optional ONNX Model)
```bash
face-detect --image RDJ.png --yunet-model path/to/face_detection_yunet_2023mar.onnx
```

---

## Python API Usage

```python
from pathlib import Path
from face_detector import DetectorConfig, FacePipeline

config = DetectorConfig(blur_mode="gaussian", show_hud=True)
pipeline = FacePipeline(config=config)

result = pipeline.process_image(
    Path("RDJ.png"),
    output_path=Path("output/RDJ_blurred.png"),
    crops_dir=Path("output/crops"),
    show=False,
)
print(result.to_dict())
```

---

## Running Tests

```bash
make test
```

---

## Contributing & License

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
Licensed under the [MIT License](LICENSE).
