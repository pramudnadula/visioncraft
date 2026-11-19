# Feature Mapping — Legacy to FaceVision

| Legacy Feature | Legacy Location | FaceVision Target | Notes |
| :--- | :--- | :--- | :--- |
| `Detection` & `DetectorConfig` | `src/face_detector/models.py` | `backend/app/detection/base.py` & `backend/app/schemas/` | Extended with `landmarks`, `tracking_id`, `model_id`, `model_version`. |
| `HaarCascadeBackend` | `src/face_detector/backends.py` | `backend/app/detection/backends/haar.py` | Implements `initialize()`, `detect()`, `get_metadata()`, `is_available()`, `release()`. |
| `YuNetBackend` | `src/face_detector/backends.py` | `backend/app/detection/backends/yunet.py` | Adds model discovery in `models/`, landmarks extraction, confidence filtering. |
| — (New) | — | `backend/app/detection/backends/opencv_dnn.py`, `scrfd.py`, `retinaface.py` | Optional adapters registered in `DetectorRegistry`. |
| `apply_privacy_blur` | `src/face_detector/overlay.py` | `backend/app/privacy/blur.py` | Adds selective face blurring (by face index/ID) alongside full-frame blur/pixelate. |
| `FacePipeline.process_image` | `src/face_detector/pipeline.py` | `backend/app/processing/image.py` | Adds dimension/signature validation, before/after outputs, crops, JSON/CSV reports. |
| `FacePipeline.process_directory` | `src/face_detector/pipeline.py` | `backend/app/processing/batch.py` | Adds per-item status, failure isolation, cancellation, ZIP export. |
| `FacePipeline.process_video` | `src/face_detector/pipeline.py` | `backend/app/processing/video.py` + `tracking.py` | Streaming frames, configurable FPS/resolution, IoU tracker, timeline JSON, cancellation. |
| Webcam loop (`cv2.VideoCapture(0)`) | `src/face_detector/pipeline.py` | `backend/app/processing/webcam.py` + Web `MediaDevices` | Browser uses `MediaDevices` + WS/frame API; desktop/CLI uses OpenCV capture. |
| `export_json_report` | `src/face_detector/pipeline.py` | `backend/app/processing/` + `/api/v1/jobs/{id}/result` | Adds CSV summary, ZIP batch archive, and structured job metadata. |
| `face-detect` CLI | `src/face_detector/cli.py` & `Face_Detector.py` | Preserved in `src/face_detector/cli.py` | Backward compatibility maintained and tested. |
