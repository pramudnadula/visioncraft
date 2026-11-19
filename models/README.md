# FaceVision Models Directory

FaceVision uses local-first model discovery in `models/`:

1. **Haar Cascade (`haar`)** — Built-in via `haarcascade_frontalface_default.xml` (BSD-3-Clause, OpenCV). Works offline with zero extra downloads.
2. **YuNet (`yunet`)** — Place `face_detection_yunet_2023mar.onnx` here (MIT License, OpenCV Zoo). Enables confidence scores and 5-point facial landmarks.
3. **OpenCV DNN (`opencv_dnn`)** — Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` here (BSD-3-Clause, OpenCV).
4. **SCRFD (`scrfd`)** — Optional adapter; requires `onnxruntime` and `scrfd_10g_bnkps.onnx`.
5. **RetinaFace (`retinaface`)** — Optional adapter; requires `onnxruntime` and `retinaface.onnx`.
6. **Face Mesh (`face_landmarker.task`)** — Google MediaPipe Face Landmarker, Apache-2.0, 3.76 MB, bundled. See `docs/models/face-mesh.md` for checksum, install and limitations.
