# FaceVision Models Directory

FaceVision uses local-first model discovery in `models/`:

1. **Haar Cascade (`haar`)** — Built-in via `haarcascade_frontalface_default.xml` (BSD-3-Clause, OpenCV). Works offline with zero extra downloads.
2. **YuNet (`yunet`)** — Place `face_detection_yunet_2023mar.onnx` here (MIT License, OpenCV Zoo). Enables confidence scores and 5-point facial landmarks.
3. **OpenCV DNN (`opencv_dnn`)** — Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` here (BSD-3-Clause, OpenCV).
4. **SCRFD (`scrfd`)** — Optional adapter; requires `onnxruntime` and `scrfd_10g_bnkps.onnx`.
5. **RetinaFace (`retinaface`)** — Optional adapter; requires `onnxruntime` and `retinaface.onnx`.
6. **Face Mesh (`face_landmarker.task`)** — Google MediaPipe Face Landmarker, Apache-2.0, 3.76 MB, bundled. See `docs/models/face-mesh.md` for checksum, install and limitations.
7. **Selfie Multiclass Segmenter (`selfie_multiclass_256x256.tflite`)** — Google MediaPipe Image Segmenter, Apache-2.0, 15.6 MB (`sha256: c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`). Powers contoured real-world face/head privacy masks (`gaussian`, `pixelate`, `mask`, `mesh`).
8. **Pose Landmarker (`pose_landmarker_full.task`, preferred; `pose_landmarker_lite.task`, fallback)** — Google MediaPipe Pose Landmarker, Apache-2.0. Full: 9.4 MB (`sha256: 4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`), from `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`. Lite: 5.5 MB (`sha256: 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`). Powers 33-point 3D human body pose estimation; group photos are split per person using YOLOv8n person boxes.
9. **Hand Landmarker (`hand_landmarker.task`)** — Google MediaPipe Hand Landmarker, Apache-2.0, 7.5 MB (`sha256: fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`). Powers 21-point 3D hand landmark detection.
10. **YOLOv8n (`yolov8n.onnx`)** — Ultralytics YOLOv8n COCO ONNX (`sha256: 162ec5e9d2886fc411e4b81811c731e81a72c8f85571de1e0ab040833d92d4f2`). Powers 80-class object & vehicle detection/counting.
11. **Face Mask Classifier (`face_mask_classifier.onnx`)** — ONNX 224×224 NHWC face mask classifier (`sha256: fcc9e8d5a19b8000e11fbae4e2cb40315c477c32c5dbf533d3a00f84fd258c00`). Powers face mask detection (`mask_detected`, `no_mask`).
12. **FER+ Emotion (`emotion-ferplus-8.onnx`)** — ONNX Model Zoo FER+ 8-class facial expression model (`sha256: a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`). Powers real-time webcam emotion analysis alongside MediaPipe blendshapes.
13. **U2-Net-P (`u2netp.onnx`)** — Universal salient-object background removal ONNX model (`sha256: 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`). Powers AI background removal (`transparent`, solid color, or blurred background).
14. **Gesture Recognizer (`gesture_recognizer.task`)** — Google MediaPipe Gesture Recognizer, Apache-2.0, 8.0 MB (`sha256: 97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`). Powers hand gesture recognition (`Thumb_Up`, `Victory`, `ILoveYou`, `Open_Palm`, etc.).
15. **ASL YOLO11n (`asl_yolo11n.onnx`)** — Ultralytics YOLO11n ASL fingerspelling ONNX (`sha256: e000613d16590ebcde988964ab56d577a2baea8190673602cc006622a6d78aec`). Powers 25-class ASL alphabet detection (`A`–`Y`).
