# FaceVision Detection Models Overview

| Model ID | Name | License | Confidence | Landmarks | Default Files |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `haar` | Haar Cascade (Frontal Face) | BSD-3-Clause (OpenCV) | Yes (`detectMultiScale3`) | No | `haarcascade_frontalface_default.xml` (bundled) |
| `yunet` | YuNet (`FaceDetectorYN`) | MIT (OpenCV Zoo) | Yes | 5-point | `models/face_detection_yunet_2023mar.onnx` |
| `opencv_dnn` | OpenCV DNN (Res10 SSD) | BSD-3-Clause (OpenCV) | Yes | No | `models/deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` |
| `scrfd` | SCRFD (Optional ONNX) | MIT code; non-commercial weights | Yes | 5-point | `models/scrfd_10g_bnkps.onnx` + `onnxruntime` |
| `retinaface` | RetinaFace (Optional ONNX) | Check exported weights license | Yes | 5-point | `models/retinaface.onnx` + `onnxruntime` |

## GPU Support
Detectors only report `supports_gpu: true` and `gpu_active: true` when the actual runtime (`cv2.cuda` or `onnxruntime` `CUDAExecutionProvider`) is available on the host.
