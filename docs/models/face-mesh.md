# Face Mesh (MediaPipe Face Landmarker)

Real-time 3D face mesh for images, videos and the live webcam, with multi-face tracking,
configurable styling and JSON/CSV/PDF report output.

## Model

| | |
| :--- | :--- |
| Model | `face_landmarker.task` (float16) — Google MediaPipe Face Landmarker |
| Contents | face detector, 478-point 3D landmark model, 52-score blendshape model, geometry pipeline |
| Library | `mediapipe` (Tasks API), Apache-2.0 |
| Model licence | Apache-2.0 (Google MediaPipe) — attribution retained in this document |
| Size | 3.76 MB, bundled in `models/` |
| SHA-256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Download | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| Offline | Yes — inference runs fully on the local CPU; nothing is sent to any server |

The checksum is verified when the model is used. A different file still loads, but a warning is
logged and the Models/Status API reports `checksum_verified: false`.

## Install

`mediapipe` hard-depends on `opencv-contrib-python`, which conflicts with the `opencv-python`
this project pins. Install it with the provided script, which uses `--no-deps`:

```bash
sh scripts/install_face_mesh.sh
```

## Compatibility (tested)

- Python 3.11, macOS arm64, NumPy 2.4, OpenCV 4.14, MediaPipe 1.2.3.
- MediaPipe >= 1.0 removed the legacy `mp.solutions` API. FaceVision uses the Tasks API for face
  mesh. The older hand and pose adapters still use `mp.solutions` and report themselves as
  unavailable on MediaPipe 1.x until they are moved to the Tasks API.

## Performance (measured, CPU only, Apple Silicon)

| Input | Detect + render |
| :--- | :--- |
| 512×512 single face | ~13 ms/frame (~76 FPS) |
| 1280×720 single face | ~12 ms/frame (~81 FPS) |

Numbers are for one face on one machine; more faces and slower CPUs cost more. The webcam sends
640 px-wide frames over a WebSocket, so end-to-end speed also depends on JPEG encode and network.

## What it does

- 478 3D landmarks per face, up to 10 faces (`max_faces`).
- Regions: mesh (tessellation), contours, face outline, eyes, eyebrows, lips, nose, irises.
- Style: mesh color, feature color, opacity, line thickness, landmark points on/off and size.
- Per-face temporary tracking ids (`Face #n`) in videos and webcam streams.
- Head pose (yaw/pitch/roll) derived from the model's transformation matrix.
- Up to 52 blendshape scores, reported as "expression indicators".

## Where it is available

- **Image / Video jobs:** `face_mesh` in `POST /api/v1/jobs`; writes `face_mesh.json` and adds a
  `face_mesh` block to `report.json` and the job manifest.
- **Webcam:** `mesh` in each `/ws/v1/webcam` message; settings change live without restarting.
- **API:** `POST /api/v1/vision/faces/analyze` (landmarks as JSON), `GET /api/v1/vision/faces/mesh/status`.
- **Reports:** JSON, CSV and PDF reports include a Face Mesh section when a job used the mesh.

## Limitations

- Head-pose angles are approximate. Frontal faces read near zero; the sign convention of
  yaw/pitch/roll has not been independently validated against ground truth.
- Tracking ids are temporary per-stream markers, never verified identities.
- Blendshape scores describe observable facial movement only. They do not reliably indicate
  emotion, intent, honesty or health, and must not be used for employment, medical, legal or other
  high-impact decisions.
- Accuracy drops with heavy occlusion, extreme angles, very small faces and low light. No accuracy
  figures are claimed here; none were independently evaluated.
