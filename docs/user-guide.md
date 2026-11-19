# FaceVision User Guide

1. **Dashboard**: View processed media counts, total detected faces, active jobs, and local/cloud execution status.
2. **Image Detection Studio**: Drag and drop any image, pick a detection model (`haar`, `yunet`, etc.), choose a privacy blur (`none`, `gaussian`, `pixelate`), toggle selective face blurring, compare Original / Annotated / Privacy views, and download PNG, face crops, JSON, or CSV reports.
3. **Video Detection Studio**: Upload MP4/MOV/AVI/MKV/WebM videos, enable temporary face tracking IDs, monitor frame-by-frame progress in real time, and export processed MP4 + timeline JSON/CSV.
4. **Live Webcam**: Select a camera, run live detection and privacy blur over WebSocket, and capture snapshots.
5. **Batch Processing**: Drop multiple images or import a folder, process them concurrently, and download a ZIP archive of all results.
6. **History & Settings**: Filter past jobs, retry failed/cancelled jobs, configure retention days, and manage optional cloud consent.
