#!/usr/bin/env sh
# Installs MediaPipe for the FaceVision face mesh feature without replacing opencv-python.
# (MediaPipe declares opencv-contrib-python as a dependency, which conflicts with opencv-python.)
set -eu
PIP="${PIP:-pip}"
$PIP install -e ".[facemesh]"
$PIP install --no-deps "mediapipe>=0.10.14"
python -c "import cv2, mediapipe; print('OK  opencv', cv2.__version__, '| mediapipe', mediapipe.__version__)"
