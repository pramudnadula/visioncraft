# Face_Detection_Python

Face detection in images, video files, and webcam streams using OpenCV Haar Cascades.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run on webcam (press `q` or `Esc` to quit):
```bash
python Face_Detector.py
```

Run on an image:
```bash
python Face_Detector.py --image RDJ.png
```

Run on a video file:
```bash
python Face_Detector.py --video clip.mp4
```

Save annotated output without opening a GUI window:
```bash
python Face_Detector.py --image RDJ.png --output output/RDJ_detected.png --no-show
```
