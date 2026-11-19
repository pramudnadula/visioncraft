"""3D Face Mesh & Head Pose Estimation using OpenCV solvePnP + Delaunay depth projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from backend.app.detection.base import Detection

# Canonical 3D face model points (in mm): nose tip, chin, left eye, right eye, left mouth, right mouth
_MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class HeadPose3D:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def estimate_head_pose_3d(
    frame_shape: tuple[int, ...],
    detection: Detection,
) -> tuple[HeadPose3D, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate 3D head pose (yaw, pitch, roll) and camera projection matrices."""
    img_h, img_w = frame_shape[:2]
    x, y, w, h = detection.x, detection.y, max(1, detection.width), max(1, detection.height)

    if detection.landmarks and len(detection.landmarks) >= 5:
        # YuNet/SCRFD order: right_eye, left_eye, nose, right_mouth, left_mouth
        r_eye = (detection.landmarks[0].x, detection.landmarks[0].y)
        l_eye = (detection.landmarks[1].x, detection.landmarks[1].y)
        nose = (detection.landmarks[2].x, detection.landmarks[2].y)
        r_mouth = (detection.landmarks[3].x, detection.landmarks[3].y)
        l_mouth = (detection.landmarks[4].x, detection.landmarks[4].y)
        chin = (x + w * 0.5, y + h * 0.96)
    else:
        r_eye = (x + w * 0.33, y + h * 0.36)
        l_eye = (x + w * 0.67, y + h * 0.36)
        nose = (x + w * 0.50, y + h * 0.56)
        r_mouth = (x + w * 0.36, y + h * 0.74)
        l_mouth = (x + w * 0.64, y + h * 0.74)
        chin = (x + w * 0.50, y + h * 0.96)

    image_points = np.array([nose, chin, l_eye, r_eye, l_mouth, r_mouth], dtype=np.float64)
    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return (
        HeadPose3D(yaw_deg=round(yaw, 1), pitch_deg=round(pitch, 1), roll_deg=round(roll, 1)),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )


def draw_3d_face_mesh(
    frame: cv2.typing.MatLike,
    detection: Detection,
) -> HeadPose3D:
    """Render a 3D depth-projected facial mesh, 3D head pose axes, and yaw/pitch/roll."""
    img_h, img_w = frame.shape[:2]
    pose, rvec, tvec, cam, dist = estimate_head_pose_3d(frame.shape, detection)

    # Project a 3D facial dome/contour grid in model space onto the image
    dome_3d = np.array(
        [
            (0.0, 260.0, -140.0),  # forehead center
            (-180.0, 230.0, -180.0),  # forehead left
            (180.0, 230.0, -180.0),  # forehead right
            (-270.0, 60.0, -240.0),  # temple left
            (270.0, 60.0, -240.0),  # temple right
            (-225.0, 170.0, -135.0),  # left eye
            (225.0, 170.0, -135.0),  # right eye
            (0.0, 140.0, -40.0),  # nose bridge
            (0.0, 0.0, 0.0),  # nose tip
            (-240.0, -90.0, -200.0),  # cheek left
            (240.0, -90.0, -200.0),  # cheek right
            (-150.0, -150.0, -125.0),  # mouth left
            (150.0, -150.0, -125.0),  # mouth right
            (0.0, -330.0, -65.0),  # chin
        ],
        dtype=np.float64,
    )
    proj_2d, _ = cv2.projectPoints(dome_3d, rvec, tvec, cam, dist)
    pts_2d = [(float(p[0][0]), float(p[0][1])) for p in proj_2d]
    clamped = [
        (min(float(img_w - 2), max(1.0, px)), min(float(img_h - 2), max(1.0, py))) for px, py in pts_2d
    ]

    subdiv = cv2.Subdiv2D((0, 0, img_w, img_h))
    for pt in clamped:
        subdiv.insert(pt)

    cyan = (255, 220, 40)
    for tri in subdiv.getTriangleList():
        p1, p2, p3 = (int(tri[0]), int(tri[1])), (int(tri[2]), int(tri[3])), (int(tri[4]), int(tri[5]))
        if all(0 <= p[0] < img_w and 0 <= p[1] < img_h for p in (p1, p2, p3)):
            cv2.line(frame, p1, p2, cyan, 1, cv2.LINE_AA)
            cv2.line(frame, p2, p3, cyan, 1, cv2.LINE_AA)
            cv2.line(frame, p3, p1, cyan, 1, cv2.LINE_AA)

    for px, py in clamped:
        cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 220), -1, cv2.LINE_AA)

    # Draw 3D orientation axes from nose tip
    axes_3d = np.array([[180.0, 0.0, 0.0], [0.0, 180.0, 0.0], [0.0, 0.0, 180.0]], dtype=np.float64)
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam, dist)
    nose_pt = (int(clamped[8][0]), int(clamped[8][1]))
    colors = [(0, 0, 255), (0, 255, 0), (255, 120, 0)]
    for axis_pt, col in zip(axes_2d, colors, strict=True):
        end = (int(axis_pt[0][0]), int(axis_pt[0][1]))
        cv2.line(frame, nose_pt, end, col, 2, cv2.LINE_AA)

    return pose


def analyze_faces_3d(frame: cv2.typing.MatLike, detections: list[Detection]) -> dict[str, Any]:
    """Return 3D head pose and projected 3D mesh points for all detected faces."""
    faces: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        pose, rvec, tvec, cam, dist = estimate_head_pose_3d(frame.shape, det)
        proj, _ = cv2.projectPoints(_MODEL_3D_POINTS, rvec, tvec, cam, dist)
        landmarks_3d = [
            {
                "x": round(float(p[0][0]), 2),
                "y": round(float(p[0][1]), 2),
                "z": round(float(_MODEL_3D_POINTS[i][2]), 2),
            }
            for i, p in enumerate(proj)
        ]
        faces.append(
            {
                "face_index": idx,
                "head_pose": {
                    "yaw_deg": pose.yaw_deg,
                    "pitch_deg": pose.pitch_deg,
                    "roll_deg": pose.roll_deg,
                },
                "landmarks_3d": landmarks_3d,
            }
        )
    return {"face_count": len(faces), "faces": faces}
