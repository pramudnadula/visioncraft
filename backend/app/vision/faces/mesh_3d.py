"""3D Face Mesh & Head Pose Estimation using OpenCV solvePnP + 3D projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from backend.app.detection.base import Detection

# Canonical 3D face model points (in mm, Y-down to match OpenCV image coordinates):
# nose tip, chin, left eye (viewer's right), right eye (viewer's left), left mouth, right mouth
_MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, 330.0, -65.0),  # Chin
        (225.0, -170.0, -135.0),  # Viewer's right eye
        (-225.0, -170.0, -135.0),  # Viewer's left eye
        (150.0, 150.0, -125.0),  # Viewer's right mouth corner
        (-150.0, 150.0, -125.0),  # Viewer's left mouth corner
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
        # YuNet/SCRFD order: right_eye (viewer left), left_eye (viewer right), nose, right_mouth, left_mouth
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
    focal_length = float(max(img_w, img_h))
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


def _draw_pose_axes(
    frame: cv2.typing.MatLike,
    nose_pt: tuple[int, int],
    pose: HeadPose3D,
    size: float = 65.0,
) -> None:
    """Draw 3D XYZ orientation axes anchored at the nose tip from Euler angles."""
    pitch = np.deg2rad(pose.pitch_deg)
    yaw = np.deg2rad(-pose.yaw_deg)
    roll = np.deg2rad(pose.roll_deg)

    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rot = rz @ ry @ rx

    axes = np.eye(3, dtype=np.float64) @ rot.T * size
    colors = ((0, 0, 255), (0, 255, 0), (255, 140, 0))  # X=Red, Y=Green, Z=Blue
    for vec, col in zip(axes, colors, strict=True):
        end = (int(nose_pt[0] + vec[0]), int(nose_pt[1] + vec[1]))
        cv2.line(frame, nose_pt, end, col, 2, cv2.LINE_AA)


def draw_3d_face_mesh(
    frame: cv2.typing.MatLike,
    detection: Detection,
    *,
    mesh_face: Any | None = None,
) -> HeadPose3D:
    """Render a 3D facial mesh and head pose axes (MediaPipe 478-pt primary, solvePnP fallback)."""
    if mesh_face is None:
        from backend.app.vision.faces.segmentation import build_face_privacy_context

        ctx = build_face_privacy_context(frame, need_segmentation=False, need_mesh=True)
        mesh_face = ctx.match_mesh(detection)

    if mesh_face is not None:
        from backend.app.vision.faces.face_mesh import FaceMeshSettings, render_face_mesh

        render_face_mesh(
            frame,
            [mesh_face],
            FaceMeshSettings.from_preset("full_mesh", opacity=0.75, thickness=1),
        )
        hp = mesh_face.head_pose
        if hp is not None:
            pose = HeadPose3D(
                yaw_deg=float(hp["yaw_deg"]),
                pitch_deg=float(hp["pitch_deg"]),
                roll_deg=float(hp["roll_deg"]),
            )
        else:
            pose, *_ = estimate_head_pose_3d(frame.shape, detection)

        nose_xy = mesh_face.landmarks_px[1] if mesh_face.landmarks_px.shape[0] > 1 else None
        nose_pt = (
            (int(nose_xy[0]), int(nose_xy[1]))
            if nose_xy is not None
            else (detection.x + detection.width // 2, detection.y + detection.height // 2)
        )
        _draw_pose_axes(frame, nose_pt, pose, size=max(40.0, detection.width * 0.45))

        pose_text = f"Yaw:{pose.yaw_deg:+.0f} Pitch:{pose.pitch_deg:+.0f} Roll:{pose.roll_deg:+.0f}"
        cv2.putText(
            frame,
            pose_text,
            (max(10, detection.x), max(22, detection.y + detection.height + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 220),
            1,
            cv2.LINE_AA,
        )
        return pose

    img_h, img_w = frame.shape[:2]
    pose, rvec, tvec, cam, dist = estimate_head_pose_3d(frame.shape, detection)

    # Fallback: Project 3D facial dome points (Y-down matching _MODEL_3D_POINTS)
    dome_3d = np.array(
        [
            (0.0, -290.0, -120.0),  # forehead center
            (-180.0, -260.0, -160.0),  # forehead left
            (180.0, -260.0, -160.0),  # forehead right
            (-280.0, -80.0, -220.0),  # temple left
            (280.0, -80.0, -220.0),  # temple right
            (-225.0, -170.0, -135.0),  # viewer's left eye
            (225.0, -170.0, -135.0),  # viewer's right eye
            (0.0, -130.0, -40.0),  # nose bridge
            (0.0, 0.0, 0.0),  # nose tip
            (-250.0, 80.0, -190.0),  # cheek left
            (250.0, 80.0, -190.0),  # cheek right
            (-150.0, 150.0, -125.0),  # mouth left
            (150.0, 150.0, -125.0),  # mouth right
            (0.0, 330.0, -65.0),  # chin
        ],
        dtype=np.float64,
    )
    proj_2d, _ = cv2.projectPoints(dome_3d, rvec, tvec, cam, dist)
    clamped = [
        (min(float(img_w - 2), max(1.0, float(p[0][0]))), min(float(img_h - 2), max(1.0, float(p[0][1]))))
        for p in proj_2d
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

    # Project 3D bounding cube around head
    rw, rh, rz_front, rz_back = 310.0, 360.0, 80.0, -340.0
    cube_3d = np.array(
        [
            (-rw, -rh, rz_back),
            (rw, -rh, rz_back),
            (rw, rh, rz_back),
            (-rw, rh, rz_back),
            (-rw, -rh, rz_front),
            (rw, -rh, rz_front),
            (rw, rh, rz_front),
            (-rw, rh, rz_front),
        ],
        dtype=np.float64,
    )
    cube_2d, _ = cv2.projectPoints(cube_3d, rvec, tvec, cam, dist)
    c_pts = [(int(p[0][0]), int(p[0][1])) for p in cube_2d]
    for i in range(4):
        cv2.line(frame, c_pts[i], c_pts[(i + 1) % 4], (180, 120, 255), 1, cv2.LINE_AA)
        cv2.line(frame, c_pts[i + 4], c_pts[((i + 1) % 4) + 4], (255, 180, 80), 2, cv2.LINE_AA)
        cv2.line(frame, c_pts[i], c_pts[i + 4], (210, 150, 220), 1, cv2.LINE_AA)

    # Draw 3D orientation axes from nose tip
    axes_3d = np.array([[220.0, 0.0, 0.0], [0.0, 220.0, 0.0], [0.0, 0.0, 220.0]], dtype=np.float64)
    axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam, dist)
    nose_pt = (int(clamped[8][0]), int(clamped[8][1]))
    colors = [(0, 0, 255), (0, 255, 0), (255, 140, 0)]
    for axis_pt, col in zip(axes_2d, colors, strict=True):
        end = (int(axis_pt[0][0]), int(axis_pt[0][1]))
        cv2.line(frame, nose_pt, end, col, 2, cv2.LINE_AA)

    # Pose angle label
    pose_text = f"Yaw:{pose.yaw_deg:+.0f} Pitch:{pose.pitch_deg:+.0f} Roll:{pose.roll_deg:+.0f}"
    cv2.putText(
        frame,
        pose_text,
        (max(10, detection.x), max(22, detection.y + detection.height + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 220),
        1,
        cv2.LINE_AA,
    )
    return pose


def analyze_faces_3d(
    frame: cv2.typing.MatLike,
    detections: list[Detection],
    *,
    models_dir: Any | None = None,
) -> dict[str, Any]:
    """Return 3D head pose and 3D landmarks for all detected faces."""
    from backend.app.vision.faces.segmentation import build_face_privacy_context

    ctx = build_face_privacy_context(
        frame,
        models_dir=models_dir,
        need_segmentation=False,
        need_mesh=True,
    )
    faces: list[dict[str, Any]] = []
    if ctx.mesh_faces and not detections:
        for idx, mf in enumerate(ctx.mesh_faces):
            hp = mf.head_pose or {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
            faces.append(
                {
                    "face_index": idx,
                    "model": "face_landmarker.task",
                    "landmark_count": int(mf.landmarks_px.shape[0]),
                    "head_pose": hp,
                    "landmarks_3d": [
                        {"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 4)}
                        for (x, y), z in zip(mf.landmarks_px, mf.landmarks_z, strict=True)
                    ],
                }
            )
        return {"face_count": len(faces), "faces": faces}

    for idx, det in enumerate(detections):
        matched_mf = ctx.match_mesh(det)
        if matched_mf is not None:
            hp = matched_mf.head_pose or {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
            faces.append(
                {
                    "face_index": idx,
                    "model": "face_landmarker.task",
                    "landmark_count": int(matched_mf.landmarks_px.shape[0]),
                    "head_pose": hp,
                    "landmarks_3d": [
                        {"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 4)}
                        for (x, y), z in zip(matched_mf.landmarks_px, matched_mf.landmarks_z, strict=True)
                    ],
                }
            )
            continue

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
                "model": "opencv_solvepnp_fallback",
                "landmark_count": len(landmarks_3d),
                "head_pose": {
                    "yaw_deg": pose.yaw_deg,
                    "pitch_deg": pose.pitch_deg,
                    "roll_deg": pose.roll_deg,
                },
                "landmarks_3d": landmarks_3d,
            }
        )
    return {"face_count": len(faces), "faces": faces}
