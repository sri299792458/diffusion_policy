"""Robot base frames and the camera transform.

The policy, the sim and vendor/ur5e_kinematics use the REP-103 base_link frame. UR's controller "Base" frame (pendant,
RTDE getActualTCPPose / moveL / speedL) is base_link rotated 180 deg about the base z axis, so x and y change sign. On
Thunder's sideways mount base_link +y is world-up, which is why the October 6 pull-out direction flipped (+y in UR Base is
world-down). Everything in the state-policy loop is in base_link; the only conversion is the camera calibration below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R_180Z = np.diag([-1.0, -1.0, 1.0])
T_BASELINK_URBASE = np.eye(4)
T_BASELINK_URBASE[:3, :3] = R_180Z        # p_base_link = T_BASELINK_URBASE @ p_ur_base (the transform is its own inverse)
FRAMES = ("base_link", "ur_base")


def load_camera_transform(path, frame):
    """4x4 T with p_base = T @ p_camera (meters; camera = L515 COLOR optical frame: x right, y down, z forward).

    path: .npy holding the 4x4 matrix, or .json holding it under "T_base_camera".
    frame: which robot frame T maps into: "base_link" (sim / our kinematics) or "ur_base" (UR controller Base, i.e. a
           calibration computed from getActualTCPPose or pendant poses). Required: the two differ by 180 deg about base z.
    Returns T_base_link_camera.
    """
    if frame not in FRAMES:
        raise ValueError(f"camera transform frame must be one of {FRAMES}")
    path = Path(path)
    if path.suffix == ".npy":
        T = np.load(path)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        if "T_base_camera" not in data:
            raise ValueError('JSON camera transform needs a 4x4 "T_base_camera" (p_base = T @ p_camera)')
        T = data["T_base_camera"]
    else:
        raise ValueError("camera transform must be .npy or .json")
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("camera transform must be a finite 4x4 matrix")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9) or not np.allclose(R @ R.T, np.eye(3), atol=1e-4) \
            or abs(np.linalg.det(R) - 1) > 1e-4:
        raise ValueError("camera transform must be a rigid transform (orthonormal rotation, det +1, last row 0 0 0 1)")
    if np.linalg.norm(T[:3, 3]) > 5.0:
        raise ValueError("camera is more than 5 m from the robot base: check units (meters) and direction (p_base = T @ p_camera)")
    return T_BASELINK_URBASE @ T if frame == "ur_base" else T


def matrix_to_pos_quat(T):
    """4x4 -> position, quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(T)[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return np.asarray(T)[:3, 3].copy(), q if q[0] >= 0 else -q


def pos_quat_to_matrix(pos, quat):
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = pos
    return T
