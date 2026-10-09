"""R214 actor observation, rebuilt from robot/cube state with NumPy (no Isaac Lab).

Contract (measured in sim, R226 probe of R214's native env; verified against 40,960 recorded observations):
one frame = 43 numbers, history of 5 frames per term (oldest first), terms concatenated in the env's ACTIVE order:
  insertive_in_receptive  (6)  carried cube pose in the bottom-cube frame          -> 30
  prev_actions            (7)  last raw policy action (6 arm + 1 gripper)           -> 35
  joint_pos              (12)  6 arm joints + 6 gripper linkage joints (sim order)  -> 60
  end_effector_pose       (6)  wrist_3_link pose in base_link                       -> 30
  insertive_pose          (6)  carried cube pose in the wrist_3_link frame          -> 30
  receptive_pose          (6)  bottom cube pose in the wrist_3_link frame           -> 30
Poses are [position (m), axis-angle (rad)] with Isaac Lab's axis_angle_from_quat (w >= 0 branch). All inputs are in the
REP-103 base_link frame used by the sim and by vendor/ur5e_kinematics (NOT UR's controller Base frame, which is rotated
180 deg about base z). After a reset the first frame fills all 5 history slots (Isaac Lab CircularBuffer semantics).
"""
from __future__ import annotations

import numpy as np

HISTORY = 5
TERMS = (("insertive_in_receptive", 6), ("prev_actions", 7), ("joint_pos", 12), ("end_effector_pose", 6),
         ("insertive_pose", 6), ("receptive_pose", 6))
OBS_DIM = HISTORY * sum(d for _, d in TERMS)   # 215
SIM_JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                   "wrist_3_joint", "finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint",
                   "right_inner_finger_joint", "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint")


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, dtype=float), -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def quat_inv(q):
    q = np.asarray(q, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, -1.0]) / np.sum(q * q, axis=-1, keepdims=True)


def quat_apply(q, v):
    q = np.asarray(q, dtype=float); v = np.asarray(v, dtype=float)
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def subtract_frame_transforms(t01, q01, t02, q02):
    """Pose of frame 2 in frame 1, given both in frame 0 (Isaac Lab math.subtract_frame_transforms)."""
    q10 = quat_inv(q01)
    return quat_apply(q10, np.asarray(t02, dtype=float) - t01), quat_mul(q10, q02)


def axis_angle_from_quat(q, eps=1.0e-6):
    """Isaac Lab math.axis_angle_from_quat: w made non-negative, then angle * axis."""
    q = np.asarray(q, dtype=float)
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = np.linalg.norm(q[..., 1:], axis=-1)
    half = np.arctan2(mag, q[..., 0])
    angle = 2.0 * half
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(angle) > eps, np.sin(half) / angle, 0.5 - angle * angle / 48)
    return q[..., 1:4] / ratio[..., None]


def pose6(pos, quat):
    return np.concatenate([np.asarray(pos, dtype=float), axis_angle_from_quat(quat)], axis=-1)


def frame_terms(last_action, joint_pos, wrist_pos, wrist_quat, ins_pos, ins_quat, rec_pos, rec_quat):
    """One frame of every term (dict term -> vector) from base_link-frame poses (quaternions wxyz)."""
    ins_rec = subtract_frame_transforms(rec_pos, rec_quat, ins_pos, ins_quat)
    ins_wrist = subtract_frame_transforms(wrist_pos, wrist_quat, ins_pos, ins_quat)
    rec_wrist = subtract_frame_transforms(wrist_pos, wrist_quat, rec_pos, rec_quat)
    values = {"insertive_in_receptive": pose6(*ins_rec), "prev_actions": np.asarray(last_action, dtype=float),
              "joint_pos": np.asarray(joint_pos, dtype=float), "end_effector_pose": pose6(wrist_pos, wrist_quat),
              "insertive_pose": pose6(*ins_wrist), "receptive_pose": pose6(*rec_wrist)}
    for name, dim in TERMS:
        if values[name].shape[-1] != dim or not np.isfinite(values[name]).all():
            raise ValueError(f"{name} must hold {dim} finite numbers")
    return values


class ObservationHistory:
    """5-frame history per term; reset() then push() every policy step; vector() is the 215-number actor input."""

    def __init__(self):
        self.buffers = None

    def reset(self):
        self.buffers = None

    def push(self, terms):
        if self.buffers is None:
            self.buffers = {name: [terms[name].copy() for _ in range(HISTORY)] for name, _ in TERMS}
        else:
            for name, _ in TERMS:
                self.buffers[name] = self.buffers[name][1:] + [terms[name].copy()]
        return self.vector()

    def vector(self):
        return np.concatenate([np.concatenate(self.buffers[name]) for name, _ in TERMS])
