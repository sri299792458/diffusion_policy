"""Relabel the cubes' faces so either cube can be placed on any face.

The cubes are physically symmetric; only their tags differ, and the tags define the cube frame. R214 always started with
the bottom cube +Z up at a yaw near -90 deg, and its success test needs the carried cube's +Z along the bottom cube's +Z
(yaw ignored). So each measured pose T is used as T @ S, with S one of the cube's 24 rotations, chosen once per run:
bottom cube +Z up at the +Z-up yaw nearest training's; carried cube +Z up with the smallest relabel. A placement like
training's gives S = identity. S is fixed to the cube, so the pose stays continuous while the cube is carried.
"""
from __future__ import annotations

import itertools

import numpy as np

# R214 starts: bottom cube +Z up, yaw -90 deg +/- 15 (upstream reset yaw +/- pi/12; R214 fixture starts -79 to -103 deg).
# Yaw = heading of the cube +X axis about world up, from base_link +X.
TRAINING_BOTTOM_YAW_DEG = -90.0
MAX_BOTTOM_YAW_ERROR_DEG = 15.0
# A cube resting on the table is flat; a start reading tilted more than this is a bad (e.g. single-tag) estimate.
MAX_RESTING_TILT_DEG = 10.0

SYMMETRIES = [S for S in (np.array(P) * np.array(signs) for P in itertools.permutations(np.eye(3))
                          for signs in itertools.product((1.0, -1.0), repeat=3)) if np.linalg.det(S) > 0]


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def resting_tilt_deg(R, up):
    """Angle between up and the cube face pointing most nearly up: 0 for a cube resting flat on any face."""
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    return float(np.degrees(np.arccos(np.clip(np.abs(R.T @ up).max(), -1.0, 1.0))))


def yaw_deg(R, up):
    flat = lambda v: (v - (v @ up) * up) / np.linalg.norm(v - (v @ up) * up)
    ref, x = flat(np.array([1.0, 0.0, 0.0])), flat(R[:, 0])
    return float(np.degrees(np.arctan2(np.cross(ref, x) @ up, ref @ x)))


def z_up_labels(R, up):
    """The four symmetries that put the cube's most upward face on +Z."""
    z = max(SYMMETRIES, key=lambda S: (R @ S)[:, 2] @ up)[:, 2]
    return [S for S in SYMMETRIES if np.array_equal(S[:, 2], z)]


def relabel(R_bottom, R_carried, up):
    """Rotations (base_link) of the bottom and carried cube -> (S_bottom, S_carried, bottom yaw error from training, deg)."""
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    S_bottom = min(z_up_labels(R_bottom, up),
                   key=lambda S: abs(wrap_deg(yaw_deg(R_bottom @ S, up) - TRAINING_BOTTOM_YAW_DEG)))
    S_carried = max(z_up_labels(R_carried, up), key=np.trace)     # identity when +Z is already up
    return S_bottom, S_carried, wrap_deg(yaw_deg(R_bottom @ S_bottom, up) - TRAINING_BOTTOM_YAW_DEG)
