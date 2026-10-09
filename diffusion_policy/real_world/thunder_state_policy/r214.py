"""R214 state policy on Thunder: policy, gripper-joint map and cube poses for eval_state_policy.py.

Everything here is what upstream's image-policy eval does not need: R214 observes cube poses (AprilCube on the L515 color
frames delivered by RealEnv) and the six simulated gripper joints, and was trained with its own action scale.
Observation rebuild (obs.py) and policy weights were verified against R214's native training env (UWLab R226: 40,960
observations to 1e-6; sim-recorded actions checked at every load).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from diffusion_policy.real_world.thunder_state_policy import cube_symmetry as CS
from diffusion_policy.real_world.thunder_state_policy import frames as F
from diffusion_policy.real_world.thunder_state_policy import obs as O

HERE = Path(__file__).resolve().parent
PKG = HERE / "r214"
CALIBRATION = HERE / "thunder_calibration.json"
# R214 Reaching start bank row 7738 (UWLab-thunder-sim2real 23b343b): S-W-E-, gripper ~4 deg from straight down, wrist
# 0.47 m above the table, arm clear of the L515's view of the trained cube region.
START_JOINTS = [0.272496, -2.544273, -1.227111, -2.524585, -2.7922, 1.67543]
THUNDER_PAYLOAD = dict(payload_mass=1.04, payload_cog=[-0.002, 0.001, 0.047])


def load_manifest():
    m = json.loads((PKG / "manifest.json").read_text())
    if hashlib.sha256((PKG / m["policy"]["file"]).read_bytes()).hexdigest() != m["policy"]["sha256"]:
        raise RuntimeError(f"{m['policy']['file']} does not match its manifest hash")
    return m


def load_calibration():
    return json.loads(CALIBRATION.read_text())


def load_policy(m):
    """NumPy R214 actor: h = (obs - mean) / (std + 0.01); 4 x elu(W h + b); action = W h + b. Checked at load against
    actions R214 produced in its training sim for stored observations."""
    w = np.load(PKG / m["policy"]["file"])
    mean, std = w["obs_mean"], w["obs_std"] + float(w["obs_eps"])
    layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(5)]

    def act(obs):
        h = (np.asarray(obs, dtype=float) - mean) / std
        for i, (W, b) in enumerate(layers):
            h = W @ h + b
            if i < 4:
                h = np.where(h > 0, h, np.expm1(h))
        return h

    error = max(float(np.abs(act(o) - a).max()) for o, a in zip(w["check_obs"], w["check_action"]))
    if error > 1e-4:
        raise RuntimeError(f"NumPy policy differs from the recorded sim actions by {error:.2e}")
    return act


class GripperMap:
    """Robotiq position (gPO) -> the 6 sim gripper joints R214 observes. finger_joint: piecewise linear through measured
    anchors (open 3 / 60 mm cube ~92 / empty closed 226 at speed 128, force 0) and the sim angles at the same states;
    the 5 linkage joints follow finger_joint as in sim (p99 error <= 0.013 rad)."""

    def __init__(self, m):
        a = m["gripper"]["position_to_finger_joint"]
        self.real = np.asarray(a["real_position"], dtype=float)
        self.finger = np.asarray(a["finger_joint_rad"], dtype=float)
        self.linkage = [np.asarray(m["gripper"]["linkage_from_finger_joint"][n], dtype=float)
                        for n in m["observation"]["sim_joint_names"][6:]]

    def sim_joints(self, position):
        fj = float(np.interp(position, self.real, self.finger))
        return np.array([fj] + [float(np.interp(fj, t[:, 0], t[:, 1])) for t in self.linkage[1:]])

    def holding(self, position, object_status, closed_cmd):
        """Commanded closed, stopped on contact while closing (gOBJ 2), between open and empty-closed."""
        return bool(closed_cmd and object_status == 2 and self.real[0] + 20 < position < self.real[-1] - 10)


class CubeEstimator:
    """Cube poses (base_link) for one policy step from AprilCube detections on the L515 frame.

    - Each cube's frame is relabelled once per episode to +Z up (cube_symmetry; the cubes are physically symmetric and
      R214 always started with the bottom cube +Z up near -90 deg yaw).
    - A cube not detected in this frame keeps its last measured pose (cubes are static unless held).
    - While the gripper holds the carried cube (often hidden from the overhead L515), its last sighting moves rigidly with
      the wrist from the later of the sighting and the grasp. Robot state and camera frame share RealEnv's aligned
      timestamp, so the wrist pose of a step and that step's detection belong together.
    """

    def __init__(self, T_base_camera, detectors, up):
        self.T_base_camera = T_base_camera
        self.detectors = detectors            # {"receptive": aprilcube detector, "insertive": ...}
        self.up = np.asarray(up, dtype=float)
        self.reset()

    def reset(self):
        self.relabel = None
        self.last = {}                        # cube -> (T_base_cube measured, wrist T at that step, step index)
        self.grasp = None                     # (wrist T, step) when the gripper started holding

    def detect(self, image_bgr, timestamp):
        out = {}
        for name, det in self.detectors.items():
            res = det.process_frame(image_bgr, timestamp=timestamp)
            if res["success"] and res["T"] is not None and not res.get("predicted"):
                T_cam = np.array(res["T"], dtype=float)
                T_cam[:3, 3] /= 1000.0        # AprilCube reports translation in millimeters
                out[name] = (self.T_base_camera @ T_cam, int(res["n_tags"]), float(res["reproj_error"]))
        return out

    def resting_flat(self, detections):
        return all(n in detections and CS.resting_tilt_deg(detections[n][0][:3, :3], self.up) <= CS.MAX_RESTING_TILT_DEG
                   for n in ("receptive", "insertive"))

    def set_relabel(self, detections):
        S_b, S_c, yaw_error = CS.relabel(detections["receptive"][0][:3, :3], detections["insertive"][0][:3, :3], self.up)
        self.relabel = {"receptive": np.eye(4), "insertive": np.eye(4)}
        self.relabel["receptive"][:3, :3], self.relabel["insertive"][:3, :3] = S_b, S_c
        return yaw_error

    def poses(self, detections, wrist_T, step, holding):
        if not holding:
            self.grasp = None
        elif self.grasp is None:
            self.grasp = (wrist_T, step)
        for name, (T, _, _) in detections.items():
            self.last[name] = (T, wrist_T, step)
        out, info = {}, {}
        for name in ("receptive", "insertive"):
            if name not in self.last:
                raise RuntimeError(f"{name} cube has never been detected")
            T, W_seen, seen_step = self.last[name]
            carried = name == "insertive" and holding and name not in detections
            if carried:
                # static until the grasp, rigid with the wrist after it: carry from the later of sighting and grasp
                W_ref = W_seen if seen_step >= self.grasp[1] else self.grasp[0]
                T = wrist_T @ np.linalg.inv(W_ref) @ T
            out[name] = F.matrix_to_pos_quat(T @ self.relabel[name])
            info[name] = dict(age_steps=step - seen_step, carried_with_wrist=carried, seen=name in detections)
        return out, info


def build_frame(last_action, arm_q, gripper_joints, wrist_pos, wrist_quat, cubes):
    (rp, rq), (ip, iq) = cubes["receptive"], cubes["insertive"]
    return O.frame_terms(last_action, np.concatenate([arm_q, gripper_joints]), wrist_pos, wrist_quat, ip, iq, rp, rq)
