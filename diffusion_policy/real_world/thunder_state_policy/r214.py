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
      R214 always started with the bottom cube +Z up near -90 deg yaw). With any_face_up=False the carried cube keeps
      the frame its tags define, as in training, whose success test needs its +Z up: the policy then turns it over.
    - Detections come from the camera frame taken at detection_time, with the wrist pose at that time, and are applied
      once (the newest result may be used by several policy steps).
    - A cube not newly detected keeps its last measured pose (cubes are static unless held).
    - While the gripper holds the carried cube (often hidden from the overhead L515), its last sighting moves rigidly with
      the wrist from the later of the sighting and the grasp; this also brings a fresh sighting from its frame time to
      now. Only a sighting within ATTACHED_MAX_WRIST_DISTANCE_M of the wrist counts as held.
    - Detections are checked against physics (check()). A free cube rests on the table or, for the carried cube, on top
      of the bottom cube, and AprilCube's error is almost all along the camera ray (dp_run5: a still cube read within
      ~1 mm across the image but 6-8 mm (p90) along the ray with 2-3 tags and 24-54 mm with one tag, e.g. when the
      stacked cube hides the bottom cube's top tag). So a free cube's detection is moved along the ray onto its support
      (onto_support), and not used if no support is within MAX_RAY_SLIDE_M; the bottom cube must also be flat. A cube
      read nearly flat is made exactly flat (a flat bottom cube read ~3 deg tilted, more than R214's 0.025 rad success
      tolerance between the cubes); a leaning cube keeps its tilt. A held cube must be between the fingers (IN_HAND_*,
      around training's grasps at (0, 0, 0.192) m in the wrist frame) and within HELD_AGREE_M of where the grasp
      carried it. A detection is judged by the cube's state when its frame was taken: detection can take up to ~0.7 s,
      so a frame from before the grasp is still a free cube (October 9 run: such a frame, applied after the grasp as an
      in-hand reading, put the held cube 35 mm low and the policy released it 3 cm above the stack).
    - When the gripper lets go, the carried cube falls straight down from where the hand released it onto the bottom
      cube's top or the table (dropped()) and stays there until it is detected again (dp_run5: left in the air where a
      tilted grasp let go 5 cm up, it kept the policy grasping at nothing for 6 s).
    - When the released carried cube comes to rest flat on a different face, it is relabelled again so +Z is up (the
      cube is symmetric; otherwise the policy sees it lying on its side and tries to turn it over). A leaning cube is not
      relabelled.
    """
    ATTACHED_MAX_WRIST_DISTANCE_M = 0.25
    MAX_RAY_SLIDE_M = 0.10                    # largest move along the camera ray onto a support (one-tag errors: <= ~0.1 m)
    SUPPORT_TOLERANCE_M = 0.015               # a released cube this little below the bottom cube's top still lands on it
    ON_TOP_RADIUS_M = 0.06                    # a detection on the bottom cube: centre within this (horizontally) of its
    IN_HAND_LOW_M = np.array([-0.03, -0.045, 0.15])   # held cube centre in the wrist frame: across the fingers (x),
    IN_HAND_HIGH_M = np.array([0.03, 0.045, 0.25])    # along the pads (y), along the tool axis (z)
    HELD_AGREE_M = 0.015                      # in-hand detection vs the rigidly carried pose (in-hand readings were off
    CUBE_SIZE_M = 0.06                        # by 18-35 mm while the grasp itself held; see the October 9 run)

    def __init__(self, T_base_camera, detectors, up, table_height, any_face_up=True):
        self.T_base_camera = T_base_camera
        self.detectors = detectors            # {"receptive": aprilcube detector, "insertive": ...}
        self.up = np.asarray(up, dtype=float)
        self.table_height = float(table_height)   # table top along up (base_link)
        self.any_face_up = any_face_up        # False: the carried cube keeps its tag faces (training's goal: tag +Z up)
        self.skipped = set()                  # cubes whose detector skipped frames since its last run
        self.reset()

    def reset(self):
        self.relabel = None
        self.last = {}                        # cube -> (T_base_cube measured, wrist T at that frame, frame time)
        self.grasp = None                     # (wrist T, time) when the gripper started holding
        self.applied_time = None              # frame time of the newest detection result already applied
        self.measured = {}                    # cube -> its pose in self.last comes from a camera reading
        self.relabel_events = []              # (frame time, cube, 3x3 symmetry) for every relabel after the start
        self.carried_now = None               # (T, wrist T, time) of the held cube at the latest holding step

    def lowest_corner(self, T):
        corners = T[:3, 3] + (np.array(np.meshgrid(*[[-0.5, 0.5]] * 3)).reshape(3, -1).T * self.CUBE_SIZE_M) @ T[:3, :3].T
        return float((corners @ self.up).min())

    def carried_at(self, W):
        """Pose of the held carried cube when the wrist is at W: its last sighting moved rigidly with the wrist from the
        later of that sighting and the grasp. None if that sighting was not in the hand."""
        T_last, W_seen, seen_time = self.last["insertive"]
        W_ref = W_seen if seen_time >= self.grasp[1] else self.grasp[0]
        cube_in_wrist = np.linalg.inv(W_ref) @ T_last
        if np.linalg.norm(cube_in_wrist[:3, 3]) >= self.ATTACHED_MAX_WRIST_DISTANCE_M:
            return None
        return W @ cube_in_wrist

    def horizontal(self, v):
        return float(np.linalg.norm(v - (v @ self.up) * self.up))

    def flattened(self, T):
        """T turned (about its centre) by the smallest rotation that puts the cube face nearest up exactly up: a cube
        resting flat. Yaw is kept. A flat cube read 2.7-2.9 deg tilted (bottom cube, dp_run5), more than R214's success
        tolerance between the cubes (0.025 rad roll + pitch)."""
        R_ = T[:3, :3]
        i = int(np.argmax(np.abs(R_.T @ self.up)))
        n = R_[:, i] * np.sign(R_[:, i] @ self.up)
        axis, c = np.cross(n, self.up), float(n @ self.up)
        s = float(np.linalg.norm(axis))
        if s < 1e-12:
            return T.copy()
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / s
        T = T.copy()
        T[:3, :3] = (np.eye(3) + s * K + (1 - c) * K @ K) @ R_
        return T

    def onto_support(self, name, T):
        """T moved along the camera ray until the cube rests on the table or, for the carried cube, on the bottom cube's
        top (whichever needs the smaller move), or None if neither is within MAX_RAY_SLIDE_M. On the bottom cube means
        centred within ON_TOP_RADIUS_M of it; on the table, not overlapping it. A cube read within
        MAX_RESTING_TILT_DEG of flat is made flat first; a leaning cube keeps its tilt."""
        if CS.resting_tilt_deg(T[:3, :3], self.up) <= CS.MAX_RESTING_TILT_DEG:
            T = self.flattened(T)
        ray = T[:3, 3] - self.T_base_camera[:3, 3]
        ray = ray / np.linalg.norm(ray)
        bottom = self.last["receptive"][0][:3, 3] if name == "insertive" and "receptive" in self.last else None
        supports = [(self.table_height, False)]
        if bottom is not None:
            supports.append((self.table_height + self.CUBE_SIZE_M, True))
        best = None
        for height, on_bottom in supports:
            slide = (height - self.lowest_corner(T)) / (ray @ self.up)
            p = T[:3, 3] + slide * ray
            if abs(slide) > self.MAX_RAY_SLIDE_M:
                continue
            if bottom is not None and (self.horizontal(p - bottom) > self.ON_TOP_RADIUS_M if on_bottom
                                       else self.horizontal(p - bottom) < self.CUBE_SIZE_M):
                continue
            if best is None or abs(slide) < best[0]:
                best = (abs(slide), p)
        if best is None:
            return None
        T = T.copy()
        T[:3, 3] = best[1]
        return T

    def check(self, name, T, holding, detection_wrist_T, detection_time):
        """(pose to use, '') for a physically possible detection, else (None, the reason it is not used). The cube's
        state is the one when the frame was taken: a frame from before the grasp shows a free cube."""
        if name == "insertive" and holding and detection_time >= self.grasp[1]:
            in_wrist = (np.linalg.inv(detection_wrist_T) @ T)[:3, 3]
            if not (np.all(in_wrist >= self.IN_HAND_LOW_M) and np.all(in_wrist <= self.IN_HAND_HIGH_M)):
                return None, "held cube detection not between the fingers"
            carried = self.carried_at(detection_wrist_T)
            if carried is not None and np.linalg.norm(T[:3, 3] - carried[:3, 3]) > self.HELD_AGREE_M:
                return None, "held cube detection disagrees with the grasp"
            return T, ""
        if name == "receptive" and CS.resting_tilt_deg(T[:3, :3], self.up) > CS.MAX_RESTING_TILT_DEG:
            return None, "bottom cube not resting flat"
        T_rest = self.onto_support(name, T)
        if T_rest is None:
            return None, ("bottom cube not on the table" if name == "receptive"
                          else "carried cube not on the table or the bottom cube")
        return T_rest, ""

    def dropped(self, T):
        """Where a cube let go at T comes to rest until it is seen again: flat on its face nearest down, straight below
        the hand, on the bottom cube's top if its centre is over that face and not below it, else on the table."""
        top = self.table_height + self.CUBE_SIZE_M
        over = self.horizontal(T[:3, 3] - self.last["receptive"][0][:3, 3]) <= self.CUBE_SIZE_M / 2
        height = top if over and self.lowest_corner(T) >= top - self.SUPPORT_TOLERANCE_M else self.table_height
        T = self.flattened(T)
        T[:3, 3] += (height - self.lowest_corner(T)) * self.up
        return T

    def relabel_if_turned(self, T, time):
        """Released carried cube resting flat on another face: smallest relabel that puts +Z up again."""
        R_now = T[:3, :3] @ self.relabel["insertive"][:3, :3]
        if R_now[:, 2] @ self.up > np.cos(np.radians(45)) or CS.resting_tilt_deg(T[:3, :3], self.up) > CS.MAX_RESTING_TILT_DEG:
            return
        S = max(CS.z_up_labels(T[:3, :3], self.up), key=lambda S: np.trace(R_now.T @ T[:3, :3] @ S))
        self.relabel["insertive"] = np.eye(4)
        self.relabel["insertive"][:3, :3] = S
        self.relabel_events.append((time, "insertive", S.copy()))

    def detect(self, image_bgr, timestamp, skip=()):
        """Measured cube poses {cube: (T_base_cube, n_tags, reprojection px)} in this frame. Every detector result, used or
        not (no tags, Kalman prediction only), is kept in self.last_raw for the episode log. Cubes in skip are not
        searched for (the held cube); a detector that skipped frames starts its next frame from scratch."""
        out, self.last_raw = {}, {}
        for name, det in self.detectors.items():
            if name in skip:
                self.skipped.add(name)
                self.last_raw[name] = dict(skipped=True)
                continue
            if name in self.skipped:
                forget_track(det)
                self.skipped.discard(name)
            res = det.process_frame(image_bgr, timestamp=timestamp)
            T_base = None
            if res["success"] and res["T"] is not None:
                T_cam = np.array(res["T"], dtype=float)
                T_cam[:3, 3] /= 1000.0        # AprilCube reports translation in millimeters
                T_base = self.T_base_camera @ T_cam
                if not res.get("predicted"):
                    out[name] = (T_base, int(res["n_tags"]), float(res["reproj_error"]))
            self.last_raw[name] = dict(
                success=bool(res["success"]), predicted=bool(res.get("predicted", False)), T_base=T_base,
                n_tags=int(res.get("n_tags") or 0), n_inliers=int(res.get("n_inliers") or 0),
                reproj_px=float(res["reproj_error"]) if res.get("reproj_error") is not None else float("nan"),
                tag_ids=[int(d[0]) for d in res.get("detections") or []],
                corners_px=[np.asarray(d[1], dtype=float).reshape(4, 2) for d in res.get("detections") or []])
        return out

    def stack_offset(self):
        """Horizontal distance (m) between the cube centres when a camera reading, not the hand's estimate, shows the
        carried cube resting flat on the bottom cube's top with the +Z face the policy sees up (with any_face_up=False:
        its tag +Z face, training's goal); else None. The caller checks that the gripper lets go."""
        if not self.measured.get("insertive") or "receptive" not in self.last:
            return None
        T, B = self.last["insertive"][0], self.last["receptive"][0]
        on_top = abs(self.lowest_corner(T) - (self.table_height + self.CUBE_SIZE_M)) < 1e-3   # onto_support's choice
        flat = CS.resting_tilt_deg(T[:3, :3], self.up) < 0.5
        z_up = self.z_from_up_deg("insertive") < 1.0
        return self.horizontal(T[:3, 3] - B[:3, 3]) if on_top and flat and z_up else None

    def z_from_up_deg(self, name):
        """Angle between up and the +Z axis the policy sees for this cube (its last pose, relabelled)."""
        z = (self.last[name][0] @ self.relabel[name])[:3, 2]
        return float(np.degrees(np.arccos(np.clip(z @ self.up, -1.0, 1.0))))

    def resting_flat(self, detections):
        return all(n in detections and CS.resting_tilt_deg(detections[n][0][:3, :3], self.up) <= CS.MAX_RESTING_TILT_DEG
                   for n in ("receptive", "insertive"))

    def set_relabel(self, detections):
        S_b, S_c, yaw_error = CS.relabel(detections["receptive"][0][:3, :3], detections["insertive"][0][:3, :3], self.up)
        self.relabel = {"receptive": np.eye(4), "insertive": np.eye(4)}
        self.relabel["receptive"][:3, :3] = S_b
        if self.any_face_up:
            self.relabel["insertive"][:3, :3] = S_c
        self.relabel_start = {name: S.copy() for name, S in self.relabel.items()}
        return yaw_error

    def poses(self, detections, detection_time, detection_wrist_T, wrist_T, now, holding):
        """detections: detect() on the frame taken at detection_time (time.time() seconds), when the wrist was at
        detection_wrist_T (detection_time None: no result yet). wrist_T, now: this policy step's wrist pose and time."""
        if not holding:
            if self.carried_now is not None and self.carried_now[2] > self.last["insertive"][2]:
                T_hand, W_hand, t_hand = self.carried_now          # released: it falls from where the hand let go
                self.last["insertive"] = (self.dropped(T_hand), W_hand, t_hand)
                self.measured["insertive"] = False
            self.grasp, self.carried_now = None, None
        elif self.grasp is None:
            self.grasp = (wrist_T, now)
        fresh = detection_time is not None and (self.applied_time is None or detection_time > self.applied_time)
        rejected = {}
        if fresh:
            self.applied_time = detection_time
            for name in ("receptive", "insertive"):          # bottom first: the carried cube's support check uses it
                if name not in detections:
                    continue
                T, reason = self.check(name, detections[name][0], holding, detection_wrist_T, detection_time)
                if reason:
                    if name in self.last:
                        rejected[name] = reason
                        continue
                    T = detections[name][0]                         # first sighting: used as measured
                self.last[name] = (T, detection_wrist_T, detection_time)
                self.measured[name] = True
                if name == "insertive" and not holding and self.relabel is not None and self.any_face_up:
                    self.relabel_if_turned(T, detection_time)
        out, info = {}, {}
        for name in ("receptive", "insertive"):
            if name not in self.last:
                raise RuntimeError(f"{name} cube has never been detected")
            T, W_seen, seen_time = self.last[name]
            carried = False
            if name == "insertive" and holding:
                # static until the grasp, rigid with the wrist after it: carry from the later of sighting and grasp
                T_carried = self.carried_at(wrist_T)
                if T_carried is not None:
                    T, carried = T_carried, True
                    self.carried_now = (T, wrist_T, now)
            out[name] = F.matrix_to_pos_quat(T @ self.relabel[name])
            info[name] = dict(age_s=now - seen_time, carried_with_wrist=carried,
                              seen=fresh and name in detections and not rejected.get(name),
                              rejected=rejected.get(name, ""))
        return out, info


def forget_track(det):
    """Make an AprilCube CubePoseEstimator (commit 80ed7c7) treat its next frame like its first. It carries the last pose,
    a Kalman filter and optical-flow corners from frame to frame; after skipped frames they are stale (the cube was
    carried away), and optical flow from the old frame could put the cube back where it was picked up."""
    det.prev_rvec = det.prev_tvec = None
    det._prev_gray = det._prev_corners_2d = det._prev_corners_3d = None
    if getattr(det, "pose_filter", None) is not None:
        det.pose_filter.reset()


class GraspHold:
    """Holds the arm at the wrist pose of a close command until the gripper reports contact (gOBJ 2), closes empty, the
    policy reopens, or max_s passes. R214's sim gripper reaches a 60 mm cube ~0.2 s after the command and the policy
    starts lifting 0.1-0.2 s after closing; the Robotiq needs ~0.36 s (speed 255) to ~0.56 s (speed 128), so without the
    hold the fingers close on air (October 9 runs). max_s = 0 disables it.

    update() returns the (wrist position, quaternion) to hold this step, or None to follow the policy; when a hold ends,
    self.ended holds the reason for that step."""

    def __init__(self, max_s, empty_closed_position, closed=True):
        self.max_s = float(max_s)
        self.empty_closed_position = float(empty_closed_position)
        self.closed = bool(closed)            # last gripper command (episodes start with the gripper closed)
        self.active = None                    # (start time, wrist position, wrist quaternion)
        self.ended = ""

    def update(self, close_cmd, gripper_position, object_status, now, wrist_pos, wrist_quat):
        self.ended = ""
        if close_cmd and not self.closed and self.max_s > 0:
            self.active = (now, np.array(wrist_pos, dtype=float), np.array(wrist_quat, dtype=float))
        self.closed = bool(close_cmd)
        if self.active is None:
            return None
        start, pos, quat = self.active
        if not close_cmd:
            self.ended = "policy reopened"
        elif object_status == 2:
            self.ended = "contact"
        elif object_status == 3 and gripper_position >= self.empty_closed_position - 10:
            self.ended = "closed empty"
        elif now - start > self.max_s:
            self.ended = "timeout"
        if self.ended:
            self.active = None
            return None
        return pos, quat


def frame_log_arrays(frames, max_tags=6):
    """Per-camera-frame detection records -> arrays for the episode .npz (prefix frame_). Each record: frame_time,
    robot_time and q of the robot state nearest the frame, detect_s, and CubeEstimator.last_raw. Missing values are NaN
    (poses, reprojection, corners) or -1 (tag ids)."""
    n = len(frames)
    out = {"frame_time": np.array([f["frame_time"] for f in frames], dtype=float).reshape(n),
           "frame_robot_time": np.array([f["robot_time"] for f in frames], dtype=float).reshape(n),
           "frame_q": np.array([f["q"] for f in frames], dtype=float).reshape(n, 6),
           "frame_detect_s": np.array([f["detect_s"] for f in frames], dtype=float).reshape(n)}
    for name in ("receptive", "insertive"):
        raw = [f["raw"].get(name, {}) for f in frames]
        T = np.full((n, 4, 4), np.nan)
        ids = np.full((n, max_tags), -1, dtype=int)
        corners = np.full((n, max_tags, 4, 2), np.nan)
        for i, r in enumerate(raw):
            if r.get("T_base") is not None:
                T[i] = r["T_base"]
            for k, (tag, c) in enumerate(zip(r.get("tag_ids", [])[:max_tags], r.get("corners_px", [])[:max_tags])):
                ids[i, k], corners[i, k] = tag, c
        out[f"frame_{name}_skipped"] = np.array([r.get("skipped", False) for r in raw], dtype=bool)
        out[f"frame_{name}_success"] = np.array([r.get("success", False) for r in raw], dtype=bool)
        out[f"frame_{name}_predicted"] = np.array([r.get("predicted", False) for r in raw], dtype=bool)
        out[f"frame_{name}_n_tags"] = np.array([r.get("n_tags", 0) for r in raw], dtype=int)
        out[f"frame_{name}_n_inliers"] = np.array([r.get("n_inliers", 0) for r in raw], dtype=int)
        out[f"frame_{name}_reproj_px"] = np.array([r.get("reproj_px", np.nan) for r in raw], dtype=float)
        out[f"frame_{name}_T_base"] = T
        out[f"frame_{name}_tag_ids"] = ids
        out[f"frame_{name}_corners_px"] = corners
    return out


def build_frame(last_action, arm_q, gripper_joints, wrist_pos, wrist_quat, cubes):
    (rp, rq), (ip, iq) = cubes["receptive"], cubes["insertive"]
    return O.frame_terms(last_action, np.concatenate([arm_q, gripper_joints]), wrist_pos, wrist_quat, ip, iq, rp, rq)
