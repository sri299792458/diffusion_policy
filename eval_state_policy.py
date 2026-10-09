"""
Evaluate the R214 state policy (UWLab OmniReset, broad 60 mm cube stacking, Thunder UR5e + Robotiq 2F-85 with UMI
fingertips) on the real robot.

Usage:
(robodiff)$ python eval_state_policy.py -o <save_dir> --robot_ip 10.33.55.89 \
    --camera_transform T_base_l515.json --camera_transform_frame ur_base -j

Same structure as eval_real_robot.py: RealEnv (cameras + RTDEInterpolationController OSC with its error clip), get_obs()
aligned to the camera frame, Cartesian target = observed pose + scale * action (apply_delta_pose), exec_actions() with the
same action timing, the same stuck-detection macro and keys. Differences, each required for this state policy on Thunder:
  - Observation: R214's 215 numbers (thunder_state_policy/obs.py, verified against R214's training env) built from the
    observed joints, the calibrated wrist pose, the Robotiq position mapped to the six sim gripper joints, and the cube
    poses from AprilCube on the L515 color frame (RealEnv camera 0) and the L515->base transform.
  - Action scale (0.02, 0.02, 0.02, 0.02, 0.2, 0.02): R214's training scale (it was not finetuned to the Stage-2 scale).
  - Robot settings passed to the controller: Thunder's kinematic calibration and payload, gripper speed 128 / force 0.
  - The gripper is closed (holding the start pose) before each episode: R214's Reaching starts have it closed.
  - No SpaceMouse (not used by the policy loop) and no extra camera-view video (RealEnv still records the L515 video).
  - Cube detection runs in a background thread on each new L515 frame (AprilCube takes ~45 ms per cube at 1280x720), with
    the L515's distortion coefficients and OpenCV's default thread count; each policy step uses the newest result, which
    carries its frame time and the wrist pose at that time. Inline and with cv2.setNumThreads(1) a step took ~300 ms.
  - While the gripper holds the carried cube, only the bottom cube is detected: the held cube's pose comes from the
    grasp (CubeEstimator), and skipping its detector halves the detection time (~80 instead of ~160 ms per frame here;
    ~350 ms more when the held cube shows no tag).
  - OpenCV's X11 helper libraries are loaded before PyAV (see share_opencv_x11_libs_with_pyav): with the pip wheels,
    importing av otherwise makes cv2.imshow hang.
  - Grasp hold (--grasp_hold_s, r214.GraspHold): after a close command the arm holds the close pose until the gripper
    reports contact (at most grasp_hold_s). The Robotiq takes ~0.36-0.56 s to reach the cube, R214's sim gripper ~0.2 s,
    and the policy starts lifting 0.1-0.2 s after closing. The policy's own actions are still what it observes.
  - Stop on a stable stack (--stack_stop_mm): the episode ends once a camera reading shows the carried cube resting
    flat on the bottom cube within stack_stop_mm of its centre, with the gripper open and not holding, for
    STACK_STEPS consecutive policy steps (UWLab's eval / data-collection config ends an episode after 5 consecutive
    successful steps). R214 was trained to keep the stack within its success test (5 mm, 0.025 rad) until the time
    limit, which is tighter than the L515 can confirm next to the gripper (5-15 mm): in dp_run6 it re-gripped a good
    stack five times until the time limit.
  - Controller-rate log (--high_rate_log, on by default): the controller records every 500 Hz loop (state, torque sent,
    OSC target, motor currents; thunder_state_policy/high_rate_log.py) to <output>/high_rate_logs/<session>/, and each
    policy step logs the time its target was sent (command_time) to align the two. For checking a simulator's arm model
    in the regime the policy uses; RealEnv itself keeps only the last 30 controller samples.

Policy in control: keep the hardware emergency stop at hand.
Keys (OpenCV window "Policy Control"): S stop the episode, R move to the start pose and start a new episode,
G open the gripper for 5 steps, Q quit.
"""
import ctypes
import importlib.util
import json
import pathlib
import threading
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np


def share_opencv_x11_libs_with_pyav():
    """The opencv-python and PyAV wheels bundle X11/xcb helper libraries under identical file names (libxcb-shm-7a199f70,
    libXau-00ec42fe, ...) but linked to different libxcb copies. Linux loads one library per name, so if PyAV's load
    first, OpenCV's Qt windows use them with the wrong libxcb and cv2.imshow hangs. Load OpenCV's copies first (before
    av is imported); PyAV only needs its own for FFmpeg's screen capture, which is not used here."""
    cv2_spec, av_spec = importlib.util.find_spec('cv2'), importlib.util.find_spec('av')
    if cv2_spec is None or av_spec is None:
        return
    cv2_libs = next(pathlib.Path(cv2_spec.origin).parent.parent.glob('opencv*.libs'), None)
    av_libs = pathlib.Path(av_spec.origin).parent.parent / 'av.libs'
    if cv2_libs is None or not av_libs.is_dir():
        return
    for name in sorted({p.name for p in cv2_libs.iterdir()} & {p.name for p in av_libs.iterdir()}):
        ctypes.CDLL(str(cv2_libs / name), mode=ctypes.RTLD_GLOBAL)


share_opencv_x11_libs_with_pyav()                           # before diffusion_policy.real_world imports av

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_env import RealEnv, DEFAULT_OBS_KEY_MAP
from diffusion_policy.real_world.rtde_interpolation_controller import install_kinematics_calibration
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose
from diffusion_policy.real_world.thunder_state_policy import frames as F
from diffusion_policy.real_world.thunder_state_policy import obs as O
from diffusion_policy.real_world.thunder_state_policy import r214 as R

OBS_KEY_MAP = dict(DEFAULT_OBS_KEY_MAP, ActualQd='arm_joint_vel', gripper_position='gripper_position',
                   gripper_object_status='gripper_object_status', gripper_state_timestamp='gripper_state_timestamp')


def l515_color_intrinsics(serial, width, height):
    """Factory intrinsics of the L515 color stream at this resolution, read before RealEnv opens the camera (RealEnv
    shares only fx, fy, cx, cy). The distortion matters: leaving it out moved a cube up to 15 mm."""
    import pyrealsense2 as rs
    for device in rs.context().query_devices():
        if device.get_info(rs.camera_info.serial_number) != serial:
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    video = profile.as_video_stream_profile()
                    if (video.width(), video.height()) == (width, height):
                        return video.get_intrinsics()
    raise RuntimeError(f"No {width}x{height} color stream on RealSense {serial}")


class CubeDetectionThread:
    """Detects the cubes on each new L515 frame in the background; latest() returns (detections, frame time, wrist pose
    at that time) for CubeEstimator.poses. The wrist pose comes from the robot state nearest the frame time.
    self.frames keeps every frame's full detector results for the episode log (r214.frame_log_arrays).
    The policy loop sets self.holding each step; while it is set, the carried cube is not detected."""

    def __init__(self, env, estimator):
        self.env, self.estimator = env, estimator
        self.holding = False
        self.lock = threading.Lock()
        self.result = ({}, None, None)
        self.frames = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='cube-detection', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def latest(self):
        if self.error is not None:
            raise RuntimeError(f"cube detection thread failed: {self.error}")
        with self.lock:
            return self.result

    def _run(self):
        last_time = None
        try:
            while not self.stop_event.is_set():
                camera = self.env.realsense.get(k=1)[0]
                frame_time = float(camera['timestamp'][-1])
                if frame_time == last_time:
                    time.sleep(0.002)
                    continue
                last_time = frame_time
                robot = self.env.robot.get_all_state()
                i = int(np.argmin(np.abs(robot['robot_receive_timestamp'] - frame_time)))
                q = np.array(robot['ActualQ'][i], dtype=float)
                wrist = F.pos_quat_to_matrix(*get_ee_pose(q))
                t_detect = time.monotonic()
                detections = self.estimator.detect(cv2.cvtColor(camera['color'][-1], cv2.COLOR_RGB2BGR), frame_time,
                                                   skip=('insertive',) if self.holding else ())
                self.frames.append(dict(frame_time=frame_time, robot_time=float(robot['robot_receive_timestamp'][i]),
                                        q=q, detect_s=time.monotonic() - t_detect, raw=self.estimator.last_raw))
                with self.lock:
                    self.result = (detections, frame_time, wrist)
        except Exception as exc:                                # surfaced to the policy loop by latest()
            self.error = f"{type(exc).__name__}: {exc}"


@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.1.10")
@click.option('--camera_transform', required=True,
              help='4x4 T with p_base = T @ p_camera for the L515 color optical frame (.npy, or .json key T_base_camera)')
@click.option('--camera_transform_frame', required=True, type=click.Choice(list(F.FRAMES)),
              help='ur_base: UR controller Base (getActualTCPPose / pendant); base_link: sim / calibrated kinematics')
@click.option('--l515_serial', default='f1380660', help='L515 serial number')
@click.option('--resolution', default='1280x720', help='L515 color resolution (detection runs at full resolution)')
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help='Move to the R214 start pose (moveJ) when the controller starts; R then returns there.')
@click.option('--max_duration', '-md', default=60.0, help='Max duration for each episode in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz (R214: 10).')
@click.option('--grasp_hold_s', default=1.0, type=float,
              help='After a close command, hold the arm still until the gripper reports contact, closes empty, the '
                   'policy reopens, or this many seconds pass (0: off). See r214.GraspHold.')
@click.option('--gripper_speed', default=128, type=click.IntRange(0, 255),
              help='Robotiq speed (force stays 0). Closing on the 60 mm cube takes ~0.56 s at 128 and ~0.36 s at 255 '
                   '(calibration/thunder_gripper/2026-10-06/speed_comparison); R214\'s sim gripper takes ~0.2 s.')
@click.option('--stack_stop_mm', default=10.0, type=float,
              help='End the episode when the carried cube is seen resting on the bottom cube within this distance '
                   '(mm, horizontal, centre to centre) with the gripper open for 5 policy steps (0: run to the time '
                   'limit).')
@click.option('--high_rate_log/--no_high_rate_log', default=True,
              help='Record every 500 Hz controller loop to <output>/high_rate_logs/<session>/ (high_rate_log.py).')
@click.option('--any_face_up/--tag_face_up', default=True,
              help='any_face_up: the carried cube may end on any face (its faces are relabelled so the face on top is '
                   '+Z, also after it lands on another face). tag_face_up: it keeps the faces its tags define, as in '
                   'training, whose goal is its tag +Z face up, so the policy turns it over when that face is not up.')
def main(output, robot_ip, camera_transform, camera_transform_frame, l515_serial, resolution, init_joints,
         max_duration, frequency, gripper_speed, grasp_hold_s, stack_stop_mm, high_rate_log, any_face_up):
    import aprilcube
    m = R.load_manifest()
    act = R.load_policy(m)
    gmap = R.GripperMap(m)
    scale = np.asarray(m['training']['arm_action_scale'], dtype=float)
    calibration = R.load_calibration()
    install_kinematics_calibration(calibration)              # this process's get_ee_pose
    T_base_camera = F.load_camera_transform(camera_transform, camera_transform_frame)
    up = np.asarray(m['scene_in_base_link']['world_up'], dtype=float)
    table_height = float(m['scene_in_base_link']['table_top_y_m'])       # along up (base_link +y on Thunder)
    width, height = (int(v) for v in resolution.split('x'))
    l515 = l515_color_intrinsics(l515_serial, width, height)
    dist_coeffs = np.asarray(l515.coeffs[:5], dtype=float)
    print(f"R214 action scale: {scale}")
    print(f"L515 color {width}x{height}: {l515.model}, distortion {np.round(dist_coeffs, 4).tolist()}")

    output = pathlib.Path(output)
    log_dir = output / 'state_policy_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    dt = 1 / frequency
    high_rate_dir = (output / 'high_rate_logs' / time.strftime('%Y%m%dT%H%M%S')) if high_rate_log else None
    if high_rate_dir is not None:
        print(f"Controller-rate log: {high_rate_dir}")

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=1,                                  # R214's 5-step history is kept per policy step below
            obs_image_resolution=(width, height),
            video_capture_resolution=(width, height),
            obs_float32=False,
            init_joints=init_joints,
            custom_init_joints=R.START_JOINTS if init_joints else None,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            rolling_action_buffer=True,
            action_mode='cartesian',
            camera_serial_numbers=[l515_serial],
            camera_configs=None,
            obs_key_map=OBS_KEY_MAP,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
            robot_kwargs=dict(kinematics_calibration=calibration, gripper_speed=gripper_speed, gripper_force=0,
                              read_gripper_state=True, high_rate_log_dir=high_rate_dir, **R.THUNDER_PAYLOAD),
        ) as env:
            # OpenCV keeps its default thread count here: cube detection is the heavy work in this process.
            # RealEnv waits only launch_timeout (3 s) for the robot; with -j its controller first moves to the start
            # pose and calibrates the gripper (longer after a power cycle), so wait until both report ready.
            print("Waiting for realsense and robot")
            time.sleep(2.0)
            deadline = time.monotonic() + 90.0
            while not env.is_ready:
                if not env.robot.is_alive():
                    raise RuntimeError("Robot controller process exited during startup (see its messages above)")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Not ready after 90 s: camera ready {env.realsense.is_ready}, robot ready "
                                       f"{env.robot.is_ready}")
                time.sleep(0.1)

            K = env.realsense.get_intrinsics()[0]
            if not np.allclose([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], [l515.fx, l515.fy, l515.ppx, l515.ppy], atol=0.5):
                raise RuntimeError(f"RealEnv camera intrinsics {K} differ from the L515 {width}x{height} profile")
            intrinsics = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
            pkg = R.PKG
            detectors = {name: aprilcube.detector(str(pkg / m['cubes'][f'{name}_detector']), intrinsics,
                                                  dist_coeffs=dist_coeffs)
                         for name in ('receptive', 'insertive')}
            estimator = R.CubeEstimator(T_base_camera, detectors, up, table_height, any_face_up=any_face_up)
            run_meta = dict(
                l515_serial=l515_serial, resolution=[width, height], camera_matrix=K.tolist(),
                dist_coeffs=dist_coeffs.tolist(), camera_transform=str(camera_transform),
                camera_transform_frame=camera_transform_frame, T_base_camera=T_base_camera.tolist(),
                frequency=frequency, action_scale=scale.tolist(), policy_sha256=m['policy']['sha256'],
                gripper=dict(speed=gripper_speed, force=0), grasp_hold_s=grasp_hold_s, any_face_up=any_face_up,
                stack_stop_mm=stack_stop_mm, high_rate_log_dir=None if high_rate_dir is None else str(high_rate_dir),
                start_joints=R.START_JOINTS, init_joints=init_joints, max_duration=max_duration,
                controller=dict(kp=np.diag(env.robot.osc_Kp).tolist(), kd=np.diag(env.robot.osc_Kd).tolist(),
                                error_clip_pos_m=env.robot.osc_error_delta_pos,
                                error_clip_rot_rad=env.robot.osc_error_delta_rot),
                code=code_version())
            print('Ready!')

            def current_pose_action(close):
                q = env.get_obs()['arm_joint_pos'][-1]
                pos, quat = get_ee_pose(q)
                return np.concatenate([pos, quat_to_axis_angle(quat), [-1.0 if close else 1.0]])[None]

            while True:
                # numbered like RealEnv's replay buffer and videos, which continue across runs into the same directory
                episode = env.replay_buffer.n_episodes
                # ========== episode setup: gripper closed at the current pose, cubes found resting flat ==========
                env.exec_actions(actions=current_pose_action(close=True), timestamps=np.array([time.time() + 0.05]),
                                 obs_actions=np.array([[0.0] * 6 + [-1.0]]))
                time.sleep(1.5)
                estimator.reset()
                deadline = time.monotonic() + 5.0
                while True:
                    obs = env.get_obs()
                    detections = estimator.detect(cv2.cvtColor(obs['front_rgb'][-1], cv2.COLOR_RGB2BGR), obs['timestamp'][-1])
                    if estimator.resting_flat(detections) or time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
                if not estimator.resting_flat(detections):
                    print('Both cubes must be detected resting flat (tilt <= 10 deg); fix the scene and press R in the '
                          'window, or Q to quit.')
                    if wait_for_key(env) == 'q':
                        return
                    continue
                yaw_error = estimator.set_relabel(detections)
                print(f"Bottom cube yaw {yaw_error:+.1f} deg from training (limit +/-{R.CS.MAX_BOTTOM_YAW_ERROR_DEG:g})")
                if abs(yaw_error) > R.CS.MAX_BOTTOM_YAW_ERROR_DEG:
                    print(f"Turn the bottom cube {-yaw_error:+.0f} deg about vertical, then press R (or Q to quit).")
                    if wait_for_key(env) == 'q':
                        return
                    continue
                # the setup reading seeds the cube poses; from here the detection thread keeps them current
                W_setup = F.pos_quat_to_matrix(*get_ee_pose(obs['arm_joint_pos'][-1]))
                t_setup = float(obs['timestamp'][-1])
                estimator.poses(detections, t_setup, W_setup, W_setup, t_setup, holding=False)
                if not any_face_up:
                    print(f"Carried cube: tag +Z face {estimator.z_from_up_deg('insertive'):.0f} deg from up "
                          "(goal: up; 90 = on its side, one turn; 180 = upside down, two turns)")
                detection_thread = CubeDetectionThread(env, estimator).start()

                # ========== policy control loop (as eval_real_robot.py) ==========
                history = O.ObservationHistory()
                last_action = np.zeros(7)
                closed_cmd = True
                grasp_hold = R.GraspHold(grasp_hold_s, gmap.real[-1], closed=True)
                gripper_open_steps_remaining = 0
                GRIPPER_OPEN_DURATION = 5
                STUCK_WINDOW_S = 2.0
                STUCK_JOINT_THRESHOLD_RAD = 0.002
                STUCK_GRIPPER_OPEN_STEPS = int(frequency)
                STACK_STEPS = 5
                stacked_steps = 0
                stuck_buffer = []
                log = []
                start_delay = 1.0
                eval_t_start = time.time() + start_delay
                t_start = time.monotonic() + start_delay
                meta = dict(run_meta, episode=episode, episode_start_time=eval_t_start,     # video frame k ~ start + k / fps
                            video=str(env.video_dir / str(episode) / '0.mp4'),
                            video_fps=env.video_capture_fps, bottom_cube_yaw_error_deg=yaw_error)
                env.start_episode(eval_t_start)
                precise_wait(eval_t_start, time_func=time.time)
                print("Started!")
                iter_idx = 0
                key = None
                try:
                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_step = time.monotonic()
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        q = obs['arm_joint_pos'][-1]
                        wpos, wquat = get_ee_pose(q)
                        W = F.pos_quat_to_matrix(wpos, wquat)
                        gpos = float(obs['gripper_position'][-1])
                        gobj = int(obs['gripper_object_status'][-1])
                        holding = gmap.holding(gpos, gobj, closed_cmd)
                        detection_thread.holding = holding
                        detections, detection_time, detection_wrist = detection_thread.latest()
                        cubes, info = estimator.poses(detections, detection_time, detection_wrist, W,
                                                      float(obs_timestamps[-1]), holding)
                        # stable stack seen with the gripper open: end the episode (see the docstring)
                        stack_offset = None if holding or closed_cmd else estimator.stack_offset()
                        stacked = (stack_stop_mm > 0 and stack_offset is not None
                                   and stack_offset * 1000 <= stack_stop_mm)
                        stacked_steps = stacked_steps + 1 if stacked else 0
                        if stacked_steps >= STACK_STEPS:
                            meta['stacked'] = dict(t=float(obs_timestamps[-1]), offset_mm=stack_offset * 1000)
                            print(f"Stacked: the top cube is seen on the bottom cube {stack_offset * 1000:.1f} mm "
                                  f"off-centre with the gripper open for {STACK_STEPS} steps; ending the episode")
                            break
                        obs_vec = history.push(R.build_frame(last_action, q, gmap.sim_joints(gpos), wpos, wquat, cubes))
                        action = act(obs_vec)
                        raw_arm_action = action[None, :6]
                        gripper_actions = action[None, 6:7]

                        # Stuck detection (eval_real_robot.py): no movement for 2 s -> open the gripper for 1 s
                        if gripper_open_steps_remaining == 0:
                            t_now = time.monotonic()
                            stuck_buffer.append((t_now, q.copy()))
                            while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                stuck_buffer.pop(0)
                            if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                jps = np.array([b[1] for b in stuck_buffer])
                                if np.max(jps.max(axis=0) - jps.min(axis=0)) < STUCK_JOINT_THRESHOLD_RAD:
                                    gripper_open_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                    stuck_buffer.clear()
                                    print("[Stuck detection] No movement for 2s, opening gripper")
                        if gripper_open_steps_remaining > 0:
                            gripper_actions = np.ones_like(gripper_actions)
                            gripper_open_steps_remaining -= 1

                        raw_actions = np.concatenate([raw_arm_action, gripper_actions], axis=1)
                        # Cartesian OSC: scale delta, compute absolute target from the observed pose
                        tgt_pos, tgt_quat = apply_delta_pose(wpos, wquat, (raw_arm_action * scale)[0])
                        # Robotiq is slower than R214's sim gripper: hold the arm at the close pose until contact
                        hold = grasp_hold.update(bool(gripper_actions[0, 0] < 0), gpos, gobj, time.monotonic(), wpos, wquat)
                        if hold is not None:
                            tgt_pos, tgt_quat = hold
                        if grasp_hold.ended:
                            print(f"[grasp hold] ended at step {iter_idx}: {grasp_hold.ended}")
                        target_actions = np.concatenate([tgt_pos, quat_to_axis_angle(tgt_quat), gripper_actions[0]])[None]

                        # deal with timing (eval_real_robot.py)
                        action_timestamps = np.arange(1, dtype=np.float64) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                        command_time = time.time()
                        env.exec_actions(actions=target_actions, timestamps=action_timestamps, obs_actions=raw_actions)
                        closed_cmd = bool(gripper_actions[0, 0] < 0)
                        last_action = raw_actions[0].copy()       # what the arm/gripper were asked to do (as in sim)

                        log.append(dict(t=float(obs_timestamps[-1]), command_time=command_time, q=q.copy(), qd=np.array(obs['arm_joint_vel'][-1]),
                                        tcp_force=np.array(obs['tcp_force'][-1]), gripper_position=gpos, gripper_object=gobj,
                                        holding=holding, obs=obs_vec, action=action, executed_action=raw_actions[0].copy(),
                                        target=target_actions[0].copy(),
                                        cubes=np.array([np.r_[cubes[n][0], cubes[n][1]] for n in ('receptive', 'insertive')]),
                                        seen=[info[n]['seen'] for n in ('receptive', 'insertive')],
                                        cube_age_s=[info[n]['age_s'] for n in ('receptive', 'insertive')],
                                        rejected=[info[n]['rejected'] for n in ('receptive', 'insertive')],
                                        carried=info['insertive']['carried_with_wrist'],
                                        grasp_hold=hold is not None, grasp_hold_end=grasp_hold.ended,
                                        applied_detection_time=np.nan if estimator.applied_time is None
                                        else estimator.applied_time,
                                        grasp_time=np.nan if estimator.grasp is None else estimator.grasp[1],
                                        stack_offset=np.nan if stack_offset is None else stack_offset,
                                        compute_s=time.monotonic() - t_step))
                        if log[-1]['compute_s'] > dt:
                            print(f"[timing] step {iter_idx} took {log[-1]['compute_s'] * 1000:.0f} ms (> {dt * 1000:.0f} ms)")

                        vis = obs['front_rgb'][-1][..., ::-1].copy()
                        cv2.putText(vis, f"Episode {episode}  t={time.monotonic() - t_start:.1f}s  gripper "
                                    f"{'close' if closed_cmd else 'open'}  seen {log[-1]['seen']}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow('Policy Control', vis)
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                        elif key_stroke in (ord('s'), ord('r'), ord('q')):
                            key = chr(key_stroke)
                            break
                        if time.monotonic() - t_start > max_duration:
                            print('Terminated by the timeout!')
                            break
                        precise_wait(t_cycle_end)
                        iter_idx += 1
                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    key = 'q'
                finally:
                    detection_thread.stop()
                    env.end_episode()
                    save_log(log_dir / f'episode_{episode:03d}.npz', log, yaw_error, estimator, detection_thread.frames, meta)
                print('Stopped.')
                if key == 'q':
                    return
                if key != 'r' and wait_for_key(env) == 'q':
                    return
                if init_joints:
                    env.robot.reset_to_initial_position()
                    time.sleep(5.0)


def code_version():
    """Git commit of this repository and whether the working tree had uncommitted changes."""
    import subprocess
    here = pathlib.Path(__file__).resolve().parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        return dict(commit=None, dirty=None)
    return dict(commit=commit, dirty=dirty)


def wait_for_key(env):
    """Block until R (new episode) or Q (quit) is pressed in the OpenCV window."""
    while True:
        obs = env.get_obs()
        vis = obs['front_rgb'][-1][..., ::-1].copy()
        cv2.putText(vis, 'R: new episode   Q: quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Policy Control', vis)
        k = cv2.pollKey()
        if k in (ord('r'), ord('q')):
            return chr(k)
        time.sleep(0.05)


def save_log(path, log, yaw_error, estimator, frames, meta):
    """Episode log: one row per policy step (what the policy saw and did), one row per camera frame (frame_*: every
    AprilCube result with tag ids, corners, reprojection and the robot state at that frame), and meta (JSON: camera
    model and transform, video file and start time, controller and policy settings, code version)."""
    if not log:
        return
    path = pathlib.Path(path)
    if path.exists():                                       # never overwrite an earlier run's log
        path = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    arrays = {k: np.array([row[k] for row in log]) for k in log[0]}
    events = estimator.relabel_events
    np.savez_compressed(path, **arrays, **R.frame_log_arrays(frames), bottom_cube_yaw_error_deg=yaw_error,
                        relabel=np.array([estimator.relabel_start['receptive'], estimator.relabel_start['insertive']]),
                        relabel_event_time=np.array([e[0] for e in events], dtype=float),
                        relabel_event_cube=np.array([e[1] for e in events], dtype=str),
                        relabel_event_S=np.array([e[2] for e in events], dtype=float).reshape(len(events), 3, 3),
                        meta=json.dumps(meta, default=float))
    print(f"Saved {len(log)} policy steps and {len(frames)} camera frames to {path}")


if __name__ == '__main__':
    main()
