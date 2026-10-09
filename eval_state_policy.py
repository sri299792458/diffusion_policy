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

Policy in control: keep the hardware emergency stop at hand.
Keys (OpenCV window "Policy Control"): S stop the episode, R move to the start pose and start a new episode,
G open the gripper for 5 steps, Q quit.
"""
import json
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_env import RealEnv, DEFAULT_OBS_KEY_MAP
from diffusion_policy.real_world.rtde_interpolation_controller import install_kinematics_calibration
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose
from diffusion_policy.real_world.thunder_state_policy import frames as F
from diffusion_policy.real_world.thunder_state_policy import obs as O
from diffusion_policy.real_world.thunder_state_policy import r214 as R

OBS_KEY_MAP = dict(DEFAULT_OBS_KEY_MAP, gripper_position='gripper_position',
                   gripper_object_status='gripper_object_status', gripper_state_timestamp='gripper_state_timestamp')


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
def main(output, robot_ip, camera_transform, camera_transform_frame, l515_serial, resolution, init_joints,
         max_duration, frequency):
    import aprilcube
    m = R.load_manifest()
    act = R.load_policy(m)
    gmap = R.GripperMap(m)
    scale = np.asarray(m['training']['arm_action_scale'], dtype=float)
    calibration = R.load_calibration()
    install_kinematics_calibration(calibration)              # this process's get_ee_pose
    T_base_camera = F.load_camera_transform(camera_transform, camera_transform_frame)
    up = np.asarray(m['scene_in_base_link']['world_up'], dtype=float)
    width, height = (int(v) for v in resolution.split('x'))
    print(f"R214 action scale: {scale}")

    output = pathlib.Path(output)
    log_dir = output / 'state_policy_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    dt = 1 / frequency

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
            robot_kwargs=dict(kinematics_calibration=calibration, gripper_speed=128, gripper_force=0,
                              read_gripper_state=True, **R.THUNDER_PAYLOAD),
        ) as env:
            cv2.setNumThreads(1)
            print("Waiting for realsense")
            time.sleep(5.0)

            K = env.realsense.get_intrinsics()[0]
            intrinsics = {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
            pkg = R.PKG
            detectors = {'receptive': aprilcube.detector(str(pkg / m['cubes']['receptive_detector']), intrinsics),
                         'insertive': aprilcube.detector(str(pkg / m['cubes']['insertive_detector']), intrinsics)}
            estimator = R.CubeEstimator(T_base_camera, detectors, up)
            print('Ready!')

            def current_pose_action(close):
                q = env.get_obs()['arm_joint_pos'][-1]
                pos, quat = get_ee_pose(q)
                return np.concatenate([pos, quat_to_axis_angle(quat), [-1.0 if close else 1.0]])[None]

            episode = 0
            while True:
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

                # ========== policy control loop (as eval_real_robot.py) ==========
                history = O.ObservationHistory()
                last_action = np.zeros(7)
                closed_cmd = True
                gripper_open_steps_remaining = 0
                GRIPPER_OPEN_DURATION = 5
                STUCK_WINDOW_S = 2.0
                STUCK_JOINT_THRESHOLD_RAD = 0.002
                STUCK_GRIPPER_OPEN_STEPS = int(frequency)
                stuck_buffer = []
                log = []
                start_delay = 1.0
                eval_t_start = time.time() + start_delay
                t_start = time.monotonic() + start_delay
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
                        image_bgr = cv2.cvtColor(obs['front_rgb'][-1], cv2.COLOR_RGB2BGR)
                        detections = estimator.detect(image_bgr, obs_timestamps[-1])
                        cubes, info = estimator.poses(detections, W, iter_idx, holding)
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
                        target_actions = np.concatenate([tgt_pos, quat_to_axis_angle(tgt_quat), gripper_actions[0]])[None]

                        # deal with timing (eval_real_robot.py)
                        action_timestamps = np.arange(1, dtype=np.float64) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                        env.exec_actions(actions=target_actions, timestamps=action_timestamps, obs_actions=raw_actions)
                        closed_cmd = bool(gripper_actions[0, 0] < 0)
                        last_action = raw_actions[0].copy()       # what the arm/gripper were asked to do (as in sim)

                        log.append(dict(t=float(obs_timestamps[-1]), q=q.copy(), gripper_position=gpos, gripper_object=gobj,
                                        holding=holding, obs=obs_vec, action=action, executed_action=raw_actions[0].copy(),
                                        target=target_actions[0].copy(),
                                        cubes=np.array([np.r_[cubes[n][0], cubes[n][1]] for n in ('receptive', 'insertive')]),
                                        seen=[info[n]['seen'] for n in ('receptive', 'insertive')],
                                        carried=info['insertive']['carried_with_wrist'],
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
                    env.end_episode()
                    save_log(log_dir / f'episode_{episode:03d}.npz', log, yaw_error, estimator)
                    episode += 1
                print('Stopped.')
                if key == 'q':
                    return
                if key != 'r' and wait_for_key(env) == 'q':
                    return
                if init_joints:
                    env.robot.reset_to_initial_position()
                    time.sleep(5.0)


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


def save_log(path, log, yaw_error, estimator):
    if not log:
        return
    arrays = {k: np.array([row[k] for row in log]) for k in log[0]}
    np.savez_compressed(path, **arrays, bottom_cube_yaw_error_deg=yaw_error,
                        relabel=np.array([estimator.relabel['receptive'], estimator.relabel['insertive']]))
    print(f"Saved {len(log)} policy steps to {path}")


if __name__ == '__main__':
    main()
