"""Record fixed-cube pull trials with the UR wrist sensor; see gripper_calibration.md."""
import argparse
import time

import numpy as np

from diffusion_policy.real_world.robotiq_gripper import (
    RobotiqGripper, DEFAULT_GRIPPER_SPEED, DEFAULT_GRIPPER_FORCE,
)
from diffusion_policy.real_world.gripper_calibration import (
    FREQUENCY, THUNDER_IP, Recording, RobotMonitor, activate_gripper, cleanup, gripper_settings,
    monitored_move, open_gripper, positive, pull_direction, verify_grasp,
)


def main(robot_ip, gripper_port, label, speed, force, trials, direction, pull_speed, max_travel, max_force,
         slip_min, slip_drop, freedrive, output, max_total_force=None):
    gripper_settings(speed, force, trials, gripper_port)
    d = pull_direction(direction)
    positive(pull_speed, 'pull_speed', 0.01)
    positive(max_travel, 'max_travel', 0.05)
    positive(max_force, 'max_force', 200.0)
    if max_total_force is None:
        max_total_force = max_force
    positive(max_total_force, 'max_total_force', 200.0)
    positive(slip_min, 'slip_min')
    if slip_min >= max_force or not np.isfinite(slip_drop) or not 0 < slip_drop < 1:
        raise ValueError('slip_min must be below max_force; slip_drop must be finite and in (0, 1)')
    if not label.strip():
        raise ValueError('label must not be empty')
    recording = Recording(output, dict(label=label, robot_ip=robot_ip, speed=speed, force=force,
                                       direction_base=d.tolist(), pull_speed=pull_speed, max_travel=max_travel,
                                       max_force=max_force, max_total_force=max_total_force,
                                       force_limit_reference='closed grasp baseline',
                                       total_force_limit_reference='tared open gripper',
                                       slip_force_reference='tared open gripper',
                                       slip_min=slip_min, slip_drop=slip_drop,
                                       slip_hold_s=0.05, slip_arm_hold_s=0.05,
                                       slip_arm_rule='sustained axial load rise from motion minimum',
                                       recorded_unix=time.time(), trials=[]))
    rtde_c = rtde_r = gripper = monitor = current_trial = None
    teaching = False
    try:
        recording.save()
        # Import lazily so --help and validation do not require the workstation SDK.
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        input('Clear all objects from the finger sweep. Empty calibration will move the fingers through their range. Press Enter ...')
        rtde_c = RTDEControlInterface(robot_ip, FREQUENCY,
                                     RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
        rtde_r = RTDEReceiveInterface(robot_ip, FREQUENCY)
        monitor = RobotMonitor(rtde_c, rtde_r)
        monitor.stationary()
        recording.info.update(payload_mass=float(monitor.payload[0]), payload_cog=monitor.payload[1:].tolist(),
                              payload_source='controller configuration, preserved')
        gripper = RobotiqGripper()
        gripper.connect(robot_ip, gripper_port)
        activate_gripper(gripper, recording)
        recording.info.update(open_position=gripper.get_open_position(), closed_position=gripper.get_closed_position(),
                              run_status='running')
        open_gripper(gripper, speed, force)
        if freedrive:
            teaching = True
            if not rtde_c.teachMode():
                raise RuntimeError('Controller rejected freedrive')
        input('Position the OPEN gripper around the fixed cube at the desired grasp depth, then press Enter ...')
        if teaching:
            if not rtde_c.endTeachMode():
                raise RuntimeError('Controller rejected leaving freedrive')
            teaching = False
        open_gripper(gripper, speed, force)
        state = monitor.stationary()
        start, preview_base = state['pose'].copy(), state['force'].copy()
        preview = start.copy(); preview[:3] += 0.005 * d
        monitored_move(rtde_c, monitor, recording, 'preview_out', preview, 0.01, preview_base, max_force)
        monitored_move(rtde_c, monitor, recording, 'preview_back', start, 0.01, preview_base, max_force)
        recording.save()
        if input('Did the hand move 5 mm in the intended pull direction and back? [y/N] ').strip().lower() != 'y':
            raise SystemExit('Direction not confirmed; no pull trials performed')
        for trial in range(trials):
            current_trial = dict(trial=trial, status='initializing', stop_reason=None)
            recording.info['trials'].append(current_trial)
            open_gripper(gripper, speed, force)
            state = monitor.stationary()
            monitored_move(rtde_c, monitor, recording, f'trial{trial}_start', start, 0.01, state['force'], max_force)
            if not rtde_c.zeroFtSensor():
                raise RuntimeError('Controller rejected force-sensor zeroing')
            time.sleep(0.5)
            state = monitor.stationary()
            open_force = state['force'].copy()
            current_trial.update(open_force=open_force.tolist(), open_pose=state['pose'].tolist(),
                                 open_robot_t=state['robot_t'], status='closing')
            grasp_t0 = time.monotonic()

            def check_grasp_load(state):
                load = state['force'][:3] - open_force[:3]
                load_norm = float(np.linalg.norm(load))
                current_trial.update(grasp_last_force=state['force'].tolist(),
                                     grasp_last_load_change_N=load_norm)
                if load_norm >= max_total_force:
                    current_trial['stop_reason'] = 'max_total_force'
                    raise RuntimeError('Overall force limit reached while establishing the grasp baseline: '
                                       f'{load_norm:.1f} N >= {max_total_force:.1f} N')
                if np.linalg.norm(state['pose'][:3] - start[:3]) > 0.001:
                    raise RuntimeError('TCP moved while establishing the baseline')

            def sample_grasp(gripper_state=None):
                if gripper_state is not None:
                    current_trial['grasp_last_gripper_sample'] = gripper_state
                    for key in ('t', 'position', 'object_status', 'requested_position', 'fault', 'go_to'):
                        recording.arrays.setdefault(f'trial{trial}_close_{key}', []).append(gripper_state[key])
                state = monitor.sample()
                for key, value in (('t', time.monotonic() - grasp_t0), ('force', state['force']),
                                   ('pose', state['pose']), ('robot_t', state['robot_t'])):
                    recording.arrays.setdefault(f'trial{trial}_grasp_{key}', []).append(value)
                check_grasp_load(state)
                return state

            try:
                pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), speed, force,
                                                          on_sample=sample_grasp)
            except BaseException:
                # Stop the fingers immediately; arm/script cleanup may take longer.
                try:
                    gripper.stop()
                except BaseException as exc:
                    recording.info['cleanup_errors'].append(dict(action='stop closing gripper', message=str(exc)))
                raise
            current_trial.update(grasp_position=int(pos), grasp_status=status.name)
            current_trial['status'] = 'settling'
            settle_until = time.monotonic() + 1.0
            while time.monotonic() < settle_until:
                sample_grasp()
                time.sleep(1 / FREQUENCY)
            state = monitor.stationary()
            current_trial['grasp_force_after_settle'] = state['force'].tolist()
            check_grasp_load(state)
            verify_grasp(status, current_trial)
            current_trial['status'] = 'baseline'
            samples, sample_times = [], []
            recording.arrays[f'trial{trial}_baseline_force'] = samples
            recording.arrays[f'trial{trial}_baseline_robot_t'] = sample_times
            for _ in range(FREQUENCY):
                state = monitor.sample()
                samples.append(state['force']); sample_times.append(state['robot_t'])
                check_grasp_load(state)
                time.sleep(1 / FREQUENCY)
            base = np.mean(samples, axis=0)
            preload = base[:3] - open_force[:3]
            current_trial.update(baseline_force=base.tolist(), preload_force_base_N=preload.tolist(),
                                 preload_norm_N=float(np.linalg.norm(preload)),
                                 preload_along_direction_N=float(np.dot(preload, d)), status='pulling')
            print(f'trial {trial + 1}: closing preload {current_trial["preload_norm_N"]:.1f} N; '
                  f'limits {max_force:.1f} N additional / {max_total_force:.1f} N overall')
            target = start.copy(); target[:3] += max_travel * d
            pulled = monitored_move(rtde_c, monitor, recording, f'trial{trial}', target, pull_speed, base,
                                    max_force, direction=d, slip_min=slip_min, slip_drop=slip_drop,
                                    load_reference=open_force, max_total_force=max_total_force)
            for key in ('peak_pull_N', 'peak_force_change_N', 'peak_load_N', 'peak_total_force_change_N',
                        'stop_reason', 'travel_at_stop_m', 'slip_candidate', 'slip_armed',
                        'slip_minimum_load_N', 'slip_loading_peak_N'):
                current_trial[key] = pulled[key]
            current_trial.update(status='returning', gripper_status_after=RobotiqGripper.ObjectStatus(
                gripper._get_var(gripper.OBJ)).name)
            open_gripper(gripper, speed, force)
            # A successful open and stationary state are prerequisites for returning.
            monitored_move(rtde_c, monitor, recording, f'trial{trial}_return', start, 0.01, open_force, max_force)
            current_trial['status'] = 'complete'
            recording.save()
            print(f'trial {trial + 1}: peak axial change from grasp baseline {pulled["peak_pull_N"]:.1f} N, '
                  f'peak overall axial load {pulled["peak_load_N"]:.1f} N, '
                  f'{pulled["stop_reason"]}, travel {pulled["travel_at_stop_m"] * 1000:.1f} mm')
            if not pulled['slip_armed']:
                print('Pull load did not build enough to arm slip detection; no slip maximum measured.')
        recording.info['run_status'] = 'complete'
    except BaseException as exc:
        recording.error(exc)
        if current_trial is not None and current_trial['status'] != 'complete':
            current_trial.update(status=recording.info['run_status'], error=recording.info['error'])
        raise
    finally:
        if monitor is not None:
            recording.info['last_robot_sample'] = monitor.last_raw
        actions = []
        if rtde_c is not None:
            actions.append(('stop arm', lambda: rtde_c.stopL(0.5)))
            if teaching:
                actions.append(('leave freedrive', rtde_c.endTeachMode))
            actions.append(('stop control script', rtde_c.stopScript))
        if gripper is not None:
            actions.extend([('gripper stop', gripper.stop), ('gripper disconnect', gripper.disconnect)])
        if rtde_c is not None:
            actions.append(('control disconnect', rtde_c.disconnect))
        if rtde_r is not None:
            actions.append(('receive disconnect', rtde_r.disconnect))
        cleanup(recording, actions)
        print(f'Saved {recording.path} ({recording.info["run_status"]})')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot_ip', '--robot-ip', '-ri', default=THUNDER_IP)
    p.add_argument('--gripper_port', '--gripper-port', type=int, default=63352)
    p.add_argument('--label', required=True)
    p.add_argument('--speed', type=int, default=DEFAULT_GRIPPER_SPEED, help='Raw 0..255 setting; 0 is minimum speed')
    p.add_argument('--force', type=int, default=DEFAULT_GRIPPER_FORCE, help='Raw 0..255 setting; 0 is minimum force, not zero N')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--direction', default='0,-1,0',
                   help='Controller BASE frame; Thunder upward preview uses -y, verify before confirming')
    p.add_argument('--pull_speed', '--pull-speed', type=float, default=0.002)
    p.add_argument('--max_travel', '--max-travel', type=float, default=0.02)
    p.add_argument('--max_force', '--max-force', type=float, default=100.0,
                   help='N of translational force change from the closed grasp baseline')
    p.add_argument('--max_total_force', '--max-total-force', type=float,
                   help='N of overall translational force change from the tared OPEN reference, '
                        'including closing preload; defaults to max_force')
    p.add_argument('--slip_min', '--slip-min', type=float, default=3.0)
    p.add_argument('--slip_drop', '--slip-drop', type=float, default=0.4)
    p.add_argument('--freedrive', dest='freedrive', action='store_true', default=True)
    p.add_argument('--no-freedrive', dest='freedrive', action='store_false')
    p.add_argument('--output', '-o', required=True, help='New .npz path; parent directories are created')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        main(**vars(args))
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
