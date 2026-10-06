"""Record empty/cube closing profiles. See gripper_calibration.md for workstation setup."""
import argparse
import time

from diffusion_policy.real_world.robotiq_gripper import (
    RobotiqGripper, DEFAULT_GRIPPER_SPEED, DEFAULT_GRIPPER_FORCE,
)
from diffusion_policy.real_world.gripper_calibration import (
    Recording, THUNDER_IP, activate_gripper, cleanup, gripper_settings, open_gripper, verify_grasp,
)


def record_move(gripper, target, speed, force, settle_s=0.5, timeout_s=5.0, record=None):
    """Keep partial samples on failure; use the same completion logic as the pull test."""
    rec = record if record is not None else {}
    rec.update(t=[], position=[], object_status=[], requested_position=[], fault=[],
               commanded=target, stopped_s=None, final_status=None, status='running')
    stable_since = None
    previous = None

    def sample(state):
        nonlocal stable_since, previous
        for key in ('t', 'position', 'object_status', 'requested_position', 'fault'):
            rec[key].append(state[key])
        rec['commanded'] = state['commanded']
        current = (state['position'], state['object_status'])
        if state['object_status'] == RobotiqGripper.ObjectStatus.MOVING.value:
            stable_since = None
        elif stable_since is None or current != previous:
            stable_since = state['t']
        previous = current

    try:
        _, status = gripper.move_and_wait_for_pos(target, speed, force, timeout_s=timeout_s,
                                                  settle_s=settle_s, on_sample=sample)
        rec.update(stopped_s=stable_since, final_status=status.name, status='complete')
    except BaseException as exc:
        rec.update(status='aborted' if isinstance(exc, KeyboardInterrupt) else 'error',
                   error=dict(type=type(exc).__name__, message=str(exc)))
        try:
            gripper.stop()
        except BaseException as stop_error:
            rec['stop_error'] = str(stop_error)
        raise
    return rec


def main(robot_ip, gripper_port, speed, force, cycles, label, output):
    gripper_settings(speed, force, cycles, gripper_port)
    if label not in ('empty', 'cube40', 'cube60'):
        raise ValueError('label must be empty, cube40 or cube60')
    recording = Recording(output, dict(label=label, speed=speed, force=force, cycles=cycles,
                                       robot_ip=robot_ip, recorded_unix=time.time()))
    gripper = None
    try:
        recording.save()
        input('Clear all objects from the finger sweep. Empty calibration will move the fingers through their range. Press Enter ...')
        gripper = RobotiqGripper()
        gripper.connect(robot_ip, gripper_port)
        activate_gripper(gripper, recording)
        recording.info.update(open_position=gripper.get_open_position(), closed_position=gripper.get_closed_position(),
                              run_status='running')
        for c in range(cycles):
            open_gripper(gripper, speed, force)
            if label != 'empty':
                input(f'[cycle {c + 1}/{cycles}] Place {label} centred between the fingertips, then press Enter ...')
            for name, target in (('close', gripper.get_closed_position()), ('open', gripper.get_open_position())):
                prefix = f'cycle{c}_{name}'
                rec = {}
                recording.info[prefix] = rec
                try:
                    record_move(gripper, target, speed, force, record=rec)
                    if name == 'open' and (rec['final_status'] != 'AT_DEST'
                                          or abs(rec['position'][-1] - gripper.get_open_position()) > 2):
                        rec['status'] = 'error'
                        raise RuntimeError('Profile opening did not reach the calibrated open position')
                    if name == 'close' and label != 'empty':
                        verify_grasp(RobotiqGripper.ObjectStatus[rec['final_status']], rec)
                    if name == 'close' and label == 'empty' and rec['final_status'] != 'AT_DEST':
                        rec['status'] = 'error'
                        raise RuntimeError('Unexpected contact in the empty profile')
                    print(f'cycle {c + 1} {name}: stopped at {rec["stopped_s"]:.3f} s, '
                          f'position {rec["position"][-1]}, {rec["final_status"]}')
                except BaseException as exc:
                    rec.update(status='aborted' if isinstance(exc, (KeyboardInterrupt, SystemExit, EOFError)) else 'error',
                               error=dict(type=type(exc).__name__, message=str(exc)))
                    raise
                finally:
                    for key in ('t', 'position', 'object_status', 'requested_position', 'fault'):
                        recording.arrays[f'{prefix}_{key}'] = rec.pop(key, [])
                    recording.save()
        recording.info['run_status'] = 'complete'
    except BaseException as exc:
        recording.error(exc)
        raise
    finally:
        actions = [('gripper stop', gripper.stop), ('gripper disconnect', gripper.disconnect)] if gripper else []
        cleanup(recording, actions)
        print(f'Saved {recording.path} ({recording.info["run_status"]})')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot_ip', '--robot-ip', '-ri', default=THUNDER_IP)
    p.add_argument('--gripper_port', '--gripper-port', type=int, default=63352)
    p.add_argument('--speed', type=int, default=DEFAULT_GRIPPER_SPEED, help='Raw 0..255 setting; 0 is minimum speed')
    p.add_argument('--force', type=int, default=DEFAULT_GRIPPER_FORCE, help='Raw 0..255 setting; 0 is minimum force, not zero N')
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--label', choices=('empty', 'cube40', 'cube60'), required=True)
    p.add_argument('--output', '-o', required=True, help='New .npz path; parent directories are created')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        main(**vars(args))
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
