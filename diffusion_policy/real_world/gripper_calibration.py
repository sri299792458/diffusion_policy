"""Validation, recording and monitored moves for offline-testable calibration scripts."""
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper

FREQUENCY = 125
THUNDER_IP = '10.33.55.89'


def positive(value, name, maximum=None):
    if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be finite and in (0, {maximum or "infinity"}]')
    return value


def gripper_settings(speed, force, count, port):
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool)
           for value in (speed, force, count, port)):
        raise ValueError('speed, force, count and port must be integers')
    if not 0 <= speed <= 255 or not 0 <= force <= 255:
        raise ValueError('speed and force must be 0..255 (0 selects the hardware minimum)')
    if count < 1 or not 1 <= port <= 65535:
        raise ValueError('cycle/trial count must be positive and port must be 1..65535')


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result


def pull_direction(text):
    d = vector([float(x) for x in text.split(',')], 3, 'direction')
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('direction must be a nonzero finite 3-vector')
    return d / norm


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class Recording:
    """Reserve the output before connecting; save atomically, including partial traces."""
    def __init__(self, output, info):
        self.path = Path(output).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental replacement of previous calibration data.
        with self.path.open('xb') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self.info = dict(info, schema_version=2, run_status='initializing', cleanup_errors=[])
        self.arrays = {}

    def error(self, exc):
        self.info.update(run_status='aborted' if isinstance(exc, (KeyboardInterrupt, SystemExit, EOFError)) else 'error',
                         error=dict(type=type(exc).__name__, message=str(exc)))
        if hasattr(exc, 'last_state'):
            self.info['error']['gripper_state'] = exc.last_state

    def save(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix='.npz', delete=False) as handle:
                temporary = Path(handle.name)
                arrays = {key: np.asarray(values).reshape(-1, 6) if key.endswith(('_force', '_pose'))
                          else np.asarray(values) for key, values in self.arrays.items()}
                np.savez(handle, info=json.dumps(json_safe(self.info), allow_nan=False), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def cleanup(recording, actions):
    """Every cleanup action runs even if another action or saving fails."""
    for name, action in actions:
        try:
            if action() is False:
                raise RuntimeError(f'{name} was rejected')
        except BaseException as exc:
            recording.info['cleanup_errors'].append(dict(action=name, type=type(exc).__name__, message=str(exc)))
    if recording.info['cleanup_errors'] and recording.info['run_status'] == 'complete':
        recording.info['run_status'] = 'cleanup_error'
    recording.save()
    if recording.info['run_status'] == 'cleanup_error':
        raise RuntimeError('Cleanup failed; see cleanup_errors in the recording')


def open_gripper(gripper, speed, force):
    position, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), speed, force)
    if status != RobotiqGripper.ObjectStatus.AT_DEST or abs(position - gripper.get_open_position()) > 2:
        raise RuntimeError(f'Fully open gripper not confirmed: position={position}, status={status.name}')
    return position


def verify_grasp(status, record):
    """Keep firmware detection separate from an operator's observation of soft tips."""
    record.update(object_detected=status == RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
                  grasp_confirmation=None, operator_confirmed=None)
    if record['object_detected']:
        record['grasp_confirmation'] = 'firmware'
        return
    if status != RobotiqGripper.ObjectStatus.AT_DEST:
        raise RuntimeError(f'Unexpected gripper closing status: {status.name}')
    # A compliant fingertip grasp can reach the same motor position as an empty
    # close. Neither the motor position nor OBJ=3 establishes absence of contact.
    record['grasp_confirmation'] = 'pending'
    try:
        answer = input('Closing reached its target without object detection. '
                       'Visually check that BOTH fingertips still contact and hold the cube; '
                       'keep hands clear. Is the grasp confirmed? [y/N] ')
    except BaseException:
        record['grasp_confirmation'] = 'interrupted'
        raise
    record['operator_confirmed'] = answer.strip().lower() == 'y'
    record['grasp_confirmation'] = 'operator' if record['operator_confirmed'] else 'rejected'
    if not record['operator_confirmed']:
        raise SystemExit('Grasp not confirmed; run aborted')


def activate_gripper(gripper, recording):
    """Retain calibration samples as well as the subsequent profile/pull samples."""
    calibration = dict(status='running', phases={})
    recording.info['calibration'] = calibration

    def sample(state):
        phase = state['phase']
        phase_info = calibration['phases'].setdefault(phase, {})
        phase_info.update(commanded=state['commanded'], last_sample=state)
        for key in ('t', 'position', 'object_status', 'requested_position', 'fault', 'go_to'):
            recording.arrays.setdefault(f'calibration_{phase}_{key}', []).append(state[key])

    try:
        gripper.activate(on_calibration_sample=sample)
    except BaseException as exc:
        calibration.update(status='aborted' if isinstance(exc, KeyboardInterrupt) else 'error',
                           error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    calibration.update(status='complete', open_position=gripper.get_open_position(),
                       closed_position=gripper.get_closed_position())


class RobotMonitor:
    """Read coherent, advancing RTDE states; stop callers on stale or invalid data."""
    def __init__(self, control, receive, stale_s=0.25):
        positive(stale_s, 'stale_s')
        self.control, self.receive, self.stale_s = control, receive, stale_s
        self.last_timestamp = None
        self.last_raw = None
        self.payload = None

    def sample(self):
        deadline = time.monotonic() + self.stale_s
        while time.monotonic() < deadline:
            r = self.receive
            if not r.isConnected() or not self.control.isConnected():
                raise RuntimeError('RTDE disconnected')
            if r.getSafetyMode() not in (1, 2) or r.getRobotMode() != 7:
                raise RuntimeError('Robot is not running in normal/reduced safety mode')
            before = float(r.getTimestamp())
            raw = dict(robot_t=before, force=r.getActualTCPForce(), pose=r.getActualTCPPose(),
                       speed=r.getActualTCPSpeed(), payload_mass=r.getPayload(), payload_cog=r.getPayloadCog())
            after = float(r.getTimestamp())
            self.last_raw = raw
            if not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0:
                raise RuntimeError('Invalid robot timestamp')
            if self.last_timestamp is not None and before < self.last_timestamp:
                raise RuntimeError('Robot timestamp regressed')
            state = dict(robot_t=before, force=vector(raw['force'], 6, 'force'),
                         pose=vector(raw['pose'], 6, 'TCP pose'), speed=vector(raw['speed'], 6, 'TCP speed'))
            payload = np.r_[float(raw['payload_mass']), vector(raw['payload_cog'], 3, 'payload CoG')]
            if not np.isfinite(payload).all() or payload[0] < 0:
                raise RuntimeError('Invalid configured payload')
            if before == after and (self.last_timestamp is None or before > self.last_timestamp):
                if self.payload is None:
                    self.payload = payload
                elif not np.allclose(payload, self.payload, atol=1e-6, rtol=0):
                    raise RuntimeError('Configured payload changed during calibration')
                self.last_timestamp = before
                return state
            time.sleep(0.001)
        raise TimeoutError(f'No fresh coherent RTDE state within {self.stale_s} s')

    def stationary(self, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        quiet_since = None
        while time.monotonic() < deadline:
            state = self.sample()
            quiet = np.linalg.norm(state['speed'][:3]) < 0.001 and np.linalg.norm(state['speed'][3:]) < 0.01
            if not quiet:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = state['robot_t']
            if quiet_since is not None and state['robot_t'] - quiet_since >= 0.1:
                return state
            time.sleep(1 / FREQUENCY)
        raise TimeoutError('Robot did not settle after stopping')


def monitored_move(control, monitor, recording, prefix, target, speed, baseline, max_force,
                   direction=None, slip_min=3.0, slip_drop=0.4, slip_hold_s=0.05,
                   load_reference=None, max_total_force=None):
    """Monitor every linear move; async completion must belong to this command."""
    positive(speed, 'speed', 0.01)
    positive(max_force, 'max_force', 200.0)
    positive(slip_min, 'slip_min')
    positive(slip_hold_s, 'slip_hold_s')
    if not np.isfinite(slip_drop) or not 0 < slip_drop < 1:
        raise ValueError('slip_drop must be finite and in (0, 1)')
    target = vector(target, 6, 'target pose')
    baseline = vector(baseline, 6, 'baseline force')
    if (load_reference is None) != (max_total_force is None):
        raise ValueError('load_reference and max_total_force must be provided together')
    if load_reference is not None:
        load_reference = vector(load_reference, 6, 'open force reference')
        positive(max_total_force, 'max_total_force', 200.0)
    start_state = monitor.stationary()
    start = start_state['pose']
    delta = target[:3] - start[:3]
    distance = np.linalg.norm(delta)
    if distance > 0.05 + 1e-9 or np.linalg.norm(target[3:] - start[3:]) > 0.02:
        raise RuntimeError('Move is outside the short translation envelope')
    if np.linalg.norm(start_state['force'][:3] - baseline[:3]) >= max_force:
        raise RuntimeError('Force limit exceeded before motion')
    if load_reference is not None and np.linalg.norm(start_state['force'][:3] - load_reference[:3]) >= max_total_force:
        raise RuntimeError('Overall force limit exceeded before motion')
    if not control.isPoseWithinSafetyLimits(target.tolist()):
        raise RuntimeError('Target violates controller safety limits')
    before = control.getAsyncOperationProgress()
    if before >= 0:
        raise RuntimeError('Another asynchronous operation is still running')
    rec = dict(phase=prefix, target_pose=target.tolist(), start_pose=start.tolist(), speed=speed,
               status='running', stop_reason=None, peak_pull_N=0.0, peak_force_change_N=0.0,
               async_before=before, baseline_force=baseline.tolist(), max_force=max_force)
    if load_reference is not None:
        initial_load = start_state['force'][:3] - load_reference[:3]
        rec.update(open_force=load_reference.tolist(), max_total_force=max_total_force,
                   peak_total_force_change_N=float(np.linalg.norm(initial_load)),
                   peak_load_N=abs(float(np.dot(initial_load, direction))) if direction is not None else 0.0,
                   slip_force_reference='open gripper')
    else:
        rec['slip_force_reference'] = 'motion baseline'
    slip_reference = load_reference if load_reference is not None else baseline
    initial_axial_load = (abs(float(np.dot(start_state['force'][:3] - slip_reference[:3], direction)))
                          if direction is not None else 0.0)
    rec.update(slip_armed=False, slip_minimum_load_N=initial_axial_load,
               slip_loading_peak_N=0.0, slip_arm_min_rise_N=slip_min, slip_arm_hold_s=slip_hold_s)
    recording.info.setdefault('motions', []).append(rec)
    times, forces, poses, robot_times = [], [], [], []
    for key, values in (('t', times), ('force', forces), ('pose', poses), ('robot_t', robot_times)):
        recording.arrays[f'{prefix}_{key}'] = values
    t0 = time.monotonic()
    deadline = t0 + distance / speed + 5.0
    loading_since = drop_since = None
    try:
        if not control.moveL(target.tolist(), speed, 0.1, True):
            raise RuntimeError('Controller rejected moveL')
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('Linear motion timed out')
            state = monitor.sample()
            times.append(time.monotonic() - t0)
            forces.append(state['force']); poses.append(state['pose']); robot_times.append(state['robot_t'])
            change = state['force'][:3] - baseline[:3]
            pull = abs(float(np.dot(change, direction))) if direction is not None else 0.0
            rec['peak_pull_N'] = max(rec['peak_pull_N'], pull)
            change_norm = float(np.linalg.norm(change))
            rec['peak_force_change_N'] = max(rec['peak_force_change_N'], change_norm)
            slip_force = pull
            total_norm = None
            if load_reference is not None:
                load = state['force'][:3] - load_reference[:3]
                total_norm = float(np.linalg.norm(load))
                rec['peak_total_force_change_N'] = max(rec['peak_total_force_change_N'], total_norm)
                slip_force = abs(float(np.dot(load, direction))) if direction is not None else 0.0
                rec['peak_load_N'] = max(rec['peak_load_N'], slip_force)
            if change_norm >= max_force:
                rec['stop_reason'] = 'max_force'
                break
            if total_norm is not None and total_norm >= max_total_force:
                rec['stop_reason'] = 'max_total_force'
                break
            if distance > 1e-9:
                travel = float(np.dot(state['pose'][:3] - start[:3], delta / distance))
                lateral = state['pose'][:3] - start[:3] - travel * delta / distance
                if travel < -0.001 or travel > distance + 0.001 or np.linalg.norm(lateral) > 0.002:
                    raise RuntimeError('TCP departed from the commanded translation')
            # Closing preload can relax as the hand starts pulling without any
            # slip. Require a sustained rise from the lowest load observed in
            # this motion before accepting a later drop as a slip candidate.
            if direction is not None and not rec['slip_armed']:
                rec['slip_minimum_load_N'] = min(rec['slip_minimum_load_N'], slip_force)
                if slip_force > slip_min and slip_force - rec['slip_minimum_load_N'] >= slip_min:
                    if loading_since is None:
                        loading_since = state['robot_t']
                    elif state['robot_t'] - loading_since >= slip_hold_s:
                        rec.update(slip_armed=True, slip_armed_robot_t=state['robot_t'],
                                   slip_loading_peak_N=slip_force)
                else:
                    loading_since = None
            if rec['slip_armed']:
                rec['slip_loading_peak_N'] = max(rec['slip_loading_peak_N'], slip_force)
            if rec['slip_armed'] and slip_force < (1 - slip_drop) * rec['slip_loading_peak_N']:
                if drop_since is None:
                    drop_since = state['robot_t']
                elif state['robot_t'] - drop_since >= slip_hold_s:
                    rec['stop_reason'] = 'slip'
                    break
            else:
                drop_since = None
            progress = control.getAsyncOperationProgress()
            if progress < 0 and progress != before:
                rec['async_after'] = progress
                if np.linalg.norm(state['pose'][:3] - target[:3]) > 0.001:
                    raise RuntimeError('Async operation ended before reaching the target')
                rec['stop_reason'] = 'max_travel' if direction is not None else 'target_reached'
                break
            time.sleep(1 / FREQUENCY)
        control.stopL(0.5)
        stopped = monitor.stationary()
        rec.update(status='complete', stopped_pose=stopped['pose'].tolist(),
                   travel_at_stop_m=float(np.dot(stopped['pose'][:3] - start[:3], direction)) if direction is not None else None,
                   slip_candidate=rec['stop_reason'] == 'slip')
        if direction is None and rec['stop_reason'] in ('max_force', 'max_total_force'):
            raise RuntimeError(f'Force limit reached during {prefix}')
        return rec
    except BaseException as exc:
        rec.update(status='aborted' if isinstance(exc, KeyboardInterrupt) else 'error',
                   error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if rec['status'] != 'complete':
            try:
                control.stopL(0.5)
            except BaseException as exc:
                recording.info['cleanup_errors'].append(dict(action=f'stop {prefix}', message=str(exc)))
