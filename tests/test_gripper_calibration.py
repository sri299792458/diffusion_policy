"""Regression tests with a fake firmware/robot and clock; never open a socket."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diffusion_policy.real_world import gripper_calibration as cal
from diffusion_policy.real_world import robotiq_gripper as driver
from scripts.sim2real import gripper_pullout_test as pull
from scripts.sim2real import record_gripper_profile as profile


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def time(self):
        return 123456.0

    def sleep(self, duration):
        self.t += duration


class FirmwareGripper(driver.RobotiqGripper):
    def __init__(self, clock, *, initial=0, delay=0.04, duration=0.06, contact=False):
        super().__init__()
        self.clock = clock
        self.position = initial
        self.initial = initial
        self.target = initial
        self.command = initial
        self.started = None
        self.delay, self.duration, self.contact = delay, duration, contact
        self.block_open = False
        self.fault = 0
        self.acknowledge = True
        self.accept = True
        self.go_to = 1
        self.stopped = False
        self.disconnected = False
        self.activation_error = False
        self.commands = []

    def connect(self, *args):
        self.socket = Mock()

    def disconnect(self):
        self.disconnected = True
        super().disconnect()

    def activate(self, *args, **kwargs):
        if self.activation_error:
            raise RuntimeError('fake activation failure')

    def _set_var(self, name, value):
        if name == self.GTO and value == 0:
            self.go_to = 0
            self.stopped = True
        return True

    def move(self, target, speed, force):
        self.initial = self.get_current_position()
        self.command = target
        self.target = 130 if target == self.get_closed_position() and self.contact else target
        self.started = self.clock.t
        self.go_to = 1
        self.commands.append(target)
        return self.accept, target

    def _status(self):
        if self.started is None:
            return 2 if self.contact and self.position == 130 else 3
        elapsed = self.clock.t - self.started
        if self.block_open and self.command == 0:
            return 2
        if elapsed < self.delay:
            return 2 if self.initial == 130 else 3
        if elapsed < self.delay + self.duration and self.target != self.initial:
            return 0
        self.position = self.target
        return 2 if self.command == self.get_closed_position() and self.contact else 3

    def _get_var(self, name):
        status = self._status()
        if name == self.POS:
            if status == 0:
                fraction = (self.clock.t - self.started - self.delay) / self.duration
                return round(self.initial + fraction * (self.target - self.initial))
            return self.position
        return {self.PRE: self.command if self.acknowledge else 17, self.OBJ: status,
                self.FLT: self.fault, self.GTO: self.go_to, self.STA: 3, self.ACT: 1}[name]


class PhysicalEndpointGripper(FirmwareGripper):
    """The measured Thunder endpoints differ from the initial requested bounds."""
    def __init__(self, clock, initial=3):
        super().__init__(clock, initial=initial)
        # Previous script finished open and sent GTO=0 during cleanup.
        self.go_to = 0

    def move(self, target, speed, force):
        accepted, requested = super().move(target, speed, force)
        self.target = min(227, max(3, self.target))
        return accepted, requested


class FakeControl:
    FLAG_VERBOSE, FLAG_UPLOAD_SCRIPT = 1, 2

    def __init__(self, clock):
        self.clock = clock
        self.position = np.zeros(6)
        self.start = self.target = self.position.copy()
        self.active = False
        self.started = 0.0
        self.duration = 0.0
        self.progress = -1
        self.moves = []
        self.reject_index = None
        self.never_start = False
        self.finish_short = False
        self.reject_teach = False
        self.reject_zero = False
        self.connected = True
        self.disconnected = False
        self.script_stopped = False
        self.teaching = False
        self.stops = 0
        self.stop_error = False
        self.safe_target = True
        self.force_mode = 'zero'
        self.payload_changed = False
        self.freeze_timestamp = False
        self.interrupt_pull = False

    def advance(self):
        if self.active:
            fraction = min(1.0, (self.clock.t - self.started) / self.duration) if self.duration else 1.0
            self.position = self.start + fraction * (self.target - self.start)
            if fraction >= 1:
                self.active = False
                self.progress = -2 if self.progress != -2 else -1
                if self.finish_short:
                    self.position[1] -= 0.003

    def isConnected(self):
        return self.connected

    def isPoseWithinSafetyLimits(self, target):
        return self.safe_target

    def moveL(self, target, speed, acceleration, asynchronous=False):
        if not asynchronous:
            raise AssertionError('All script motion must be monitored asynchronously')
        self.moves.append((target, speed))
        if len(self.moves) - 1 == self.reject_index:
            return False
        self.start, self.target = self.position.copy(), np.array(target)
        self.duration = np.linalg.norm(self.target[:3] - self.start[:3]) / speed
        self.started = self.clock.t
        self.active = not self.never_start
        return True

    def getAsyncOperationProgress(self):
        self.advance()
        return 0 if self.active else self.progress

    def stopL(self, *args):
        self.stops += 1
        if self.stop_error:
            raise RuntimeError('fake stop failure')
        self.advance()
        if self.active:
            self.progress = -2 if self.progress != -2 else -1
        self.active = False

    def stopScript(self):
        self.script_stopped = True

    def disconnect(self):
        self.disconnected = True

    def teachMode(self):
        self.teaching = True
        return not self.reject_teach

    def endTeachMode(self):
        self.teaching = False
        return True

    def zeroFtSensor(self):
        return not self.reject_zero


class FakeReceive:
    def __init__(self, clock, control):
        self.clock, self.control = clock, control
        self.disconnected = False
        self.bad_baseline = False
        self.safety_mode = 1
        self.disconnect_pull = False
        self.force_shape = 6
        self.preview_force = 0.0
        self.timestamp_value = None

    def isConnected(self):
        return not (self.disconnect_pull and len(self.control.moves) >= 4)

    def getRobotMode(self):
        return 7

    def getSafetyMode(self):
        return self.safety_mode

    def getTimestamp(self):
        if self.control.freeze_timestamp and len(self.control.moves) >= 4:
            if self.timestamp_value is None:
                self.timestamp_value = 100.0 + int(self.clock.t * 125) / 125
            return self.timestamp_value
        return 100.0 + int(self.clock.t * 125) / 125

    def getActualTCPPose(self):
        self.control.advance()
        return self.control.position.tolist()

    def getActualTCPSpeed(self):
        c = self.control
        c.advance()
        return ((c.target - c.start) / c.duration).tolist() if c.active and c.duration else [0.0] * 6

    def getActualTCPForce(self):
        c = self.control
        if self.bad_baseline and len(c.moves) == 3:
            return [0.0, np.nan, 0.0, 0.0, 0.0, 0.0]
        force = [0.0] * self.force_shape
        if len(c.moves) == 1 and c.active:
            force[1] = self.preview_force
        if len(c.moves) == 4 and c.active:
            elapsed = self.clock.t - c.started
            # Interrupt after real simulated travel, rather than after a fixed
            # number of register reads that can include stale-state retries.
            if c.interrupt_pull and elapsed >= 0.03:
                raise KeyboardInterrupt('fake interrupted pull')
            if c.force_mode == 'cap':
                force[1] = 101.0
            elif c.force_mode == 'slip':
                force[1] = 12.0 if elapsed < 0.3 else 2.0
            elif c.force_mode == 'noise':
                force[1] = 2.0 if 0.3 <= elapsed < 0.316 else 12.0
            elif c.force_mode == 'nan':
                force[1] = np.nan
        return force

    def getPayload(self):
        return 0.98 if self.control.payload_changed and len(self.control.moves) >= 4 else 1.04

    def getPayloadCog(self):
        return [-0.002, 0.001, 0.047]

    def disconnect(self):
        self.disconnected = True


class GraspVerificationTest(unittest.TestCase):
    def test_detected_grasp_does_not_need_operator_confirmation(self):
        record = {}
        with patch('builtins.input') as prompt:
            cal.verify_grasp(driver.RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT, record)
        prompt.assert_not_called()
        self.assertTrue(record['object_detected'])
        self.assertEqual(record['grasp_confirmation'], 'firmware')
        self.assertIsNone(record['operator_confirmed'])

    def test_unflagged_grasp_keeps_detection_false_after_confirmation(self):
        record = dict(final_status='AT_DEST')
        with patch('builtins.input', return_value=' Y '):
            cal.verify_grasp(driver.RobotiqGripper.ObjectStatus.AT_DEST, record)
        self.assertFalse(record['object_detected'])
        self.assertEqual(record['final_status'], 'AT_DEST')
        self.assertEqual(record['grasp_confirmation'], 'operator')
        self.assertTrue(record['operator_confirmed'])

    def test_unflagged_grasp_needs_explicit_y(self):
        for answer in ('', 'n', 'yes', 'anything'):
            with self.subTest(answer=answer), patch('builtins.input', return_value=answer):
                record = {}
                with self.assertRaisesRegex(SystemExit, 'Grasp not confirmed'):
                    cal.verify_grasp(driver.RobotiqGripper.ObjectStatus.AT_DEST, record)
                self.assertEqual(record['grasp_confirmation'], 'rejected')
                self.assertFalse(record['operator_confirmed'])

    def test_interrupted_confirmation_is_recorded(self):
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), patch('builtins.input', side_effect=error):
                record = {}
                with self.assertRaises(type(error)):
                    cal.verify_grasp(driver.RobotiqGripper.ObjectStatus.AT_DEST, record)
                self.assertEqual(record['grasp_confirmation'], 'interrupted')
                self.assertIsNone(record['operator_confirmed'])

    def test_unexpected_closing_status_cannot_be_confirmed(self):
        with patch('builtins.input') as prompt:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
                cal.verify_grasp(driver.RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT, {})
        prompt.assert_not_called()


class GripperCompletionTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.time_patch = patch.object(driver, 'time', self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def test_stale_contact_does_not_confirm_open(self):
        g = FirmwareGripper(self.clock, initial=130, contact=True, delay=0.4)
        pos, status = g.move_and_wait_for_pos(0, 128, 128)
        self.assertEqual((pos, status), (0, g.ObjectStatus.AT_DEST))
        self.assertGreaterEqual(self.clock.t, 0.56)

    def test_recorder_waits_through_delayed_motion(self):
        g = FirmwareGripper(self.clock, delay=0.4, duration=1.0, contact=True)
        rec = profile.record_move(g, 255, 128, 128)
        self.assertEqual(rec['final_status'], 'STOPPED_INNER_OBJECT')
        self.assertGreaterEqual(rec['stopped_s'], 1.4)
        self.assertGreaterEqual(rec['t'][-1], 1.9)

    def test_already_at_target_no_op_is_allowed(self):
        g = FirmwareGripper(self.clock)
        self.assertEqual(g.move_and_wait_for_pos(0, 128, 128)[0], 0)
        self.assertLess(self.clock.t, 0.5)

    def test_missed_short_move_can_complete_from_changed_position(self):
        g = FirmwareGripper(self.clock, duration=0.0001, contact=True)
        self.assertEqual(g.move_and_wait_for_pos(255, 128, 128)[0], 130)

    def test_stale_status_without_motion_times_out(self):
        g = FirmwareGripper(self.clock, initial=130, contact=True)
        g.block_open = True
        with self.assertRaises(TimeoutError):
            g.move_and_wait_for_pos(0, 128, 128, timeout_s=0.2)

    def test_missing_ack_times_out(self):
        g = FirmwareGripper(self.clock)
        g.acknowledge = False
        with self.assertRaises(TimeoutError):
            g.move_and_wait_for_pos(255, 128, 128, timeout_s=0.2)

    def test_rejected_command_and_fault_fail(self):
        g = FirmwareGripper(self.clock)
        g.accept = False
        with self.assertRaises(RuntimeError):
            g.move_and_wait_for_pos(255, 128, 128)
        g.accept, g.fault = True, 14
        with self.assertRaisesRegex(RuntimeError, 'fault 14'):
            g.move_and_wait_for_pos(255, 128, 128)

    def test_timeout_preserves_profile_and_stops_fingers(self):
        g = FirmwareGripper(self.clock, delay=2.0)
        rec = {}
        with self.assertRaises(TimeoutError):
            profile.record_move(g, 255, 128, 128, timeout_s=0.2, record=rec)
        self.assertTrue(rec['t'])
        self.assertEqual(rec['status'], 'error')
        self.assertTrue(g.stopped)

    def test_activation_and_reset_have_deadlines(self):
        g = driver.RobotiqGripper()
        g._set_var = Mock(return_value=True)
        g._get_var = Mock(return_value=1)
        with self.assertRaises(TimeoutError):
            g._reset(timeout_s=0.05)
        g._reset = Mock()
        g._get_var = Mock(side_effect=lambda key: 0 if key == g.ACT else 1)
        with self.assertRaises(TimeoutError):
            g.activate(auto_calibrate=False, timeout_s=0.05)

    def test_settling_resets_if_motion_resumes(self):
        g = FirmwareGripper(self.clock)
        original = g._get_var
        def bounce(name):
            if name == g.OBJ and g.started is not None:
                elapsed = self.clock.t - g.started
                return 0 if elapsed < 0.04 or 0.08 <= elapsed < 0.2 else 3
            return original(name)
        g._get_var = bounce
        g.move_and_wait_for_pos(255, 128, 128)
        self.assertGreaterEqual(self.clock.t, 0.3)

    def test_overlapping_gripper_command_is_refused(self):
        g = FirmwareGripper(self.clock)
        g.move(255, 128, 128)
        self.clock.sleep(0.05)
        with self.assertRaisesRegex(RuntimeError, 'already moving'):
            g.move_and_wait_for_pos(0, 128, 128)

    def test_auto_calibration_refuses_an_object(self):
        g = FirmwareGripper(self.clock, contact=True)
        with self.assertRaisesRegex(RuntimeError, 'object'):
            driver.RobotiqGripper.auto_calibrate(g, log=False)

    def test_calibration_when_already_open_at_physical_endpoint(self):
        g = PhysicalEndpointGripper(self.clock)
        driver.RobotiqGripper.auto_calibrate(g, log=False)
        self.assertEqual((g.get_open_position(), g.get_closed_position()), (3, 227))
        self.assertEqual(g.get_current_position(), 3)

    def test_calibration_when_already_closed_at_physical_endpoint(self):
        g = PhysicalEndpointGripper(self.clock, initial=227)
        driver.RobotiqGripper.auto_calibrate(g, log=False)
        self.assertEqual((g.get_open_position(), g.get_closed_position()), (3, 227))

    def test_calibration_when_already_at_interior_position(self):
        g = PhysicalEndpointGripper(self.clock, initial=127)
        driver.RobotiqGripper.auto_calibrate(g, log=False)
        self.assertEqual((g.get_open_position(), g.get_closed_position()), (3, 227))

    def test_new_process_can_calibrate_after_previous_process_stopped_open(self):
        for _ in range(2):
            g = PhysicalEndpointGripper(self.clock)
            # Use the real activation path, not the general test double's override.
            driver.RobotiqGripper.activate(g)
            self.assertEqual((g.get_open_position(), g.get_closed_position()), (3, 227))
            g.stop()
            self.assertEqual((g.get_current_position(), g.go_to), (3, 0))

    def test_failed_sweep_does_not_install_partial_calibration(self):
        g = FirmwareGripper(self.clock, contact=True)
        with self.assertRaises(RuntimeError):
            driver.RobotiqGripper.auto_calibrate(g, log=False)
        self.assertEqual((g.get_open_position(), g.get_closed_position()), (0, 255))

    def test_timeout_exposes_last_register_values(self):
        g = PhysicalEndpointGripper(self.clock)
        with self.assertRaises(TimeoutError) as caught:
            g.move_and_wait_for_pos(0, 64, 1, timeout_s=0.2)
        self.assertEqual(caught.exception.last_state['commanded'], 0)
        self.assertEqual(caught.exception.last_state['position'], 3)
        self.assertFalse(caught.exception.last_state['seen_motion'])


class PulloutTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'new-parent' / 'out.npz'
        self.clock = Clock()
        self.c = FakeControl(self.clock)
        self.r = FakeReceive(self.clock, self.c)
        self.g = FirmwareGripper(self.clock, contact=True)
        self.factory = Mock(return_value=self.c)
        self.factory.FLAG_VERBOSE, self.factory.FLAG_UPLOAD_SCRIPT = 1, 2
        self.recv_factory = Mock(return_value=self.r)
        self.g_factory = Mock(return_value=self.g)
        self.kwargs = dict(robot_ip='offline-only', gripper_port=63352, label='cube60', speed=128, force=128,
                           trials=1, direction='0,1,0', pull_speed=0.002, max_travel=0.02, max_force=100.0,
                           slip_min=3.0, slip_drop=0.4, freedrive=False, output=str(self.output))

    @contextlib.contextmanager
    def mocks(self, answers=None):
        with contextlib.ExitStack() as stack:
            for module in (driver, cal, pull, profile):
                stack.enter_context(patch.object(module, 'time', self.clock))
            stack.enter_context(patch.dict(sys.modules, {
                'rtde_control': types.SimpleNamespace(RTDEControlInterface=self.factory),
                'rtde_receive': types.SimpleNamespace(RTDEReceiveInterface=self.recv_factory),
            }))
            stack.enter_context(patch.object(pull, 'RobotiqGripper', self.g_factory))
            stack.enter_context(patch.object(profile, 'RobotiqGripper', self.g_factory))
            # Production code uses the class enum; factories expose it explicitly.
            self.g_factory.ObjectStatus = driver.RobotiqGripper.ObjectStatus
            stack.enter_context(patch('builtins.input', side_effect=answers or ['', '', 'y']))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def run_script(self, **kwargs):
        with self.mocks():
            return pull.main(**dict(self.kwargs, **kwargs))

    def saved(self):
        with np.load(self.output) as saved:
            return json.loads(str(saved['info'])), {key: saved[key] for key in saved.files if key != 'info'}

    def assert_cleaned(self):
        self.assertTrue(self.c.script_stopped)
        self.assertTrue(self.c.disconnected)
        self.assertTrue(self.r.disconnected)
        self.assertTrue(self.g.disconnected)
        self.assertGreater(self.c.stops, 0)

    def test_success_reaches_limit_preserves_payload_and_creates_directory(self):
        self.run_script()
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual(info['payload_mass'], 1.04)
        self.assertEqual(info['payload_cog'], [-0.002, 0.001, 0.047])
        self.assertEqual(info['trials'][0]['stop_reason'], 'max_travel')
        self.assertAlmostEqual(info['trials'][0]['travel_at_stop_m'], 0.02)
        self.assertTrue(np.all(np.diff(arrays['trial0_robot_t']) > 0))
        self.assert_cleaned()

    def test_negative_y_pull_reports_positive_travel(self):
        self.run_script(direction='0,-1,0', speed=0, force=0)
        info, arrays = self.saved()
        self.assertEqual(info['direction_base'], [0.0, -1.0, 0.0])
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual(info['trials'][0]['stop_reason'], 'max_travel')
        self.assertAlmostEqual(info['trials'][0]['travel_at_stop_m'], 0.02)
        self.assertLess(np.min(arrays['trial0_pose'][:, 1]), -0.019)
        self.assert_cleaned()

    def test_declined_direction_never_closes_or_pulls(self):
        with self.mocks(answers=['', '', 'n']):
            with self.assertRaisesRegex(SystemExit, 'Direction not confirmed'):
                pull.main(**dict(self.kwargs, direction='0,-1,0'))
        info, _ = self.saved()
        self.assertEqual(info['run_status'], 'aborted')
        self.assertFalse(info['trials'])
        self.assertEqual(len(self.c.moves), 2)
        self.assertNotIn(self.g.get_closed_position(), self.g.commands)
        self.assert_cleaned()

    def test_unflagged_grasp_can_pull_only_after_confirmation(self):
        self.g.contact = False
        with self.mocks(answers=['', '', 'y', 'y']):
            pull.main(**self.kwargs)
        info, _ = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        trial = info['trials'][0]
        self.assertEqual(trial['grasp_status'], 'AT_DEST')
        self.assertFalse(trial['object_detected'])
        self.assertEqual(trial['grasp_confirmation'], 'operator')
        self.assertTrue(trial['operator_confirmed'])
        self.assertEqual(trial['stop_reason'], 'max_travel')
        self.assert_cleaned()

    def test_declined_unflagged_grasp_never_pulls_or_returns(self):
        self.g.contact = False
        with self.mocks(answers=['', '', 'y', '']):
            with self.assertRaisesRegex(SystemExit, 'Grasp not confirmed'):
                pull.main(**self.kwargs)
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'aborted')
        self.assertEqual(info['trials'][0]['grasp_confirmation'], 'rejected')
        self.assertEqual(len(self.c.moves), 3)
        self.assertNotIn('trial0_force', arrays)
        self.assert_cleaned()

    def test_eof_during_unflagged_grasp_never_pulls(self):
        self.g.contact = False
        with self.mocks(answers=['', '', 'y', EOFError()]):
            with self.assertRaises(EOFError):
                pull.main(**self.kwargs)
        info, _ = self.saved()
        self.assertEqual(info['trials'][0]['status'], 'aborted')
        self.assertEqual(info['trials'][0]['grasp_confirmation'], 'interrupted')
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_force_limit_before_unflagged_grasp_prompt_never_pulls(self):
        self.g.contact = False
        self.r.getActualTCPForce = lambda: [0.0, 150.0 if self.g.position == 255 else 0.0, 0.0, 0.0, 0.0, 0.0]
        with self.mocks(answers=['', '', 'y']):
            with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
                pull.main(**self.kwargs)
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_force_limit_after_unflagged_grasp_prompt_never_pulls(self):
        self.g.contact = False
        confirmed = [False]
        def answer(prompt):
            if 'Is the grasp confirmed?' in prompt:
                confirmed[0] = True
            return 'y'
        self.r.getActualTCPForce = lambda: [0.0, 150.0 if confirmed[0] else 0.0, 0.0, 0.0, 0.0, 0.0]
        with self.mocks(), patch('builtins.input', side_effect=answer):
            with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
                pull.main(**self.kwargs)
        self.assertTrue(confirmed[0])
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_persistent_force_drop_is_slip(self):
        self.c.force_mode = 'slip'
        self.run_script()
        info, _ = self.saved()
        self.assertEqual(info['trials'][0]['stop_reason'], 'slip')
        self.assertLess(info['trials'][0]['travel_at_stop_m'], 0.002)
        self.assert_cleaned()

    def test_one_sample_force_drop_is_not_slip(self):
        self.c.force_mode = 'noise'
        self.run_script()
        self.assertEqual(self.saved()[0]['trials'][0]['stop_reason'], 'max_travel')

    def test_force_limit_stops_pull(self):
        self.c.force_mode = 'cap'
        self.run_script()
        self.assertEqual(self.saved()[0]['trials'][0]['stop_reason'], 'max_force')
        self.assert_cleaned()

    def preloaded_force(self, extra=0.0, release=False):
        # Include a nonzero residual tare: guards must compare vectors, not raw
        # sensor magnitudes. Only the moving pull adds load to the 26 N grasp.
        load = 26.0 if self.g.position == 130 else 0.0
        if len(self.c.moves) == 4 and self.c.active and load:
            elapsed = self.clock.t - self.c.started
            load = 0.0 if release and elapsed >= 0.3 else load + extra
        return [0.0, 5.0 - load, 0.0, 0.0, 0.0, 0.0]

    def test_explicit_overall_limit_preserves_preload_and_both_references(self):
        self.r.getActualTCPForce = lambda: self.preloaded_force(extra=10.0)
        self.run_script(max_force=20.0, max_total_force=40.0, direction='0,-1,0', speed=0, force=0)
        info, arrays = self.saved()
        trial = info['trials'][0]
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual(info['max_force'], 20.0)
        self.assertEqual(info['max_total_force'], 40.0)
        self.assertEqual(trial['open_force'][1], 5.0)
        self.assertAlmostEqual(trial['baseline_force'][1], -21.0)
        self.assertAlmostEqual(trial['preload_norm_N'], 26.0)
        self.assertAlmostEqual(trial['preload_along_direction_N'], 26.0)
        self.assertAlmostEqual(trial['peak_pull_N'], 10.0)
        self.assertAlmostEqual(trial['peak_load_N'], 36.0)
        self.assertEqual(trial['stop_reason'], 'max_travel')
        self.assertGreater(len(arrays['trial0_close_position']), 0)
        self.assertGreater(len(arrays['trial0_grasp_force']), 0)
        self.assertTrue(np.all(np.diff(arrays['trial0_grasp_robot_t']) > 0))
        self.assertEqual(len(arrays['trial0_grasp_force']), len(arrays['trial0_grasp_pose']))
        self.assert_cleaned()

    def test_overall_limit_stops_below_additional_pull_limit(self):
        self.r.getActualTCPForce = lambda: self.preloaded_force(extra=15.0)
        self.run_script(max_force=20.0, max_total_force=40.0)
        info, _ = self.saved()
        trial = info['trials'][0]
        self.assertEqual(trial['stop_reason'], 'max_total_force')
        self.assertAlmostEqual(trial['peak_force_change_N'], 15.0)
        self.assertAlmostEqual(trial['peak_total_force_change_N'], 41.0)
        self.assertFalse(trial['slip_candidate'])
        self.assertLess(trial['travel_at_stop_m'], 0.002)
        self.assert_cleaned()

    def test_additional_limit_still_stops_with_higher_overall_limit(self):
        self.r.getActualTCPForce = lambda: self.preloaded_force(extra=21.0)
        self.run_script(max_force=20.0, max_total_force=60.0)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'max_force')
        self.assertAlmostEqual(trial['peak_force_change_N'], 21.0)
        self.assertAlmostEqual(trial['peak_total_force_change_N'], 47.0)
        self.assertFalse(trial['slip_candidate'])
        self.assert_cleaned()

    def test_preloaded_release_uses_open_reference_for_slip(self):
        # A 26 -> 36 -> 0 N load releases the grasp. Closed-baseline subtraction
        # alone produces magnitudes 0 -> 10 -> 26 N and would hide the drop.
        self.r.getActualTCPForce = lambda: self.preloaded_force(extra=10.0, release=True)
        self.run_script(max_force=40.0, max_total_force=60.0)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'slip')
        self.assertTrue(trial['slip_candidate'])
        self.assertAlmostEqual(trial['peak_load_N'], 36.0)
        self.assertLess(trial['travel_at_stop_m'], 0.002)
        self.assert_cleaned()

    def test_default_overall_limit_does_not_admit_26_N_preload_at_20_N(self):
        self.r.getActualTCPForce = self.preloaded_force
        with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
            self.run_script(max_force=20.0)
        info, arrays = self.saved()
        self.assertEqual(info['max_total_force'], 20.0)
        self.assertEqual(len(self.c.moves), 3)
        self.assertNotIn('trial0_force', arrays)
        self.assertGreater(len(arrays['trial0_grasp_force']), 0)
        self.assert_cleaned()

    def test_overall_limit_stops_fingers_during_closing_and_keeps_trace(self):
        def force_during_closing():
            load = 41.0 if self.g.get_current_position() > 30 else 0.0
            return [0.0, -load, 0.0, 0.0, 0.0, 0.0]
        self.r.getActualTCPForce = force_during_closing
        with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
            self.run_script(max_force=20.0, max_total_force=40.0)
        info, arrays = self.saved()
        trial = info['trials'][0]
        self.assertEqual(len(self.c.moves), 3)
        self.assertTrue(self.g.stopped)
        self.assertEqual(trial['stop_reason'], 'max_total_force')
        self.assertEqual(trial['grasp_last_gripper_sample']['object_status'], 0)
        self.assertNotIn('grasp_position', trial)
        self.assertEqual(arrays['trial0_grasp_force'][-1, 1], -41.0)
        self.assertNotIn('trial0_force', arrays)
        self.assert_cleaned()

    def test_transient_overall_limit_during_settling_is_not_hidden_by_delay(self):
        def settling_force():
            if self.g.position != 130:
                return [0.0] * 6
            age = self.clock.t - self.g.started
            return [0.0, 41.0 if 0.25 <= age < 0.30 else 26.0, 0.0, 0.0, 0.0, 0.0]
        self.r.getActualTCPForce = settling_force
        with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
            self.run_script(max_force=20.0, max_total_force=40.0)
        info, arrays = self.saved()
        self.assertEqual(info['trials'][0]['grasp_position'], 130)
        self.assertEqual(info['trials'][0]['stop_reason'], 'max_total_force')
        self.assertEqual(arrays['trial0_grasp_force'][-1, 1], 41.0)
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_immediate_unloading_keeps_peak_but_does_not_confirm_slip(self):
        def immediate_release():
            if len(self.c.moves) == 4 and self.c.active:
                return [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
            return self.preloaded_force()
        self.r.getActualTCPForce = immediate_release
        self.run_script(max_force=40.0, max_total_force=60.0)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'max_travel')
        self.assertFalse(trial['slip_armed'])
        self.assertFalse(trial['slip_candidate'])
        self.assertAlmostEqual(trial['peak_load_N'], 26.0)
        self.assert_cleaned()

    def test_measured_preload_unloading_does_not_trigger_slip(self):
        # Representative axial loads from Thunder setup_preload, 2026-10-06.
        # The operator confirmed that the cube stayed between the pads.
        times = [0, .016, .040, .064, .080, .104, .128, .144, .168, .192, .208, .232, .264]
        loads = [23.146, 21.655, 18.899, 19.069, 18.427, 18.106, 16.773,
                 16.258, 14.958, 14.281, 13.442, 12.264, 10.616]
        def unloading_force():
            if len(self.c.moves) == 4 and self.c.active:
                load = np.interp(self.clock.t - self.c.started, times, loads)
                return [0.0, 5.0 - load, 0.0, 0.0, 0.0, 0.0]
            return self.preloaded_force()
        self.r.getActualTCPForce = unloading_force
        self.run_script(max_force=20.0, max_total_force=40.0, max_travel=0.005)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'max_travel')
        self.assertFalse(trial['slip_armed'])
        self.assertFalse(trial['slip_candidate'])
        self.assertAlmostEqual(trial['travel_at_stop_m'], 0.005)
        self.assert_cleaned()

    def test_pull_load_can_build_after_initial_unloading_then_drop(self):
        def unloading_then_loading_force():
            if len(self.c.moves) == 4 and self.c.active:
                elapsed = self.clock.t - self.c.started
                # Unload the initial contact, then build load on the opposite
                # side of zero. Keep the sensor sign instead of hiding it.
                axial = np.interp(elapsed, [0, .3, .5, .8, .9], [26, 0, -12, -12, -2])
                return [0.0, 5.0 - axial, 0.0, 0.0, 0.0, 0.0]
            return self.preloaded_force()
        self.r.getActualTCPForce = unloading_then_loading_force
        self.run_script(max_force=40.0, max_total_force=40.0)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'slip')
        self.assertTrue(trial['slip_armed'])
        self.assertTrue(trial['slip_candidate'])
        self.assertAlmostEqual(trial['slip_loading_peak_N'], 12.0)
        self.assertLess(trial['travel_at_stop_m'], 0.004)
        self.assert_cleaned()

    def test_short_load_rise_during_unloading_does_not_arm_slip(self):
        def noisy_unloading_force():
            if len(self.c.moves) == 4 and self.c.active:
                elapsed = self.clock.t - self.c.started
                load = 30.0 if elapsed < .016 else 12.0
                return [0.0, 5.0 - load, 0.0, 0.0, 0.0, 0.0]
            return self.preloaded_force()
        self.r.getActualTCPForce = noisy_unloading_force
        self.run_script(max_force=20.0, max_total_force=40.0, max_travel=0.005)
        trial = self.saved()[0]['trials'][0]
        self.assertEqual(trial['stop_reason'], 'max_travel')
        self.assertFalse(trial['slip_armed'])
        self.assertFalse(trial['slip_candidate'])
        self.assert_cleaned()

    def test_rejected_pull_is_error_not_max_travel(self):
        self.c.reject_index = 3
        with self.assertRaisesRegex(RuntimeError, 'rejected moveL'):
            self.run_script()
        info, _ = self.saved()
        self.assertEqual(info['trials'][0]['status'], 'error')
        self.assertIsNone(info['trials'][0]['stop_reason'])
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_invalid_baseline_refuses_to_pull(self):
        self.r.bad_baseline = True
        with self.assertRaisesRegex(ValueError, 'force'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 3)
        self.assertEqual(self.saved()[0]['run_status'], 'error')
        self.assertIsNone(self.saved()[0]['last_robot_sample']['force'][1])
        self.assert_cleaned()

    def test_invalid_force_during_pull_stops_and_preserves_trace(self):
        self.c.force_mode = 'nan'
        with self.assertRaisesRegex(ValueError, 'force'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assertEqual(self.saved()[0]['run_status'], 'error')
        self.assert_cleaned()

    def test_stale_state_stops_without_returning(self):
        self.c.freeze_timestamp = True
        with self.assertRaisesRegex(TimeoutError, 'fresh'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_disconnection_stops_without_returning(self):
        self.r.disconnect_pull = True
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_payload_change_stops_without_returning(self):
        self.c.payload_changed = True
        with self.assertRaisesRegex(RuntimeError, 'payload changed'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_interrupt_preserves_partial_trial(self):
        self.c.interrupt_pull = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_script()
        info, arrays = self.saved()
        self.assertEqual(info['trials'][0]['status'], 'aborted')
        self.assertGreater(len(arrays['trial0_force']), 0)
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_unconfirmed_open_never_returns_arm(self):
        original = self.g.move
        def block_after_pull(target, speed, force):
            self.g.block_open = target == 0 and len(self.c.moves) >= 4
            return original(target, speed, force)
        self.g.move = block_after_pull
        with self.assertRaises(TimeoutError):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_activation_failure_cleans_all_interfaces(self):
        self.g.activation_error = True
        with self.assertRaisesRegex(RuntimeError, 'activation'):
            self.run_script()
        self.assertFalse(self.c.moves)
        self.assert_cleaned()

    def test_receive_constructor_failure_stops_control(self):
        self.recv_factory.side_effect = RuntimeError('fake receive constructor failure')
        with self.assertRaisesRegex(RuntimeError, 'constructor'):
            self.run_script()
        self.assertTrue(self.c.script_stopped and self.c.disconnected)
        self.g_factory.assert_not_called()

    def test_interrupt_during_freedrive_leaves_freedrive(self):
        with self.mocks(answers=['', KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                pull.main(**dict(self.kwargs, freedrive=True))
        self.assertFalse(self.c.teaching)
        self.assert_cleaned()

    def test_rejected_freedrive_is_not_used(self):
        self.c.reject_teach = True
        with self.assertRaisesRegex(RuntimeError, 'freedrive'):
            self.run_script(freedrive=True)
        self.assertFalse(self.c.teaching)
        self.assert_cleaned()

    def test_rejected_zeroing_refuses_to_close_and_pull(self):
        self.c.reject_zero = True
        with self.assertRaisesRegex(RuntimeError, 'zeroing'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_protective_stop_refuses_all_motion(self):
        self.r.safety_mode = 3
        with self.assertRaisesRegex(RuntimeError, 'safety mode'):
            self.run_script()
        self.assertFalse(self.c.moves)
        self.assertTrue(self.c.script_stopped and self.r.disconnected)

    def test_cleanup_continues_after_stop_failure(self):
        self.c.stop_error = True
        self.g.activation_error = True
        with self.assertRaises(RuntimeError):
            self.run_script()
        self.assertTrue(self.saved()[0]['cleanup_errors'])
        self.assert_cleaned()

    def test_final_save_failure_still_cleans_every_interface(self):
        original = cal.Recording.save
        def fail_final(recording):
            if recording.info['run_status'] == 'complete':
                raise OSError('fake disk full')
            return original(recording)
        with patch.object(cal.Recording, 'save', fail_final):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.run_script()
        self.assert_cleaned()

    def test_unwritable_path_fails_before_connecting(self):
        blocker = Path(self.directory.name) / 'not-a-directory'
        blocker.write_text('file')
        with self.assertRaises(OSError):
            self.run_script(output=str(blocker / 'out.npz'))
        self.factory.assert_not_called()
        self.g_factory.assert_not_called()

    def test_existing_output_is_preserved_before_connecting(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b'previous calibration')
        with self.assertRaises(FileExistsError):
            self.run_script()
        self.assertEqual(self.output.read_bytes(), b'previous calibration')
        self.factory.assert_not_called()

    def test_invalid_arguments_fail_before_connecting(self):
        for name, values in {'max_travel': [-0.5, 0, 0.051, np.nan], 'pull_speed': [-1, 0, 0.011, np.inf],
                             'max_force': [-1, 0, 201, np.nan], 'max_total_force': [-1, 0, 201, np.nan, np.inf],
                             'direction': ['0,0,0', 'nan,1,0', '1,0', 'inf,0,0'],
                             'slip_drop': [-0.1, 0, 1, np.nan], 'force': [-1, 256], 'speed': [-1, 256],
                             'trials': [0, -1, 1.5], 'gripper_port': [0, 65536]}.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.run_script(**{name: value})
        self.factory.assert_not_called()
        self.g_factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_idle_async_status_never_counts_as_completion(self):
        self.c.never_start = True
        with self.assertRaisesRegex(TimeoutError, 'motion timed out'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 1)
        self.assert_cleaned()

    def test_early_async_completion_is_error(self):
        self.c.finish_short = True
        with self.assertRaisesRegex(RuntimeError, 'before reaching'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 1)
        self.assert_cleaned()

    def test_profile_preserves_close_if_opening_fails(self):
        original = self.g.move
        def block_open_after_close(target, speed, force):
            self.g.block_open = target == 0 and 255 in self.g.commands
            return original(target, speed, force)
        self.g.move = block_open_after_close
        with self.mocks(answers=['', '']):
            with self.assertRaises(TimeoutError):
                profile.main('offline-only', 63352, 128, 128, 1, 'cube60', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['cycle0_close']['status'], 'complete')
        self.assertGreater(len(arrays['cycle0_close_t']), 0)
        self.assertGreater(len(arrays['cycle0_open_t']), 0)
        self.assertTrue(self.g.stopped and self.g.disconnected)

    def test_preview_force_limit_aborts_before_grasping(self):
        self.r.preview_force = 101.0
        with self.assertRaisesRegex(RuntimeError, 'Force limit reached during preview'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 1)
        self.assertFalse(self.saved()[0]['trials'])
        self.assert_cleaned()

    def test_baseline_does_not_hide_existing_grasp_load(self):
        self.r.getActualTCPForce = lambda: [0.0, 150.0 if self.g.position == 130 else 0.0, 0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(RuntimeError, 'grasp baseline'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 3)
        self.assert_cleaned()

    def test_incoherent_states_time_out_before_any_motion(self):
        counter = [0]
        def timestamp():
            counter[0] += 1
            return counter[0] * 0.008
        self.r.getTimestamp = timestamp
        with self.assertRaisesRegex(TimeoutError, 'coherent'):
            self.run_script()
        self.assertFalse(self.c.moves)
        self.assertTrue(self.c.script_stopped and self.r.disconnected)

    def test_safety_limit_rejects_preview_target(self):
        self.c.safe_target = False
        with self.assertRaisesRegex(RuntimeError, 'controller safety limits'):
            self.run_script()
        self.assertFalse(self.c.moves)
        self.assert_cleaned()

    def test_protective_stop_during_pull_prevents_return(self):
        self.r.getSafetyMode = lambda: 3 if len(self.c.moves) >= 4 else 1
        with self.assertRaisesRegex(RuntimeError, 'safety mode'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_path_departure_stops_pull(self):
        original = self.r.getActualTCPPose
        def depart():
            pose = original()
            if len(self.c.moves) == 4 and self.c.active:
                pose[0] += 0.003
            return pose
        self.r.getActualTCPPose = depart
        with self.assertRaisesRegex(RuntimeError, 'departed'):
            self.run_script()
        self.assertEqual(len(self.c.moves), 4)
        self.assert_cleaned()

    def test_profile_empty_success_never_constructs_rtde(self):
        self.g.contact = False
        with self.mocks(answers=['']):
            profile.main('offline-only', 63352, 128, 128, 1, 'empty', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual(info['cycle0_close']['final_status'], 'AT_DEST')
        self.assertTrue(len(arrays['cycle0_open_t']))
        self.factory.assert_not_called()
        self.assertTrue(self.g.stopped and self.g.disconnected)

    def test_minimum_pendant_settings_are_valid_for_profile(self):
        self.g.contact = False
        with self.mocks(answers=['']):
            profile.main('offline-only', 63352, 0, 0, 1, 'empty', str(self.output))
        info, _ = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual((info['speed'], info['force']), (0, 0))
        self.factory.assert_not_called()

    def test_profile_and_pull_defaults_match_selected_deployment_settings(self):
        self.assertEqual((driver.DEFAULT_GRIPPER_SPEED, driver.DEFAULT_GRIPPER_FORCE), (0, 0))
        for script in (profile, pull):
            with self.subTest(script=script.__name__):
                args = script.parser().parse_args(['--label', 'cube60', '--output', str(self.output)])
                self.assertEqual((args.speed, args.force), (0, 0))
                if script is pull:
                    self.assertEqual(args.direction, '0,-1,0')

    def test_unflagged_cube_profile_confirms_each_cycle_and_keeps_raw_trace(self):
        self.g.contact = False
        with self.mocks(answers=['', '', 'y', '', 'y', '', 'y']):
            profile.main('offline-only', 63352, 0, 0, 3, 'cube60', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        for cycle in range(3):
            rec = info[f'cycle{cycle}_close']
            self.assertEqual(rec['final_status'], 'AT_DEST')
            self.assertFalse(rec['object_detected'])
            self.assertEqual(rec['grasp_confirmation'], 'operator')
            self.assertEqual(arrays[f'cycle{cycle}_close_position'][-1], 255)
            self.assertEqual(arrays[f'cycle{cycle}_open_position'][-1], 0)
        self.factory.assert_not_called()
        self.assertTrue(self.g.stopped and self.g.disconnected)

    def test_unflagged_cube_profile_decline_keeps_close_without_opening(self):
        self.g.contact = False
        with self.mocks(answers=['', '', 'n']):
            with self.assertRaisesRegex(SystemExit, 'Grasp not confirmed'):
                profile.main('offline-only', 63352, 0, 0, 1, 'cube60', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'aborted')
        self.assertEqual(info['cycle0_close']['status'], 'aborted')
        self.assertEqual(info['cycle0_close']['grasp_confirmation'], 'rejected')
        self.assertGreater(len(arrays['cycle0_close_t']), 0)
        self.assertNotIn('cycle0_open_t', arrays)
        self.assertEqual(self.g.commands[-1], 255)
        self.assertTrue(self.g.stopped and self.g.disconnected)

    def test_profile_save_failure_still_stops_and_disconnects(self):
        self.g.contact = False
        original = cal.Recording.save
        def fail_final(recording):
            if recording.info['run_status'] == 'complete':
                raise OSError('fake disk full')
            return original(recording)
        with self.mocks(answers=['']), patch.object(cal.Recording, 'save', fail_final):
            with self.assertRaisesRegex(OSError, 'disk full'):
                profile.main('offline-only', 63352, 128, 128, 1, 'empty', str(self.output))
        self.assertTrue(self.g.stopped and self.g.disconnected)

    def test_invalid_force_shape_is_saved_as_an_error(self):
        self.r.getActualTCPForce = lambda: np.array(np.nan)
        with self.assertRaisesRegex(ValueError, 'force'):
            self.run_script()
        self.assertEqual(self.saved()[0]['run_status'], 'error')
        self.assertIsNone(self.saved()[0]['last_robot_sample']['force'])
        self.assertTrue(self.c.script_stopped and self.r.disconnected)

    def test_success_with_cleanup_failure_is_not_reported_as_complete(self):
        self.c.stopScript = Mock(side_effect=RuntimeError('fake script-stop failure'))
        with self.assertRaisesRegex(RuntimeError, 'Cleanup failed'):
            self.run_script()
        info, _ = self.saved()
        self.assertEqual(info['run_status'], 'cleanup_error')
        self.assertTrue(self.c.disconnected and self.r.disconnected and self.g.disconnected)

    def test_profile_with_real_activation_records_physical_endpoint_sweep(self):
        self.g = PhysicalEndpointGripper(self.clock)
        self.g.activate = lambda **kwargs: driver.RobotiqGripper.activate(self.g, **kwargs)
        self.g_factory.return_value = self.g
        with self.mocks(answers=['']):
            profile.main('offline-only', 63352, 128, 128, 1, 'empty', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['run_status'], 'complete')
        self.assertEqual(info['open_position'], 3)
        self.assertEqual(info['closed_position'], 227)
        self.assertEqual(info['calibration']['status'], 'complete')
        for phase in ('preposition', 'open_start', 'close', 'open_finish'):
            self.assertGreater(len(arrays[f'calibration_{phase}_t']), 0)
        self.assertEqual(arrays['cycle0_open_position'][-1], 3)
        self.assertTrue(self.g.stopped and self.g.disconnected)
        self.factory.assert_not_called()

    def test_failed_calibration_retains_phase_and_register_samples(self):
        self.g = PhysicalEndpointGripper(self.clock)
        self.g.acknowledge = False
        self.g.activate = lambda **kwargs: driver.RobotiqGripper.activate(self.g, **kwargs)
        self.g_factory.return_value = self.g
        with self.mocks(answers=['']):
            with self.assertRaises(TimeoutError):
                profile.main('offline-only', 63352, 128, 128, 1, 'empty', str(self.output))
        info, arrays = self.saved()
        self.assertEqual(info['calibration']['status'], 'error')
        self.assertEqual(info['error']['gripper_state']['requested_position'], 17)
        self.assertGreater(len(arrays['calibration_preposition_t']), 0)
        self.assertTrue(self.g.stopped and self.g.disconnected)


if __name__ == '__main__':
    unittest.main()
