"""Mock-only test of the robot-state-locked torque loop (no robot, no sockets).

A fake host clock drives a mock robot that publishes a 2 ms-stamped state every `robot_period_s` host seconds
(values other than 0.002 model host/robot clock drift). The loop must consume every robot cycle exactly once, send one
torque command per cycle, and hand gripper moves to the gripper thread.
"""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

# rtde_control / rtde_receive are only needed on the robot workstation; stub them before importing the controller.
sys.modules.setdefault('rtde_control', types.SimpleNamespace(RTDEControlInterface=Mock()))
sys.modules.setdefault('rtde_receive', types.SimpleNamespace(RTDEReceiveInterface=Mock()))
sys.modules.setdefault('atomics', types.SimpleNamespace(atomicview=Mock(), MemoryOrder=Mock(), UINT=Mock()))  # shared-memory dep only
from diffusion_policy.real_world import rtde_interpolation_controller as ctrl  # noqa: E402
from diffusion_policy.shared_memory.shared_memory_queue import Empty  # noqa: E402


def make_controller():
    c = object.__new__(ctrl.RTDEInterpolationController)
    c.robot_ip, c.gripper_port, c.frequency = 'test-double-only', 0, 500
    c.joints_init, c.joints_init_speed, c.soft_real_time, c.verbose = None, 1.0, False, False
    c.tcp_offset = None
    c.torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    c.osc_error_delta_pos, c.osc_error_delta_rot = 0.05, 0.3
    stiffness = np.array([1000.0] * 3 + [50.0] * 3)
    c.osc_Kp, c.osc_Kd = np.diag(stiffness), np.diag(2 * np.sqrt(stiffness))
    c.receive_keys = ['ActualQ', 'ActualQd']
    c.ring_buffer, c.ready_event = Mock(), Mock()
    return c


class RobotLockedLoopTest(unittest.TestCase):
    def run_loop(self, robot_period_s, cycles=300, close_at=100):
        clock = [0.0]
        cycle = lambda: int(np.floor(clock[0] / robot_period_s + 1e-9))
        q0 = np.array([0.5, -1.4, -2.4, -2.0, -2.2, 0.08])

        def timestamp():
            clock[0] += 1e-5
            return 100.0 + cycle() * 0.002
        rtde_r = Mock()
        rtde_r.getTimestamp.side_effect = timestamp
        rtde_r.getActualQ.side_effect = lambda: (q0 + 1e-4 * cycle()).tolist()
        rtde_r.getActualQd.side_effect = lambda: [0.0] * 6
        rtde_c = Mock()
        rtde_c.directTorque.return_value = True
        gripper = Mock()
        gripper.get_closed_position.return_value, gripper.get_open_position.return_value = 255, 0
        c = make_controller()
        states = []
        c.ring_buffer.put.side_effect = lambda s: states.append(dict(s))
        calls = [0]

        def get_all():
            calls[0] += 1
            if calls[0] == close_at:
                return {'cmd': np.array([ctrl.Command.CartesianOSCControl.value]),
                        'target_ee_pos': np.zeros((1, 3)), 'target_ee_quat': np.array([[1.0, 0, 0, 0]]),
                        'close_gripper': np.array([True]), 'target_joints': np.zeros((1, 6))}
            if calls[0] >= cycles:
                return {'cmd': np.array([ctrl.Command.STOP.value])}
            raise Empty()
        c.input_queue = Mock(); c.input_queue.get_all.side_effect = get_all
        factory = Mock(return_value=rtde_c); factory.FLAG_VERBOSE, factory.FLAG_UPLOAD_SCRIPT = 1, 2
        with patch.object(ctrl, 'RTDEControlInterface', factory), \
             patch.object(ctrl, 'RTDEReceiveInterface', Mock(return_value=rtde_r)), \
             patch.object(ctrl, 'RobotiqGripper', Mock(return_value=gripper)), \
             patch.object(ctrl.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(ctrl.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0] + dt)), \
             patch('builtins.print'):
            c.run()
        return states, rtde_c, gripper

    def test_every_robot_cycle_once_under_drift(self):
        for period in (0.0019, 0.002, 0.0021):
            with self.subTest(robot_period_s=period):
                states, rtde_c, gripper = self.run_loop(period)
                t = np.array([s['robot_timestamp'] for s in states])
                self.assertEqual(len(states), 300)
                self.assertTrue(np.allclose(np.diff(t), 0.002))                 # no duplicate, no skipped cycle
                self.assertTrue(all(s['robot_cycle_gap'] == 1 for s in states))
                self.assertEqual(rtde_c.directTorque.call_count, 300 + 1)       # one per cycle, then zero torque
                rtde_c.initPeriod.assert_not_called(); rtde_c.waitPeriod.assert_not_called()
                gripper.move.assert_called_once_with(255, 128, 128)             # sent by the gripper thread


if __name__ == '__main__':
    unittest.main()
