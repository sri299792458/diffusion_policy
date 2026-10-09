"""CPU checks for the controller-rate log (thunder_state_policy/high_rate_log.py) and its use in the OSC control loop (no
robot or gripper: RTDE, the Robotiq and the shared-memory queue are replaced by fakes)."""
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault('rtde_control', SimpleNamespace(RTDEControlInterface=object))
sys.modules.setdefault('rtde_receive', SimpleNamespace(RTDEReceiveInterface=object))
try:                                           # shared-memory dependency of the controller module (import only)
    import atomics  # noqa: F401
except ImportError:
    sys.modules['atomics'] = SimpleNamespace(atomicview=None, MemoryOrder=None, UINT=None)
from diffusion_policy.real_world import rtde_interpolation_controller as C  # noqa: E402
from diffusion_policy.real_world.thunder_state_policy import high_rate_log as H  # noqa: E402
from diffusion_policy.real_world.thunder_state_policy import r214 as R  # noqa: E402


class HighRateLogTests(unittest.TestCase):
    def test_rows_survive_chunking_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            log = H.HighRateLog(d, meta=dict(note='test'), chunk_rows=7, spare_blocks=2)
            for i in range(25):
                log.append(t_host=float(i), q=np.full(6, i), torque=np.arange(6) + i, target_seq=i // 5)
            self.assertEqual(log.close(), [])
            data, meta = H.load(d)
            self.assertEqual(meta['note'], 'test')
            self.assertEqual(len(data['t_host']), 25)
            np.testing.assert_array_equal(data['t_host'], np.arange(25))
            np.testing.assert_array_equal(data['q'][:, 3], np.arange(25))
            np.testing.assert_array_equal(data['torque'][24], np.arange(6) + 24)
            np.testing.assert_array_equal(data['target_seq'], np.arange(25) // 5)
            self.assertTrue(np.isnan(data['current']).all())           # never given: NaN, not stale numbers
            self.assertEqual(len(list(__import__('pathlib').Path(d).glob('chunk_*.npy'))), 4)


class FakeQueue:
    """Controller input queue: commands become visible at given loop counts; STOP after the last."""
    def __init__(self, schedule):
        self.schedule = schedule          # {loop: command dict}
        self.loop = 0

    def get_all(self):
        cmd = self.schedule.get(self.loop)
        if cmd is None:
            raise C.Empty()
        return {k: np.array([v]) for k, v in cmd.items()}


class FakeControl:
    def __init__(self, *args, **kwargs):
        self.sent = []

    def initPeriod(self):
        return 0.0

    def directTorque(self, torque, **kwargs):
        self.sent.append(np.array(torque))
        return True

    def waitPeriod(self, t):
        FAKE['queue'].loop += 1

    def setPayload(self, *a):
        pass

    def servoJ(self, *a):
        pass

    servoStop = stopScript = disconnect = lambda self, *a: None


class FakeReceive:
    def __init__(self, *args, **kwargs):
        self.t = 0.0

    def getActualQ(self):
        return list(R.START_JOINTS)

    def getActualQd(self):
        return [0.01] * 6

    def getTimestamp(self):
        self.t += 0.002
        return self.t

    def getActualCurrent(self):
        return [0.5, 1.0, 1.5, 0.1, 0.2, 0.3]

    def getActualTCPForce(self):
        return [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]

    def disconnect(self):
        pass


class FakeGripper:
    def connect(self, *a):
        pass

    def activate(self):
        pass

    def move(self, *a):
        pass

    def get_closed_position(self):
        return 227

    def get_open_position(self):
        return 3


FAKE = {}


def command(cmd, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), close=False):
    return dict(cmd=cmd.value, target_joints=np.zeros(6), target_ee_pos=np.array(pos), target_ee_quat=np.array(quat),
                close_gripper=close)


class ControllerLoopLogTests(unittest.TestCase):
    def test_every_loop_is_logged_with_the_torque_sent_and_its_target(self):
        ctrl = C.RTDEInterpolationController.__new__(C.RTDEInterpolationController)
        stiffness = np.array([1000.0] * 3 + [50.0] * 3)
        with tempfile.TemporaryDirectory() as d:
            ctrl.__dict__.update(
                robot_ip='fake', gripper_port=0, frequency=500, soft_real_time=False, kinematics_calibration=None,
                payload_mass=1.04, payload_cog=[0, 0, 0.05], high_rate_log_dir=d, osc_Kp=np.diag(stiffness),
                osc_Kd=np.diag(2 * np.sqrt(stiffness)), osc_error_delta_pos=0.05, osc_error_delta_rot=0.3,
                torque_max=np.array([150.0] * 3 + [28.0] * 3), gripper_speed=255, gripper_force=0, verbose=False,
                joints_init=None, joints_init_speed=1.0, read_gripper_state=False, receive_keys=['ActualTCPForce'],
                ring_buffer=SimpleNamespace(put=lambda state: None), ready_event=threading.Event())
            pos, _ = C.get_ee_pose(np.array(R.START_JOINTS))
            target = pos + np.array([0.0, 0.03, 0.0])               # 3 cm up from loop 6
            FAKE['queue'] = ctrl.input_queue = FakeQueue({
                5: command(C.Command.CartesianOSCControl, target, C.get_ee_pose(np.array(R.START_JOINTS))[1]),
                40: command(C.Command.STOP)})
            control = FakeControl()

            def make_control(*a, **k):
                return control
            make_control.FLAG_VERBOSE, make_control.FLAG_UPLOAD_SCRIPT = 1, 2
            with patch.object(C, 'RTDEControlInterface', make_control), \
                    patch.object(C, 'RTDEReceiveInterface', FakeReceive), patch.object(C, 'RobotiqGripper', FakeGripper):
                ctrl.run()
            data, meta = H.load(d)
        n = len(control.sent) - 1                                    # the last send is the zero torque at shutdown
        self.assertEqual(len(data['t_host']), n)
        self.assertEqual(n, 41)
        np.testing.assert_allclose(data['torque'], np.array(control.sent[:n]))
        np.testing.assert_allclose(data['q'], np.tile(R.START_JOINTS, (n, 1)))
        np.testing.assert_allclose(np.diff(data['t_robot']), 0.002)  # one robot timestamp per loop
        self.assertEqual(int(data['target_seq'][5]), 0)              # loop 5 still uses the initial target
        self.assertEqual(int(data['target_seq'][6]), 1)              # the command read at loop 5 acts from loop 6
        np.testing.assert_allclose(data['target_pos'][6], target)
        np.testing.assert_allclose(data['current'][0], [0.5, 1.0, 1.5, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(data['tcp_force'][0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        self.assertGreater(np.linalg.norm(data['torque'][6]), np.linalg.norm(data['torque'][5]))   # target moved
        self.assertEqual(meta['frequency'], 500)


if __name__ == '__main__':
    unittest.main()
