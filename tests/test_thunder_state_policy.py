"""CPU checks for the Thunder R214 state-policy additions (no robot, camera or gripper).

Fixture tests/thunder_r214_contract_fixture.npz: 4 episodes (one per reset family, first 40 policy steps) recorded in R214's
native training env (UWLab R226): poses in base_link, the env's actor observation and the actor output.
"""
import importlib
from pathlib import Path
import sys
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
from diffusion_policy.real_world import ur5e_kinematics as kin  # noqa: E402
from diffusion_policy.real_world.thunder_state_policy import frames as F  # noqa: E402
from diffusion_policy.real_world.thunder_state_policy import obs as O  # noqa: E402
from diffusion_policy.real_world.thunder_state_policy import r214 as R  # noqa: E402

FIX = np.load(Path(__file__).resolve().parent / 'thunder_r214_contract_fixture.npz')
UP_Z_TO_BASE_Y = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # cube +Z -> base_link +y (world up on Thunder)


def cube(pos, R_=UP_Z_TO_BASE_Y):
    T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = pos
    return T


class ControllerAdditionsTests(unittest.TestCase):
    def test_direct_torque_compensation_off_for_every_ur_rtde_version(self):
        self.assertEqual(C.direct_torque_kwargs((1, 6, 2)), {'friction_comp': False})      # upstream's pinned version
        self.assertEqual(C.direct_torque_kwargs((1, 6, 3)), {'friction_comp': False})
        self.assertEqual(C.direct_torque_kwargs((1, 6, 5)), {'viscous_scale': [0.0] * 6, 'coulomb_scale': [0.0] * 6})

    def test_thunder_calibration_matches_sim_wrist_pose(self):
        saved = (kin.CALIBRATED_JOINTS, kin.LINK_INERTIAS)
        try:
            C.install_kinematics_calibration(R.load_calibration())
            for e in range(4):
                pos, quat = kin.get_ee_pose(FIX['joint_pos'][e, 0, :6])
                np.testing.assert_allclose(pos, FIX['wrist_pos'][e, 0], atol=1e-5)
                self.assertLess(1 - abs(float(quat @ FIX['wrist_quat'][e, 0])), 1e-6)
        finally:
            kin.CALIBRATED_JOINTS, kin.LINK_INERTIAS = saved

    def test_upstream_target_rule_matches_training(self):
        pos, quat = FIX['wrist_pos'][0, 0], FIX['wrist_quat'][0, 0]
        scaled = np.array([1.5, -2.0, 0.5, 3.0, -0.7, 0.2]) * np.array([0.02, 0.02, 0.02, 0.02, 0.2, 0.02])
        tp, tq = kin.apply_delta_pose(pos, quat, scaled)
        angle = np.linalg.norm(scaled[3:])
        delta = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * scaled[3:] / angle])   # RelCartesianOSCAction
        np.testing.assert_allclose(tp, pos + scaled[:3])
        np.testing.assert_allclose(tq, O.quat_mul(delta, quat), atol=1e-9)


class StatePolicyTests(unittest.TestCase):
    def test_observation_and_policy_reproduce_training(self):
        act = R.load_policy(R.load_manifest())
        for e in range(4):
            h = O.ObservationHistory()
            for t in range(FIX['obs'].shape[1]):
                terms = O.frame_terms(FIX['last_action'][e, t], FIX['joint_pos'][e, t], FIX['wrist_pos'][e, t],
                                      FIX['wrist_quat'][e, t], FIX['ins_pos'][e, t], FIX['ins_quat'][e, t],
                                      FIX['rec_pos'][e, t], FIX['rec_quat'][e, t])
                obs = h.push(terms)
                np.testing.assert_allclose(obs, FIX['obs'][e, t], atol=2e-6)
                np.testing.assert_allclose(act(obs), FIX['action'][e, t], atol=2e-4)

    def test_gripper_map(self):
        g = R.GripperMap(R.load_manifest())
        np.testing.assert_allclose(g.sim_joints(3), np.zeros(6), atol=0.01)
        self.assertAlmostEqual(g.sim_joints(92)[0], 0.2884)
        self.assertAlmostEqual(g.sim_joints(226)[0], 0.8194)
        held = FIX['joint_pos'][2, 0, 6:]                                        # Grasped episode: cube in hand
        np.testing.assert_allclose(g.sim_joints(np.interp(held[0], g.finger, g.real)), held, atol=0.02)
        self.assertTrue(g.holding(92, 2, True))
        self.assertFalse(g.holding(92, 3, True))                                  # not stopped on contact
        self.assertFalse(g.holding(226, 2, True))                                 # closed empty
        self.assertFalse(g.holding(92, 2, False))                                 # commanded open

    def test_cube_estimator_relabel_static_and_carried(self):
        est = R.CubeEstimator(np.eye(4), {}, [0, 1, 0])
        bottom, carried = cube([-0.5, -0.5935, 0.0]), cube([-0.4, -0.5935, -0.1])
        det = {'receptive': (bottom, 2, 0.3), 'insertive': (carried, 2, 0.3)}
        self.assertTrue(est.resting_flat(det))
        est.set_relabel({'receptive': (cube(bottom[:3, 3], UP_Z_TO_BASE_Y @ np.diag([-1.0, -1.0, 1.0])), 2, 0.3),
                         'insertive': det['insertive']})
        W0 = cube([-0.4, -0.40, -0.1], np.eye(3))
        out, info = est.poses(det, W0, 0, holding=False)
        np.testing.assert_allclose(F.pos_quat_to_matrix(*out['insertive']), carried, atol=1e-12)
        # last seen at step 0 on the table; grasped at step 3 with the wrist at W3; lifted 10 cm by step 5, unseen
        W3 = cube([-0.4, -0.40, -0.1], np.eye(3))
        est.poses({}, W3, 3, holding=True)
        W5 = cube([-0.4, -0.30, -0.1], np.eye(3))
        out, info = est.poses({}, W5, 5, holding=True)
        self.assertTrue(info['insertive']['carried_with_wrist'])
        np.testing.assert_allclose(out['insertive'][0], [-0.4, -0.4935, -0.1], atol=1e-12)
        np.testing.assert_allclose(out['receptive'][0], bottom[:3, 3], atol=1e-12)        # static bottom cube
        # released: the carried cube keeps its last measured pose (on the table)
        out, info = est.poses({}, W5, 6, holding=False)
        self.assertFalse(info['insertive']['carried_with_wrist'])
        np.testing.assert_allclose(out['insertive'][0], carried[:3, 3], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
