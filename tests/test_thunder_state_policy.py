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
WRIST_DOWN = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])        # tool z -> base_link -y (gripper pointing down)
TABLE = -0.62353                                                       # table top height (base_link y is up)
TABLE_CENTRE = TABLE + 0.03                                            # centre height of a cube resting on it
CAMERA = np.eye(4)
CAMERA[:3, 3] = [-0.158, 0.122, -0.267]                                # Thunder's L515 position (base_link)


def cube(pos, R_=UP_Z_TO_BASE_Y):
    T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = pos
    return T


def estimator(detectors=None, **kw):
    return R.CubeEstimator(CAMERA, detectors or {}, [0, 1, 0], TABLE, **kw)


def along_ray(T, d):
    """T moved d metres away from the camera along its ray: how AprilCube's estimates err."""
    ray = (T[:3, 3] - CAMERA[:3, 3]) / np.linalg.norm(T[:3, 3] - CAMERA[:3, 3])
    T = T.copy(); T[:3, 3] += d * ray
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
        est = estimator()
        bottom, carried = cube([-0.5, TABLE_CENTRE, 0.0]), cube([-0.4, TABLE_CENTRE, -0.1])
        det = {'receptive': (bottom, 2, 0.3), 'insertive': (carried, 2, 0.3)}
        self.assertTrue(est.resting_flat(det))
        est.set_relabel({'receptive': (cube(bottom[:3, 3], UP_Z_TO_BASE_Y @ np.diag([-1.0, -1.0, 1.0])), 2, 0.3),
                         'insertive': det['insertive']})
        W0 = cube([-0.4, TABLE_CENTRE + 0.1935, -0.1], WRIST_DOWN)
        out, info = est.poses(det, 0.0, W0, W0, 0.0, holding=False)          # frame at t = 0.0, wrist at W0
        np.testing.assert_allclose(F.pos_quat_to_matrix(*out['insertive']), carried, atol=1e-12)
        self.assertTrue(info['insertive']['seen'])
        # the same result reused by later steps is applied once; grasped at t = 0.3 (wrist still at W0)
        out, info = est.poses(det, 0.0, W0, W0, 0.3, holding=True)
        self.assertFalse(info['insertive']['seen'])
        # lifted 10 cm by t = 0.5, not seen again: the table sighting is carried with the wrist from the grasp
        W5 = cube([-0.4, TABLE_CENTRE + 0.2935, -0.1], WRIST_DOWN)
        out, info = est.poses(det, 0.0, W0, W5, 0.5, holding=True)
        self.assertTrue(info['insertive']['carried_with_wrist'])
        np.testing.assert_allclose(out['insertive'][0], [-0.4, TABLE_CENTRE + 0.1, -0.1], atol=1e-12)
        np.testing.assert_allclose(out['receptive'][0], bottom[:3, 3], atol=1e-12)        # static bottom cube
        self.assertAlmostEqual(info['insertive']['age_s'], 0.5)
        # seen in hand on the frame at t = 0.55 (wrist at W5); by t = 0.6 the wrist moved 5 cm: brought to now
        held = cube([-0.4, TABLE_CENTRE + 0.1, -0.1])
        W6 = cube([-0.35, TABLE_CENTRE + 0.2935, -0.1], WRIST_DOWN)
        out, info = est.poses({'insertive': (held, 2, 0.3)}, 0.55, W5, W6, 0.6, holding=True)
        self.assertTrue(info['insertive']['seen'] and info['insertive']['carried_with_wrist'])
        np.testing.assert_allclose(out['insertive'][0], [-0.35, TABLE_CENTRE + 0.1, -0.1], atol=1e-12)
        # released 10 cm up, away from the bottom cube: it falls from where the hand let go (not from where it
        # was last measured, at t = 0.55) straight down onto the table
        out, info = est.poses({'insertive': (held, 2, 0.3)}, 0.55, W5, W6, 0.7, holding=False)
        self.assertFalse(info['insertive']['carried_with_wrist'])
        released = [-0.35, TABLE_CENTRE, -0.1]
        np.testing.assert_allclose(out['insertive'][0], released, atol=1e-12)
        # "holding" with the last sighting far from the wrist (e.g. a missed grasp): not carried
        far = cube([-0.9, -0.2, 0.3], WRIST_DOWN)
        out, info = est.poses({}, None, None, far, 0.8, holding=True)
        self.assertFalse(info['insertive']['carried_with_wrist'])
        np.testing.assert_allclose(out['insertive'][0], released, atol=1e-12)

    def test_cube_estimator_moves_detections_onto_their_support_and_relabels_a_turned_cube(self):
        est = estimator()
        rot_x = lambda deg: np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                                      [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
        bottom, carried = cube([-0.5, TABLE_CENTRE, 0.0]), cube([-0.4, TABLE_CENTRE, -0.1])
        det = {'receptive': (bottom, 2, 0.3), 'insertive': (carried, 2, 0.3)}
        est.set_relabel(det)
        W0 = cube([-0.4, TABLE_CENTRE + 0.193, -0.1], WRIST_DOWN)
        est.poses(det, 0.0, W0, W0, 0.0, holding=False)
        check = lambda d, t: est.poses(d, t, W0, W0, t, holding=False)
        # one-tag readings 5 cm off along the camera ray (dp_run5: the stacked cube hid the bottom cube's top tag):
        # moved back onto the table
        for i, d in enumerate((0.05, -0.05)):
            out, info = check({'receptive': (along_ray(bottom, d), 1, 0.3)}, 0.1 + 0.01 * i)
            self.assertEqual(info['receptive']['rejected'], '')
            np.testing.assert_allclose(out['receptive'][0], bottom[:3, 3], atol=1e-9)
        # read 3 deg tilted (a flat cube read 2.7-2.9 deg tilted in dp_run5): made flat, yaw kept
        out, info = check({'receptive': (cube(bottom[:3, 3], rot_x(3) @ UP_Z_TO_BASE_Y), 2, 0.3)}, 0.12)
        np.testing.assert_allclose(F.pos_quat_to_matrix(*out['receptive']), bottom @ est.relabel['receptive'], atol=1e-9)
        # 15 cm off along the ray, or tilted 30 deg: not used
        out, info = check({'receptive': (along_ray(bottom, 0.15), 1, 0.3)}, 0.2)
        self.assertEqual(info['receptive']['rejected'], 'bottom cube not on the table')
        out, info = check({'receptive': (cube(bottom[:3, 3], rot_x(30) @ UP_Z_TO_BASE_Y), 1, 0.3)}, 0.25)
        self.assertEqual(info['receptive']['rejected'], 'bottom cube not resting flat')
        np.testing.assert_allclose(out['receptive'][0], bottom[:3, 3], atol=1e-9)
        # the carried cube stacked, read 4 cm off along the ray either way: moved onto the bottom cube's top (on the
        # table along that ray it would be inside the bottom cube)
        on_top = cube([-0.49, TABLE_CENTRE + 0.06, 0.005])
        for i, d in enumerate((0.04, -0.04)):
            out, info = check({'insertive': (along_ray(on_top, d), 1, 0.3)}, 0.3 + 0.01 * i)
            self.assertEqual(info['insertive']['rejected'], '')
            np.testing.assert_allclose(out['insertive'][0], on_top[:3, 3], atol=1e-9)
        # 20 cm up beside the bottom cube, beyond MAX_RAY_SLIDE_M from any support: not used
        out, info = check({'insertive': (cube([-0.3, TABLE_CENTRE + 0.2, 0.1]), 1, 0.3)}, 0.35)
        self.assertEqual(info['insertive']['rejected'], 'carried cube not on the table or the bottom cube')
        np.testing.assert_allclose(out['insertive'][0], on_top[:3, 3], atol=1e-9)
        # leaning 30 deg with an edge on the table: used, and not relabelled (it does need turning over)
        leaning = cube([-0.3, TABLE + 0.03 * (np.cos(np.radians(30)) + np.sin(np.radians(30))), 0.1],
                       rot_x(30) @ UP_Z_TO_BASE_Y)
        out, info = check({'insertive': (leaning, 2, 0.3)}, 0.4)
        self.assertEqual(info['insertive']['rejected'], '')
        np.testing.assert_allclose(out['insertive'][0], leaning[:3, 3], atol=1e-9)
        self.assertEqual(est.relabel_events, [])
        # lying flat on another face: used and relabelled, so the policy sees +Z up again
        turned = cube([-0.3, TABLE_CENTRE, 0.1], rot_x(90) @ UP_Z_TO_BASE_Y)
        out, info = check({'insertive': (turned, 2, 0.3)}, 0.5)
        self.assertEqual(len(est.relabel_events), 1)
        self.assertLess(np.degrees(np.arccos(F.pos_quat_to_matrix(*out['insertive'])[1, 2])), 1e-6)

    def test_cube_estimator_released_cube_falls_onto_its_support(self):
        est = estimator()
        bottom = cube([-0.5, TABLE_CENTRE, 0.0])
        det = {'receptive': (bottom, 2, 0.3), 'insertive': (cube([-0.4, TABLE_CENTRE, -0.1]), 2, 0.3)}
        est.set_relabel(det)
        W0 = cube([-0.4, TABLE_CENTRE + 0.1935, -0.1], WRIST_DOWN)
        est.poses(det, 0.0, W0, W0, 0.0, holding=False)
        est.poses({}, None, None, W0, 0.1, holding=True)                           # grasped at t = 0.1
        for hand_over, t, expected in (([-0.495, 0.08, 0.005], 0.2, [-0.495, TABLE_CENTRE + 0.06, 0.005]),   # on top
                                       ([-0.5, 0.05, 0.05], 0.4, [-0.5, TABLE_CENTRE, 0.05])):              # beside it
            # carried to the hand position (cube centre hand_over[1] above the table), let go
            W = cube([hand_over[0], TABLE_CENTRE + hand_over[1] + 0.1935, hand_over[2]], WRIST_DOWN)
            est.poses({}, None, None, W, t, holding=True)
            out, info = est.poses({}, None, None, W, t + 0.05, holding=False)
            np.testing.assert_allclose(out['insertive'][0], expected, atol=1e-9)
            est.poses({}, None, None, W, t + 0.1, holding=True)                    # grasped again where it landed
        # let go from a hand tilted 6 deg beside the bottom cube: lands flat on the table below the hand
        rot_x = lambda deg: np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                                      [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
        W = cube([-0.4, TABLE_CENTRE + 0.25, 0.05], rot_x(6) @ WRIST_DOWN)
        held, _ = est.poses({}, None, None, W, 0.7, holding=True)
        self.assertGreater(R.CS.resting_tilt_deg(F.pos_quat_to_matrix(*held['insertive'])[:3, :3], [0, 1, 0]), 5.9)
        out, _ = est.poses({}, None, None, W, 0.75, holding=False)
        T = F.pos_quat_to_matrix(*out['insertive'])
        self.assertLess(R.CS.resting_tilt_deg(T[:3, :3], [0, 1, 0]), 1e-6)
        np.testing.assert_allclose(T[:3, 3], [held['insertive'][0][0], TABLE_CENTRE, held['insertive'][0][2]], atol=1e-9)
        # stack_offset: only a camera reading counts, not the hand's estimate of where the cube landed
        self.assertIsNone(est.stack_offset())
        seen_on_top = cube([-0.495, TABLE_CENTRE + 0.06, 0.006])
        est.poses({'insertive': (along_ray(seen_on_top, 0.03), 1, 0.3)}, 0.8, W, W, 0.8, holding=False)
        self.assertAlmostEqual(est.stack_offset(), np.hypot(0.005, 0.006), places=9)
        on_table = cube([-0.4, TABLE_CENTRE, 0.0])
        est.poses({'insertive': (on_table, 2, 0.3)}, 0.85, W, W, 0.85, holding=False)
        self.assertIsNone(est.stack_offset())

    def test_cube_estimator_tag_face_up_keeps_the_carried_cubes_tag_frame(self):
        rot_x = lambda deg: np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                                      [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
        table_centre = -0.62353 + 0.03
        bottom = cube([-0.5, table_centre, 0.0], UP_Z_TO_BASE_Y @ np.diag([-1.0, -1.0, 1.0]))   # tag +Z up
        upside_down = cube([-0.4, table_centre, -0.1], rot_x(180) @ UP_Z_TO_BASE_Y)            # tag +Z down
        det = {'receptive': (bottom, 2, 0.3), 'insertive': (upside_down, 2, 0.3)}
        est = estimator(any_face_up=False)
        est.set_relabel(det)
        W0 = cube([-0.4, table_centre + 0.193, -0.1], WRIST_DOWN)
        out, _ = est.poses(det, 0.0, W0, W0, 0.0, holding=False)
        np.testing.assert_allclose(est.relabel['insertive'], np.eye(4))
        self.assertAlmostEqual(F.pos_quat_to_matrix(*out['insertive'])[1, 2], -1.0)            # policy sees +Z down
        # flat on yet another face later: still the tag frame, no relabel
        turned = cube([-0.3, table_centre, 0.1], rot_x(90) @ UP_Z_TO_BASE_Y)
        out, _ = est.poses({'insertive': (turned, 2, 0.3)}, 0.1, W0, W0, 0.1, holding=False)
        self.assertEqual(est.relabel_events, [])
        np.testing.assert_allclose(F.pos_quat_to_matrix(*out['insertive'])[:3, :3], turned[:3, :3], atol=1e-12)
        # stacked with the tag +Z face sideways: not training's goal, so no stack; with it up: stacked
        self.assertAlmostEqual(est.z_from_up_deg('insertive'), 90.0, places=6)
        sideways = cube([-0.497, TABLE_CENTRE + 0.06, 0.004], rot_x(90) @ UP_Z_TO_BASE_Y)
        est.poses({'insertive': (sideways, 2, 0.3)}, 0.2, W0, W0, 0.2, holding=False)
        self.assertIsNone(est.stack_offset())
        upright = cube([-0.497, TABLE_CENTRE + 0.06, 0.004])
        est.poses({'insertive': (upright, 2, 0.3)}, 0.3, W0, W0, 0.3, holding=False)
        self.assertAlmostEqual(est.stack_offset(), 0.005, places=9)
        # the default relabels the same start so the face on top is +Z
        est_any = estimator()
        est_any.set_relabel(det)
        out, _ = est_any.poses(det, 0.0, W0, W0, 0.0, holding=False)
        self.assertAlmostEqual(F.pos_quat_to_matrix(*out['insertive'])[1, 2], 1.0)

    def test_grasp_hold(self):
        pos, quat = np.array([-0.4, -0.4, -0.1]), np.array([0.0, 1.0, 0.0, 0.0])
        moved = pos + [0.0, 0.05, 0.0]
        h = R.GraspHold(1.0, 226, closed=False)
        self.assertIsNone(h.update(False, 3, 3, 0.0, pos, quat))                  # open: follow the policy
        held = h.update(True, 3, 3, 0.1, pos, quat)                               # close commanded: hold this pose
        np.testing.assert_allclose(held[0], pos)
        np.testing.assert_allclose(h.update(True, 60, 0, 0.3, moved, quat)[0], pos)   # still closing: keep holding
        self.assertIsNone(h.update(True, 92, 2, 0.5, moved, quat))                # contact: release to the policy
        self.assertEqual(h.ended, 'contact')
        self.assertIsNone(h.update(True, 92, 2, 0.6, moved, quat))                # still closed: no new hold
        for status, position, t_end, reason in ((3, 226, 0.9, 'closed empty'), (0, 120, 1.2, 'timeout')):
            h = R.GraspHold(1.0, 226, closed=False)
            h.update(True, 3, 3, 0.0, pos, quat)
            self.assertIsNone(h.update(True, position, status, t_end, moved, quat))
            self.assertEqual(h.ended, reason)
        h = R.GraspHold(1.0, 226, closed=False)
        h.update(True, 3, 3, 0.0, pos, quat)
        self.assertIsNone(h.update(False, 40, 0, 0.2, moved, quat))               # the policy reopened
        self.assertEqual(h.ended, 'policy reopened')
        self.assertIsNone(R.GraspHold(0.0, 226, closed=False).update(True, 3, 3, 0.0, pos, quat))   # disabled
        self.assertIsNone(R.GraspHold(1.0, 226, closed=True).update(True, 226, 3, 0.0, pos, quat))  # start closed

    def test_cube_estimator_held_detection_must_be_between_the_fingers(self):
        est = estimator()
        table_centre = -0.62353 + 0.03
        carried = cube([-0.4, table_centre, -0.1])
        det = {'receptive': (cube([-0.5, table_centre, 0.0]), 2, 0.3), 'insertive': (carried, 2, 0.3)}
        est.set_relabel(det)
        W0 = cube([-0.4, table_centre + 0.193, -0.1], WRIST_DOWN)
        est.poses(det, 0.0, W0, W0, 0.0, holding=False)
        est.poses(det, 0.0, W0, W0, 0.3, holding=True)                            # grasped at t = 0.3
        W5 = cube([-0.4, table_centre + 0.293, -0.1], WRIST_DOWN)                  # lifted 10 cm
        lifted = [-0.4, table_centre + 0.1, -0.1]
        out, info = est.poses({'insertive': (cube([-0.32, table_centre + 0.1, -0.1]), 1, 0.3)}, 0.45, W5, W5, 0.5, True)
        self.assertTrue(info['insertive']['rejected'])                            # 8 cm beside the fingers
        np.testing.assert_allclose(out['insertive'][0], lifted, atol=1e-12)
        close = cube([-0.405, table_centre + 0.1, -0.1])
        out, info = est.poses({'insertive': (close, 2, 0.3)}, 0.55, W5, W5, 0.6, True)
        self.assertTrue(info['insertive']['seen'])                                # between the fingers, agrees: used
        np.testing.assert_allclose(out['insertive'][0], close[:3, 3], atol=1e-12)
        # between the fingers but 25 mm lower along the tool axis than the grasp carried it: not used
        lower = cube([-0.405, table_centre + 0.075, -0.1])
        out, info = est.poses({'insertive': (lower, 1, 0.3)}, 0.65, W5, W5, 0.7, True)
        self.assertEqual(info['insertive']['rejected'], 'held cube detection disagrees with the grasp')
        np.testing.assert_allclose(out['insertive'][0], close[:3, 3], atol=1e-12)
        # a frame from before the grasp, applied late: judged as a free cube, so its 4 cm error along the camera ray
        # (~35 mm low, as on October 9) is moved back onto the table, and that sighting is carried from the grasp
        est2 = estimator()
        est2.set_relabel(det)
        est2.poses(det, 0.0, W0, W0, 0.0, holding=False)
        est2.poses({}, None, None, W0, 0.3, holding=True)                         # grasped at t = 0.3
        sunk = along_ray(carried, 0.04)                                           # frame at t = 0.2
        out, info = est2.poses({'insertive': (sunk, 1, 0.3)}, 0.2, W0, W5, 0.9, True)
        self.assertEqual(info['insertive']['rejected'], '')
        np.testing.assert_allclose(out['insertive'][0], lifted, atol=1e-9)

    def test_cube_estimator_skips_the_held_cube_and_restarts_its_tracking(self):
        class FakeDetector:                                                       # AprilCube's per-frame tracking state
            def __init__(self):
                self.calls, self.filter_reset = 0, False
                self.prev_rvec, self.prev_tvec = np.zeros(3), np.zeros(3)
                self._prev_gray = self._prev_corners_2d = self._prev_corners_3d = np.zeros(1)
                self.pose_filter = SimpleNamespace(reset=lambda: setattr(self, 'filter_reset', True))

            def process_frame(self, image, timestamp):
                self.calls += 1
                return dict(success=True, T=np.eye(4), n_tags=2, reproj_error=0.3, predicted=False)

        dets = {'receptive': FakeDetector(), 'insertive': FakeDetector()}
        est = estimator(dets)
        image, raws = np.zeros((4, 4, 3), np.uint8), []
        self.assertEqual(set(est.detect(image, 0.0)), {'receptive', 'insertive'})
        raws.append(est.last_raw)
        self.assertEqual(set(est.detect(image, 0.1, skip=('insertive',))), {'receptive'})     # held: not searched
        raws.append(est.last_raw)
        self.assertEqual((dets['receptive'].calls, dets['insertive'].calls), (2, 1))
        self.assertIsNotNone(dets['insertive'].prev_rvec)
        self.assertEqual(set(est.detect(image, 0.2)), {'receptive', 'insertive'})             # released: from scratch
        raws.append(est.last_raw)
        held = dets['insertive']
        self.assertEqual(held.calls, 2)
        self.assertTrue(held.prev_rvec is None and held._prev_gray is None and held.filter_reset)
        self.assertFalse(dets['receptive'].filter_reset)
        logged = R.frame_log_arrays([dict(frame_time=t, robot_time=t, q=np.zeros(6), detect_s=0.05, raw=raw)
                                     for t, raw in zip((0.0, 0.1, 0.2), raws)])
        self.assertEqual(logged['frame_insertive_skipped'].tolist(), [False, True, False])
        self.assertEqual(logged['frame_insertive_success'].tolist(), [True, False, True])
        self.assertFalse(logged['frame_receptive_skipped'].any())


if __name__ == '__main__':
    unittest.main()
