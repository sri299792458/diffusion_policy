"""Pull-out test: how much force the real gripper (UMI fingertips) can hold before a grasped cube slips.

No load cell needed: the UR5e's built-in wrist force/torque sensor measures the pull. The cube is FIXED to the table
(clamped, or taped into a stop block), the gripper grasps it with the deployment settings (speed 128, force 128 by default),
then the arm moves very slowly along --direction while the wrist force is logged. The force rises until the cube slips
in the fingers (force drops) or a safety limit is reached. The same test is then reproduced in simulation to calibrate the
simulated gripper (one drive for every task, matched to how the deployment code commands the real gripper).

Safety (read before running):
- Keep the e-stop in reach. The arm only moves in short, slow straight lines (default 2 mm/s, at most --max_travel).
- The pull stops when |force change| exceeds --max_force (default 100 N), when slip is detected, or at --max_travel.
- The robot's own protective stop still applies. Make sure the cube fixture can take more than --max_force.
- Before grasping, the script previews the pull direction with the gripper OPEN (moves 5 mm along it and back) and asks
  you to confirm. On Thunder (sideways mount) world-up is the robot BASE +y, so the default --direction is 0,1,0.
- At force 128 the hold may exceed --max_force without slipping; that trial is recorded as "no slip up to X N". Lower force
  settings (e.g. --force 0 and --force 32) usually slip within the limit and give the force-setting -> holding-force trend.

Usage (diffusion_policy repo root, robodiff_real env):
    python scripts/sim2real/gripper_pullout_test.py --robot_ip 192.168.1.10 --label cube60 --force 128 --trials 3 \
        --output data/pullout_cube60_f128.npz
"""
import json
import time

import click
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper
from diffusion_policy.real_world.ur5e_kinematics import PAYLOAD_MASS, PAYLOAD_COG

FREQUENCY = 125


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10')
@click.option('--gripper_port', default=63352, type=int)
@click.option('--label', required=True, help='e.g. cube40 or cube60 (recorded in the output)')
@click.option('--speed', default=128, type=int, help='Robotiq speed 0-255 (deployment: 128)')
@click.option('--force', default=128, type=int, help='Robotiq force 0-255 (deployment: 128)')
@click.option('--trials', default=3, type=int)
@click.option('--direction', default='0,1,0', help='Pull direction in the robot BASE frame (Thunder: 0,1,0 = world up)')
@click.option('--pull_speed', default=0.002, type=float, help='m/s')
@click.option('--max_travel', default=0.02, type=float, help='m, per trial')
@click.option('--max_force', default=100.0, type=float, help='N, stop if the force change exceeds this')
@click.option('--slip_min', default=3.0, type=float, help='N, peak needed before a drop counts as slip')
@click.option('--slip_drop', default=0.4, type=float, help='slip = force falls below (1 - slip_drop) x peak')
@click.option('--freedrive/--no-freedrive', default=True, help='Enable freedrive while you position the open gripper')
@click.option('--output', '-o', required=True)
def main(robot_ip, gripper_port, label, speed, force, trials, direction, pull_speed, max_travel, max_force,
         slip_min, slip_drop, freedrive, output):
    d = np.array([float(x) for x in direction.split(',')]); d = d / np.linalg.norm(d)
    assert pull_speed <= 0.01 and max_travel <= 0.05 and max_force <= 200, 'refusing unsafe limits'
    rtde_c = RTDEControlInterface(robot_ip, FREQUENCY, RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
    rtde_r = RTDEReceiveInterface(robot_ip, FREQUENCY)
    rtde_c.setPayload(PAYLOAD_MASS, PAYLOAD_COG)   # same payload as the deployment controller
    gripper = RobotiqGripper(); gripper.connect(robot_ip, gripper_port); gripper.activate()
    info = dict(label=label, speed=speed, force=force, direction_base=d.tolist(), pull_speed=pull_speed, max_travel=max_travel,
                max_force=max_force, slip_min=slip_min, slip_drop=slip_drop, payload_mass=PAYLOAD_MASS, payload_cog=list(PAYLOAD_COG),
                open_position=gripper.get_open_position(), closed_position=gripper.get_closed_position(), trials=[])
    arrays = {}
    try:
        gripper.move_and_wait_for_pos(gripper.get_open_position(), speed, force)
        if freedrive:
            rtde_c.teachMode()
        input('Position the OPEN gripper around the fixed cube at the grasp depth you want to test, then press Enter ...')
        if freedrive:
            rtde_c.endTeachMode()
        start = np.array(rtde_r.getActualTCPPose())
        # Preview the pull direction with the gripper open.
        preview = start.copy(); preview[:3] += 0.005 * d
        rtde_c.moveL(preview.tolist(), 0.01, 0.1); rtde_c.moveL(start.tolist(), 0.01, 0.1)
        if input('Did the hand move 5 mm in the intended pull direction and back? [y/N] ').strip().lower() != 'y':
            raise SystemExit('Direction not confirmed; nothing pulled.')
        for trial in range(trials):
            gripper.move_and_wait_for_pos(gripper.get_open_position(), speed, force)
            rtde_c.moveL(start.tolist(), 0.01, 0.1)
            time.sleep(0.5)
            rtde_c.zeroFtSensor(); time.sleep(0.5)                     # zero with the gripper open, not touching the cube
            pos, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), speed, force)
            status = RobotiqGripper.ObjectStatus(status)
            if status not in (RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT, RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT):
                raise SystemExit(f'No object detected after closing ({status.name}); check the cube position.')
            time.sleep(1.0)
            samples = []
            for _ in range(FREQUENCY):                                   # 1 s baseline while holding, before pulling
                samples.append(rtde_r.getActualTCPForce()); time.sleep(1 / FREQUENCY)
            base = np.mean(samples, axis=0)
            target = start.copy(); target[:3] += max_travel * d
            rtde_c.moveL(target.tolist(), pull_speed, 0.1, True)          # asynchronous slow pull
            t0 = time.perf_counter(); ts, F, P = [], [], []
            peak, reason = 0.0, 'max_travel'
            while True:
                f = np.array(rtde_r.getActualTCPForce()); p = np.array(rtde_r.getActualTCPPose())
                ts.append(time.perf_counter() - t0); F.append(f); P.append(p)
                pull = abs(float(np.dot(f[:3] - base[:3], d)))           # force change along the pull axis (sign-agnostic)
                peak = max(peak, pull)
                if np.linalg.norm(f[:3] - base[:3]) > max_force:
                    reason = 'max_force'; break
                if peak > slip_min and pull < (1 - slip_drop) * peak:
                    reason = 'slip'; break
                if rtde_c.getAsyncOperationProgress() < 0 and ts[-1] > 0.5:   # < 0: the asynchronous moveL has finished
                    break
                time.sleep(1 / FREQUENCY)
            rtde_c.stopL(0.5)
            travel = float(np.dot(np.array(P[-1][:3]) - start[:3], d))
            obj_after = RobotiqGripper.ObjectStatus(gripper._get_var(gripper.OBJ)).name
            gripper.move_and_wait_for_pos(gripper.get_open_position(), speed, force)
            rtde_c.moveL(start.tolist(), 0.01, 0.1)
            rec = dict(trial=trial, grasp_position=int(pos), grasp_status=status.name, peak_pull_N=peak, stop_reason=reason,
                       travel_at_stop_m=travel, gripper_status_after=obj_after, baseline_force=base.tolist())
            info['trials'].append(rec)
            arrays[f'trial{trial}_t'] = np.array(ts); arrays[f'trial{trial}_force'] = np.array(F); arrays[f'trial{trial}_pose'] = np.array(P)
            print(f'trial {trial + 1}: peak pull {peak:.1f} N, stopped by {reason} after {travel * 1000:.1f} mm '
                  f'(grasp position {pos}, {status.name})')
    finally:
        try:
            rtde_c.stopL(0.5)
        except Exception:
            pass
        np.savez(output, info=json.dumps(info), **arrays)
        print(f'Saved {output}')
        gripper.disconnect(); rtde_c.stopScript()


if __name__ == '__main__':
    main()
