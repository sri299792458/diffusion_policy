"""Record the real Robotiq 2F-85 open/close profile at the deployment settings (gripper only; the arm does not move).

The OmniReset sim2real guide asks for a gripper sanity check: run open/close cycles on the real and simulated gripper
and compare the trajectories. The deployment controller (rtde_interpolation_controller.py) commands only fully open or
fully closed with speed 128 and force 128, so those are the defaults here.

Each cycle: open fully and wait, then command "closed" and poll the finger position (0-255) and the object-detection
status as fast as the socket allows until the gripper stops, hold, then command "open" and record the same way.
Run it three ways (use --label):
  empty   - nothing between the fingers
  cube40  - the 40 mm cube held centred between the open fingertips (by hand or on a stand)
  cube60  - the 60 mm AprilCube held the same way
For the cube runs the script waits for Enter before each close, so the cube can be placed.

Usage (from the diffusion_policy repo root, robodiff_real env):
    python scripts/sim2real/record_gripper_profile.py --robot_ip 192.168.1.10 --label empty --output data/gripper_profile_empty.npz
    python scripts/sim2real/record_gripper_profile.py --robot_ip 192.168.1.10 --label cube60 --output data/gripper_profile_cube60.npz
"""
import json
import time

import click
import numpy as np

from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


def record_move(gripper, target, speed, force, settle_s=0.5, timeout_s=5.0):
    """Command one move and poll position/status until the gripper stops, then for settle_s longer."""
    t0 = time.perf_counter()
    ok, cmd = gripper.move(target, speed, force)
    if not ok:
        raise RuntimeError('Gripper did not accept the move command')
    ts, pos, obj = [], [], []
    stopped_at = None
    acknowledged = seen_moving = False
    while True:
        now = time.perf_counter() - t0
        p = gripper.get_current_position()
        o = gripper._get_var(gripper.OBJ)
        ts.append(now); pos.append(p); obj.append(o)
        # The status register can still show the previous stop for a few ms: only look for the stop once the gripper has
        # echoed the new target (PRE) and reported motion (or 0.3 s have passed, e.g. when already at the target).
        acknowledged = acknowledged or gripper._get_var(gripper.PRE) == cmd
        seen_moving = seen_moving or o == RobotiqGripper.ObjectStatus.MOVING.value
        if (stopped_at is None and acknowledged and (seen_moving or now > 0.3)
                and o != RobotiqGripper.ObjectStatus.MOVING.value):
            stopped_at = now
        if stopped_at is not None and now - stopped_at > settle_s:
            break
        if now > timeout_s:
            print('  warning: move did not finish within the timeout')
            break
    return dict(t=np.array(ts), position=np.array(pos), object_status=np.array(obj), commanded=cmd,
                stopped_s=stopped_at, final_status=RobotiqGripper.ObjectStatus(obj[-1]).name)


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10', help='UR5e IP address (the Robotiq URCap socket runs on it)')
@click.option('--gripper_port', default=63352, type=int)
@click.option('--speed', default=128, type=int, help='Robotiq speed 0-255 (deployment: 128)')
@click.option('--force', default=128, type=int, help='Robotiq force 0-255 (deployment: 128)')
@click.option('--cycles', default=3, type=int)
@click.option('--label', type=click.Choice(['empty', 'cube40', 'cube60']), required=True)
@click.option('--output', '-o', required=True, help='Output .npz file')
def main(robot_ip, gripper_port, speed, force, cycles, label, output):
    gripper = RobotiqGripper()
    print(f'Connecting to the gripper at {robot_ip}:{gripper_port} ...')
    gripper.connect(robot_ip, gripper_port)
    gripper.activate()   # same call as the deployment controller (auto-calibrates the open/closed range)
    info = dict(label=label, speed=speed, force=force, cycles=cycles, robot_ip=robot_ip,
                open_position=gripper.get_open_position(), closed_position=gripper.get_closed_position(),
                recorded_unix=time.time())
    print(f'Calibrated range: open={info["open_position"]} closed={info["closed_position"]}; speed={speed} force={force}')
    arrays = {}
    try:
        for c in range(cycles):
            gripper.move_and_wait_for_pos(gripper.get_open_position(), speed, force)
            time.sleep(0.5)
            if label != 'empty':
                input(f'[cycle {c + 1}/{cycles}] Place the {label} centred between the fingertips, then press Enter to close ...')
            close = record_move(gripper, gripper.get_closed_position(), speed, force)
            print(f'  cycle {c + 1}: close stopped after {close["stopped_s"]:.3f} s at position {close["position"][-1]} '
                  f'({close["final_status"]}), {len(close["t"])} samples')
            time.sleep(0.5)
            opening = record_move(gripper, gripper.get_open_position(), speed, force)
            print(f'  cycle {c + 1}: open stopped after {opening["stopped_s"]:.3f} s at position {opening["position"][-1]}')
            for name, rec in (('close', close), ('open', opening)):
                for key in ('t', 'position', 'object_status'):
                    arrays[f'cycle{c}_{name}_{key}'] = rec[key]
                info[f'cycle{c}_{name}'] = {k: rec[k] for k in ('commanded', 'stopped_s', 'final_status')}
    finally:
        np.savez(output, info=json.dumps(info), **arrays)
        print(f'Saved {output}')
        gripper.disconnect()


if __name__ == '__main__':
    main()
