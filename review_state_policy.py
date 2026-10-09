"""
Review an eval_state_policy.py episode log without watching the run.

Usage:
(robodiff)$ python review_state_policy.py <save_dir>/state_policy_logs/episode_000.npz [--every 0.5] [--video review.mp4]

Prints the run settings, a timeline (wrist, gripper, where the carried cube sits relative to the gripper, cube heights and
tilts, cube ages), the events (gripper commands, grasps and releases, cube tilt changes, contact forces), and every
camera frame whose cube detection looks wrong (one tag only, high reprojection error, or a jump from where the cube should
be). --video draws, on the L515 episode video, the detected tags, each detected cube (thin axes) and the pose the policy
used (thick axes, after face relabelling), with the step's gripper state.
"""
import json
import pathlib
import sys

import click
import cv2
import numpy as np

from diffusion_policy.real_world.rtde_interpolation_controller import install_kinematics_calibration
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose
from diffusion_policy.real_world.thunder_state_policy import frames as F
from diffusion_policy.real_world.thunder_state_policy import r214 as R

CUBES = ('receptive', 'insertive')
LABEL = {'receptive': 'bottom', 'insertive': 'carried'}
GRASP_DEPTH_M = 0.192        # training grasps hold the cube centre here on the wrist z axis (manifest held_cube_center)
JUMP_M = 0.02                # a detection this far from where the cube should be is flagged
CONTACT_N = 15.0             # TCP force change flagged as contact


def wrist(q):
    return F.pos_quat_to_matrix(*get_ee_pose(q))


def tilt_deg(T, up):
    return float(np.degrees(np.arccos(np.clip(T[:3, 2] @ up, -1.0, 1.0))))


def nearest(times, t):
    return int(np.argmin(np.abs(np.asarray(times) - t)))


@click.command()
@click.argument('log_path')
@click.option('--every', default=0.5, help='Timeline row spacing in seconds.')
@click.option('--video', default='', help='Write an annotated copy of the L515 episode video here.')
def main(log_path, every, video):
    z = np.load(log_path)
    m = R.load_manifest()
    install_kinematics_calibration(R.load_calibration())
    scene = m['scene_in_base_link']
    up, table = np.asarray(scene['world_up'], dtype=float), scene['table_top_y_m']
    held = m['scene_in_base_link']['held_cube_center_in_wrist_m']
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    has_frames = 'frame_time' in z.files
    t0 = meta.get('episode_start_time', float(z['t'][0]))
    t = z['t'] - t0
    relabel = z['relabel']

    print(f"== {log_path}")
    if meta:
        c = meta['controller']
        print(f"code {meta['code']['commit'][:9]}{' (uncommitted changes)' if meta['code']['dirty'] else ''}; "
              f"policy {meta['policy_sha256'][:9]}; {meta['frequency']:g} Hz; OSC kp {c['kp'][0]:g}/{c['kp'][3]:g}, "
              f"error clip {c['error_clip_pos_m']} m / {c['error_clip_rot_rad']} rad"
              + (f"; gripper speed {meta['gripper']['speed']} force {meta['gripper']['force']}" if 'gripper' in meta else "")
              + (f"; grasp hold up to {meta['grasp_hold_s']} s" if 'grasp_hold_s' in meta else ""))
        print(f"camera {meta['l515_serial']} {meta['resolution'][0]}x{meta['resolution'][1]}, distortion "
              f"{np.round(meta['dist_coeffs'], 3).tolist()}, transform {meta['camera_transform']} ({meta['camera_transform_frame']})")
    print(f"bottom cube yaw error {float(z['bottom_cube_yaw_error_deg']):+.1f} deg; relabel bottom "
          f"{relabel[0][:3, :3].astype(int).tolist()} carried {relabel[1][:3, :3].astype(int).tolist()}"
          + ("; goal: carried cube tag +Z up (--tag_face_up)" if meta.get('any_face_up') is False else ""))
    print(f"{len(t)} policy steps over {t[-1]:.1f} s; step compute median {np.median(z['compute_s']) * 1000:.0f} ms, "
          f"max {z['compute_s'].max() * 1000:.0f} ms" + (f"; {len(z['frame_time'])} camera frames" if has_frames else ""))
    if 'stacked' in meta:
        print(f"ended on a stable stack at {meta['stacked']['t'] - t0:.1f} s: top cube seen "
              f"{meta['stacked']['offset_mm']:.1f} mm off-centre with the gripper open")
    print(f"training grasps hold the carried cube at {held['median']} m in the wrist frame (p05 {held['p05']}, p95 {held['p95']})")

    # ---------- timeline ----------
    print("\n   t   wrist_h tool_tilt  grip  gPO obj hold carry  carried cube in wrist (mm)  carried h/tilt   bottom h/tilt  age b/c (s)")
    next_t = -np.inf
    for i in range(len(t)):
        if t[i] < next_t and i != len(t) - 1:
            continue
        next_t = t[i] + every
        W = wrist(z['q'][i])
        Tc = F.pos_quat_to_matrix(z['cubes'][i, 1, :3], z['cubes'][i, 1, 3:])
        Tb = F.pos_quat_to_matrix(z['cubes'][i, 0, :3], z['cubes'][i, 0, 3:])
        rel = (np.linalg.inv(W) @ Tc)[:3, 3] * 1000
        h = lambda T: (T[:3, 3] @ up - table - scene['cube_size_m'] / 2) * 1000
        print(f"{t[i]:5.1f}  {(W[:3, 3] @ up - table):6.3f}  {180 - tilt_deg(W, up):6.1f}  "
              f"{'CLOSE' if z['executed_action'][i, 6] < 0 else 'open ':5} {z['gripper_position'][i]:4.0f} {int(z['gripper_object'][i]):3d} "
              f"{int(z['holding'][i]):4d} {int(z['carried'][i]):5d}  {np.round(rel).astype(int)!s:>26}  "
              f"{h(Tc):+6.1f} {tilt_deg(Tc, up):5.1f}   {h(Tb):+6.1f} {tilt_deg(Tb, up):5.1f}   "
              f"{z['cube_age_s'][i, 0]:.2f}/{z['cube_age_s'][i, 1]:.2f}")

    # ---------- events ----------
    print("\nEvents:")
    close = z['executed_action'][:, 6] < 0
    for i in range(1, len(t)):
        if close[i] != close[i - 1]:
            print(f"  {t[i]:5.1f} s  gripper {'CLOSE' if close[i] else 'OPEN'} commanded (position {z['gripper_position'][i]:.0f})")
        if z['holding'][i] != z['holding'][i - 1]:
            W = wrist(z['q'][i])
            Tc = F.pos_quat_to_matrix(z['cubes'][i, 1, :3], z['cubes'][i, 1, 3:])
            rel = np.round((np.linalg.inv(W) @ Tc)[:3, 3] * 1000).astype(int)
            print(f"  {t[i]:5.1f} s  {'HOLDING' if z['holding'][i] else 'released'} (position {z['gripper_position'][i]:.0f}, "
                  f"object {int(z['gripper_object'][i])}); carried cube in wrist {rel.tolist()} mm")
    tilts = np.array([tilt_deg(F.pos_quat_to_matrix(c[1, :3], c[1, 3:]), up) for c in z['cubes']])
    for i in range(1, len(t)):
        if abs(tilts[i] - tilts[i - 1]) > 30 and not z['holding'][i]:
            print(f"  {t[i]:5.1f} s  carried cube tilt {tilts[i - 1]:.0f} -> {tilts[i]:.0f} deg (fell or turned over)")
    if 'tcp_force' in z.files:
        f = np.linalg.norm(z['tcp_force'][:, :3] - np.median(z['tcp_force'][:, :3], axis=0), axis=1)
        for i in np.flatnonzero((f > CONTACT_N) & (np.r_[0, f[:-1]] <= CONTACT_N)):
            print(f"  {t[i]:5.1f} s  TCP force {f[i]:.0f} N above its median (contact?)")
    if 'grasp_hold' in z.files:
        hold = z['grasp_hold'].astype(bool)
        for i in range(len(t)):
            if hold[i] and (i == 0 or not hold[i - 1]):
                print(f"  {t[i]:5.1f} s  grasp hold: arm held at the close pose (gripper position {z['gripper_position'][i]:.0f})")
            if str(z['grasp_hold_end'][i]):
                print(f"  {t[i]:5.1f} s  grasp hold ended: {z['grasp_hold_end'][i]} (gripper position "
                      f"{z['gripper_position'][i]:.0f}, object {int(z['gripper_object'][i])})")
    if 'relabel_event_time' in z.files:
        for et, cube in zip(z['relabel_event_time'], z['relabel_event_cube']):
            print(f"  {et - t0:5.1f} s  {LABEL[str(cube)]} cube came to rest on another face: relabelled +Z up")
    if 'rejected' in z.files:
        for c, name in enumerate(CUBES):
            idx = [i for i in range(len(t)) if z['rejected'][i, c]]
            if idx:
                reasons = sorted(set(str(z['rejected'][i, c]) for i in idx))
                print(f"  {LABEL[name]} cube: {len(idx)} detections not used ({'; '.join(reasons)}) at "
                      f"{', '.join(f'{t[i]:.1f}' for i in idx[:12])}{' ...' if len(idx) > 12 else ''} s")

    if not has_frames:
        print("\n(no frame_ records: this log predates per-frame detection logging)")
        return

    # ---------- detection quality per camera frame ----------
    ft = z['frame_time'] - t0
    print("\nCamera frames with doubtful cube detections (one tag, reprojection > 2 px, or a jump > 2 cm):")
    flagged = 0
    for name in CUBES:
        T = z[f'frame_{name}_T_base']
        ok = z[f'frame_{name}_success'] & ~z[f'frame_{name}_predicted']
        last = None
        for k in np.flatnonzero(ok):
            i = nearest(z['t'] - t0, ft[k])
            expected = last
            if expected is not None and name == 'insertive' and z['holding'][i]:
                kp, Tp = expected
                expected = (k, wrist(z['frame_q'][k]) @ np.linalg.inv(wrist(z['frame_q'][kp])) @ Tp)   # moved with the wrist
            jump = np.linalg.norm(T[k][:3, 3] - expected[1][:3, 3]) if expected is not None else 0.0
            n_tags, reproj = z[f'frame_{name}_n_tags'][k], z[f'frame_{name}_reproj_px'][k]
            if n_tags <= 1 or reproj > 2.0 or jump > JUMP_M:
                flagged += 1
                ids = [int(x) for x in z[f'frame_{name}_tag_ids'][k] if x >= 0]
                print(f"  {ft[k]:5.2f} s  {LABEL[name]:7s} tags {ids} reproj {reproj:.2f} px  jump {jump * 1000:5.0f} mm"
                      f"{'  (held)' if z['holding'][i] else ''}")
            last = (k, T[k])
    print(f"  {flagged} flagged of {int(sum((z[f'frame_{n}_success'] & ~z[f'frame_{n}_predicted']).sum() for n in CUBES))} detections")
    for name in CUBES:
        skipped = z[f'frame_{name}_skipped'] if f'frame_{name}_skipped' in z.files else np.zeros(len(ft), dtype=bool)
        n = z[f'frame_{name}_n_tags'][~skipped]
        print(f"  {LABEL[name]}: seen in {np.mean(n > 0) * 100:.0f}% of {len(n)} searched frames"
              + (f" (not searched in {skipped.sum()} while held)" if skipped.any() else "")
              + f"; tags per detection {np.bincount(n, minlength=4)[:4].tolist()} (0,1,2,3)")
    ds, gap = z['frame_detect_s'] * 1000, np.diff(ft) * 1000
    print(f"  detection per frame: median {np.median(ds):.0f} ms, p90 {np.percentile(ds, 90):.0f}, max {ds.max():.0f}; "
          f"frame to frame median {np.median(gap):.0f} ms, max {gap.max():.0f}")

    if video:
        recorded = pathlib.Path(meta['video'])
        if not recorded.exists():                                   # archived run: the video sits next to the logs
            meta['video'] = str(pathlib.Path(log_path).resolve().parent.parent / 'videos' / recorded.parent.name
                                / recorded.name)
        write_video(z, meta, t0, relabel, video)


def write_video(z, meta, t0, relabel, out_path):
    K = np.asarray(meta['camera_matrix'], dtype=float)
    dist = np.asarray(meta['dist_coeffs'], dtype=float)
    cam_T_base = np.linalg.inv(np.asarray(meta['T_base_camera'], dtype=float))
    cap = cv2.VideoCapture(meta['video'])
    fps = cap.get(cv2.CAP_PROP_FPS) or meta['video_fps']
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    ft, st = z['frame_time'] - t0, z['t'] - t0

    def axes(img, T, length, thickness):
        pts = (cam_T_base @ np.c_[np.vstack([T[:3, 3], T[:3, 3] + length * T[:3, :3].T]), np.ones(4)].T)[:3].T
        if np.any(pts[:, 2] <= 0):
            return
        uv = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).astype(int)
        for j, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):          # x red, y green, z blue (BGR)
            cv2.line(img, tuple(uv[0]), tuple(uv[j + 1]), color, thickness, cv2.LINE_AA)

    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        tv = k / fps
        k += 1
        f = nearest(ft, tv)
        if abs(ft[f] - tv) < 0.2:
            for name in CUBES:
                for c in z[f'frame_{name}_corners_px'][f]:
                    if np.isfinite(c).all():
                        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
                if z[f'frame_{name}_success'][f] and not z[f'frame_{name}_predicted'][f]:
                    axes(img, z[f'frame_{name}_T_base'][f], 0.04, 1)
        i = nearest(st, tv)
        if abs(st[i] - tv) < 0.2:
            for c, name in enumerate(CUBES):
                axes(img, F.pos_quat_to_matrix(z['cubes'][i, c, :3], z['cubes'][i, c, 3:]), 0.05, 3)
            text = (f"t={tv:5.2f}s  grip {'CLOSE' if z['executed_action'][i, 6] < 0 else 'open'} pos {z['gripper_position'][i]:.0f}"
                    f"  hold {int(z['holding'][i])}  carried {int(z['carried'][i])}  seen b/c {z['seen'][i].astype(int).tolist()}")
            cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"\nWrote {k} frames to {out_path} (thick axes: pose the policy used; thin: this frame's detection; yellow: tags)")


if __name__ == '__main__':
    sys.exit(main())
