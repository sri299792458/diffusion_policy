# R214 state policy on Thunder with `eval_state_policy.py`: October 9, 2026

Eight runs of the R214 state policy (UWLab OmniReset cube stacking) on Thunder: UR5e, Robotiq 2F-85 with UMI fingertips,
and the L515 for the cube poses (AprilCube). The runs used the RealEnv OSC controller (Kp 1000/50, error clip 0.05 m /
0.3 rad) at 10 Hz.

All runs used uncommitted code on top of `67b9349`, which changed between runs as listed below; `ec9139e` is the code
after the last run. Read a log with:

    python review_state_policy.py validation_results/thunder_r214_state_policy/2026-10-09/dp_run6/state_policy_logs/episode_000.npz

The camera transform is the Spark calibration `T_base_l515_20261008.json` (frame `ur_base`), archived in UWLab under
`scripts_v2/tools/thunder_sim2real/validation_results/state_policy_dry_run_20261009/camera/`. Each log's `meta` holds its
base_link form, the L515 intrinsics and distortion, and the controller and gripper settings.

| File | What |
| --- | --- |
| `dp_runN/state_policy_logs/episode_NNN.npz` | one row per policy step (observation, action, target, cube poses, gripper, grasp hold, rejected readings), one row per camera frame (`frame_*`: every AprilCube result with tag ids and corners, and the robot state at that frame), and `meta` (JSON) |
| `dp_runN/replay_buffer.zarr` | RealEnv's own recording: robot state, actions and timestamps for every episode |
| `dp_runN/videos/<episode>/0.mp4` | RealEnv's L515 color video of each episode (71 MB in all) |
| `SHA256SUMS` | every file in this folder |

Log `episode_NNN` goes with video folder `NNN`, except in `dp_run2`, which has no `meta`. `review_state_policy.py --video
out.mp4` draws the detections and the poses the policy used on that video; when the workstation path recorded in `meta` is
missing, it uses the video in this folder.

## Runs

| Run | Code at the time | What happened |
| --- | --- | --- |
| dp_run1 | first upstream eval | Hung in the first policy step: after PyAV was imported, `cv2.imshow` hung (both wheels ship X11 libraries with identical names). The robot did not move. No step log. |
| dp_run2 | X11 fix, background detection | After the carried cube fell and landed on another face, the policy tried to turn it over and repeatedly missed it. No per-frame records yet. |
| dp_run3 | relabel after landing, plausibility checks, per-frame logs | Repeated close-then-reopen: the Robotiq at speed 128 needs ~0.56 s to reach the cube, the sim gripper ~0.2 s, and the policy lifts before the fingers arrive. |
| dp_run4 | grasp hold, gripper speed 255 | The grasp hold worked (contact every time). Releases ~3 cm above the stack: a camera frame from before the grasp, applied late, was taken as an in-hand reading and put the held cube 35 mm low. |
| dp_run5 | readings judged by frame time, held cube not searched while holding | Episode 0 stacked at 3.5 s, then re-gripped and hovered; once the top tag was covered, the bottom cube's single-tag readings jumped up to 38 mm. Episode 1: grasped tilted (fingers at 154), let go ~5 cm up, and the estimate stayed in the air for 6 s; it recovered and ended stacked, rotated. |
| dp_run6 | readings moved along the camera ray onto their support and made flat; released cube drops onto its support | First stack fell off after a re-grip; the second (8.8 s) held. The policy then re-gripped it 5 times until 16 s: it saw the top cube 5-17 mm off-centre, and R214's success test is 5 mm / 0.025 rad. |
| dp_run7 | stop on a stable stack (5 steps within 10 mm, gripper open) | Episode 3 ended on a stable stack at 9.0 s (6.2 mm). Episodes 0-2 ran to 16 s. |
| dp_run8 | `--tag_face_up` (training's goal: tag 24 up), then the default | See below. |

### dp_run8

Two sessions were written to the same output directory. The eval then numbered its logs from `episode_000` in every
session, so the second session's logs overwrote the first's; `ec9139e` fixes this. The two surviving logs are renamed here
to their video folders.

- **Session 1 (videos 0 and 1), `--tag_face_up`, 30 s.** Step logs lost; `session1_tag_face_up_review.txt` is the review
  printout of them. Its robot data is in `replay_buffer.zarr` (episodes 0 and 1).
  - Episode 0 started with tag 24 already up, and stacked and stopped at 6.0 s (5.7 mm).
  - Episode 1 started with tag 24 on its side. The policy did not turn the cube over in 30 s. About 20 times it brought
    its fingertips 5-7 cm beside the cube's upper edge, then tipped the cube partway twice and let it fall back.
  - Where the cube was visible, the estimate matched it in the video.
  - The arm moved much less per unit action than in sim. In free space it moved 11% as much along base x, 21% about
    vertical and 65-75% on the other two axes, compared with the stored sim rollouts. Its commands exceeded the
    controller's error clip on 54-69% of steps per translation axis, against 19-27% in sim.
- **Session 2 (videos 2 and 3), default mode, 30 s.** `state_policy_logs/episode_002.npz` and `episode_003.npz`. Not
  analysed yet.
