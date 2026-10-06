# Thunder gripper calibration

These scripts collect real gripper trajectories and fixed-cube pull curves for comparison with simulation. The selected Thunder deployment and calibration settings are now raw speed 0 / force 0, matching the minimum settings displayed on the pendant. The deployment gripper worker and both calibration CLIs share these defaults from `robotiq_gripper.py`. They are independent of the diffusion-policy training dependencies.

Speed and force arguments are raw settings in 0..255, whereas the pendant displays percentages. Raw 128 is about 50% of the setting range, not 128 N. Raw 0 selects the hardware minimum, not zero motion or zero gripping force. The pendant photo provided on 2026-10-06 displayed Speed 0% / Force 0%; use `--speed 0 --force 0` to collect a comparison at the minimum settings rather than assuming it matches 128/128. Keep speed/force pairs explicit and do not combine runs at different settings when fitting one operating point. Force 0 also disables automatic re-grasp; 128 selects high-torque mode with re-grasp enabled. See the [Robotiq speed, force and picking-feature definitions](https://assets.robotiq.com/website-assets/support_documents/document/online/2F-85_2F-140_Instruction_Manual_Gen_HTML_20190524.zip/2F-85_2F-140_Instruction_Manual_Gen_HTML/Content/4.%20Control.htm).

The [2026-10-06 measurement package](../../calibration/thunder_gripper/2026-10-06/README.md) contains the new 0/0 empty and 60 mm cube profiles, three provisional resisted-pull curves, a loose-cube control, three original photos and the first 10 mm trial's Drive video link. The cube profiles used different vertical contact heights, and the resisted pulls used hand restraint rather than an instrumented fixture; keep those distinctions when fitting. The earlier `profile_empty.npz` in `data/gripper_calibration_20261006_075305` was collected at 128/128 and remains a separate local reference. Collect fixture-controlled pulls at consistent contact height and corresponding 40 mm cube data before validating a fit across cube sizes. Closing/opening trajectories constrain timing and motor motion; resisted-pull curves constrain overall holding behavior. Neither motor position nor one pull peak separately identifies TPU stiffness, friction and drive effort. First test whether a single effective rigid-gripper fit reproduces timing and holding behavior on both cube sizes. If it cannot, retain the mismatch and add compliance measurements/modeling rather than compensating with unrelated drive or friction values.

## Workstation environment

A separate Python 3.11 environment is installed on this workstation at `/home/srinivas/Documents/ChatGPT/omnireset/.venv-gripper-calibration`, with NumPy 1.26.4, ur-rtde 1.6.5, and an editable installation of this fork. From the fork root:

```bash
cd /home/srinivas/Documents/ChatGPT/omnireset/diffusion_policy
source ../.venv-gripper-calibration/bin/activate
python scripts/sim2real/record_gripper_profile.py --help
python scripts/sim2real/gripper_pullout_test.py --help
```

To recreate that environment on a workstation with Python 3.9–3.12:

```bash
python3 -m venv ../.venv-gripper-calibration
../.venv-gripper-calibration/bin/python -m pip install -r scripts/sim2real/gripper_calibration_requirements.txt
../.venv-gripper-calibration/bin/python -m pip install --no-deps -e .
```

The default robot address is Thunder's recorded address, `10.33.55.89`; override `--robot_ip` for another setup. Output directories are created and checked before any connection. Existing outputs are refused: choose a fresh filename to preserve earlier measurements. `--help` does not connect to hardware.

## Closing profiles

The arm receives no commands during profile recording. Before connecting, the script asks you to clear all objects from the finger sweep. Empty activation first moves to an interior position, then opens, closes and reopens at speed 64 / force 1. This permits calibration when the actual endpoints differ from nominal requests 0/255, including Thunder's measured 3/227 endpoints. Keep the cube out until the subsequent placement prompt. Each cycle records closing and opening at the requested deployment settings.

```bash
python scripts/sim2real/record_gripper_profile.py --label empty --speed 0 --force 0 --output data/gripper_profile_empty_s0_f0.npz
python scripts/sim2real/record_gripper_profile.py --label cube40 --speed 0 --force 0 --output data/gripper_profile_cube40_s0_f0.npz
python scripts/sim2real/record_gripper_profile.py --label cube60 --speed 0 --force 0 --output data/gripper_profile_cube60_s0_f0.npz
```

A gripper move must acknowledge its target and show motion or a changed position, then remain stopped for the settling interval. An already-open gripper can complete as a verified no-op. A stale stop alone cannot complete a move. Commands, faults, requested positions and partial trajectories are recorded; timeout is an error, rather than a completed profile.

The yellow TPU tips can bend elastically while the motor reaches its empty-close position. `AT_DEST` therefore does not prove that the cube is absent. If closing completes without firmware object detection, the script pauses before opening and asks you to confirm visually that both fingertips still contact and hold the cube. Keep hands clear; answer `y` only after checking the grasp. Enter, `n`, EOF or Ctrl+C aborts the run and retains the closing trace. This confirmation is requested separately for each unflagged grasp. The recording distinguishes `object_detected` from `grasp_confirmation` (`firmware` or `operator`) and `operator_confirmed`; it never relabels an operator observation as firmware detection.

The rigid simulation asset omits TPU bending. Motor position alone is not a measurement of the loaded fingertip gap or deformation. Preserve this distinction when comparing these profiles and pull curves with simulation. Robotiq also documents that its [object-detection status can miss a successful fingertip grasp](https://assets.robotiq.com/website-assets/support_documents/document/online/2F-85_2F-140_Instruction_Manual_Gen_HTML_20190524.zip/2F-85_2F-140_Instruction_Manual_Gen_HTML/Content/4.%20Control.htm).

## Fixed-cube pull trials

Fix the cube securely to the table and clear the intended translation path. Keep the e-stop accessible. Remove objects from the fingers for activation. The script verifies that the gripper is fully open, enables freedrive for positioning, and then previews 5 mm along the pull direction and back. The first motion must go upward, away from the table, before returning. Confirm the direction only after observing that motion. Thunder's supervised preview on 2026-10-06 showed that **controller base +y moves downward**; the operator rejected the preview and no pull trial started. The corrected default is **controller base -y** (`--direction 0,-1,0`), which reverses that motion. Preview and verify it again before confirming. RTDE moves use the controller base frame; do not substitute a URDF/REP-103 frame's axis labels.

```bash
python scripts/sim2real/gripper_pullout_test.py --label cube60 --speed 0 --force 0 --trials 3 --output data/pullout_cube60_s0_f0.npz
python scripts/sim2real/gripper_pullout_test.py --label cube40 --speed 0 --force 0 --trials 3 --output data/pullout_cube40_s0_f0.npz
```

For an additional supervised fixture check, start with one 5 mm trial using the same 40 N closed-baseline and open-reference change limits as the archived completed pulls. Choose a fresh filename. An earlier failed 2026-10-06 trial recorded approximately 26 N after closing and never started a pull; its open reference was not saved, so that value is not an exact preload measurement. Current recordings preserve the reference and closing trace. These chosen software thresholds do not establish the fixture's or robot's safe load rating; inspect the short trial before extending it.

```bash
python scripts/sim2real/gripper_pullout_test.py --robot_ip 10.33.55.89 --label cube60 --speed 0 --force 0 --trials 1 --direction 0,-1,0 --pull_speed 0.002 --max_travel 0.005 --max_force 40 --max_total_force 40 --output data/gripper_calibration_pendant_0_0/pullout_cube60_s0_f0_fixture_5mm.npz
```

The script preserves the controller's configured payload and records its mass and CoG. It refuses changes during the run. It zeroes the force sensor only with the gripper open and saves the resulting open reference. The overall force-change limit (`--max_total_force`) compares the norm of the three translational force components against that reference, including closing preload. It monitors closing and the settling interval and stops the fingers immediately if closing fails. A grasp must produce closing object detection or an explicit visual confirmation as described above before a pull can start. It then collects a fresh stationary grasp baseline, checking the overall limit and TCP position again after the prompt. Force values and the direction use the UR controller's base orientation, as specified by the [UR wrist-force API](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-scriptmanual/all_scripts/get_tcp_force.htm).

During the pull, `--max_force` limits the norm of the additional translational force change from the closed-grasp baseline. Both force limits remain active. `--max_total_force` defaults to `--max_force` when omitted, so separating the limits requires an explicit argument. The recording saves the open and closed references, signed preload along the pull direction, preload vector/norm, peak additional axial change (`peak_pull_N`), peak overall axial load (`peak_load_N`), and both vector-norm peaks. The wrist load is the external net reaction on the hand; it does not measure the opposing normal squeeze forces between the fingers and cube.

Every scripted arm move, including the preview and return, runs asynchronously with monitoring. Rejected commands, invalid sensor values, stale or inconsistent states, protective/emergency stops, changed payload, path departure and timeouts abort the run. Async completion must belong to the newly accepted operation; idle status alone does not count as completion, following the [ur-rtde API contract](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html).

The pull defaults to 2 mm/s, 20 mm travel and 100 N for both force-change limits. User limits must be positive and finite; the maximum permitted settings are 10 mm/s, 50 mm and 200 N. These are software thresholds, not guarantees against overshoot or loss of communication. A stop request decelerates at 0.5 m/s².

Slip detection first requires overall axial load to rise by at least `--slip_min` from the lowest load observed during this motion and remain above that rise threshold for 50 ms. Only then is it armed to detect a 40% drop from the subsequent loading peak, sustained for another 50 ms (the drop fraction is configurable). Initial monotonic unloading of the closing preload does not arm it. Both force limits remain active before and after arming. The recording retains `slip_armed`, `slip_minimum_load_N`, and `slip_loading_peak_N`. A run that never arms the detector does not establish a slip maximum. A very early release before loading cannot be identified from force alone. The recorded flag is still `slip_candidate`: a force drop can also come from fixture movement or elastic unloading. Inspect the fixture and the signed force/pose traces before using that result to fit the simulated gripper.

The 2026-10-06 `pullout_cube60_s0_f0_setup_preload.npz` trial completed normally, but the previous detector labeled monotonic preload unloading as `slip`. The operator confirmed the cube stayed in place. Its closing preload was 24.20 N (norm), and the signed axial load decreased from 23.15 N to 10.62 N during the recorded pull; stopped travel was 0.476 mm. The reported `peak_pull_N=13.41` is a decrease from the closed baseline, not a holding-force measurement. Preserve its raw arrays as preload/unloading data and exclude its slip label from grip-strength fitting. The corrected detector requires loading before a drop can qualify.

Following a normal trial stop, the script confirms the arm has settled, opens the gripper, verifies its calibrated open position and status, and monitors the return to the starting pose. On an error or Ctrl+C it requests an arm stop, exits freedrive if necessary, stops the control script and finger motion, and attempts to disconnect all interfaces. It does not command a recovery motion. Cleanup failures are recorded and a saving error cannot skip cleanup.

## Inspecting data and offline validation

The existing `info` JSON and `cycleN_close_*`, `cycleN_open_*`, `trialN_t`, `trialN_force`, and `trialN_pose` keys are retained. Schema version 2 adds run/trial/motion status, error and cleanup details, advancing robot timestamps, baseline samples, and preview/return traces. Calibration phases also keep `calibration_PHASE_*` arrays and their last register readings, including when activation fails. Pull trials now also save finger-register traces in `trialN_close_*` and wrist force/pose/timestamps during closing and settling in `trialN_grasp_*`. Check `run_status` and each trial's `status` before treating data as complete. A completed pull with `max_force` or `max_total_force` is a threshold-capped observation, not a measured slip maximum. `max_travel` means the requested motion completed without a detected slip. None measures the true maximum holding force. Interrupted trials keep their collected samples.

```bash
python -m unittest discover -s tests -p test_gripper_calibration.py -v
python -O -m unittest discover -s tests -p test_gripper_calibration.py
```

These tests use a fake robot, gripper firmware and clock. They never connect to the real robot. The workstation SDK and CLI can be checked offline; physical force sensing, stopping behavior, fixture strength and actual fingertip slip still require supervised hardware validation.
