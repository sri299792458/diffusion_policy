# Thunder gripper measurements — 2026-10-06

This package preserves the selected real-world measurements for tuning the simulated Thunder gripper at **raw speed 0 / force 0**, together with three original setup photos and the first 10 mm trial's video link. The pull curves show the combined behavior of the Robotiq mechanism and deformable TPU tips. Simulation parameters have not yet been fitted.

The recordings and photos are unchanged copies of the workstation originals. Reviews add operator observations and interpretation without modifying the raw measurements. [manifest.json](manifest.json) records SHA-256 checksums for every other file in this package and the original recording directory. Failed runs, early misleading force-drop results, and profiles collected at 128/128 are excluded from this package.

## Recordings and selection

All six recordings completed at speed 0 / force 0, with empty-calibration endpoints 3/226 and no recorded cleanup errors. Raw 0 selects the hardware minimum; it does not mean zero motion or zero force. Pulls used controller-base direction `[0, -1, 0]` (upward on this mounted Thunder), 2 mm/s, and two 40 N software force-change limits: one from the closed-grasp baseline and one from the tared open reference. Requested travel differs by run.

| Recording | Setup and intended use |
| --- | --- |
| [profile_empty.npz](recordings/profile_empty.npz) | Three empty close/open cycles; use for motor travel and timing. Closing took approximately 4.29–4.33 s. |
| [profile_cube60.npz](recordings/profile_cube60.npz) | Three closes on the nominal 60 mm cube at different vertical contact heights between the tips. Contact motor positions were 112, 85, 137. Use each trajectory separately; these are not three repetitions of one grasp depth. Contact heights were not measured. |
| [pullout_cube60_s0_f0_setup_preload_retry2.npz](recordings/pullout_cube60_s0_f0_setup_preload_retry2.npz) | Loose cube resting on the table. Control for preload unloading and lifting; exclude from resisted-pull fitting. |
| [pullout_cube60_s0_f0_setup_preload_retry4.npz](recordings/pullout_cube60_s0_f0_setup_preload_retry4.npz) | Cube held stationary by hand on its other faces; operator observed TPU bending and sliding. Requested 5 mm; provisional resisted-pull curve. |
| [pullout_cube60_s0_f0_travel10mm.npz](recordings/pullout_cube60_s0_f0_travel10mm.npz) | First requested 10 mm trial, hand restraint, observed bending/sliding; associated with the linked video. |
| [pullout_cube60_s0_f0_travel10mm_1.npz](recordings/pullout_cube60_s0_f0_travel10mm_1.npz) | Repeat requested 10 mm trial. Same hand restraint is inferred from conversation, not independently confirmed; no separate visual-slip confirmation for this repeat. |

Each recording has an adjacent `.review.json`. The [two-run comparison](recordings/pullout_cube60_s0_f0_travel10mm_repeatability.json) preserves the detailed measurements and limitations.

| Pull | Initial axial preload (N) | Peak opposing load (N) | Mean opposing load over 1.5–4.8 mm (N) | Stop travel (mm) | Recorded stop |
| --- | ---: | ---: | ---: | ---: | --- |
| retry2 — loose control | 23.92 | 0.83 | — | 4.93 | max_travel |
| retry4 — resisted 5 mm | 27.89 | 8.96 | 8.40 | 4.99 | max_travel |
| first requested 10 mm | 24.84 | 8.17 | 7.09 | 7.17 | slip candidate |
| repeat requested 10 mm | 25.45 | 7.99 | 6.59 | 6.18 | slip candidate |

The two requested 10 mm trials differed by 2.23% in opposing-load peak, 7.03% in the same-window mean, and 0.98 mm in stop travel. Their binned force-curve difference was 0.76 N RMS over 1.5–5.75 mm. This supports a provisional peak target near 8 N, with weakening during further travel. Two trials do not establish a statistical repeatability tolerance. Keep both curves and their differing off-axis forces/torques when assessing a fit.

## Force interpretation for simulation

Read the files with `numpy.load(path, allow_pickle=False)`; `info` is a JSON scalar and the measurement arrays are numeric. For a pull trial:

```python
import json
import numpy as np

with np.load("recordings/pullout_cube60_s0_f0_travel10mm.npz", allow_pickle=False) as data:
    info = json.loads(str(data["info"]))
    direction = np.asarray(info["direction_base"])
    trial = info["trials"][0]
    open_force = np.asarray(trial["open_force"])
    signed_load_N = (data["trial0_force"][:, :3] - open_force[:3]) @ direction
    opposing_load_N = -signed_load_N
    motion = next(item for item in info["motions"] if item["phase"] == "trial0")
    travel_mm = (data["trial0_pose"][:, :3] - motion["start_pose"][:3]) @ direction * 1000
```

The initial positive signed load assists the pull and unloads before the opposing load develops. `peak_pull_N` is an absolute change from the closed baseline; it includes the approximately 24–28 N preload unloading/reversal. The approximately 33–37 N changes in resisted trials must not be treated as holding strength. Use the signed curve relative to the open reference and the opposing loading branch.

The wrist sensor measures net external reaction, so these runs do not separately identify normal finger squeeze, TPU stiffness, or friction. The force-drop stop occurred after visible sliding in the first 10 mm trial; it does not timestamp first slip or establish maximum static breakaway force. Steady sliding in retry4 did not trigger a force drop. Hand restraint and cube displacement were not instrumented, so these are provisional effective-resistance curves rather than fixed-fixture measurements.

Match empty motor timing first, retain the different contact-height profiles, and compare the complete resisted force-versus-travel curves under equivalent geometry. A rigid-tip fit must be judged on its measured mismatch with TPU bending and weakening; matching only an 8 N scalar is insufficient. A controlled fixture repeat and corresponding 40 mm cube data remain to be collected before validation across cube sizes. The exact contact height and first-slip time are unknown.

## Three original photos

| Photo | Context |
| --- | --- |
| [01_tpu_grasp_initial.png](photos/01_tpu_grasp_initial.png) | Initial bent-TPU grasp; the operator reported that the tips spring back. The raw settings of this photo are not established. |
| [02_pendant_grasp.png](photos/02_pendant_grasp.png) | Grasp the operator considered satisfactory using the pendant. |
| [03_pendant_settings_0_0.png](photos/03_pendant_settings_0_0.png) | Original pendant screenshot showing Speed 0% / Force 0%. Its nominal position display is not a measurement of the loaded TPU fingertip gap. Orientation is preserved. |

The photos are setup evidence, not synchronized measurement frames or force measurements.

## Video

[Watch the first 10 mm trial: gripper_slip_test.mp4](https://drive.google.com/file/d/1Gqe8o3dv6_r9Unr9PKLsz1e4-YbefEPV/view?usp=drivesdk).

[video_evidence.json](video_evidence.json) associates the video with `pullout_cube60_s0_f0_travel10mm.npz` based on timestamps and the recorded sequence, and preserves the observations and approximate video windows. It shows the cube against the table under hand restraint, pad deformation and upward relative sliding, then opening. Video and RTDE have no shared synchronization, so these windows cannot determine the first microscopic slip.

The video remains hosted on Drive. Metadata was readable through the connected account on 2026-10-06 and its reported size matched the local reviewed video (45,387,379 bytes). The remote content checksum and access for other viewers were not verified. The recorded local video SHA-256 is provenance for the reviewed copy; no sharing settings were changed.

## Code and validation

The code on this branch uses common 0/0 defaults for calibration and deployment; waits for acknowledged finger motion and settled completion; preserves calibration, closing, open-reference and partial-error traces; monitors asynchronous arm motion; and requires a sustained loading rise before treating a force drop as a slip candidate. See [the operating instructions](../../../scripts/sim2real/gripper_calibration.md) for setup and command details. Archive packaging and validation do not connect to the robot.

Offline validation passed 83 calibration tests, the same 83 tests with Python optimization enabled, and the RTDE controller timing test. These use simulated interfaces and do not replace supervised hardware validation. Verify archived file integrity from this directory:

```python
from pathlib import Path
import hashlib
import json

root = Path(".")
manifest = json.loads((root / "manifest.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    assert path.stat().st_size == item["size_bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print("All archived checksums match")
```
