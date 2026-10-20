# Verification results

> **Verification note for the repaired archive:** the numerical results below were already present in the supplied project and describe a prior integrated build. The repaired source package has been re-checked with 37 non-MuJoCo regression tests; the two physical MuJoCo/rendering tests must be rerun in the team virtual environment before treating the historical numbers below as current evidence. See `docs/APPLIED_FIXES.md`.

Generated during integration on 15 October 2026 (UTC). These are simulation measurements from the included code. They are not hardware results or an assurance of performance on other scenes.

## Automated checks

- `python -m pytest -q`: **39 passed**. Includes the adapted original frame/noise tests, motion Jacobian/rank checks, covariance and innovation gating, lateral-target control, uninterrupted docking dwell, planner detours, truth isolation, rendered-marker detection and low-speed physical turning.

- `python -m ruff check --select E4,E7,E9,F ...`: passed for integrated runtime scripts and tests. Reference snapshots are intentionally excluded.

- `python part_a_check.py`: largest numerical Jacobian discrepancy 1.4e-10; controllability ranks 2 at rest and 3 while moving; point-landmark observability ranks 2 for one point and 3 for two.

## Repeated end-to-end trials

Command: `python evaluate.py --seeds 1 2 3 --duration 65`

All 12 runs completed the estimated-state visual docking conditions and passed the independent true-pose/collision checks. Maximum run duration was 22.60 s. Three seeds per condition are a small tested sample, not a population reliability estimate.

| Condition | Successes | Mean EKF position RMSE (m) | Mean final dock error (m) | Collisions |
|---|---:|---:|---:|---:|
| nominal | 3/3 | 0.0314 | 0.0543 | 0 |
| noisy | 3/3 | 0.0413 | 0.0503 | 0 |
| dropout | 3/3 | 0.0100 | 0.0553 | 0 |
| disturbed | 3/3 | 0.0138 | 0.0548 | 0 |

Full per-trial records are in `results/robustness/trials.csv`; each run includes its raw time series, parameters and plot.

| Condition | Seed | EKF position RMSE (m) | Odometry-only RMSE (m) | Heading RMSE (deg) | Mean NEES |
|---|---:|---:|---:|---:|---:|
| nominal | 1 | 0.0209 | 0.0134 | 0.745 | 1.475 |
| nominal | 2 | 0.0539 | 0.0134 | 1.371 | 6.020 |
| nominal | 3 | 0.0193 | 0.0135 | 0.523 | 1.306 |
| noisy | 1 | 0.0331 | 0.0134 | 1.127 | 2.238 |
| noisy | 2 | 0.0428 | 0.0278 | 1.264 | 2.625 |
| noisy | 3 | 0.0479 | 0.0140 | 2.235 | 9.666 |
| dropout | 1 | 0.0099 | 0.0133 | 0.312 | 0.075 |
| dropout | 2 | 0.0100 | 0.0133 | 0.306 | 0.044 |
| dropout | 3 | 0.0100 | 0.0134 | 0.310 | 0.060 |
| disturbed | 1 | 0.0118 | 0.0156 | 0.394 | 1.006 |
| disturbed | 2 | 0.0169 | 0.0156 | 0.518 | 0.929 |
| disturbed | 3 | 0.0126 | 0.0155 | 0.328 | 1.008 |

### Interpretation

The camera does **not** improve position RMSE in every trial. Nominal wheel odometry is particularly accurate after the contact-model correction, while far/oblique planar-marker estimates can introduce pose bias and ambiguity. The nominal seed-2 and noisy seed-3 mean NEES values exceed the state dimension of three, showing that local covariance and innovation gating do not establish consistency. The camera-blackout condition can have lower RMSE because it avoids some poorer early views. This is a limitation to discuss and improve, not a reason to hide those trials.

The disturbed condition uses a modest 1.5 N lateral force for 0.5 s, partly resisted by floor friction, plus a 0.12 m obstacle shift. It does not demonstrate recovery from a large slip. The moving-reference tracking metric contains the intentional 0.28 m lookahead and must not be described as fixed-path cross-track error.

## Camera-loss negative check

Command: `python main.py --headless --scenario no_camera --duration 25`

The robot timed out with **zero camera updates and no docking declaration**, despite reaching within 0.0669 m of the true target. Exit status 2 is expected. This confirms that arrival alone is not presented as successful visual docking.

## Independent calibration / model validation

Command: `python calibrate.py`

| Motion | Commanded wheel speed (rad/s) | Identified quantity | Measured value (m) |
|---|---:|---|---:|
| straight | 1.5 | radius | 0.032775 |
| straight | 3.0 | radius | 0.032777 |
| straight | 4.5 | radius | 0.032777 |
| spin | 1.5 | separation | 0.287997 |
| spin | 3.0 | separation | 0.287990 |
| spin | 4.5 | separation | 0.287985 |

The default 0.033/0.288 m parameters were not silently replaced by fitted values. The measured radius reflects contact compliance as well as geometry.

Independent mixed-motion prediction: position RMSE **0.00182 m**, heading RMSE **0.114°**.

All **5/5** rendered camera validation views were detected. Maximum pose errors were **0.00527 m** and **0.283°**. These views use no added corner jitter, though rendering and subpixel detection still introduce error. Intrinsics were derived analytically from the ideal MuJoCo camera; this is not a real-lens calibration.

## Reproduction limits

Linux/EGL and the pinned direct dependencies were used. Windows/macOS viewer execution was not tested. Seeds reproduce the random draws; different MuJoCo, numerical-library or graphics-driver versions can change contact integration or rasterised marker corners. The application and results are suitable for inspection and further development; project-page publication, the team video and actual member contributions still need to be supplied by the team.

## Clean-install and extracted-package check

The ZIP was extracted into a separate directory. A new Python virtual environment installed only `requirements.txt`, and the extracted `main.py` was run from outside the project directory with `--headless --quiet --duration 30`. It completed docking in 19.15 s with zero collisions and the same numerical metrics as nominal seed 1. The metrics and package versions from that check are retained in `docs/clean_install_metrics.json` and `docs/clean_install_config.json`. The archive was checked for ZIP integrity and case-only duplicate paths.
