# Applied integration fixes

This archive was repaired from the supplied Part A–D snapshots and the integration design already present in `docs/`.

## Runtime structure

A single lowercase `main.py` now launches a real `track_a/` package. The package contains central configuration, the shared midpoint motion model, frame utilities, sensor-noise handling, the Part C EKF, LiDAR mapping/A*, the non-ROS state-feedback controller, MuJoCo simulation, ArUco sensing, truth-only evaluation and the integrated runner.

## Important fixes applied

- Added the missing `track_a` package expected by `main.py` and the regression tests.
- Removed the root `Main.py` / `main.py` case mismatch; the entry point is now lowercase `main.py`.
- Centralised robot geometry at wheel radius `0.033 m` and wheelbase `0.288 m`.
- Reused the Part A midpoint model and its exact Jacobians inside the Part C EKF.
- Added zero-range protection to the range/bearing model and Jacobian.
- Fixed Part B marker rotation assignment, quaternion validation and frame-cycle detection.
- Retained LiDAR invalid-return handling and median filtering.
- Added a MuJoCo scene containing the Waffle Pi, walls, obstacles, charging station, wheel encoders, LiDAR site, RGB camera and a real rendered ArUco ID 7 marker.
- Added 180-beam LiDAR ray casting with failed-scan safe handling.
- Changed EKF prediction to use measured encoder increments rather than commanded motion.
- Added encoder-derived input covariance and time-scaled continuous process noise.
- Replaced explicit covariance inverses with linear solves.
- Replaced `(I-KH)P` with the Joseph covariance update and symmetry protection.
- Added NIS innovation gates for pose and range/bearing measurements.
- Added rendered-image ArUco detection, PnP, frame conversion and measurement covariance.
- Camera loss now produces prediction-only EKF operation; docking cannot be declared without a recent accepted camera observation.
- Replaced the ROS 1 Part D runtime with a direct Python estimated-state controller.
- Restored lateral correction using nonzero forward reference speed.
- Replaced time-based waypoint publishing with LiDAR occupancy mapping, obstacle inflation, A* and geometric lookahead.
- Added local front-LiDAR slow/stop safety and complete-scan failure stop.
- Added explicit navigation/docking controller modes and docking dwell reset logic.
- Wheel-speed saturation uses one common scale so curvature is preserved.
- Added truth-only RMSE/NEES/docking/collision/control-effort evaluation and output logs.
- Added root `pytest.ini` so historical/reference snapshots are not collected as the integrated application.
- Added compatibility wrappers in Part B/C/D folders so old imports point to the integrated implementation rather than stale duplicate logic.

## Verification performed in the repair environment

The following tests were executed successfully:

```text
python -m pytest -q tests/test_frames.py tests/test_noise.py tests/test_integration_contracts.py
37 passed
```

`part_a_check.py` also reports numerical Jacobian errors below `1.5e-10`, controllability rank 2 at rest / 3 while moving, and the expected one- vs two-landmark observability ranks.

The two tests in `tests/test_physical_sensors.py` require the `mujoco` Python package and an offscreen rendering backend. MuJoCo could not be installed in the repair container because that environment has no package-network access, so those two physical/rendering checks were not re-run here. They remain in the archive and should be run in the project's `.venv` after `pip install -r requirements.txt`.

The pre-existing `results/` and older numerical claims in `docs/TEST_RESULTS.md` came from a prior integrated build and should be treated as historical until the current repaired package is re-run on the team's machine.

## Charging-zone perimeter and random start

- Added a configurable rectangular docking/charging acceptance zone centred on the robot docking goal.
- Added a green visual-only perimeter in the MuJoCo scene. It has no collision and is excluded from LiDAR returns.
- Added `Config.in_docking_zone(...)` and `Config.docking_zone_bounds` for deterministic IN/OUT checks.
- Runtime status now prints `dock_box=IN/OUT` from the estimated pose.
- Evaluation logs both estimated and ground-truth zone membership and saves final/ever/time-in-zone metrics.
- Normal runs now use a seed-reproducible random x/y start in an obstacle-free region on the left side of the room.
- The EKF and odometry are initialised from the same known commanded start pose; ground truth is not used for this initialisation.
- Added `--fixed-start` for deterministic debugging using `Config.start`.
- Added regression tests for zone membership and reproducible random starts.
