# Integrated Project — Track A

TurtleBot3 Waffle Pi visual waypoint navigation for 41014 Sensors and Control in Mechatronics Systems.

The integrated runtime is in `track_a/`. The older `PartA/`, `PartB/`, `PartC/`, `PartD/` and `reference_sources/` folders are retained as provenance/compatibility material; they are **not** separate applications.

## Runtime pipeline

`MuJoCo encoders -> EKF prediction -> rendered ArUco camera -> EKF correction -> estimated pose -> LiDAR mapping/A* -> state-feedback controller -> wheel commands -> MuJoCo`

Ground truth is isolated to evaluation/calibration and is never passed to the estimator, planner or controller.

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\steph\OneDrive\Desktop\sensors and cotrol\Integrated-Project-Track-A-main"
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code Python interpreter.

## Run

```powershell
python main.py --headless --scenario nominal --duration 65
# Use the original deterministic start only when debugging:
python main.py --headless --scenario nominal --duration 65 --fixed-start
python main.py --headless --scenario no_camera --duration 25
python main.py
```

## Tests

```powershell
python -m pytest -q
```

## Important files

- `track_a/config.py` — one shared set of robot, sensor and controller parameters.
- `track_a/model.py` — midpoint differential-drive model and Jacobians.
- `track_a/frames.py` — world/base/camera/LiDAR/marker transforms.
- `track_a/noise.py` — LiDAR cleaning, ArUco gating and encoder covariance.
- `track_a/estimation.py` — Part C EKF using encoder prediction and camera correction.
- `track_a/sensing.py` — rendered ArUco detection and PnP pose recovery.
- `track_a/planning.py` — LiDAR occupancy map and inflated-grid A*.
- `track_a/controller.py` — estimated-state path tracking and visual docking.
- `track_a/simulator.py` — MuJoCo robot, encoders, LiDAR rays, RGB camera and disturbance scenarios.
- `track_a/runner.py` — the single end-to-end integration loop.
- `track_a/evaluation.py` — truth-only metrics and output plots/logs.

See `docs/DESIGN.md` and `docs/INTEGRATION.md` for the reasoning behind the model, frames, EKF, controller and integration fixes.

## Random start and charging-zone box

Normal runs now sample a reproducible random x/y start on the obstacle-free left side of the room.
The `--seed` controls the start as well as sensor noise/dropout, so the same seed repeats exactly.
Use `--fixed-start` to return to `Config.start` for debugging.

A green visual-only rectangle marks the charging acceptance zone in front of the station.
Its bounds are configured by `docking_zone_center` and `docking_zone_half_size` in `track_a/config.py`.
Runtime status prints `dock_box=IN/OUT`, and `metrics.json` records estimated and true zone membership.
