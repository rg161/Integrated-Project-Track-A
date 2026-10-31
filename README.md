# Track A — TurtleBot3 Visual Waypoint Navigation

A minimal simulation-only implementation for **41014 Sensors and Control in Mechatronics Systems — Assessment 4**.

The robot uses:

- wheel encoders for EKF prediction,
- a rendered RGB camera + ArUco marker for EKF correction,
- 2D LiDAR for occupancy mapping and obstacle avoidance,
- A* for path planning,
- an estimated-state feedback controller for path tracking and docking,
- MuJoCo ground truth **only for evaluation**.

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Camera and visual docking

The front camera is rigidly mounted to the TurtleBot. During normal navigation, obstacles naturally move in and out of its field of view as the robot turns. Near the charging station, the controller uses the detected ArUco marker bearing as a visual-servo correction so the marker is kept near the centre of the camera while the robot aligns and docks.

## Run

The interface is intentionally small.

### Standard layout — visual viewer

```powershell
python main.py
```

### Random-obstacle layout

Six obstacles are randomised from the seed while keeping the start and dock clear.
The generated layout is path-checked, and the online LiDAR map uses both obstacle hits and no-return free-space rays so stale detections cannot create permanent ghost walls.

```powershell
python main.py --layout random --seed 2
```

### Headless robustness run

```powershell
python main.py --headless --scenario noisy --seed 2
```

Available layouts:

```text
standard   original two-obstacle room
random     six seed-reproducible random obstacles
```

Available scenarios:

```text
nominal    normal sensor noise
noisy      increased sensor noise + some camera dropout
dropout    camera dropout + temporary blackout
disturbed  brief sideways force disturbance
```


Other useful options:

```text
--seed N       repeatable random start/obstacle/noise seed
--duration S   maximum simulation duration in seconds
--headless     run without the MuJoCo viewer
```

## Tests

```powershell
python -m pytest -q
```

The MuJoCo scene tests automatically skip if MuJoCo is not installed. All runtime functions and methods are required by a regression test to contain a `"""docstring"""`.

## Runtime files

```text
main.py
track_a/
├── config.py        parameters, scenarios, random start/obstacles
├── model.py         differential-drive model and Jacobians
├── frames.py        camera/marker coordinate transforms
├── noise.py         LiDAR cleaning, camera gating, odometry covariance
├── sensing.py       ArUco detection + PnP pose measurement
├── estimation.py    Part C EKF
├── planning.py      LiDAR occupancy map + A*
├── controller.py    estimated-state path tracking + docking
├── simulator.py     MuJoCo sensors, physics and collision counting
├── ground_truth.py  evaluation-only true pose
├── evaluation.py    RMSE/NEES/docking metrics + result export
└── runner.py        one integrated sensor-estimation-control loop

models/
├── track_a_scene.xml
└── robotis_tb3/assets/

tests/
├── test_core.py
└── test_simulator.py
```

## Results

Each run creates:

```text
results/<layout>_<scenario>_seed<seed>/
├── metrics.json
├── log.csv
├── run_config.json
├── map.npz
├── camera.png
└── summary.png
```

For the final project page, run repeated seeds for the standard and random layouts and report success rate, EKF RMSE, docking error, collisions, NEES/coverage, and camera acceptance/rejection.

## Layout targets

The standard layout docks at `(1.12, 0.00)` with the ArUco marker at `(1.60, 0.00, 0.24)`.
The random layout uses a different docking location at `(1.12, 0.80)` with the marker at `(1.60, 0.80, 0.24)`, while keeping the same forward-facing camera geometry.

The random layout also moves the ArUco charging target and charging-station geometry, not just the obstacles.
