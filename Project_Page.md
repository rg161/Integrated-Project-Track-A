# Integrated Project — Track A

## Team

**Group number:** [add group number]

- Rohan
- Stephen
- Daniel
- Jake

**Repository:** [add final GitHub repository link]

## Project overview

This project implements **Track A: mobile robot visual waypoint navigation** using a simulated TurtleBot3 Waffle Pi in MuJoCo. The robot navigates an obstacle-filled room using 2D LiDAR, estimates its pose using an Extended Kalman Filter (EKF), and visually docks at an ArUco-marked charging station. The controller acts only on the **estimated state**, never simulator ground truth.

Two environment layouts are included:

- **standard** — the original two fixed obstacles,
- **random** — six seed-reproducible obstacles with random positions/sizes, while keeping the start and docking areas clear.

The robot start x/y position is also randomised by seed.

## System pipeline

```text
MuJoCo wheel encoders ──► EKF prediction ───────────────┐
                                                        │
MuJoCo RGB camera ─► ArUco/PnP pose ─► EKF correction ─┤
                                                        ▼
                                              estimated [x,y,theta], P
                                                        │
MuJoCo LiDAR ─► cleaning ─► occupancy map ─► A* path ──┤
                                                        ▼
                                                state-feedback control
                                                        │
                                                        ▼
                                                   wheel commands
                                                        │
                                                        ▼
                                                      MuJoCo

MuJoCo ground truth ─────────────────────────────► evaluation only
```

## Part A — modelling

The planar state and input are:

```text
state: x = [x, y, theta]^T
input: u = [v, omega]^T
```

The midpoint discrete unicycle model is used:

```text
theta_mid = theta + 0.5*omega*dt
x_next     = x + v*cos(theta_mid)*dt
y_next     = y + v*sin(theta_mid)*dt
theta_next = wrap(theta + omega*dt)
```

Key parameters:

- wheel radius: **0.033 m**
- wheel separation: **0.288 m**
- control/estimation timestep: **0.05 s**

`model.py` contains the shared motion model plus the analytical state/input Jacobians used by the EKF. The regression tests compare those Jacobians against finite differences.

### Observability / controllability interpretation

The full camera pose measurement is `[x, y, theta]`, so its local measurement matrix is `H = I3` and all three states are directly observable while the marker pose measurement is available. Wheel encoders provide motion increments but do not independently remove global position/heading drift, which is why the camera correction is needed. Differential-drive motion is controlled through `v` and `omega`; translation and rotation over time allow the mobile robot to reach planar poses subject to the non-holonomic no-sideways-motion constraint.

## Part B — sensing and frames

### LiDAR

- 180 simulated beams over 360 degrees
- maximum range: **3.5 m**
- nominal range noise standard deviation: **0.008 m**
- scan period: **0.10 s**
- invalid/out-of-range values are rejected and valid neighbours are median filtered

### Camera / ArUco

- image size: **480 x 360**
- vertical FOV: **58 degrees**
- marker ID: **7**
- marker side: **0.24 m**
- camera period: **0.25 s**

The camera pipeline detects the marker, solves its pose with OpenCV `solvePnP`, transforms the result through the camera/base/world frames, rejects poor detections using range/size/incidence/reprojection checks, and estimates a 3x3 pose covariance using finite differences of image-corner locations.

## Part C — estimation

The EKF estimates:

```text
x_hat = [x, y, theta]^T
```

Prediction uses measured wheel-encoder increments rather than commanded wheel speed. Encoder/odometry uncertainty and continuous process noise are propagated through the model Jacobians.

When the ArUco pose is available, the EKF performs a full-pose correction. The correction includes:

- heading innovation wrapping,
- NIS measurement gating (`16.27`, approximately 99.9% chi-square for 3 DOF),
- `solve()` rather than an explicit matrix inverse,
- Joseph-form covariance update,
- covariance symmetry/eigenvalue protection.

Outputs used by the rest of the pipeline are:

```text
estimated state x_hat = [x, y, theta]
covariance P
```

## Part D — control

The planner creates a LiDAR occupancy map, inflates occupied cells by the robot radius plus safety margin, and computes an A* route to the docking goal.

The controller receives the estimated state/covariance, current path, LiDAR scan and camera age. It performs:

- turn-to-path alignment,
- estimated-state path tracking,
- docking approach,
- final heading alignment,
- local LiDAR slow/stop protection,
- recent-camera confirmation before declaring docking complete.

Key values include:

- maximum forward speed: **0.18 m/s**
- maximum yaw rate: **1.2 rad/s**
- path gains: `k1=0.8`, `k2=2.0`, `k3=1.8`
- lookahead: **0.28 m**
- docking goal tolerance: **0.055 m** and **5 degrees**

## Part E — integration and robustness

The same pipeline can be tested under:

```text
nominal
noisy
dropout
disturbed
```

and under both:

```text
standard layout
random six-obstacle layout
```

The random layout is reproducible from `--seed`, and its exact obstacle coordinates/sizes are saved in `run_config.json`.

Ground truth is isolated to `ground_truth.py` and is passed only to `evaluation.py`. It is never imported by sensing, estimation, planning or control.

The evaluator reports:

- EKF position RMSE,
- EKF heading RMSE,
- odometry-only position RMSE,
- reference tracking RMSE,
- final docking position/heading error,
- mean NEES and 2-sigma coverage,
- control effort,
- collision count,
- accepted/rejected camera updates,
- docking-zone entry and task success.

## Quantitative results

**Replace this table with fresh results generated from the submitted build before final submission.** Use repeated seeds rather than a single visual run.

| Layout / scenario | Seeds | Success rate | Position RMSE | Final dock error | Collisions |
|---|---:|---:|---:|---:|---:|
| standard / nominal | [ ] | [ ] | [ ] | [ ] | [ ] |
| standard / noisy | [ ] | [ ] | [ ] | [ ] | [ ] |
| standard / dropout | [ ] | [ ] | [ ] | [ ] | [ ] |
| standard / disturbed | [ ] | [ ] | [ ] | [ ] | [ ] |
| random / nominal | [ ] | [ ] | [ ] | [ ] | [ ] |

For the demo video, show at least one nominal run and one varied/disturbed or random-obstacle run, and show the estimate/uncertainty alongside ground truth in the quantitative result plots.

## Install and run

### Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Normal visual simulation

```powershell
python main.py
```

### Random-obstacle layout

```powershell
python main.py --layout random --seed 2
```

### Headless robustness test

```powershell
python main.py --headless --scenario noisy --seed 2
```

### Automated tests

```powershell
python -m pytest -q
```

## Demo video

**5-minute demo:** [add final video link]

## Individual contributions

See `Contribution Statement`.

## Generative AI declaration

OpenAI ChatGPT was used to assist with code integration review, debugging, refactoring, comments/docstrings, automated-test generation and documentation. Team members are responsible for reviewing, testing and being able to explain the submitted implementation and results.

## Environment layouts

The standard layout uses the original docking target on `y = 0.00 m`. The random layout uses six seed-randomised obstacles and relocates the complete charging target (dock, perimeter and ArUco marker) to `y = 0.80 m`. This gives the second layout a different final visual-docking location while retaining the same sensing, EKF and controller pipeline.


### Random-layout navigability protection

The random environment validates each generated obstacle set before the run. Obstacles are inflated by the same robot radius plus safety margin used by the runtime planner, and the layout is accepted only when a grid route remains from the sampled start to the docking goal. The controller also holds with `no_path` if A* cannot provide a route, and its local LiDAR hard-stop keeps linear velocity at zero while rotating toward the clearer side to recover from a close obstacle.
