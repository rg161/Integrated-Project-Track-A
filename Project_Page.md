# Integrated Project Track A

## Group Number
Group Number: [Insert group number]

## Team Members
- Rohan
- Stephen
- Daniel
- Jake

## GitHub Repository
Repository: https://github.com/rg161/Integrated-Project-Track-A
Running branch: backuprunning

## Project Overview
This project develops a mobile robot navigation and docking system for a TurtleBot operating in a simulated obstacle-filled room. The robot uses LiDAR for mapping and obstacle avoidance, and a camera with an ArUco marker for visual docking. The system estimates the robot pose online, plans a collision-free route, tracks the planned path, and docks at the charging station while handling sensor noise, partial camera dropouts, and minor disturbances.

The robot state is a planar unicycle pose:
- x, y: position in the world frame
- theta: heading angle

The main task is to navigate from a start pose to the goal while avoiding obstacles and visually aligning with the charging marker using the camera.

## System Diagram / Coordinate Frames

The system uses the following frames:
- W: world frame
- B: robot base frame
- C: camera frame
- M: ArUco marker frame
- L: LiDAR frame

Simple reference diagram:

```text
        World (W)
            |
            | known transform
            v
      Robot Base (B)
      /      |      \
     /       |       \
    L        C        wheel joints
             |
             v
         Marker (M)
```

The LiDAR is mounted at a fixed known transform relative to the robot base. The camera observes the ArUco marker, and the pose of the marker relative to the camera is recovered using OpenCV detection and solvePnP. This gives a measurement update for the estimator.

## Model and Parameters
The robot follows the standard unicycle model:

- State: x = [x, y, theta]^T
- Input: u = [v, omega]^T

The implementation uses the supplied midpoint discretisation consistent with the project design:

- wheel radius: 0.033 m
- wheel separation: 0.288 m
- dt: 0.05 s

The robot is validated against MuJoCo with the following representative model parameters:
- nominal wheel radius = 0.033 m
- nominal wheel separation = 0.288 m
- LiDAR and odometry noise are added as part of the simulated sensing pipeline

The project design also reports calibration checks showing the effective wheel size is consistent with the chosen simulation model and that the robot motion matches the physical model to a good degree.

## Estimation
The estimator is an Extended Kalman Filter (EKF) that predicts the robot pose using the unicycle motion model and updates using sensor measurements.

Key implementation points:
- Prediction uses wheel encoder increments and the discrete motion model.
- The process covariance includes odometry uncertainty and additional continuous process noise.
- The filter uses measurement updates from the detected camera marker pose and LiDAR-informed map constraints where relevant.
- Innovation gating is used to reject extreme outliers.
- The covariance update follows Joseph-form correction for numerical stability.

Important design note: ground truth from the simulator is only used for evaluation and validation. It is not used directly in the real controller or estimator loop.

This is visible in the code structure and documentation, especially in the integration notes and evaluation workflow.

## Controller Design
The controller is a state-feedback planner and dock controller built around the estimated robot state.

Main control strategy:
- LiDAR occupancy map is built from the environment.
- A* path planning is performed on an inflated cost map.
- A local reference point is selected ahead of the robot along the path.
- Heading toward the look-ahead point is used for path tracking.
- A receding-horizon reference point is used rather than a fixed trajectory.

Key parameters from the project design:
- look-ahead distance: 0.28 m
- turn-in-place threshold: entering at 0.65 rad, leaving at 0.35 rad
- docking region: within 0.28 m of goal
- tighter stopping distance: translational motion reduced below 0.50 m, fully stopped below 0.24 m
- position uncertainty safeguard: excessive uncertainty halts motion

The docking controller reduces forward speed as the robot approaches the target and rotates to align with the desired heading before declaring completion. A dwell condition based on accepted camera data is required before docking is considered complete.

## Project Results
The repository includes quantitative validation results from simulation.

### Automated checks
- pytest: 39 passed
- ruff/linting checks passed for runtime and test code
- Jacobian and controllability checks completed

### End-to-end trial results
The project reports the following with 3 seeds per condition:

| Condition | Successes | Mean EKF position RMSE | Mean final dock error | Collisions |
|---|---:|---:|---:|---:|
| nominal | 3/3 | 0.0314 m | 0.0543 m | 0 |
| noisy | 3/3 | 0.0413 m | 0.0503 m | 0 |
| dropout | 3/3 | 0.0100 m | 0.0553 m | 0 |
| disturbed | 3/3 | 0.0138 m | 0.0548 m | 0 |

Additional validation indicates the robot reaches the target and successfully docks in the tested scenarios, with the controller and EKF remaining robust to moderate noise and disturbance. The no-camera negative test also confirms that docking does not occur based on proximity alone without valid camera confirmation.

## Install and Run
The project is organised for a fresh Python environment and can be run from a clean setup using the repository requirements.

### Requirements
The repository includes a pinned dependency list in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the simulator
From the repository root:

```bash
python main.py
```

For headless execution:

```bash
python main.py --headless --scenario nominal --duration 25
```

For a different scenario:

```bash
python main.py --scenario noisy
python main.py --scenario dropout
python main.py --scenario disturbed
python main.py --scenario no_camera
```

The code is structured so that the runtime entry point is the main script in the project root, and the simulation/evaluation pipeline is separated from the estimator/controller logic.

## Demo Video
Demo video: Demo Vid

## Individual Contribution Statement
- Rohan: Part A - Modelling, state-space model analysis, observability and controllability analysis.
- Stephen: [Insert contribution]
- Daniel: [Insert contribution]
- Jake: [Insert contribution]

## Summary
This project combines LiDAR-based navigation, visual marker docking, and an EKF-based state estimator into a complete autonomous mobile robot pipeline. It demonstrates a robust navigation-and-docking workflow in simulation with quantitative validation, clear coding structure, and explicit separation between ground-truth evaluation and online estimation/control.

The implementation is designed to be run from a fresh environment, uses the code in this repository as the core system, and is suitable for a project page summary with technical discussion and results.

---

Project Page prepared for the repository: https://github.com/rg161/Integrated-Project-Track-A
