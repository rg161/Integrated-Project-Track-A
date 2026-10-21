# Integration decisions and fixes

## Source selection

The six supplied archives were read separately before combining them. Their
original Python/text files are in `reference_sources/`. Relevant original robot
XML files are also retained there. Unrelated manipulator/OP3 models from the
TurtleBot archive are not dependencies of this project and are omitted from the
deliverable. The required Waffle Pi meshes and their licence are included.

| Finding in the supplied sections | Integrated resolution |
|---|---|
| Several empty `Main.py` files | One lowercase `main.py`, avoiding case-only duplicate filenames on Windows |
| Environment ran two identical random-action loops | One scheduled simulation loop, explicit close/cleanup |
| Environment was a fixed blue sphere with no actuators | Actual free-base Waffle Pi, wheel actuators and sensor interfaces |
| Environment path assumed `Integrated-Project-Track-A/Scene.xml` beneath current directory | Paths derive from `__file__`; scene assembled from the robot XML |
| `Frames.py` versus `from frames import ...` | Consistent lowercase package imports across platforms |
| Marker rotation branch did not assign `R` | Correct assignment, with a regression test |
| Zero quaternion could divide by zero | Explicit finite/nonzero quaternion validation |
| Frame-tree cycles could recurse indefinitely | Cycle/root/self-edge checks |
| TurtleBot copy's F/G array expressions were malformed | Use the corrected Part A version and check against finite differences |
| Part A used midpoint integration, Part C forward Euler | EKF and odometry both use Part A's midpoint model and Jacobians |
| Part A fitted r=0.0283, L=0.353; TurtleBot used nominal geometry | Fix contact behaviour, retain 0.033/0.288 geometry, rerun independent calibration |
| Mesh collision shape and wheel joint friction distorted motion | Rounded analytic wheel contacts, chassis collision box, joint friction 0.002 and armature 0.0002 |
| Rear supports intended to be frictionless were overridden by floor contact defaults | Set their contact priority to 1, so their `condim=1` governs the floor pair |
| Camera, encoder and LiDAR acquisition were absent | Named joint-position sensors, 360° ray casting and real RGB rendering |
| No rendered/detected ArUco marker | Marker cells generated from OpenCV dictionary in the 3-D scene, then detected from image pixels |
| Part C used fixed per-step Q and direct matrix inverses | Time-scaled process noise, encoder-derived input covariance, linear solves and Joseph covariance update |
| No innovation gating | Chi-square-style NIS thresholds: 16.27 for full pose, 13.82 for range/bearing |
| Part D required ROS 1 topics and used an odometry-relay stub | Controller receives the real Part C filter output directly; stub is reference-only |
| `run_partd.py` expected `ekf.py`, but archive supplied `efk.py` | No subprocess launcher or ambiguous duplicate estimator in runtime |
| Waypoint publisher advanced every second regardless of progress | LiDAR map, A* path, geometric lookahead and explicit docking states |
| v_ref=0 removed the original law's lateral correction | Nonzero feedforward during tracking; turn-to-path mode and docking position/heading feedback |
| Controller could cut corners or become trapped in an inflated start cell | Reference points toward lookahead; hysteretic turning; permit a monotonic escape through extra planning margin while preserving the footprint bound |
| Close marker was missed by default OpenCV contour grouping | Narrower candidate grouping, verified against a real rendered close-up |
| No integrated evidence | Seeded trials, versioned configuration snapshots, logs, plots, calibration and regression tests |



## Truth isolation

`track_a/ground_truth.py` is imported only by evaluation and offline calibration.
`track_a/simulator.py` necessarily accesses physical state to emulate optical
rays, render images, apply forces and evaluate contacts. It exposes only sensor
samples to the runner. Neither the controller nor estimator imports MuJoCo or
receives the simulator object. The runner passes `ekf.get_state()` and
`ekf.get_covariance()` to control and mapping.

The configured start pose is a prior known before the trial. It is not
continually reset from the simulator. The offline calibration script does use
ground truth, and is explicitly separate from the navigation pipeline.

## Scope of the original files

Reference snapshots are intentionally not repaired: they document the starting
point. Do not run them as the integrated application. Original results quoted in
those READMEs are historical source claims. New measurements are under `results/`
and in `docs/TEST_RESULTS.md`.

## Primary implementation references

- MuJoCo contact parameter rules:
  <https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters>
- MuJoCo ray casting API:
  <https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-ray>
- OpenCV ArUco detection and PnP:
  <https://docs.opencv.org/5.0/tutorials/objdetect/aruco_detection/aruco_detection.html>

The attached Sensors and Control marking guide informed the separation of
modelling, sensing, estimation, control and integration evidence. Meeting every
submission requirement still requires the team's contribution statement,
project page, video and individual understanding.
