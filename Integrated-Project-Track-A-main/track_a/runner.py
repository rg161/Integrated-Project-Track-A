"""One end-to-end pipeline. Truth goes only to the evaluator."""

import time
from pathlib import Path
import cv2
import numpy as np
from .simulator import Simulator
from .sensing import ArucoCamera
from .estimation import MobileRobotEKF
from .planning import LidarPlanner
from .controller import StateFeedbackController
from .evaluation import Evaluator
from .model import f
from .noise import wheel_increments_to_motion, clean_scan


def run(config, trial, seed, output, viewer=False, realtime=False, verbose=True):
    pose = np.array(config.start, dtype=float)
    ekf = MobileRobotEKF(pose)
    odometry = pose.copy()
    camera = ArucoCamera(config)
    planner = LidarPlanner(config)
    controller = StateFeedbackController(config)
    camera_rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    last_camera = -float("inf")
    ranges = None
    last_wheels = None
    next_plan = 0.0
    done = False
    termination = "timeout"
    last_mode = None
    with Simulator(config, trial, seed, viewer) as sim:
        evaluator = Evaluator(sim, config, trial, seed, output)
        while True:
            wall_start = time.perf_counter()
            packet = sim.sample()
            if last_wheels is not None:
                increments = packet.wheel_angles - last_wheels
                ekf.predict_encoders(increments, config.dt, config, trial.noise_scale)
                ds, dtheta = wheel_increments_to_motion(
                    *increments, config.wheel_radius, config.wheel_base
                )
                odometry = f(odometry, [ds / config.dt, dtheta / config.dt], config.dt)
            last_wheels = packet.wheel_angles.copy()
            seen = False
            if packet.image is not None:
                measurement = camera.measure(
                    packet.image, camera_rng, trial.noise_scale
                )
                if measurement is not None:
                    seen = ekf.update_pose(measurement.pose, measurement.covariance)
                    if seen:
                        last_camera = packet.time
                if seen or not (Path(output) / "camera.png").exists():
                    annotated = packet.image.copy()
                    if camera.last_corners is not None:
                        cv2.polylines(
                            annotated,
                            [np.rint(camera.last_corners).astype(np.int32)],
                            True,
                            (0, 200, 80),
                            2,
                        )
                    cv2.imwrite(
                        str(Path(output) / "camera.png"),
                        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                    )
            estimate = ekf.get_state()
            P = ekf.get_covariance()
            if packet.ranges is not None:
                # Keep infinities as no-return rays so free-space carving can reach max range.
                ranges = clean_scan(packet.ranges, r_min=0.04, r_max=config.lidar_max)
                ranges[np.isinf(packet.ranges)] = np.inf
                planner.update(estimate, ranges)
            if packet.time + 1e-9 >= next_plan:
                planner.plan(estimate[:2], config.goal[:2])
                next_plan = packet.time + 0.5
            control = controller.command(
                estimate, P, planner.path, ranges, packet.time - last_camera
            )
            evaluator.record(
                packet.time,
                estimate,
                P,
                odometry,
                control,
                packet.time - last_camera,
                ekf,
                seen,
            )
            if verbose and control.mode != last_mode:
                print(
                    f"t={packet.time:5.1f}s  {control.mode:18s}  estimated=({estimate[0]:+.2f}, {estimate[1]:+.2f})  camera updates={ekf.accepted}",
                    flush=True,
                )
                last_mode = control.mode
            if control.done:
                done = True
                termination = "docked"
                break
            if packet.time + 1e-9 >= config.max_duration:
                break
            if sim.viewer is not None and not sim.viewer.is_running():
                termination = "viewer_closed"
                break
            sim.step(control.wheels)
            if realtime:
                time.sleep(max(0, config.dt - (time.perf_counter() - wall_start)))
        metrics = evaluator.save(sim, ekf, done, termination, planner)
    return metrics
