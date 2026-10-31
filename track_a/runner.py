"""Run the complete Track A loop: sensors -> EKF -> planning -> control -> robot."""

from pathlib import Path
import time as walltime
import numpy as np

from .config import sample_start_pose
from .controller import StateFeedbackController
from .estimation import MobileRobotEKF
from .evaluation import Evaluator
from .ground_truth import GroundTruth
from .model import f, v_w_to_wheels, wheels_to_v_w
from .noise import clean_scan
from .planning import LidarPlanner
from .sensing import ArucoCamera
from .simulator import Simulator


def run(config, trial, seed, output, viewer=False):
    """Run one reproducible simulation and return its quantitative result dictionary."""
    output = Path(output)
    rng_enc = np.random.default_rng(seed * 10 + 1)
    rng_lidar = np.random.default_rng(seed * 10 + 2)
    rng_camera = np.random.default_rng(seed * 10 + 3)
    rng_dropout = np.random.default_rng(seed * 10 + 4)
    rng_start = np.random.default_rng(seed * 10 + 5)
    start_pose = sample_start_pose(config, rng_start)

    planner = LidarPlanner(config)
    controller = StateFeedbackController(config)
    camera = ArucoCamera(config)
    ekf = MobileRobotEKF(start_pose)
    odom = start_pose.copy()
    evaluator = Evaluator(config, trial, seed, output, start_pose=start_pose)

    viewer_handle = None
    last_camera_image = None
    control = None

    with Simulator(config, trial, seed=seed, start_pose=start_pose) as sim:
        truth = GroundTruth(sim.model, sim.data)
        xmin, xmax, ymin, ymax = config.docking_zone_bounds
        print(
            f"layout={config.layout} scenario={trial.name} seed={seed} "
            f"start=({start_pose[0]:+.2f},{start_pose[1]:+.2f}) "
            f"dock_zone=x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}]",
            flush=True,
        )

        if viewer:
            try:
                import mujoco.viewer

                viewer_handle = mujoco.viewer.launch_passive(sim.model, sim.data)
            except Exception as exc:
                print(f"Viewer unavailable ({exc}); continuing headless.")
                viewer_handle = None

        initial = sim.sample()
        prev_encoder = initial.wheel_angles + rng_enc.normal(
            0.0, config.encoder_sigma * trial.noise_scale, 2
        )
        latest_scan = initial.lidar_ranges.copy()
        finite = np.isfinite(latest_scan)
        latest_scan[finite] += rng_lidar.normal(
            0.0,
            config.lidar_sigma * trial.noise_scale,
            np.sum(finite),
        )
        latest_scan = clean_scan(latest_scan, r_max=config.lidar_max, window=3)
        planner.update(ekf.get_state(), latest_scan)
        path = planner.plan(ekf.get_state()[:2], config.goal[:2])

        camera_age = np.inf
        latest_marker_bearing = None
        next_lidar = 0.0
        next_camera = 0.0
        next_status = 0.0
        done = False
        termination = "timeout"

        while sim.time < config.max_duration:
            loop_start = walltime.perf_counter()
            sample = sim.sample()
            t = sample.time

            # 1) Wheel encoders -> EKF prediction and odometry-only comparison.
            enc = sample.wheel_angles + rng_enc.normal(
                0.0, config.encoder_sigma * trial.noise_scale, 2
            )
            increments = enc - prev_encoder
            prev_encoder = enc
            ekf.predict_encoders(increments, config.dt, config)
            wl, wr = increments / config.dt
            v_odo, w_odo = wheels_to_v_w(
                wl, wr, config.wheel_radius, config.wheel_base
            )
            odom = f(odom, [v_odo, w_odo], config.dt)
            camera_age += config.dt

            # 2) LiDAR -> cleaned scan -> occupancy map -> A* path.
            if t + 1e-10 >= next_lidar:
                raw = sample.lidar_ranges.copy()
                valid = np.isfinite(raw)
                raw[valid] += rng_lidar.normal(
                    0.0,
                    config.lidar_sigma * trial.noise_scale,
                    np.sum(valid),
                )
                latest_scan = clean_scan(raw, r_max=config.lidar_max, window=3)
                if np.any(np.isfinite(latest_scan)):
                    planner.update(ekf.get_state(), latest_scan)
                path = planner.plan(ekf.get_state()[:2], config.goal[:2])
                next_lidar += config.lidar_period

            # 3) RGB camera -> ArUco/PnP -> NIS-gated EKF correction.
            if t + 1e-10 >= next_camera:
                in_blackout = trial.blackout[0] <= t <= trial.blackout[1]
                dropped = in_blackout or rng_dropout.random() < trial.camera_dropout
                if not dropped:
                    last_camera_image = sim.camera_image()
                    measurement = camera.measure(
                        last_camera_image,
                        rng_camera,
                        noise_scale=trial.noise_scale,
                    )
                    if measurement is not None:
                        # Marker visibility and EKF acceptance are deliberately
                        # tracked separately. A geometrically valid marker can
                        # still be NIS-rejected by the EKF, but it remains useful
                        # for keeping the camera pointed at the charging tag.
                        camera_age = 0.0
                        latest_marker_bearing = float(measurement.bearing)
                        accepted = ekf.update_pose(measurement.pose, measurement.R)
                        evaluator.note_camera(accepted)
                    elif camera.last_reason not in {"not_found", "wrong_id"}:
                        evaluator.note_camera(False)
                next_camera += config.camera_period

            # 4) Estimated state + path + LiDAR -> control -> wheel speeds.
            x_hat = ekf.get_state()
            P = ekf.get_covariance()
            control = controller.command(
                x_hat,
                P,
                path,
                latest_scan,
                camera_age,
                marker_bearing=latest_marker_bearing,
            )
            wheels = v_w_to_wheels(
                control.v,
                control.omega,
                config.wheel_radius,
                config.wheel_base,
                limit=config.wheel_limit,
            )
            sim.step(wheels)

            # 5) Ground truth is read only here for evaluation, never for control.
            true_pose = truth.pose()
            reference = np.asarray(control.reference, dtype=float)
            if reference.shape != (3,) or not np.all(np.isfinite(reference)):
                reference = np.asarray(config.goal, dtype=float)
            evaluator.record(
                t=sim.time,
                estimate=x_hat,
                P=P,
                truth=true_pose,
                odom=odom,
                reference=reference,
                wheels=wheels,
                command=(control.v, control.omega),
                mode=control.mode,
                nis=ekf.last_nis,
            )

            if viewer_handle is not None:
                try:
                    viewer_handle.sync()
                    if not viewer_handle.is_running():
                        termination = "viewer_closed"
                        break
                except Exception:
                    pass

            if sim.time >= next_status:
                zone = "IN" if config.in_docking_zone(x_hat) else "OUT"
                bearing_text = (
                    f"{np.degrees(latest_marker_bearing):+.0f}deg"
                    if latest_marker_bearing is not None
                    and np.isfinite(latest_marker_bearing)
                    and camera_age <= config.camera_max_age
                    else "---"
                )
                print(
                    f"t={sim.time:5.1f}s mode={control.mode:20s} "
                    f"est=({x_hat[0]:+.2f},{x_hat[1]:+.2f},"
                    f"{np.degrees(x_hat[2]):+.0f}deg) "
                    f"dock={zone:3s} cam_age={camera_age:4.1f}s "
                    f"bearing={bearing_text:>7s} "
                    f"path_pts={len(path):2d} collisions={sim.collisions}",
                    flush=True,
                )
                next_status += 2.0

            if control.done:
                done = True
                termination = "docked"
                break

            if viewer_handle is not None:
                elapsed = walltime.perf_counter() - loop_start
                if elapsed < config.dt:
                    walltime.sleep(config.dt - elapsed)

        if viewer_handle is not None:
            try:
                viewer_handle.close()
            except Exception:
                pass

        metrics = evaluator.metrics(
            controller_done=done,
            collisions=sim.collisions,
            termination=termination,
        )
        evaluator.save(
            metrics,
            planner=planner,
            camera_image=last_camera_image,
            obstacle_layout=sim.obstacle_layout,
        )
        return metrics
