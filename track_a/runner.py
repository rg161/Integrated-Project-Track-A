"""Single integrated Track A loop: sensors -> EKF -> map/plan -> controller -> robot."""

from dataclasses import asdict
from pathlib import Path
import time as walltime
import numpy as np

from .controller import StateFeedbackController
from .estimation import MobileRobotEKF
from .evaluation import Evaluator
from .ground_truth import GroundTruth
from .model import f, v_w_to_wheels, wheels_to_v_w
from .noise import clean_scan
from .planning import LidarPlanner
from .sensing import ArucoCamera
from .camera_view import CameraView
from .simulator import Simulator
from .config import sample_start_pose


def run(config, trial, seed, output, viewer=False, realtime=False, verbose=True,
        camera_video=True, camera_window=False):
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
    camera_view = CameraView(output, record=camera_video, window=camera_window, fps=1.0 / config.camera_period)
    control = None
    with Simulator(config, trial, start_pose=start_pose) as sim:
        truth = GroundTruth(sim.model, sim.data)
        if verbose:
            xmin, xmax, ymin, ymax = config.docking_zone_bounds
            print(
                f"start=({start_pose[0]:+.2f},{start_pose[1]:+.2f},{np.degrees(start_pose[2]):+.0f}deg) "
                f"dock_zone=x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}]",
                flush=True,
            )
        if viewer:
            try:
                import mujoco.viewer
                viewer_handle = mujoco.viewer.launch_passive(sim.model, sim.data)
            except Exception as exc:
                if verbose:
                    print(f"Viewer unavailable ({exc}); continuing headless.")
                viewer_handle = None

        initial = sim.sample()
        prev_encoder = initial.wheel_angles + rng_enc.normal(
            0.0, config.encoder_sigma * trial.noise_scale, 2
        )
        latest_scan = initial.lidar_ranges.copy()
        finite = np.isfinite(latest_scan)
        latest_scan[finite] += rng_lidar.normal(0.0, config.lidar_sigma * trial.noise_scale, np.sum(finite))
        latest_scan = clean_scan(latest_scan, r_max=config.lidar_max, window=3)
        planner.update(ekf.get_state(), latest_scan)
        path = planner.plan(ekf.get_state()[:2], config.goal[:2])
        camera_age = np.inf
        next_lidar = 0.0
        next_camera = 0.0
        done = False
        termination = "timeout"
        next_status = 0.0

        while sim.time < config.max_duration:
            loop_start = walltime.perf_counter()
            sample = sim.sample()
            t = sample.time

            # 1) Encoder measurement -> shared-model EKF prediction and odometry baseline.
            enc = sample.wheel_angles + rng_enc.normal(
                0.0, config.encoder_sigma * trial.noise_scale, 2
            )
            increments = enc - prev_encoder
            prev_encoder = enc
            ekf.predict_encoders(increments, config.dt, config)
            wl, wr = increments / config.dt
            v_odo, w_odo = wheels_to_v_w(wl, wr, config.wheel_radius, config.wheel_base)
            odom = f(odom, [v_odo, w_odo], config.dt)
            camera_age += config.dt

            # 2) LiDAR update/map/replan at its own sample period.
            if t + 1e-10 >= next_lidar:
                raw = sample.lidar_ranges.copy()
                ok = np.isfinite(raw)
                raw[ok] += rng_lidar.normal(0.0, config.lidar_sigma * trial.noise_scale, np.sum(ok))
                latest_scan = clean_scan(raw, r_max=config.lidar_max, window=3)
                if np.any(np.isfinite(latest_scan)):
                    planner.update(ekf.get_state(), latest_scan)
                path = planner.plan(ekf.get_state()[:2], config.goal[:2])
                next_lidar += config.lidar_period

            # 3) Rendered RGB -> ArUco -> EKF correction. Dropout means predict-only.
            if t + 1e-10 >= next_camera:
                in_blackout = trial.blackout[0] <= t <= trial.blackout[1]
                dropped = in_blackout or (rng_dropout.random() < trial.camera_dropout)
                accepted = None
                if not dropped:
                    last_camera_image = sim.camera_image()
                    measurement = camera.measure(last_camera_image, rng_camera, noise_scale=trial.noise_scale)
                    if measurement is not None:
                        accepted = ekf.update_pose(measurement.pose, measurement.R)
                        evaluator.note_camera(accepted)
                        if accepted:
                            camera_age = 0.0
                    elif camera.last_reason not in {"not_found", "wrong_id"}:
                        evaluator.note_camera(False)
                if camera_video or camera_window:
                    shown = last_camera_image if last_camera_image is not None else sim.camera_image()
                    frame = camera_view.draw(
                        shown,
                        None if dropped else measurement,
                        camera.last_reason,
                        t,
                        control.mode if control is not None else "start",
                        camera_age,
                        ekf.get_state(),
                        accepted=accepted,
                        dropped=dropped,
                    )
                    camera_view.show(frame)
                next_camera += config.camera_period

            x_hat, P = ekf.get_state(), ekf.get_covariance()
            control = controller.command(x_hat, P, path, latest_scan, camera_age)
            wheels = v_w_to_wheels(
                control.v, control.omega, config.wheel_radius, config.wheel_base, limit=config.wheel_limit
            )
            sim.step(wheels)

            true_pose = truth.pose()
            ref = np.asarray(control.reference, dtype=float)
            if ref.shape != (3,) or not np.all(np.isfinite(ref)):
                ref = np.asarray(config.goal, dtype=float)
            evaluator.record(
                t=sim.time,
                estimate=x_hat,
                P=P,
                truth=true_pose,
                odom=odom,
                reference=ref,
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
            if verbose and sim.time >= next_status:
                zone = "IN" if config.in_docking_zone(x_hat) else "OUT"
                print(
                    f"t={sim.time:5.1f}s mode={control.mode:15s} "
                    f"est=({x_hat[0]:+.2f},{x_hat[1]:+.2f},{np.degrees(x_hat[2]):+.0f}deg) "
                    f"dock_box={zone:3s} cam_age={camera_age:4.1f}s collisions={sim.collisions}",
                    flush=True,
                )
                next_status += 2.0
            if control.done:
                done = True
                termination = "docked"
                break
            if realtime:
                elapsed = walltime.perf_counter() - loop_start
                if elapsed < config.dt:
                    walltime.sleep(config.dt - elapsed)

        if viewer_handle is not None:
            try:
                viewer_handle.close()
            except Exception:
                pass
        camera_view.close()
        metrics = evaluator.metrics(
            controller_done=done, collisions=sim.collisions, termination=termination
        )
        evaluator.save(metrics, planner=planner, camera_image=last_camera_image)
        return metrics
