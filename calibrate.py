"""Offline wheel and camera validation; truth is permitted for this calibration.

Does not silently change runtime parameters. The ideal camera intrinsics come
from the MuJoCo optical specification; this is not real-camera calibration.

- straight/spin trials at diff wheel speeds. to measure effective wheel radius and wheel seperation

- mixed motion run. pose predicted from encoder increments, compared with ground truth, gives position and heading

- camera checks. use aruco marker with five start poses, compare pose with ground truth

"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np


def calibrate(output):

    """
    1. straight/spin trials at diff wheel speeds. to measure effective wheel radius and wheel seperation

    2. mixed motion run. pose predicted from encoder increments, compared with ground truth, gives position and heading

    3. camera checks. use aruco marker with five start poses, compare pose with ground truth
    """

    from track_a.config import Config, TRIALS
    from track_a.simulator import Simulator
    from track_a.ground_truth import GroundTruth
    from track_a.model import f, wrap
    from track_a.noise import wheel_increments_to_motion
    from track_a.sensing import ArucoCamera

    c = Config() #default project configurations
    rows = []
    radius = []


    for mode in ["straight", "spin"]:
        """
        Stage 1.
        new simulation for each mode and speed
        Straight - both wheels same speed
        Spin - left reversed,
        Use ground truth for calibration
        """
        for speed in [1.5, 3.0, 4.5]:
            with Simulator(c, TRIALS["no_camera"]) as sim:
                truth = GroundTruth(sim.model, sim.data)
                command = np.array([speed, speed])
                if mode == "spin":
                    command[0] *= -1
                for _ in range(40):
                    sim.step(command)
                start = truth.pose()
                enc0 = sim.sample().wheel_angles
                yaw_previous = start[2]
                yaw_change = 0.0
                for _ in range(40):
                    sim.step(command)
                    yaw_now = truth.pose()[2]
                    yaw_change += wrap(yaw_now - yaw_previous)
                    yaw_previous = yaw_now
                delta = truth.pose() - start
                dphi = sim.sample().wheel_angles - enc0
                row = {
                    "mode": mode,
                    "command_rad_s": speed,
                    "distance_m": float(np.linalg.norm(delta[:2])),
                    "yaw_change_rad": float(yaw_change),
                    "encoder_increments_rad": dphi.tolist(),
                }
                if mode == "straight":
                    #effective radius = distance travelled / wheel rotation
                    row["effective_radius_m"] = float(
                        np.linalg.norm(delta[:2]) / np.mean(dphi)
                    )
                    radius.append(row["effective_radius_m"])
                else:
                    #using the mean radius from the straight trials above
                    row["effective_base_m"] = float(
                        np.mean(radius) * (dphi[1] - dphi[0]) / yaw_change
                    )
                rows.append(row)
    
    
   
    with Simulator(c, TRIALS["no_camera"]) as sim:
        """
        Stage 2.
        Independent mixed-motion validation, driven by encoder increments.
        pose is predicted purely from nominal data and encoder, then compared with ground
        """
        truth = GroundTruth(sim.model, sim.data)
        pred = truth.pose()
        enc = sim.sample().wheel_angles
        errors = []
        for command in [[3, 3], [2, 4], [3, 3], [4, 2]]:
            for _ in range(30):
                sim.step(command)
                new = sim.sample().wheel_angles
                #encoder increments -> distance travelled ds and heading change dth
                ds, dth = wheel_increments_to_motion(
                    *(new - enc), c.wheel_radius, c.wheel_base
                )
                #divide by dt to get speeds, then move up a step
                pred = f(pred, [ds / c.dt, dth / c.dt], c.dt)
                enc = new
                error = pred - truth.pose()
                error[2] = wrap(error[2])
                errors.append(error)
    errors = np.array(errors)



  #Stage 3.
    camera = ArucoCamera(c)
    views = []
    for pose in [
        """
        Stage 3.
        start robot at 5 diff poses (final is aim pose)
        use aroco to detect marker and compare measured pose w/ ground truth
        """
        (0.45, -0.12, 0.08),
        (0.65, 0.10, -0.10),
        (0.8, -0.10, 0.15),
        (0.95, 0.04, -0.07),
        c.goal,
    ]:
        with Simulator(replace(c, start=pose), TRIALS["nominal"]) as sim:
            rgb = sim.camera_image()
            m = camera.measure(rgb, np.random.default_rng(123), noise_scale=0.0)
            entry = {
                "configured_view": list(pose),
                "detected": m is not None,
                "reason": camera.last_reason,
            }
            if m is not None:
                #measured pose vs ground truth
                error = m.pose - GroundTruth(sim.model, sim.data).pose()
                error[2] = wrap(error[2])
                entry.update(
                    {
                        "pose_error": error.tolist(),
                        "reprojection_px": m.reprojection_px,
                        "position_error_m": float(np.linalg.norm(error[:2])),
                        "heading_error_deg": float(np.degrees(abs(error[2]))),
                    }
                )
            views.append(entry)


    
    result = {
        "wheel_trials": rows,
        "camera_intrinsics": camera.K.tolist(),
        "distortion": camera.distortion.tolist(),
        "camera_views": views,
        "camera_calibration_method": "Analytic ideal pinhole from MuJoCo fovy; verified with rendered marker views",
        "nominal_parameters": {"radius_m": c.wheel_radius, "base_m": c.wheel_base},
        "mixed_motion": {
            "position_rmse_m": float(
                np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1)))
            ),
            "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(errors[:, 2] ** 2)))),
        },
        "runtime_parameters_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results/calibration.json",
    )
    args = parser.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    print(json.dumps(calibrate(args.output), indent=2))
