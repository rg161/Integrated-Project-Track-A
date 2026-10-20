"""Real MuJoCo rendering/contact checks; no mocked camera pose shortcut."""

import os
import sys
from dataclasses import replace
import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from track_a.config import Config, TRIALS
from track_a.simulator import Simulator
from track_a.sensing import ArucoCamera
from track_a.ground_truth import GroundTruth
from track_a.noise import wheel_increments_to_motion
from track_a.model import wrap


def test_rendered_marker_near_dock_and_blank_image():
    c = replace(Config(), start=Config().goal)
    with Simulator(c, TRIALS["nominal"]) as sim:
        camera = ArucoCamera(c)
        measurement = camera.measure(
            sim.camera_image(), np.random.default_rng(0), noise_scale=0.0
        )
        assert measurement is not None, camera.last_reason
        truth = GroundTruth(sim.model, sim.data).pose()
        assert np.linalg.norm(measurement.pose[:2] - truth[:2]) < 0.025
        assert abs(wrap(measurement.pose[2] - truth[2])) < np.deg2rad(4)
        assert (
            camera.measure(
                np.full((c.camera_height, c.camera_width, 3), 255, np.uint8),
                np.random.default_rng(0),
            )
            is None
        )


def test_small_turn_command_moves_the_robot_without_caster_braking():
    c = Config()
    with Simulator(c, TRIALS["no_camera"]) as sim:
        truth = GroundTruth(sim.model, sim.data)
        initial = truth.pose()
        enc0 = sim.sample().wheel_angles
        for _ in range(40):
            sim.step(np.array([-1.0, 1.0]))
        actual = wrap(truth.pose()[2] - initial[2])
        increments = sim.sample().wheel_angles - enc0
        _, predicted = wheel_increments_to_motion(
            *increments, c.wheel_radius, c.wheel_base
        )
        assert actual > 0.3
        assert abs(actual - predicted) < 0.025
        assert sim.collisions == 0
