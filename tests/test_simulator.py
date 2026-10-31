"""MuJoCo-only checks for the physical scene, random layout and rendered ArUco sensor."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from track_a.config import Config, TRIALS
from track_a.sensing import ArucoCamera
from track_a.simulator import Simulator


def test_standard_and_random_layout_sensor_contracts():
    """Verify both layouts expose the same encoder/LiDAR interface and random layout has six boxes."""
    start = np.array([-1.45, -0.70, 0.0])
    for layout in ["standard", "random"]:
        config = Config(layout=layout)
        with Simulator(config, TRIALS["nominal"], seed=2, start_pose=start) as sim:
            sample = sim.sample()
            assert sample.wheel_angles.shape == (2,)
            assert sample.lidar_ranges.shape == (config.lidar_beams,)
            expected = 2 if layout == "standard" else config.random_obstacle_count
            assert len(sim.obstacle_layout) == expected


def test_rendered_marker_can_produce_a_camera_pose():
    """Place the robot facing the dock and confirm the rendered ArUco marker is detectable."""
    config = Config(layout="standard")
    start = np.array([0.7, 0.0, 0.0])
    camera = ArucoCamera(config)
    with Simulator(config, TRIALS["nominal"], seed=1, start_pose=start) as sim:
        image = sim.camera_image()
        measurement = camera.measure(image, np.random.default_rng(1), noise_scale=0.0)
        assert measurement is not None
        assert measurement.pose.shape == (3,)
        assert measurement.R.shape == (3, 3)


def test_random_layout_moves_rendered_marker_and_charging_station():
    """Confirm the second layout physically moves the tag and dock to its configured target."""
    start = np.array([-1.45, -0.70, 0.0])
    standard = Config(layout="standard")
    random_config = Config(layout="random")
    assert standard.marker_pos != random_config.marker_pos

    with Simulator(random_config, TRIALS["nominal"], seed=2, start_pose=start) as sim:
        marker_id = sim._geom_id("aruco_backing")
        station_id = sim._geom_id("charging_station")
        assert marker_id >= 0 and station_id >= 0
        # The backing remains offset slightly toward the camera from marker centre.
        assert np.isclose(sim.model.geom_pos[marker_id][1], random_config.marker_pos[1])
        assert np.isclose(sim.model.geom_pos[station_id][1], random_config.marker_pos[1])
