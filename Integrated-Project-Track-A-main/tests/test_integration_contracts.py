"""Regression tests for the faults found when joining the supplied sections."""

import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from track_a.config import Config
from track_a.frames import (
    FrameTree,
    Transform,
    build_static_frames,
    rot_z,
    quat_wxyz_to_R,
)
from track_a.estimation import MobileRobotEKF
from track_a.controller import StateFeedbackController
from track_a.model import f, H_jac, v_w_to_wheels, wheels_to_v_w
from track_a.planning import LidarPlanner
from part_a_check import analyse


def test_model_derivatives_and_rank_claims():
    result = analyse()
    assert max(result["jacobian_max_errors"].values()) < 1e-8
    assert result["controllability"]["0.0"]["rank"] == 2
    assert result["controllability"]["0.15"]["rank"] == 3
    assert result["observability"]["single_point"]["rank"] == 2
    assert result["observability"]["two_points"]["rank"] == 3


def test_marker_rotation_bug_and_invalid_frame_graph():
    tree = FrameTree()
    R = rot_z(0.6)
    build_static_frames(
        tree,
        [0.1, 0, 0.1],
        [1, 0, 0, 0],
        [0, 0, 0.2],
        marker_pos_W=[2, 0, 0.1],
        marker_R_W=R,
    )
    np.testing.assert_allclose(tree.get("W", "M").R, R)
    tree.set_static(Transform("W", "A"))
    tree.set_static(Transform("A", "D"))
    with pytest.raises(ValueError):
        tree.set_static(Transform("D", "A"))
    with pytest.raises(ValueError):
        quat_wxyz_to_R([0, 0, 0, 0])


def test_filter_wrap_gating_and_covariance_stability():
    ekf = MobileRobotEKF((0, 0, np.pi - 0.01))
    assert ekf.update_pose([0, 0, -np.pi + 0.01])
    assert abs(abs(ekf.x[2]) - np.pi) < 0.02
    x, P = ekf.get_state(), ekf.get_covariance()
    assert not ekf.update_pose([100, 100, 0])
    np.testing.assert_array_equal(ekf.x, x)
    np.testing.assert_array_equal(ekf.P, P)
    for i in range(150):
        ekf.predict(0.1, 0.15, 0.05)
        ekf.update_pose(ekf.x + np.array([0.002, -0.001, 0.001]))
        assert np.linalg.eigvalsh(ekf.P).min() > 0
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-14)
    with pytest.raises(ValueError):
        H_jac([0, 0, 0], [0, 0])
    assert not MobileRobotEKF().update_range_bearing(0, 0, [0, 0])


def test_encoder_prediction_uses_the_shared_midpoint_model():
    config = Config()
    ekf = MobileRobotEKF((0.1, 0.2, 0.3))
    increments = np.array([0.1, 0.2])
    dt = 0.05
    v, w = wheels_to_v_w(*(increments / dt), config.wheel_radius, config.wheel_base)
    expected = f(ekf.x, [v, w], dt)
    ekf.predict_encoders(increments, dt, config)
    np.testing.assert_allclose(ekf.x, expected)


def test_controller_steers_to_lateral_target_and_requires_camera_confirmation():
    c = replace(Config(), goal=(0, 1, 0))
    controller = StateFeedbackController(c)
    output = controller.command(
        np.zeros(3),
        np.eye(3) * 0.01,
        np.array([[0, 0], [0, 1]]),
        np.ones(c.lidar_beams) * 3.5,
        0.0,
    )
    assert output.omega > 0 and output.v == 0
    for _ in range(30):
        output = controller.command(
            np.array(c.goal),
            np.eye(3) * 0.01,
            np.empty((0, 2)),
            np.ones(c.lidar_beams) * 3.5,
            100.0,
        )
    assert not output.done
    for _ in range(10):
        output = controller.command(
            np.array(c.goal),
            np.eye(3) * 0.01,
            np.empty((0, 2)),
            np.ones(c.lidar_beams) * 3.5,
            0.0,
        )
    assert output.done


def test_docking_dwell_resets_after_heading_leaves_tolerance():
    c = Config()
    controller = StateFeedbackController(c)
    P = np.eye(3) * 0.001
    pose = np.array(c.goal)
    scan = np.ones(c.lidar_beams) * 3.5
    for _ in range(9):
        assert not controller.command(pose, P, np.empty((0, 2)), scan, 0.0).done
    bad_pose = pose.copy()
    bad_pose[2] += 0.2
    assert not controller.command(bad_pose, P, np.empty((0, 2)), scan, 0.0).done
    assert not controller.command(pose, P, np.empty((0, 2)), scan, 0.0).done


def test_wheel_saturation_preserves_curvature_and_limits():
    wheels = v_w_to_wheels(0.22, 2.8, limit=7.88)
    assert np.max(abs(wheels)) <= 7.88 + 1e-12
    v, w = wheels_to_v_w(*wheels)
    assert np.isclose(w / v, 2.8 / 0.22)


def test_planner_detours_around_observed_obstacle():
    c = Config()
    planner = LidarPlanner(c)
    # Range samples from a simple vertical segment 1 m ahead in the sensor frame.
    a = planner.angles
    r = np.full(len(a), np.inf)
    hit = (np.cos(a) > 0.5) & (abs(np.tan(a)) < 0.35)
    r[hit] = 1 / np.cos(a[hit])
    pose = np.array([-1.0, 0.0, 0.0])
    planner.update(pose, r)
    path = planner.plan(pose[:2], [1.0, 0.0])
    assert len(path) >= 3
    assert np.max(abs(path[:, 1])) > 0.35


def test_control_estimation_and_sensor_processing_cannot_read_simulator_truth():
    root = Path(__file__).resolve().parents[1] / "track_a"
    for name in [
        "controller",
        "estimation",
        "planning",
        "sensing",
        "model",
        "frames",
        "noise",
    ]:
        tree = ast.parse((root / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not {"mujoco", "ground_truth", "simulator", "evaluation"} & set(
                    imports
                ), name
            if isinstance(node, ast.Attribute):
                assert node.attr not in {
                    "qpos",
                    "qvel",
                    "xpos",
                    "xmat",
                    "xquat",
                    "sensordata",
                }, name
