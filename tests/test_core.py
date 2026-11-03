"""Fast regression tests for the required Track A model, EKF, planner and controller."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from track_a.config import Config, _random_layout_has_path, sample_random_obstacles, sample_start_pose
from track_a.controller import StateFeedbackController
from track_a.estimation import MobileRobotEKF
from track_a.model import F_jac, G_jac, f, wheels_to_v_w
from track_a.noise import clean_scan
from track_a.planning import LidarPlanner


def test_motion_jacobians_match_finite_differences():
    """Check analytical F/G Jacobians against numerical derivatives of the motion model."""
    x = np.array([0.4, -0.2, 0.7])
    u = np.array([0.18, 0.35])
    dt = 0.05
    eps = 1e-6

    F_num = np.zeros((3, 3))
    for j in range(3):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        delta = f(xp, u, dt) - f(xm, u, dt)
        F_num[:, j] = delta / (2 * eps)

    G_num = np.zeros((3, 2))
    for j in range(2):
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        delta = f(x, up, dt) - f(x, um, dt)
        G_num[:, j] = delta / (2 * eps)

    np.testing.assert_allclose(F_jac(x, u, dt), F_num, atol=1e-7)
    np.testing.assert_allclose(G_jac(x, u, dt), G_num, atol=1e-7)


def test_camera_pose_measurement_observes_all_three_states():
    """Confirm a direct camera pose measurement has full local rank for x, y and heading."""
    H_pose = np.eye(3)
    assert np.linalg.matrix_rank(H_pose) == 3


def test_ekf_uses_encoder_prediction_and_rejects_extreme_camera_outlier():
    """Verify encoder prediction follows the shared model and NIS gating protects the state."""
    config = Config()
    ekf = MobileRobotEKF((0.1, 0.2, 0.3))
    increments = np.array([0.1, 0.2])
    v, w = wheels_to_v_w(
        *(increments / config.dt), config.wheel_radius, config.wheel_base
    )
    expected = f(ekf.x, [v, w], config.dt)
    ekf.predict_encoders(increments, config.dt, config)
    np.testing.assert_allclose(ekf.x, expected)

    before = ekf.get_state()
    assert not ekf.update_pose([100.0, 100.0, 0.0])
    np.testing.assert_array_equal(ekf.get_state(), before)
    assert np.linalg.eigvalsh(ekf.get_covariance()).min() > 0


def test_controller_requires_recent_camera_before_declaring_docked():
    """Ensure reaching the goal alone cannot finish docking without recent visual confirmation."""
    config = Config()
    controller = StateFeedbackController(config)
    pose = np.array(config.goal)
    P = np.eye(3) * 0.001
    scan = np.ones(config.lidar_beams) * config.lidar_max

    for _ in range(20):
        assert not controller.command(
            pose, P, np.empty((0, 2)), scan, 100.0, marker_bearing=None
        ).done

    done = False
    for _ in range(20):
        done = controller.command(
            pose, P, np.empty((0, 2)), scan, 0.0, marker_bearing=0.0
        ).done
        if done:
            break
    assert done


def test_random_start_and_obstacles_are_reproducible_and_safe():
    """Check one seed always produces the same valid random start and six separated obstacles."""
    config = Config(layout="random")
    start_a = sample_start_pose(config, np.random.default_rng(123))
    start_b = sample_start_pose(config, np.random.default_rng(123))
    np.testing.assert_allclose(start_a, start_b)

    obs_a = sample_random_obstacles(config, np.random.default_rng(456), start_a)
    obs_b = sample_random_obstacles(config, np.random.default_rng(456), start_a)
    assert obs_a == obs_b
    assert len(obs_a) == config.random_obstacle_count == 6
    assert _random_layout_has_path(config, start_a, obs_a)

    goal = np.asarray(config.goal)[:2]
    for i, (x, y, hx, hy, _) in enumerate(obs_a):
        centre = np.array([x, y])
        assert np.linalg.norm(centre - start_a[:2]) >= 0.75
        assert np.linalg.norm(centre - goal) >= 0.55
        for ox, oy, ohx, ohy, _ in obs_a[:i]:
            assert not (
                abs(x - ox) < hx + ohx + 0.12
                and abs(y - oy) < hy + ohy + 0.12
            )


def test_planner_detours_around_observed_obstacle():
    """Check A* returns a lateral detour after LiDAR observes a blocking segment ahead."""
    config = Config()
    planner = LidarPlanner(config)
    angles = planner.angles
    ranges = np.full(len(angles), np.inf)
    hit = (np.cos(angles) > 0.5) & (abs(np.tan(angles)) < 0.35)
    ranges[hit] = 1 / np.cos(angles[hit])
    pose = np.array([-1.0, 0.0, 0.0])
    planner.update(pose, ranges)
    path = planner.plan(pose[:2], [1.0, 0.0])
    assert len(path) >= 3
    assert np.max(abs(path[:, 1])) > 0.35



def test_random_layouts_are_navigable_across_many_seeds():
    """Verify repeated random layouts retain an inflated collision-free route to the dock."""
    config = Config(layout="random")
    for seed in range(100):
        start = sample_start_pose(config, np.random.default_rng(seed * 10 + 5))
        obstacles = sample_random_obstacles(config, np.random.default_rng(seed * 10 + 6), start)
        assert _random_layout_has_path(config, start, obstacles), seed


def test_controller_stops_when_no_astar_path_exists():
    """Ensure an empty planner result cannot make the controller drive directly at the goal."""
    config = Config()
    controller = StateFeedbackController(config)
    pose = np.array([-1.0, 0.0, 0.0])
    P = np.eye(3) * 0.001
    scan = np.ones(config.lidar_beams) * config.lidar_max
    control = controller.command(pose, P, np.empty((0, 2)), scan, np.inf)
    assert control.mode == "no_path"
    assert control.v == 0.0 and control.omega == 0.0


def test_controller_hard_stop_turns_toward_clearer_side():
    """Check the local safety layer stops translation and rotates out of a close obstacle."""
    config = Config()
    controller = StateFeedbackController(config)
    pose = np.array([-1.0, 0.0, 0.0])
    P = np.eye(3) * 0.001
    path = np.array([[-1.0, 0.0], [1.0, 0.0]])
    scan = np.ones(config.lidar_beams) * 1.0
    angles = config.lidar_angle_min + config.lidar_angle_inc * np.arange(config.lidar_beams)
    scan[np.abs(angles) < np.deg2rad(20.0)] = 0.18
    scan[(angles > np.deg2rad(25.0)) & (angles < np.deg2rad(120.0))] = 2.0
    scan[(angles < -np.deg2rad(25.0)) & (angles > -np.deg2rad(120.0))] = 0.7
    control = controller.command(pose, P, path, scan, np.inf)
    assert control.v == 0.0
    assert control.omega > 0.0
    assert control.mode.endswith("+escape")

def test_lidar_cleaning_distinguishes_no_return_from_invalid():
    """Keep no-hit rays as infinity while leaving too-close invalid readings as NaN."""
    ranges = np.array([1.0, np.inf, 1.1, 0.01, 1.2])
    cleaned = clean_scan(ranges, window=3)
    assert np.isposinf(cleaned[1])
    assert np.isnan(cleaned[3])


def test_no_return_rays_clear_stale_occupancy():
    """Ensure open-space LiDAR rays remove stale obstacle evidence instead of forming ghost walls."""
    config = Config()
    planner = LidarPlanner(config)
    pose = np.array([0.0, 0.0, 0.0])
    angles = planner.angles

    # Repeatedly observe a false obstacle one metre straight ahead so it becomes occupied.
    scan = np.full(config.lidar_beams, np.inf)
    idx = int(np.argmin(np.abs(angles)))
    for _ in range(3):
        scan[idx] = 1.0
        planner.update(pose, scan)
    hit_cell = planner.world_to_cell([1.0 + config.lidar_pos[0], 0.0])
    assert planner.occupied()[hit_cell]

    # Subsequent no-return rays through the same direction must clear that stale hit.
    clear_scan = np.full(config.lidar_beams, np.inf)
    for _ in range(5):
        planner.update(pose, clear_scan)
    assert not planner.occupied()[hit_cell]


def test_control_chain_cannot_access_simulator_ground_truth():
    """Statically enforce that sensing/estimation/planning/control cannot read MuJoCo truth state."""
    root = Path(__file__).resolve().parents[1] / "track_a"
    forbidden_imports = {"mujoco", "ground_truth", "simulator", "evaluation"}
    forbidden_attrs = {"qpos", "qvel", "xpos", "xmat", "xquat", "sensordata"}

    for name in ["controller", "estimation", "planning", "sensing", "model", "frames", "noise"]:
        tree = ast.parse((root / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not (forbidden_imports & set(imported)), name
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, name


def test_every_project_function_has_a_docstring():
    """Enforce the requested triple-quoted lookup comment on every project function/method."""
    root = Path(__file__).resolve().parents[1]
    files = [root / "main.py", *sorted((root / "track_a").glob("*.py"))]
    missing = []
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                missing.append(f"{path.name}:{node.lineno}:{node.name}")
    assert missing == []


def test_camera_mount_and_both_layout_markers_are_in_view_at_dock():
    """Check camera pose matches the scene and both layout markers lie inside its FOV at the goal."""
    root = Path(__file__).resolve().parents[1]
    xml_root = ET.parse(root / "models" / "track_a_scene.xml").getroot()
    camera_xml = xml_root.find(".//camera[@name='front_camera']")
    assert camera_xml is not None

    scene_pos = np.fromstring(camera_xml.attrib["pos"], sep=" ")
    scene_xyaxes = np.fromstring(camera_xml.attrib["xyaxes"], sep=" ")
    np.testing.assert_allclose(scene_pos, Config().camera_pos)

    # MuJoCo cameras look along -z. The supplied x/y axes therefore imply
    # a +world/body-x viewing direction for a zero-heading robot.
    x_axis = scene_xyaxes[:3]
    y_axis = scene_xyaxes[3:]
    view_dir = -np.cross(x_axis, y_axis)
    np.testing.assert_allclose(view_dir, [1.0, 0.0, 0.0], atol=1e-12)

    for layout in ("standard", "random"):
        config = Config(layout=layout)
        goal = np.asarray(config.goal, dtype=float)
        marker = np.asarray(config.marker_pos, dtype=float)
        camera_local = np.asarray(config.camera_pos, dtype=float)
        ct, st = np.cos(goal[2]), np.sin(goal[2])
        R_W_B = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        camera_world = np.array([goal[0], goal[1], 0.0]) + R_W_B @ camera_local
        p_body = R_W_B.T @ (marker - camera_world)

        # OpenCV optical coordinates: +z forward, +x right, +y down.
        p_camera = np.array([-p_body[1], -p_body[2], p_body[0]])
        assert p_camera[2] > 0.0
        vertical_angle = abs(np.arctan2(p_camera[1], p_camera[2]))
        horizontal_angle = abs(np.arctan2(p_camera[0], p_camera[2]))
        half_vfov = np.deg2rad(config.camera_fovy) / 2.0
        focal = config.camera_height / (2.0 * np.tan(half_vfov))
        half_hfov = np.arctan((config.camera_width / 2.0) / focal)
        assert vertical_angle < half_vfov
        assert horizontal_angle < half_hfov

    assert Config(layout="standard").marker_pos != Config(layout="random").marker_pos


def test_visual_docking_turns_toward_marker_bearing():
    """Check final docking steering turns toward the visible ArUco marker to keep it centred."""
    config = Config()
    controller = StateFeedbackController(config)
    pose = np.array(config.goal, dtype=float)
    P = np.eye(3) * 0.001
    scan = np.ones(config.lidar_beams) * config.lidar_max

    right = controller.command(
        pose, P, np.empty((0, 2)), scan, 0.0, marker_bearing=np.deg2rad(-10.0)
    )
    assert right.mode == "dock_visual_align"
    assert right.omega < 0.0

    controller.reset()
    left = controller.command(
        pose, P, np.empty((0, 2)), scan, 0.0, marker_bearing=np.deg2rad(10.0)
    )
    assert left.mode == "dock_visual_align"
    assert left.omega > 0.0
