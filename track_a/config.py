"""
Central configuration for the integrated Track A project.

all adjustable values
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

#Project root folder
#to build paths, models and results.
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """All settings for one run of the simulation."""
    # Timing / differential-drive geometry

    dt: float = 0.05 #controller and estimator step
    physics_dt: float = 0.002 #MuJoCo physical steps
    wheel_radius: float = 0.033
    wheel_base: float = 0.288 #wheel seperation
    wheel_limit: float = 7.88 #the actuator range

    # Controller (controller.py)
    max_v: float = 0.18 #forward speed limit
    max_omega: float = 1.2 #yaw rate limit
    k1: float = 0.8 #gain on the along-path error ex
    k2: float = 2.0 #gain on the cross-path error ey
    k3: float = 1.8 #gain on the heading error etheta
    robot_radius: float = 0.235
    safety_margin: float = 0.09
    grid_resolution: float = 0.06
    lookahead: float = 0.28

    # LiDAR
    lidar_beams: int = 180 #rays per full 360-degree scan
    lidar_max: float = 3.5
    lidar_sigma: float = 0.008 #range noise standard deviation
    lidar_period: float = 0.10 #time between scans

    # Camera / marker
    camera_period: float = 0.25 #time between frames
    camera_width: int = 480
    camera_height: int = 360
    camera_fovy: float = 58.0 #vert fov
    corner_sigma: float = 0.25
    marker_side: float = 0.24
    marker_id: int = 7
    marker_pos: tuple = (1.6, 0.0, 0.24) #marker position in the world
    camera_pos: tuple = (0.078, 0.0, 0.18) #camera position on the robot
    lidar_pos: tuple = (-0.049, 0.0, 0.17) #LiDAR position on the robot

    # Encoder / estimator assumptions
    encoder_sigma: float = 0.0005 #noise on each wheel angle reading
    odom_k: float = 0.0002 # odometry noise coefficient

    # Task geometry
    # ``start`` remains the deterministic/debug start. Normal runs sample x/y
    # from the obstacle-free left-hand start region using the run seed.
    start: tuple = (-1.45, -0.7, 0.0)
    random_start: bool = True
    start_x_range: tuple = (-1.75, -1.15)
    start_y_range: tuple = (-1.10, 1.10)

    goal: tuple = (1.12, 0.0, 0.0)

    # Rectangular robot-centre acceptance area immediately in front of the
    # charging station.  This is also drawn as a green perimeter in MuJoCo.
    docking_zone_center: tuple = (1.12, 0.0)
    docking_zone_half_size: tuple = (0.12, 0.16)
    docking_zone_line_width: float = 0.012

    bounds: tuple = (-2.2, 2.2, -1.6, 1.6) #room limitations
    goal_position_tolerance: float = 0.055
    goal_heading_tolerance: float = np.deg2rad(5.0)
    camera_max_age: float = 0.75
    max_duration: float = 100.0 #run time limit

    @property
    def docking_zone_bounds(self):
        """
        Docking zone as (xmin, xmax, ymin, ymax), from its centre and half sizes
        """
        cx, cy = map(float, self.docking_zone_center)
        hx, hy = map(float, self.docking_zone_half_size)
        return (cx - hx, cx + hx, cy - hy, cy + hy)

    def in_docking_zone(self, pose):
        """
        Return True when pose x/y lies inside the charging-zone rectangle.
        only checks x and y (not heading)
        """
        p = np.asarray(pose, dtype=float).reshape(-1)
        if p.size < 2 or not np.all(np.isfinite(p[:2])):
            return False
        xmin, xmax, ymin, ymax = self.docking_zone_bounds
        return bool(xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)

    @property
    def lidar_angle_min(self):
        """
        Angle of the first LiDAR beam
        """
        return -np.pi

    @property
    def lidar_angle_inc(self):
        """
        Angle between neighbouring beams
        """
        return 2.0 * np.pi / self.lidar_beams


@dataclass(frozen=True)
class Trial:
    """
    One test scenario: which disturbances and sensor faults to apply
    """
    name: str
    noise_scale: float = 1.0
    camera_dropout: float = 0.0
    blackout: tuple = (-1.0, -1.0)
    push_force: float = 0.0
    push_window: tuple = (9.0, 9.5)
    obstacle_shift: float = 0.0

#The scenarios that main.py --scenario and evaluate.py can run
TRIALS = {
    "nominal": Trial("nominal"),
    "noisy": Trial("noisy", noise_scale=3.0, camera_dropout=0.20),
    "dropout": Trial("dropout", camera_dropout=0.15, blackout=(6.0, 14.0)),
    "disturbed": Trial(
        "disturbed", push_force=1.5, push_window=(9.0, 9.5), obstacle_shift=0.12
    ),
    "no_camera": Trial("no_camera", camera_dropout=1.0),
}


def sample_start_pose(config, rng):
    """Sample a reproducible known initial pose for one run.

    Only x/y are randomized; heading remains the configured start heading.
    The sampling rectangle is deliberately on the obstacle-free left side of
    the room so the robot is never spawned inside a wall or obstacle.
    """
    if not config.random_start:
        return np.asarray(config.start, dtype=float).copy()
    x = rng.uniform(*map(float, config.start_x_range))
    y = rng.uniform(*map(float, config.start_y_range))
    return np.array([x, y, float(config.start[2])], dtype=float)
