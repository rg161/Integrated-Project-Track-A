"""One source of truth for units, sensor mounting, gains and trial settings."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    dt: float = 0.05
    physics_dt: float = 0.002
    wheel_radius: float = 0.033
    wheel_base: float = 0.288
    wheel_limit: float = 7.88
    max_v: float = 0.18
    max_omega: float = 1.2
    k1: float = 0.8
    k2: float = 2.0
    k3: float = 1.8
    robot_radius: float = 0.235  # conservative radius about wheel axle
    safety_margin: float = 0.09
    grid_resolution: float = 0.06
    lookahead: float = 0.28
    lidar_beams: int = 180
    lidar_max: float = 3.5
    lidar_sigma: float = 0.008
    lidar_period: float = 0.10
    camera_period: float = 0.25
    camera_width: int = 480
    camera_height: int = 360
    camera_fovy: float = 58.0
    corner_sigma: float = 0.25
    encoder_sigma: float = 0.0005  # rad / sample, cumulative encoder reading
    odom_k: float = 2e-4  # wheel travel variance / absolute travel (m)
    marker_side: float = 0.24  # black border to black border
    marker_id: int = 7
    marker_pos: tuple = (1.60, 0.0, 0.24)
    camera_pos: tuple = (0.078, 0.0, 0.18)
    lidar_pos: tuple = (-0.049, 0.0, 0.17)
    start: tuple = (-1.45, -0.70, 0.0)
    goal: tuple = (1.12, 0.0, 0.0)
    bounds: tuple = (-2.2, 2.2, -1.6, 1.6)
    goal_position_tolerance: float = 0.055
    goal_heading_tolerance: float = np.deg2rad(5)
    camera_max_age: float = 0.75
    max_duration: float = 100.0

    def __post_init__(self):
        if self.dt <= 0 or self.physics_dt <= 0:
            raise ValueError("Time steps must be positive")
        if not np.isclose(self.dt / self.physics_dt, round(self.dt / self.physics_dt)):
            raise ValueError("Control step must be a whole number of physics steps")
        if self.wheel_radius <= 0 or self.wheel_base <= 0:
            raise ValueError("Wheel dimensions must be positive")


@dataclass(frozen=True)
class Trial:
    name: str = "nominal"
    noise_scale: float = 1.0
    camera_dropout: float = 0.0
    blackout: tuple = (-1.0, -1.0)
    push_force: float = 0.0
    push_window: tuple = (9.0, 9.5)
    obstacle_shift: float = 0.0


TRIALS = {
    "nominal": Trial(),
    "noisy": Trial("noisy", noise_scale=3.0, camera_dropout=0.20),
    "dropout": Trial("dropout", camera_dropout=0.15, blackout=(6.0, 14.0)),
    "disturbed": Trial("disturbed", push_force=1.5, obstacle_shift=0.12),
    "no_camera": Trial("no_camera", camera_dropout=1.0),
}
