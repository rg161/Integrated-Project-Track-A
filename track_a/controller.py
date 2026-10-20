"""Estimated-state path following and visual-docking controller."""

from dataclasses import dataclass
import numpy as np
from .model import wrap


@dataclass
class ControlOutput:
    v: float
    omega: float
    done: bool = False
    mode: str = "track"
    reference: tuple = (np.nan, np.nan, np.nan)


class StateFeedbackController:
    def __init__(self, config):
        self.c = config
        self.turning = False
        self.dwell = 0
        self.dwell_required = 10

    def reset(self):
        self.turning = False
        self.dwell = 0

    @staticmethod
    def _closest_index(path, p):
        return int(np.argmin(np.sum((path - p[None, :]) ** 2, axis=1)))

    def _lookahead_reference(self, pose, path):
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        if len(path) == 0:
            goal = np.asarray(self.c.goal, dtype=float)
            return goal.copy()
        i = self._closest_index(path, pose[:2])
        # Walk forward by physical path length until the lookahead distance is met.
        target = path[-1]
        remain = self.c.lookahead
        prev = pose[:2]
        for j in range(i, len(path)):
            seg = float(np.linalg.norm(path[j] - prev))
            if seg >= remain and seg > 1e-12:
                target = prev + (remain / seg) * (path[j] - prev)
                break
            remain -= seg
            prev = path[j]
        heading = np.arctan2(target[1] - pose[1], target[0] - pose[0])
        if np.linalg.norm(target - pose[:2]) < 1e-8:
            heading = self.c.goal[2]
        return np.array([target[0], target[1], float(wrap(heading))])

    def _front_clearance(self, scan):
        scan = np.asarray(scan, dtype=float).reshape(-1)
        if len(scan) != self.c.lidar_beams:
            return np.nan
        finite = np.isfinite(scan)
        if not np.any(finite):
            return np.nan
        angles = self.c.lidar_angle_min + self.c.lidar_angle_inc * np.arange(len(scan))
        front = finite & (np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= np.deg2rad(28.0))
        if not np.any(front):
            return np.inf
        return float(np.min(scan[front]))

    def command(self, x_hat, P, path, lidar_scan, camera_age):
        x_hat = np.asarray(x_hat, dtype=float).reshape(3)
        P = np.asarray(P, dtype=float).reshape(3, 3)
        goal = np.asarray(self.c.goal, dtype=float)

        if not (np.all(np.isfinite(x_hat)) and np.all(np.isfinite(P))):
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "invalid_estimate")
        if np.sqrt(max(P[0, 0] + P[1, 1], 0.0)) > 0.45:
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "uncertainty_stop")

        clearance = self._front_clearance(lidar_scan)
        if not np.isfinite(clearance):
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "lidar_failure")

        to_goal = goal[:2] - x_hat[:2]
        goal_dist = float(np.linalg.norm(to_goal))
        heading_err_goal = float(wrap(goal[2] - x_hat[2]))

        # Fine docking state machine.
        if goal_dist <= self.c.lookahead:
            if goal_dist <= self.c.goal_position_tolerance:
                v = 0.0
                omega = np.clip(2.0 * heading_err_goal, -0.7, 0.7)
                good = (
                    abs(heading_err_goal) <= self.c.goal_heading_tolerance
                    and camera_age <= self.c.camera_max_age
                )
                self.dwell = self.dwell + 1 if good else 0
                done = self.dwell >= self.dwell_required
                if done:
                    v = omega = 0.0
                return ControlOutput(
                    float(v), float(omega), done, "docked" if done else "dock_align",
                    tuple(goal),
                )
            self.dwell = 0
            bearing = float(wrap(np.arctan2(to_goal[1], to_goal[0]) - x_hat[2]))
            v = min(0.10, 0.6 * goal_dist) * max(0.0, np.cos(bearing))
            omega = np.clip(2.2 * bearing + 0.35 * heading_err_goal, -self.c.max_omega, self.c.max_omega)
            mode = "dock_approach"
            ref = goal
        else:
            self.dwell = 0
            ref = self._lookahead_reference(x_hat, path)
            dx, dy = ref[0] - x_hat[0], ref[1] - x_hat[1]
            ct, st = np.cos(x_hat[2]), np.sin(x_hat[2])
            ex = ct * dx + st * dy
            ey = -st * dx + ct * dy
            etheta = float(wrap(ref[2] - x_hat[2]))
            alpha = float(wrap(np.arctan2(dy, dx) - x_hat[2]))
            if self.turning:
                if abs(alpha) < 0.35:
                    self.turning = False
            elif abs(alpha) > 0.65:
                self.turning = True
            if self.turning:
                v = 0.0
                omega = np.clip(1.8 * alpha, -self.c.max_omega, self.c.max_omega)
                mode = "turn_to_path"
            else:
                v_ref = 0.7 * self.c.max_v
                v = (v_ref * np.cos(etheta) + self.c.k1 * ex) * np.cos(alpha) ** 2
                omega = self.c.k2 * v_ref * ey + self.c.k3 * np.sin(etheta)
                v = np.clip(v, 0.0, self.c.max_v)
                omega = np.clip(omega, -self.c.max_omega, self.c.max_omega)
                mode = "track"

        # Local LiDAR safety layer. Rotation remains available so the robot can escape.
        if clearance < 0.24:
            v = 0.0
            mode += "+stop"
        elif clearance < 0.50:
            scale = np.clip((clearance - 0.24) / (0.50 - 0.24), 0.0, 1.0)
            v *= scale
            mode += "+slow"
        return ControlOutput(float(v), float(omega), False, mode, tuple(ref))
