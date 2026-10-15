"""Part D state feedback adapted from Controller.py without ROS dependencies."""

from dataclasses import dataclass
import numpy as np
from .model import wrap, v_w_to_wheels, wheels_to_v_w
from .planning import reference_on_path


@dataclass(frozen=True)
class ControlOutput:
    wheels: np.ndarray
    v: float
    omega: float
    reference: np.ndarray
    mode: str
    done: bool = False


class StateFeedbackController:
    def __init__(self, config):
        self.config = config
        self.mode = "navigate"
        self._dwell = 0

    def command(self, estimate, covariance, path, ranges, camera_age):
        c = self.config
        x, y, theta = np.asarray(estimate)
        goal = np.array(c.goal)
        delta = goal[:2] - [x, y]
        distance = np.linalg.norm(delta)
        heading_error = wrap(goal[2] - theta)
        if (
            distance >= c.goal_position_tolerance
            or abs(heading_error) >= c.goal_heading_tolerance
            or camera_age > c.camera_max_age
            or not np.all(np.isfinite(estimate))
            or np.trace(covariance[:2, :2]) > 0.25
        ):
            self._dwell = 0
        ref = goal.copy()
        v = 0.0
        omega = 0.0
        done = False
        if not np.all(np.isfinite(estimate)) or np.trace(covariance[:2, :2]) > 0.25:
            self.mode = "uncertain_stop"
        elif distance < c.goal_position_tolerance:
            self.mode = "align_dock"
            omega = 2.0 * heading_error
            if abs(heading_error) < c.goal_heading_tolerance:
                omega = 0.0
                self.mode = "confirm_marker"
                self._dwell = self._dwell + 1 if camera_age <= c.camera_max_age else 0
                if self._dwell >= round(0.5 / c.dt):
                    self.mode = "docked"
                    done = True
        elif len(path) < 2:
            self.mode = "no_path"
        else:
            self._dwell = 0
            ref = reference_on_path(estimate, path, c.lookahead)
            dx, dy = ref[:2] - [x, y]
            ex = np.cos(theta) * dx + np.sin(theta) * dy
            ey = -np.sin(theta) * dx + np.cos(theta) * dy
            alpha = wrap(np.arctan2(dy, dx) - theta)
            # Point the local reference toward the lookahead position. Using the
            # next segment's tangent too early cuts across an inflated corner.
            ref[2] = wrap(theta + alpha)
            etheta = alpha
            turning = abs(alpha) > (0.35 if self.mode == "turn_to_path" else 0.65)
            if turning:
                self.mode = "turn_to_path"
                omega = 2.0 * alpha
            elif distance < 0.28:
                self.mode = "dock_approach"
                alpha = wrap(np.arctan2(delta[1], delta[0]) - theta)
                v = min(0.10, 0.65 * distance) * max(0.0, np.cos(alpha))
                omega = 2.0 * alpha
            else:
                self.mode = "track"
                vref = c.max_v * 0.7
                # Original Part D body-frame state-feedback structure.
                v = (vref * np.cos(etheta) + c.k1 * ex) * np.cos(alpha) ** 2
                omega = c.k2 * vref * ey + c.k3 * np.sin(etheta)
            v = float(np.clip(v, 0, c.max_v))
            omega = float(np.clip(omega, -c.max_omega, c.max_omega))
            # Fresh LiDAR is also checked in the body frame before translation.
            # Unknown/no-return rays are not hits; a missing entire scan stops motion.
            if ranges is None:
                v = 0.0
                omega = 0.0
                self.mode = "waiting_lidar"
            else:
                angles = np.linspace(-np.pi, np.pi, len(ranges), endpoint=False)
                valid = np.isfinite(ranges)
                px = c.lidar_pos[0] + ranges[valid] * np.cos(angles[valid])
                py = c.lidar_pos[1] + ranges[valid] * np.sin(angles[valid])
                corridor = (px > 0) & (np.abs(py) < 0.195)
                nearest = np.min(px[corridor], initial=np.inf)
                if nearest < 0.24:
                    v = 0.0
                    self.mode = "obstacle_stop"
                elif nearest < 0.50:
                    v *= float(np.clip((nearest - 0.24) / 0.26, 0, 1))
        omega = float(np.clip(omega, -c.max_omega, c.max_omega))
        wheels = v_w_to_wheels(v, omega, c.wheel_radius, c.wheel_base, c.wheel_limit)
        v, omega = wheels_to_v_w(*wheels, c.wheel_radius, c.wheel_base)
        return ControlOutput(wheels, float(v), float(omega), ref, self.mode, done)
