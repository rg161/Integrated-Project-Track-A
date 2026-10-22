"""Estimated-state path following and visual-docking controller.

This module implements a state-feedback controller for a robot that tracks a
planned path and then aligns with a final goal pose for docking. The controller
uses an estimated vehicle state, a path lookahead strategy, and a local LiDAR
safety layer to prevent collisions while still allowing rotational escape
maneuvers when needed.
"""

from dataclasses import dataclass
import numpy as np
from .model import wrap


@dataclass
class ControlOutput:
    """Container for the controller's commanded motion and execution status.

    Attributes:
        v: Forward linear speed command in meters per second.
        omega: Angular velocity command in radians per second.
        done: True when the docking task is considered complete.
        mode: Human-readable controller mode string for debugging or logging.
        reference: The target pose tuple used for the current control cycle.
    """

    v: float
    omega: float
    done: bool = False
    mode: str = "track"
    reference: tuple = (np.nan, np.nan, np.nan)


class StateFeedbackController:
    """Controller that tracks a path and performs final docking alignment.

    The controller maintains a small state machine for path tracking, turning in
    place to align with the path heading, and docking when the robot reaches the
    goal position. It also applies conservative LiDAR-based safety scaling to
    prevent driving into obstacles while keeping rotation available to escape
    blocked situations.
    """

    def __init__(self, config):
        """Store config and initialize internal motion-state tracking.

        The turning flag and dwell timer are used to decide when the robot should
        continue rotating in place versus returning to forward path tracking.
        """
        self.c = config
        self.turning = False
        self.dwell = 0
        self.dwell_required = 10

    def reset(self):
        """Reset transient docking/turning state for a new task cycle."""
        self.turning = False
        self.dwell = 0

    @staticmethod
    def _closest_index(path, p):
        """Return the index of the closest point on the path to a pose position."""
        return int(np.argmin(np.sum((path - p[None, :]) ** 2, axis=1)))

    def _lookahead_reference(self, pose, path):
        """Compute the path-following lookahead target and heading.

        This walks along the path from the nearest waypoint until the requested
        lookahead distance is reached and returns the corresponding target point.
        """
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        if len(path) == 0:
            # No path is available: fall back to the configured goal pose.
            goal = np.asarray(self.c.goal, dtype=float)
            return goal.copy()

        # Find the nearest point on the path to the robot pose.
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

        # Use the target point's heading toward the next point, unless the robot is
        # already effectively on top of the target.
        heading = np.arctan2(target[1] - pose[1], target[0] - pose[0])
        if np.linalg.norm(target - pose[:2]) < 1e-8:
            heading = self.c.goal[2]
        return np.array([target[0], target[1], float(wrap(heading))])

    def _front_clearance(self, scan):
        """Return the minimum valid LiDAR distance in the forward field of view.

        The clearance is computed over the front-facing beams and is used by the
        safety layer to slow or stop the robot near obstacles.
        """
        scan = np.asarray(scan, dtype=float).reshape(-1)
        if len(scan) != self.c.lidar_beams:
            return np.nan

        finite = np.isfinite(scan)
        if not np.any(finite):
            return np.nan

        # Build the beam angle array and select only the front-most directions.
        angles = self.c.lidar_angle_min + self.c.lidar_angle_inc * np.arange(len(scan))
        front = finite & (np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= np.deg2rad(28.0))
        if not np.any(front):
            return np.inf
        return float(np.min(scan[front]))

    def command(self, x_hat, P, path, lidar_scan, camera_age):
        """Compute a motion command from the state estimate, path, and sensors.

        The method first validates the estimate and covariance, then chooses a
        path-tracking or docking strategy, and finally applies LiDAR-based safety
        scaling before returning a ControlOutput object.
        """
        x_hat = np.asarray(x_hat, dtype=float).reshape(3)
        P = np.asarray(P, dtype=float).reshape(3, 3)
        goal = np.asarray(self.c.goal, dtype=float)

        # Reject invalid or highly uncertain state estimates before controlling.
        if not (np.all(np.isfinite(x_hat)) and np.all(np.isfinite(P))):
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "invalid_estimate")
        if np.sqrt(max(P[0, 0] + P[1, 1], 0.0)) > 0.45:
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "uncertainty_stop")

        # Measure the nearest obstacle in the forward-facing LiDAR sector.
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
                # Once the robot is within tolerance, hold position and align heading.
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
            # Approach the goal while rotating to reduce final bearing error.
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

            # Trigger a pure-rotation mode when the robot is pointing away from the
            # desired path direction, allowing it to realign before resuming motion.
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
                # Standard path-following controller using cross-track and heading errors.
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
