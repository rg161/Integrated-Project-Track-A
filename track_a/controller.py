"""Estimated-state path following and visual-docking controller.

This module implements a state-feedback controller for a mobile robot that:
- follows a planned path using a lookahead reference,
- aligns with a target goal pose before docking,
- monitors LiDAR clearance for local collision avoidance,
- and reports a structured control output with mode and target information.
"""

from dataclasses import dataclass
import numpy as np
from .model import wrap


@dataclass
class ControlOutput:
    """Stores the controller's velocity command and status metadata.

    This dataclass is returned by the controller every control cycle. It captures
    the commanded linear and angular velocities, whether the docking task is
    complete, the current mode label, and the reference pose used to compute the
    command.
    """

    v: float
    omega: float
    done: bool = False
    mode: str = "track"
    reference: tuple = (np.nan, np.nan, np.nan)


class StateFeedbackController:
    """State-feedback controller for path tracking and visual docking.

    The controller estimates the robot's current pose and uncertainty, chooses a
    behavior based on the goal and path geometry, and emits a safe command for
    the robot to execute. It combines path tracking, turning behavior, goal
    approach, final docking alignment, and local obstacle avoidance.
    """

    def __init__(self, config):
        """Initialize the controller with configuration values and internal state."""
        self.c = config
        # True while the controller is intentionally rotating the robot to regain path alignment.
        self.turning = False
        # Number of consecutive good docking cycles required to declare completion.
        self.dwell = 0
        self.dwell_required = 10

    def reset(self):
        """Reset transient controller state before a new execution cycle."""
        self.turning = False
        self.dwell = 0

    @staticmethod
    def _closest_index(path, p):
        """Find the index of the path point closest to a given 2D pose."""
        return int(np.argmin(np.sum((path - p[None, :]) ** 2, axis=1)))

    def _lookahead_reference(self, pose, path):
        """Compute a local reference point ahead of the robot along the path.

        The robot tracks a point located a fixed distance ahead on the path to
        create a local reference for heading and lateral error.

        The logic is:
        1) find the nearest path point to the robot,
        2) march forward while accumulating arc length until the cumulative
           distance reaches the lookahead distance L, and
        3) interpolate along the segment to obtain a reference pose at distance L.

        If the robot is already at or past the target, the heading is replaced by
        the goal heading to avoid numerical discontinuities and unstable yaw
        commands.
        """
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        if len(path) == 0:
            # If no path is available, fall back to the configured goal.
            goal = np.asarray(self.c.goal, dtype=float)
            return goal.copy()
        i = self._closest_index(path, pose[:2])
        # Walk forward along the path until the lookahead distance is met.
        target = path[-1]
        remain = self.c.lookahead
        prev = pose[:2]
        for j in range(i, len(path)):
            # Segment length between path[j-1] and path[j] (or robot pose to the
            # first path point if j == i). This is the distance accumulated along
            # the path before the lookahead distance is reached.
            seg = float(np.linalg.norm(path[j] - prev))
            if seg >= remain and seg > 1e-12:
                # Interpolate along the segment to land exactly at the lookahead
                # distance: p_ref = p_prev + (L / ||p_j - p_prev||) * (p_j - p_prev).
                target = prev + (remain / seg) * (path[j] - prev)
                break
            remain -= seg
            prev = path[j]
        # The reference heading is the angle from the robot to the lookahead point.
        # This is the desired yaw direction used by the path-tracking controller.
        heading = np.arctan2(target[1] - pose[1], target[0] - pose[0])
        if np.linalg.norm(target - pose[:2]) < 1e-8:
            # Prevent unstable heading when the robot is already at the target.
            # In that case, align with the goal heading instead of the undefined
            # direction from a point to itself.
            heading = self.c.goal[2]
        return np.array([target[0], target[1], float(wrap(heading))])

    def _front_clearance(self, scan):
        """Measure the nearest obstacle distance in front of the robot.

        Only LiDAR beams within a narrow forward cone are considered so that the
        safety check reflects obstacles relevant to the robot's immediate motion.

        For a beam at angle theta_i, the beam is kept if its angle is within the
        forward cone of ±28 degrees. The safety measure is the minimum range over
        all valid front-facing beams:

            clearance = min_{i in front cone} scan[i]
        """
        scan = np.asarray(scan, dtype=float).reshape(-1)
        if len(scan) != self.c.lidar_beams:
            # Invalid scan length means the measurement cannot be trusted.
            return np.nan
        finite = np.isfinite(scan)
        if not np.any(finite):
            return np.nan
        # Compute the beam angles and keep only the forward-facing cone.
        angles = self.c.lidar_angle_min + self.c.lidar_angle_inc * np.arange(len(scan))
        # For a beam with angle theta, we test whether it lies within the forward
        # cone by measuring the signed angle error relative to the robot's x-axis.
        # The implementation uses atan2(cos(theta), sin(theta)) to map the beam
        # angle into a canonical angle domain and then clip it to ±28°.
        front = finite & (np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= np.deg2rad(28.0))
        if not np.any(front):
            return np.inf
        return float(np.min(scan[front]))

    def command(self, x_hat, P, path, lidar_scan, camera_age):
        """Compute a velocity command from state estimate, path, and environment.

        The controller validates the estimate, checks safety limits, selects a
        behavior based on how close the robot is to the goal, and then returns a
        command with the appropriate mode and reference pose.
        """
        x_hat = np.asarray(x_hat, dtype=float).reshape(3)
        P = np.asarray(P, dtype=float).reshape(3, 3)
        goal = np.asarray(self.c.goal, dtype=float)

        # Reject invalid pose or covariance values before issuing any motion.
        if not (np.all(np.isfinite(x_hat)) and np.all(np.isfinite(P))):
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "invalid_estimate")
        # Stop if the estimator covariance is too large to trust the current state.
        # The covariance is interpreted as the error variance in x and y, and we
        # enforce a loose safety threshold on its RMS magnitude:
        # sqrt(P_xx + P_yy) <= 0.45 m.
        if np.sqrt(max(P[0, 0] + P[1, 1], 0.0)) > 0.45:
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "uncertainty_stop")

        # Read the nearest obstacle clearance in the robot's forward direction.
        clearance = self._front_clearance(lidar_scan)
        if not np.isfinite(clearance):
            self.dwell = 0
            return ControlOutput(0.0, 0.0, False, "lidar_failure")

        to_goal = goal[:2] - x_hat[:2]
        goal_dist = float(np.linalg.norm(to_goal))
        # Goal heading error in the robot's yaw frame. wrap() keeps the difference
        # in the principal interval [-pi, pi]: e_yaw = wrap(yaw_goal - yaw_robot).
        heading_err_goal = float(wrap(goal[2] - x_hat[2]))

        # Fine docking state machine.
        if goal_dist <= self.c.lookahead:
            if goal_dist <= self.c.goal_position_tolerance:
                # Final goal alignment: hold position while the heading and camera
                # observation remain within tolerance for a dwell period.
                v = 0.0
                # Use the yaw error as a direct proportional steering command:
                # omega = k * e_yaw, clipped to the maximum allowable turn rate.
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
            # Approach the goal with a slow, heading-aware motion.
            # bearing is the angle from the robot heading to the goal direction:
            # bearing = wrap(atan2(goal_y - y, goal_x - x) - yaw).
            bearing = float(wrap(np.arctan2(to_goal[1], to_goal[0]) - x_hat[2]))
            # Distance-based speed scaling: the robot slows as it approaches the
            # final goal, while the cosine term prevents reversing the robot when
            # the goal is behind it.
            v = min(0.10, 0.6 * goal_dist) * max(0.0, np.cos(bearing))
            # The angular velocity is a proportional controller on bearing and the
            # residual goal yaw error: omega = k_bearing * bearing + k_yaw * e_yaw.
            omega = np.clip(2.2 * bearing + 0.35 * heading_err_goal, -self.c.max_omega, self.c.max_omega)
            mode = "dock_approach"
            ref = goal
        else:
            self.dwell = 0
            # Track a point ahead of the robot along the planned path.
            ref = self._lookahead_reference(x_hat, path)
            dx, dy = ref[0] - x_hat[0], ref[1] - x_hat[1]
            ct, st = np.cos(x_hat[2]), np.sin(x_hat[2])
            # Rotate the planar error into the robot frame:
            # e_x = cos(yaw) * dx + sin(yaw) * dy
            # e_y = -sin(yaw) * dx + cos(yaw) * dy
            # These are the forward and lateral tracking errors expressed in the
            # robot's local coordinate system.
            ex = ct * dx + st * dy
            ey = -st * dx + ct * dy
            # Reference heading error relative to current robot yaw.
            etheta = float(wrap(ref[2] - x_hat[2]))
            # Angle from the robot to the lookahead point, measured in the robot frame.
            # This acts as a local heading error for the path segment direction.
            alpha = float(wrap(np.arctan2(dy, dx) - x_hat[2]))
            if self.turning:
                if abs(alpha) < 0.35:
                    self.turning = False
            elif abs(alpha) > 0.65:
                self.turning = True
            if self.turning:
                # If the robot is badly misaligned with the path, rotate to re-orient.
                # This is a pure heading controller that steers the robot toward the
                # line from its current pose to the lookahead target.
                v = 0.0
                omega = np.clip(1.8 * alpha, -self.c.max_omega, self.c.max_omega)
                mode = "turn_to_path"
            else:
                # Normal path-following mode: use lateral and heading errors.
                # The forward speed command is a nonlinear approximation of a path
                # tracking law: v = (v_ref cos(e_theta) + k1 e_x) cos^2(alpha).
                # The angular rate uses a PD-like correction on lateral error and
                # heading mismatch: omega = k2 * v_ref * e_y + k3 * sin(e_theta).
                v_ref = 0.7 * self.c.max_v
                v = (v_ref * np.cos(etheta) + self.c.k1 * ex) * np.cos(alpha) ** 2
                omega = self.c.k2 * v_ref * ey + self.c.k3 * np.sin(etheta)
                v = np.clip(v, 0.0, self.c.max_v)
                omega = np.clip(omega, -self.c.max_omega, self.c.max_omega)
                mode = "track"

        # Local LiDAR safety layer. Rotation remains available so the robot can escape.
        if clearance < 0.24:
            # Hard stop when the obstacle is within the minimum permitted distance.
            # This enforces a critical safety boundary: v = 0.
            v = 0.0
            mode += "+stop"
        elif clearance < 0.50:
            # Scale down forward speed when the obstacle is nearby but not an immediate stop.
            # Linear interpolation between the stop threshold and the warning threshold:
            # scale = (clearance - 0.24) / (0.50 - 0.24), clipped to [0, 1].
            scale = np.clip((clearance - 0.24) / (0.50 - 0.24), 0.0, 1.0)
            v *= scale
            mode += "+slow"
        return ControlOutput(float(v), float(omega), False, mode, tuple(ref))
