"""Part C EKF, sharing Part A's midpoint model and Part B's wheel noise.

This module has no simulator handle and cannot read ground truth.
"""

import numpy as np
from .model import f, F_jac, G_jac, H_jac, h_range_bearing, wrap
from .noise import wheel_increments_to_motion, odom_cov


class MobileRobotEKF:
    def __init__(self, initial_pose=(0.0, 0.0, 0.0), covariance=None):
        self.x = np.array(initial_pose, dtype=float)
        self.P = (
            np.diag([0.08**2, 0.08**2, np.deg2rad(6) ** 2])
            if covariance is None
            else np.array(covariance, dtype=float)
        )
        self.Q = np.diag([0.003**2, 0.003**2, np.deg2rad(0.3) ** 2])  # per second
        self.R_pose = np.diag([0.05**2, 0.05**2, np.deg2rad(4) ** 2])
        self.R_range_bearing = np.diag([0.05**2, np.deg2rad(2) ** 2])
        self.last_nis = float("nan")
        self.accepted = 0
        self.rejected = 0

    def predict(self, v, omega, dt, input_covariance=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        u = np.array([v, omega])
        F, G = F_jac(self.x, u, dt), G_jac(self.x, u, dt)
        Q = self.Q * dt
        if input_covariance is not None:
            Q = Q + G @ np.asarray(input_covariance) @ G.T
        self.x = f(self.x, u, dt)
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) / 2
        return self.get_state()

    def predict_encoders(self, increments, dt, config, noise_scale=1.0):
        ds, dtheta = wheel_increments_to_motion(
            *increments, config.wheel_radius, config.wheel_base
        )
        M = odom_cov(
            *increments,
            config.wheel_radius,
            config.wheel_base,
            config.odom_k * noise_scale**2,
        )
        # Differences of two independently noisy encoder samples have 2*sigma^2.
        J = config.wheel_radius * np.array(
            [[0.5, 0.5], [-1 / config.wheel_base, 1 / config.wheel_base]]
        )
        M += J @ (np.eye(2) * 2 * (config.encoder_sigma * noise_scale) ** 2) @ J.T
        return self.predict(ds / dt, dtheta / dt, dt, M / dt**2)

    def _update(self, innovation, H, R, threshold):
        if not np.all(np.isfinite(innovation)) or not np.all(np.isfinite(R)):
            self.rejected += 1
            return False
        S = H @ self.P @ H.T + R
        self.last_nis = float(innovation @ np.linalg.solve(S, innovation))
        if self.last_nis > threshold:
            self.rejected += 1
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        self.x[2] = wrap(self.x[2])
        A = np.eye(3) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T  # Joseph form
        self.P = (self.P + self.P.T) / 2
        self.accepted += 1
        return True

    def update_pose(self, camera_pose, covariance=None):
        innovation = np.asarray(camera_pose, dtype=float) - self.x
        innovation[2] = wrap(innovation[2])
        return self._update(
            innovation,
            np.eye(3),
            self.R_pose if covariance is None else covariance,
            16.27,
        )

    def update_range_bearing(
        self, measured_range, measured_bearing, marker_position, covariance=None
    ):
        """Measurement must be planar range/bearing from BASE, not optical camera."""
        if measured_range <= 0:
            self.rejected += 1
            return False
        try:
            zhat = h_range_bearing(self.x, marker_position)
            H = H_jac(self.x, marker_position)
        except ValueError:
            self.rejected += 1
            return False
        innovation = np.array([measured_range, measured_bearing]) - zhat
        innovation[1] = wrap(innovation[1])
        return self._update(
            innovation,
            H,
            self.R_range_bearing if covariance is None else covariance,
            13.82,
        )

    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()
