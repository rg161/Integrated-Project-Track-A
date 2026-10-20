"""Part C: EKF fusing wheel-encoder odometry and camera measurements."""

import numpy as np
from .model import f, F_jac, G_jac, H_jac, h_range_bearing, wheels_to_v_w, wrap
from .noise import odom_cov


class MobileRobotEKF:
    POSE_NIS_LIMIT = 16.27          # approx 99.9% chi-square, dof=3
    RANGE_BEARING_NIS_LIMIT = 13.82 # approx 99.9% chi-square, dof=2

    def __init__(self, x0=(0.0, 0.0, 0.0), P0=None):
        self.x = np.asarray(x0, dtype=float).reshape(3).copy()
        self.x[2] = float(wrap(self.x[2]))
        self.P = (
            np.diag([0.08**2, 0.08**2, np.deg2rad(6.0) ** 2])
            if P0 is None
            else np.asarray(P0, dtype=float).reshape(3, 3).copy()
        )
        # Continuous unmodelled process covariance; multiplied by dt.
        self.Qc = np.diag([0.003**2, 0.003**2, np.deg2rad(0.3) ** 2])
        self.R_pose = np.diag([0.025**2, 0.025**2, np.deg2rad(2.0) ** 2])
        self.R_range_bearing = np.diag([0.05**2, np.deg2rad(2.0) ** 2])
        self.last_nis = np.nan
        self.last_update_accepted = False
        self.last_innovation = None

    @staticmethod
    def _symmetrise(P):
        P = 0.5 * (P + P.T)
        # Very small numerical negative eigenvalues can otherwise accumulate.
        vals, vecs = np.linalg.eigh(P)
        if vals.min() <= 1e-14:
            vals = np.maximum(vals, 1e-14)
            P = (vecs * vals) @ vecs.T
        return 0.5 * (P + P.T)

    def reset(self, x0, P0=None):
        self.__init__(x0, P0)

    def predict(self, v, omega, dt, input_cov=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        x_old = self.x.copy()
        u = np.array([v, omega], dtype=float)
        F = F_jac(x_old, u, dt)
        G = G_jac(x_old, u, dt)
        self.x = f(x_old, u, dt)
        Sigma_u = np.zeros((2, 2)) if input_cov is None else np.asarray(input_cov, dtype=float).reshape(2, 2)
        self.P = F @ self.P @ F.T + G @ Sigma_u @ G.T + self.Qc * dt
        self.P = self._symmetrise(self.P)
        self.last_update_accepted = False
        self.last_nis = np.nan
        self.last_innovation = None
        return self.x.copy(), self.P.copy()

    def predict_encoders(self, increments, dt, config):
        """Predict from measured wheel-angle increments, not wheel commands."""
        inc = np.asarray(increments, dtype=float).reshape(2)
        wl, wr = inc / dt
        v, omega = wheels_to_v_w(wl, wr, config.wheel_radius, config.wheel_base)

        # Part B wheel-travel model + variance from differencing two independent
        # noisy encoder readings at consecutive samples.
        motion_cov = odom_cov(
            inc[0], inc[1], config.wheel_radius, config.wheel_base, k=config.odom_k
        )
        enc_var = 2.0 * (config.wheel_radius * config.encoder_sigma) ** 2
        J = np.array([[0.5, 0.5], [-1.0 / config.wheel_base, 1.0 / config.wheel_base]])
        motion_cov += J @ np.diag([enc_var, enc_var]) @ J.T
        input_cov = motion_cov / (dt * dt)
        return self.predict(v, omega, dt, input_cov=input_cov)

    def _linear_update(self, z, h, H, R, angle_indices, nis_limit):
        z = np.asarray(z, dtype=float).reshape(-1)
        h = np.asarray(h, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(R))):
            self.last_update_accepted = False
            return False
        nu = z - h
        for idx in angle_indices:
            nu[idx] = float(wrap(nu[idx]))
        S = H @ self.P @ H.T + R
        try:
            Sinv_nu = np.linalg.solve(S, nu)
            nis = float(nu @ Sinv_nu)
        except np.linalg.LinAlgError:
            self.last_update_accepted = False
            return False
        self.last_nis = nis
        self.last_innovation = nu.copy()
        if not np.isfinite(nis) or nis > nis_limit:
            self.last_update_accepted = False
            return False

        # K = P H' S^-1, computed with a solve rather than explicit inverse.
        PHt = self.P @ H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError:
            self.last_update_accepted = False
            return False
        self.x = self.x + K @ nu
        self.x[2] = float(wrap(self.x[2]))
        I = np.eye(3)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = self._symmetrise(self.P)
        self.last_update_accepted = True
        return True

    def update_pose(self, camera_pose, R=None):
        R = self.R_pose if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        z = np.asarray(camera_pose, dtype=float).reshape(3)
        return self._linear_update(
            z, self.x.copy(), np.eye(3), R, angle_indices=(2,), nis_limit=self.POSE_NIS_LIMIT
        )

    def update_range_bearing(self, measured_range, measured_bearing, marker_position, R=None):
        marker = np.asarray(marker_position, dtype=float).reshape(2)
        try:
            h = h_range_bearing(self.x, marker)
            H = H_jac(self.x, marker)
        except ValueError:
            self.last_update_accepted = False
            return False
        R = self.R_range_bearing if R is None else np.asarray(R, dtype=float).reshape(2, 2)
        z = np.array([measured_range, measured_bearing], dtype=float)
        return self._linear_update(
            z, h, H, R, angle_indices=(1,), nis_limit=self.RANGE_BEARING_NIS_LIMIT
        )

    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()
