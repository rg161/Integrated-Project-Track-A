"""Part C EKF: wheel-encoder prediction plus ArUco full-pose correction."""

import numpy as np
from .model import F_jac, G_jac, f, wheels_to_v_w, wrap
from .noise import odom_cov


class MobileRobotEKF:
    """Estimate the TurtleBot3 state ``[x, y, theta]`` and its covariance."""

    POSE_NIS_LIMIT = 16.27  # ~99.9% chi-square threshold, 3 measurement DOF.

    def __init__(self, x0=(0.0, 0.0, 0.0), P0=None):
        """Initialise state, covariance, process noise and camera noise assumptions."""
        self.x = np.asarray(x0, dtype=float).reshape(3).copy()
        self.x[2] = float(wrap(self.x[2]))
        self.P = (
            np.diag([0.08**2, 0.08**2, np.deg2rad(6.0) ** 2])
            if P0 is None
            else np.asarray(P0, dtype=float).reshape(3, 3).copy()
        )
        self.Qc = np.diag([0.003**2, 0.003**2, np.deg2rad(0.3) ** 2])
        self.R_pose = np.diag([0.025**2, 0.025**2, np.deg2rad(2.0) ** 2])
        self.last_nis = np.nan
        self.last_update_accepted = False
        self.last_innovation = None

    @staticmethod
    def _symmetrise(P):
        """Force covariance symmetry and floor tiny negative numerical eigenvalues."""
        P = 0.5 * (P + P.T)
        vals, vecs = np.linalg.eigh(P)
        if vals.min() <= 1e-14:
            vals = np.maximum(vals, 1e-14)
            P = (vecs * vals) @ vecs.T
        return 0.5 * (P + P.T)

    def reset(self, x0, P0=None):
        """Reset the estimator to a new initial state and optional covariance."""
        self.__init__(x0, P0)

    def predict(self, v, omega, dt, input_cov=None):
        """Predict state/covariance from body velocity and optional input covariance."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        x_old = self.x.copy()
        u = np.array([v, omega], dtype=float)
        F = F_jac(x_old, u, dt)
        G = G_jac(x_old, u, dt)
        self.x = f(x_old, u, dt)
        Sigma_u = (
            np.zeros((2, 2))
            if input_cov is None
            else np.asarray(input_cov, dtype=float).reshape(2, 2)
        )
        self.P = F @ self.P @ F.T + G @ Sigma_u @ G.T + self.Qc * dt
        self.P = self._symmetrise(self.P)
        self.last_update_accepted = False
        self.last_nis = np.nan
        self.last_innovation = None
        return self.x.copy(), self.P.copy()

    def predict_encoders(self, increments, dt, config):
        """Predict from measured left/right wheel-angle increments and encoder uncertainty."""
        inc = np.asarray(increments, dtype=float).reshape(2)
        wl, wr = inc / dt
        v, omega = wheels_to_v_w(wl, wr, config.wheel_radius, config.wheel_base)

        motion_cov = odom_cov(
            inc[0], inc[1], config.wheel_radius, config.wheel_base, k=config.odom_k
        )
        enc_var = 2.0 * (config.wheel_radius * config.encoder_sigma) ** 2
        J = np.array(
            [[0.5, 0.5], [-1.0 / config.wheel_base, 1.0 / config.wheel_base]]
        )
        motion_cov += J @ np.diag([enc_var, enc_var]) @ J.T
        input_cov = motion_cov / (dt * dt)
        return self.predict(v, omega, dt, input_cov=input_cov)

    def _pose_update(self, z, R):
        """Perform NIS-gated Kalman correction for a full ``[x, y, theta]`` camera pose."""
        z = np.asarray(z, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(R))):
            self.last_update_accepted = False
            return False

        H = np.eye(3)
        innovation = z - self.x
        innovation[2] = float(wrap(innovation[2]))
        S = H @ self.P @ H.T + R

        try:
            solved = np.linalg.solve(S, innovation)
            nis = float(innovation @ solved)
        except np.linalg.LinAlgError:
            self.last_update_accepted = False
            return False

        self.last_nis = nis
        self.last_innovation = innovation.copy()
        if not np.isfinite(nis) or nis > self.POSE_NIS_LIMIT:
            self.last_update_accepted = False
            return False

        PHt = self.P @ H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError:
            self.last_update_accepted = False
            return False

        self.x = self.x + K @ innovation
        self.x[2] = float(wrap(self.x[2]))
        I = np.eye(3)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = self._symmetrise(self.P)
        self.last_update_accepted = True
        return True

    def update_pose(self, camera_pose, R=None):
        """Correct the EKF with an ArUco-derived world pose and its covariance."""
        covariance = self.R_pose if R is None else R
        return self._pose_update(camera_pose, covariance)

    def get_state(self):
        """Return a copy of the current estimated state ``[x, y, theta]``."""
        return self.x.copy()

    def get_covariance(self):
        """Return a copy of the current 3x3 state covariance matrix."""
        return self.P.copy()
