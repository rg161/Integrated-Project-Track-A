"""Part C: EKF fusing wheel-encoder odometry and camera measurements."""

import numpy as np
from .model import f, F_jac, G_jac, H_jac, h_range_bearing, wheels_to_v_w, wrap
from .noise import odom_cov


class MobileRobotEKF:
    """
    Estimator class
    """
    #Normalized innovation squared used to reject unrealistic camera measurments
    POSE_NIS_LIMIT = 16.27          # approx 99.9% chi-square, dof=3
    RANGE_BEARING_NIS_LIMIT = 13.82 # approx 99.9% chi-square, dof=2

    def __init__(self, x0=(0.0, 0.0, 0.0), P0=None):
        """
        Creates new EKF
        """
        # Gurantees that x is stored as exactly three floating point values
        self.x = np.asarray(x0, dtype=float).reshape(3).copy()
        self.x[2] = float(wrap(self.x[2]))
        # determins uncertainty about current state
        self.P = (
            np.diag([0.08**2, 0.08**2, np.deg2rad(6.0) ** 2])
            if P0 is None
            else np.asarray(P0, dtype=float).reshape(3, 3).copy()
        )
        # Continuous unmodelled process covariance; multiplied by dt.
        #process noise
        self.Qc = np.diag([0.003**2, 0.003**2, np.deg2rad(0.3) ** 2])
        # R represents measurment uncertainty small R means trust sensor
        self.R_pose = np.diag([0.025**2, 0.025**2, np.deg2rad(2.0) ** 2])
        self.R_range_bearing = np.diag([0.05**2, np.deg2rad(2.0) ** 2])
        #debugging / evaluation variables
        self.last_nis = np.nan
        self.last_update_accepted = False
        self.last_innovation = None

    @staticmethod
    def _symmetrise(P):
        """
        Keep covariance matrix mathematically valid and symmetric
        if not symmetric take the average

        takes in the convariance matric.

        decomposes covariance matrix into eigenvalues/ eigenvectors
        """
        P = 0.5 * (P + P.T)
        # Very small numerical negative eigenvalues can otherwise accumulate.
        vals, vecs = np.linalg.eigh(P)
        if vals.min() <= 1e-14:
            vals = np.maximum(vals, 1e-14)
            # make symmetric again
            P = (vecs * vals) @ vecs.T
        return 0.5 * (P + P.T)

    def reset(self, x0, P0=None):
        """
        reset the EKF
        """
        self.__init__(x0, P0)

    def predict(self, v, omega, dt, input_cov=None):
        """
        performs EKF prediction step

        inputs 
        v = robot linear velocity
        w = robot angular velocity
        dt = time step
        input_cov uncertainty in motion input OPTIONAL

        returns the copys by returning copies it prevents outside code from modifying EKS internal state
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        #save old state
        x_old = self.x.copy()
        # create input vector
        u = np.array([v, omega], dtype=float)
        # transition jacobian
        F = F_jac(x_old, u, dt)
        # calculate uncertainty in v,w x,y,0
        G = G_jac(x_old, u, dt)
        # predict state
        self.x = f(x_old, u, dt)
        # input covariance
        Sigma_u = np.zeros((2, 2)) if input_cov is None else np.asarray(input_cov, dtype=float).reshape(2, 2)
        #predect covariance
        self.P = F @ self.P @ F.T + G @ Sigma_u @ G.T + self.Qc * dt
        # clear covariance
        self.P = self._symmetrise(self.P)
        #clear camera updaate information
        self.last_update_accepted = False
        self.last_nis = np.nan
        self.last_innovation = None
        # returns the copies
        return self.x.copy(), self.P.copy()

    def predict_encoders(self, increments, dt, config):
        """Predict from measured wheel-angle increments, not wheel commands."""

        # read wheel incements
        inc = np.asarray(increments, dtype=float).reshape(2)
        #convert increment to angular speed
        wl, wr = inc / dt
        # convert wheels into robot motion
        v, omega = wheels_to_v_w(wl, wr, config.wheel_radius, config.wheel_base)

        # Part B wheel-travel model + variance from differencing two independent
        # noisy encoder readings at consecutive samples.
        motion_cov = odom_cov(
            inc[0], inc[1], config.wheel_radius, config.wheel_base, k=config.odom_k
        )
        #encoder measurment variance
        enc_var = 2.0 * (config.wheel_radius * config.encoder_sigma) ** 2
        #jacobian from wheel movement to robot movement
        J = np.array([[0.5, 0.5], [-1.0 / config.wheel_base, 1.0 / config.wheel_base]])
        # add encoder uncertainty
        motion_cov += J @ np.diag([enc_var, enc_var]) @ J.T
        # convert displacement covariance to velocity covariance
        input_cov = motion_cov / (dt * dt)
        return self.predict(v, omega, dt, input_cov=input_cov)

    def _linear_update(self, z, h, H, R, angle_indices, nis_limit):
        """
        measurment correction update function
        arguments 
        z = actual sensor measurement
        h = predicted sensor measurement
        H = measurement jacobian
        R = sensor covariance
        angle_indices = which measurement entries are angles
        nis_limit = how large an innovation can  be before rejecting measurment
        
        return true if accepted
        """
        # Force correct array shapes 
        z = np.asarray(z, dtype=float).reshape(-1)
        h = np.asarray(h, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        # reject invalid measurements
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(R))):
            self.last_update_accepted = False
            return False
        # innovation
        # how different is what was measured to what i expected to measure
        nu = z - h
        for idx in angle_indices:
            nu[idx] = float(wrap(nu[idx]))
        # innovation covariance
        # how uncertain should i expect the measuremnt difference to be
        S = H @ self.P @ H.T + R
        try:
            #calculate NIS without matrix inverse
            Sinv_nu = np.linalg.solve(S, nu)
            nis = float(nu @ Sinv_nu)
            #singuar matrix prtection
        except np.linalg.LinAlgError:
            self.last_update_accepted = False
            return False
        self.last_nis = nis
        self.last_innovation = nu.copy()
        #suspicious camera measurment does not midify EKF
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
        #correct the state
        self.x = self.x + K @ nu
        #wrap heading
        self.x[2] = float(wrap(self.x[2]))
        # joseph covariance update
        I = np.eye(3)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = self._symmetrise(self.P)
        self.last_update_accepted = True
        #return true if measurerment accepted
        return True

    def update_pose(self, camera_pose, R=None):
        """
        Update pose 3 dimensional measurment
        """
        R = self.R_pose if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        z = np.asarray(camera_pose, dtype=float).reshape(3)
        return self._linear_update(
            z, self.x.copy(), np.eye(3), R, angle_indices=(2,), nis_limit=self.POSE_NIS_LIMIT
        )

    def update_range_bearing(self, measured_range, measured_bearing, marker_position, R=None):
        """
        update_range_bearing 
        """
        marker = np.asarray(marker_position, dtype=float).reshape(2)
        try:
            #predict what sensor should be
            h = h_range_bearing(self.x, marker)
            #measurement jacobian
            H = H_jac(self.x, marker)
        #invalid landmark geometry protection
        except ValueError:
            self.last_update_accepted = False
            return False
        R = self.R_range_bearing if R is None else np.asarray(R, dtype=float).reshape(2, 2)
        z = np.array([measured_range, measured_bearing], dtype=float)
        return self._linear_update(
            z, h, H, R, angle_indices=(1,), nis_limit=self.RANGE_BEARING_NIS_LIMIT
        )

    def get_state(self):
        """
        returns x,y,angle
        """
        return self.x.copy()
    
    def get_covariance(self):
        """
        return current 3x3 covariance matrix
        """
        return self.P.copy()
