"""Camera-side ArUco processing.  This module never reads simulator truth.

1. detect aruco markers
2. add simulated corner noise
3. solve markers pose relative to camera
4. reject unreliable detections
5. estimate a covariance for the pose

"""

from dataclasses import dataclass
import numpy as np
import cv2

from .frames import Transform, transform_to_pose2d
from .model import wrap
from .noise import aruco_check, incidence_cos

# Optical camera axes expressed in the robot base at zero mount rotation:
# optical +x = body -y, optical +y = body -z, optical +z = body +x.
R_B_C = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Marker object convention used by IPPE_SQUARE:
# +x = marker right, +y = marker up, +z = out of the printed face.
# The charging marker faces toward world -x.
R_W_M = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class ArucoMeasurement:
    """
    One accepted camera measurement of the robot pose
    """
    pose: np.ndarray
    R: np.ndarray
    reprojection_px: float
    range_m: float
    incidence_cosine: float
    corners: np.ndarray


class ArucoCamera:
    """
    Detects the charging-station marker and turns it into a robot pose
    """
    def __init__(self, config):
        """
        Build the camera model, detector and the marker's known geometry
        """
        self.c = config
        f = config.camera_height / (2.0 * np.tan(np.deg2rad(config.camera_fovy) / 2.0))
        self.K = np.array(
            [[f, 0.0, (config.camera_width - 1) / 2.0], [0.0, f, (config.camera_height - 1) / 2.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        self.distortion = np.zeros(5, dtype=float)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        # Slightly narrower grouping helps the large close-up marker remain a candidate.
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.02
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 6.0
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params) if hasattr(cv2.aruco, "ArucoDetector") else None
        s = config.marker_side / 2.0
        self.object_points = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
            dtype=np.float32,
        )
        self.T_B_C = Transform("B", "C", R_B_C, np.asarray(config.camera_pos, dtype=float))
        self.T_W_M = Transform("W", "M", R_W_M, np.asarray(config.marker_pos, dtype=float))
        self.last_reason = "not_run"

    def _detect(self, gray):
        """
        Detect markers in a grayscale image
        """
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary)

    def _solve_pose(self, corners):
        """
        Robot pose from the 4 marker corners
        """
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners,
            self.K,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        t = np.asarray(tvec, dtype=float).reshape(3)
        if not np.all(np.isfinite(t)) or t[2] <= 0:
            return None
        R_cm, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, self.K, self.distortion)
        projected = projected.reshape(4, 2)
        reproj = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
        T_C_M = Transform("C", "M", R_cm, t)
        T_W_B = self.T_W_M @ T_C_M.inv() @ self.T_B_C.inv()
        pose = np.array(transform_to_pose2d(T_W_B), dtype=float)
        pose[2] = float(wrap(pose[2]))
        return pose, R_cm, t, reproj

    def _pose_covariance(self, corners, base_pose, sigma_px):
        """Finite-difference corner-to-planar-pose covariance around the chosen PnP branch."""
        c = np.asarray(corners, dtype=float).reshape(4, 2)
        J = np.zeros((3, 8), dtype=float)
        eps = 0.25
        for j in range(8):
            cp, cm = c.copy(), c.copy()
            cp.flat[j] += eps
            cm.flat[j] -= eps
            sp, sm = self._solve_pose(cp), self._solve_pose(cm)
            if sp is None or sm is None:
                continue
            dp = sp[0] - sm[0]
            dp[2] = float(wrap(dp[2]))
            J[:, j] = dp / (2.0 * eps)
        R = J @ (sigma_px**2 * np.eye(8)) @ J.T
        R += np.diag([0.012**2, 0.012**2, np.deg2rad(1.5) ** 2])
        R = 0.5 * (R + R.T)
        vals, vecs = np.linalg.eigh(R)
        vals = np.maximum(vals, 1e-10)
        return (vecs * vals) @ vecs.T

    def measure(self, rgb, rng=None, noise_scale=1.0):
        """
        Measure the robot pose from one RGB camera frame
        """
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] < 3:
            self.last_reason = "invalid_image"
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detect(gray)
        if ids is None or len(ids) == 0:
            self.last_reason = "not_found"
            return None
        ids = np.asarray(ids).reshape(-1)
        hits = np.flatnonzero(ids == self.c.marker_id)
        if hits.size == 0:
            self.last_reason = "wrong_id"
            return None
        detected = np.asarray(corners[int(hits[0])], dtype=float).reshape(4, 2)
        if rng is not None and noise_scale > 0 and self.c.corner_sigma > 0:
            detected = detected + rng.normal(0.0, self.c.corner_sigma * noise_scale, detected.shape)

        solved = self._solve_pose(detected)
        if solved is None:
            self.last_reason = "pnp"
            return None
        pose, R_cm, tvec, reproj = solved
        ok, reason = aruco_check(tvec, R_cm, detected, reproj)
        if not ok:
            self.last_reason = reason
            return None
        cos_inc = float(np.clip(incidence_cos(tvec, R_cm), 0.0, 1.0))
        sigma_px = max(0.5, self.c.corner_sigma * max(noise_scale, 1.0))
        cov = self._pose_covariance(detected, pose, sigma_px)
        self.last_reason = "ok"
        return ArucoMeasurement(
            pose=pose,
            R=cov,
            reprojection_px=reproj,
            range_m=float(np.linalg.norm(tvec)),
            incidence_cosine=cos_inc,
            corners=detected.copy(),
        )
