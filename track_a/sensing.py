"""ArUco camera processing that converts one RGB frame into a robot world-pose measurement."""

from dataclasses import dataclass
import cv2
import numpy as np

from .frames import Transform, transform_to_pose2d
from .model import wrap
from .noise import aruco_check

# OpenCV optical axes expressed in the robot base at zero mount rotation:
# optical +x = body -y, optical +y = body -z, optical +z = body +x.
R_B_C = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Marker axes in the world. The charging marker faces toward world -x.
R_W_M = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class ArucoMeasurement:
    """Store one accepted ArUco measurement used by estimation and final visual docking.

    ``pose`` and ``R`` are consumed by the EKF. ``bearing`` is the marker
    direction in the robot/camera horizontal plane with positive meaning the
    marker is to the robot's left; ``range_m`` is the camera-to-marker distance.
    """

    pose: np.ndarray
    R: np.ndarray
    bearing: float
    range_m: float


class ArucoCamera:
    """Detect marker ID 7, solve its pose, and return the robot pose in world coordinates."""

    def __init__(self, config):
        """Build camera intrinsics, ArUco detector, marker geometry and fixed frame transforms."""
        self.c = config
        focal = config.camera_height / (
            2.0 * np.tan(np.deg2rad(config.camera_fovy) / 2.0)
        )
        self.K = np.array(
            [
                [focal, 0.0, (config.camera_width - 1) / 2.0],
                [0.0, focal, (config.camera_height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.distortion = np.zeros(5, dtype=float)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.02
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 6.0
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

        half = config.marker_side / 2.0
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        self.T_B_C = Transform(
            "B", "C", R_B_C, np.asarray(config.camera_pos, dtype=float)
        )
        self.T_W_M = Transform(
            "W", "M", R_W_M, np.asarray(config.marker_pos, dtype=float)
        )
        self.last_reason = "not_run"

    def _detect(self, gray):
        """Detect all ArUco candidates in one grayscale image."""
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary)

    def _solve_pose(self, corners):
        """Convert four image corners into robot world pose using IPPE square PnP and frames."""
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

        translation = np.asarray(tvec, dtype=float).reshape(3)
        if not np.all(np.isfinite(translation)) or translation[2] <= 0:
            return None

        R_cm, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            self.object_points, rvec, tvec, self.K, self.distortion
        )
        projected = projected.reshape(4, 2)
        reprojection = float(
            np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1)))
        )

        T_C_M = Transform("C", "M", R_cm, translation)
        T_W_B = self.T_W_M @ T_C_M.inv() @ self.T_B_C.inv()
        pose = np.array(transform_to_pose2d(T_W_B), dtype=float)
        pose[2] = float(wrap(pose[2]))
        return pose, R_cm, translation, reprojection

    def _pose_covariance(self, corners, sigma_px):
        """Estimate 3x3 pose covariance by finite-differencing each noisy image corner coordinate."""
        corners = np.asarray(corners, dtype=float).reshape(4, 2)
        J = np.zeros((3, 8), dtype=float)
        eps = 0.25
        for j in range(8):
            plus = corners.copy()
            minus = corners.copy()
            plus.flat[j] += eps
            minus.flat[j] -= eps
            solved_plus = self._solve_pose(plus)
            solved_minus = self._solve_pose(minus)
            if solved_plus is None or solved_minus is None:
                continue
            delta = solved_plus[0] - solved_minus[0]
            delta[2] = float(wrap(delta[2]))
            J[:, j] = delta / (2.0 * eps)

        covariance = J @ (sigma_px**2 * np.eye(8)) @ J.T
        covariance += np.diag([0.012**2, 0.012**2, np.deg2rad(1.5) ** 2])
        covariance = 0.5 * (covariance + covariance.T)
        vals, vecs = np.linalg.eigh(covariance)
        vals = np.maximum(vals, 1e-10)
        return (vecs * vals) @ vecs.T

    def measure(self, rgb, rng=None, noise_scale=1.0):
        """Return an accepted ArUco robot-pose measurement, or ``None`` when detection is unusable."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            self.last_reason = "invalid_image"
            return None
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
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
            detected += rng.normal(
                0.0,
                self.c.corner_sigma * noise_scale,
                detected.shape,
            )

        solved = self._solve_pose(detected)
        if solved is None:
            self.last_reason = "pnp"
            return None
        pose, R_cm, tvec, reprojection = solved

        accepted, reason = aruco_check(tvec, R_cm, detected, reprojection)
        if not accepted:
            self.last_reason = reason
            return None

        sigma_px = max(0.5, self.c.corner_sigma * max(noise_scale, 1.0))
        covariance = self._pose_covariance(detected, sigma_px)

        # OpenCV optical coordinates use +x to the image right and +z forward.
        # The robot convention used by the controller treats positive bearing as
        # left, so negate atan2(x, z): marker right -> negative bearing.
        bearing = float(-np.arctan2(tvec[0], tvec[2]))
        range_m = float(np.linalg.norm(tvec))

        self.last_reason = "ok"
        return ArucoMeasurement(
            pose=pose,
            R=covariance,
            bearing=bearing,
            range_m=range_m,
        )
