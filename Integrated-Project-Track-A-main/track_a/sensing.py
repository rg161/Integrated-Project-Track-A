"""Part B camera processing. Inputs are RGB pixels and fixed calibration only."""

from dataclasses import dataclass
import cv2
import numpy as np
from .frames import (
    Transform,
    camera_mount,
    marker_mount,
    transform_to_pose2d,
    wrap_angle,
)
from .noise import aruco_check, ArucoGate


@dataclass(frozen=True)
class CameraMeasurement:
    pose: np.ndarray
    covariance: np.ndarray
    reprojection_px: float
    marker_id: int
    range_m: float


class ArucoCamera:
    def __init__(self, config):
        self.config = config
        f = config.camera_height / (2 * np.tan(np.deg2rad(config.camera_fovy) / 2))
        self.K = np.array(
            [
                [f, 0, (config.camera_width - 1) / 2],
                [0, f, (config.camera_height - 1) / 2],
                [0, 0, 1.0],
            ]
        )
        self.distortion = np.zeros(
            5
        )  # ideal MuJoCo pinhole, not a real camera calibration
        self.T_B_C, self.T_W_M = camera_mount(config), marker_mount(config)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Preserve the outer border candidate when a close marker fills the image.
        params.minMarkerDistanceRate = 0.05
        if hasattr(params, "minGroupDistance"):
            params.minGroupDistance = 0.05
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        s = config.marker_side / 2
        self.object_points = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=float
        )
        self.gate = ArucoGate(max_range=3.5, min_side_px=16.0)
        self.last_reason = "not_seen"
        self.last_corners = None

    def _to_pose(self, rvec, tvec):
        T_C_M = Transform("C", "M", cv2.Rodrigues(rvec)[0], np.asarray(tvec).ravel())
        T_W_B = self.T_W_M @ T_C_M.inv() @ self.T_B_C.inv()
        return np.array(transform_to_pose2d(T_W_B))

    def _pose_covariance(self, corners, rvec, tvec, noise_scale):
        # Local pixel-to-planar-pose Jacobian with a fixed PnP solution branch.
        # This keeps position/heading cross covariance caused by the camera lever arm.
        J = np.zeros((3, 8))
        eps = 0.1
        for j in range(8):
            values = []
            for sign in (-1, 1):
                c = corners.copy().reshape(-1)
                c[j] += sign * eps
                ok, rv, tv = cv2.solvePnP(
                    self.object_points,
                    c.reshape(4, 2),
                    self.K,
                    self.distortion,
                    rvec.copy(),
                    tvec.copy(),
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    return np.diag([0.15**2, 0.15**2, np.deg2rad(12) ** 2])
                values.append(self._to_pose(rv, tv))
            diff = values[1] - values[0]
            diff[2] = wrap_angle(diff[2])
            J[:, j] = diff / (2 * eps)
        sigma = max(0.5, self.config.corner_sigma * noise_scale)
        return J @ J.T * sigma**2 + np.diag([0.012**2, 0.012**2, np.deg2rad(1.5) ** 2])

    def measure(self, rgb, rng, noise_scale=1.0):
        self.last_corners = None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.config.marker_id not in ids:
            self.last_reason = "not_seen"
            return None
        index = list(ids.ravel()).index(self.config.marker_id)
        pts = np.asarray(corners[index], dtype=float).reshape(4, 2)
        pts += rng.normal(0, self.config.corner_sigma * noise_scale, (4, 2))
        ok, rv, tv = cv2.solvePnP(
            self.object_points,
            pts,
            self.K,
            self.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.last_reason = "pnp_failed"
            return None
        projected = cv2.projectPoints(
            self.object_points, rv, tv, self.K, self.distortion
        )[0].reshape(4, 2)
        reproj = float(np.sqrt(np.mean(np.sum((projected - pts) ** 2, axis=1))))
        accepted, self.last_reason = aruco_check(
            tv, cv2.Rodrigues(rv)[0], pts, reproj, self.gate
        )
        self.last_corners = pts
        if not accepted:
            return None
        return CameraMeasurement(
            self._to_pose(rv, tv),
            self._pose_covariance(pts, rv, tv, noise_scale),
            reproj,
            self.config.marker_id,
            float(np.linalg.norm(tv)),
        )
