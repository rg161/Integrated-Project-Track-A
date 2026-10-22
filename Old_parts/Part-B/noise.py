import warnings
from dataclasses import dataclass
import numpy as np

def clean_scan(ranges, r_min: float = 0.12, r_max: float = 3.5, window: int = 3, circular: bool = True) -> np.ndarray:
    """
    Clean a 1D lidar range scan.
 
    Steps:
      1. Mark non-finite readings and readings outside [r_min, r_max] as NaN.
      2. Apply a sliding-window median filter to suppress isolated spikes.
      3. Keep readings that were invalid as NaN (the filter never "fills in" holes).
 
    Args:
        ranges:   Raw range readings (metres), one per beam.
        r_min:    Minimum valid range (sensor blind zone).
        r_max:    Maximum valid range (sensor limit).
        window:   Median window size in beams; must be a positive odd integer.
        circular: True for a 360-degree scan (beam 0 neighbours the last beam),
                  False for a partial field of view with hard edges.
 
    Returns:
        Array of the same length as `ranges` with invalid beams set to NaN.
    """
    # Window must be centred on a beam, so it must be odd and at least 1
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd number")
    # Copy into a float array so the caller's data is not modified in place
    r = np.array(ranges, dtype = float)
    # Invalidate NaN/inf, too-close and too-far readings in a single boolean mask
    r[~np.isfinite(r) | (r < r_min) | (r > r_max)] = np.nan
    # A window of 1 means no filtering; return the masked scan directly
    if window == 1:
        return r
    # Half-width: number of neighbours on each side of the centre beam
    h = window // 2
    if circular:
        # Shift the scan by -h..+h so each row holds one neighbour offset
        # (np.roll wraps around, which is what we want for a full 360-degree scan)
        stack = np.stack([np.roll(r, s) for s in range(-h, h + 1)])
    else:
        # Pad both ends with NaN so edge beams have "missing" neighbours instead of wrapped ones
        pad = np.pad(r, h, constant_values=np.nan)
        # Take `window` shifted slices of the padded array, each the same length as r
        stack = np.stack([pad[i:i + len(r)] for i in range(window)]) # all-NaN windows
        # NaN-aware median down each column, i.e. across the window for every beam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN windows
        med = np.nanmedian(stack, axis=0)
    # Do not let the median invent values for beams that were themselves invalid
    med[np.isnan(r)] = np.nan
    return med

def scan_to_points(ranges, angle_min: float, angle_inc: float) -> np.ndarray:
    """
    Convert polar lidar readings to 2D Cartesian points in the sensor frame.
 
    Args:
        ranges:    Range per beam (metres); NaN/inf beams are dropped.
        angle_min: Angle of the first beam (rad).
        angle_inc: Angular step between consecutive beams (rad).
 
    Returns:
        (N, 2) array of [x, y] points, where N is the number of finite beams.
    """
    r = np.asarray(ranges, dtype=float)
    # Bearing of every beam: start angle plus index times the angular resolution
    ang = angle_min + angle_inc * np.arange(len(r))
    # Only convert beams that have a valid range
    ok = np.isfinite(r)
    # x = r cos(theta), y = r sin(theta), stacked into columns
    return np.column_stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])

def cluster_scan(ranges, angle_inc: float, sigma_r: float = 0.01, lam: float = np.deg2rad(10.0), min_points: int = 3) -> list:
    """
    Segment a scan into clusters of consecutive beams using an adaptive breakpoint test.
 
    Two neighbouring valid beams belong to the same cluster if the Euclidean distance
    between their points is below a threshold that grows with range (far points are
    naturally more widely spaced). The threshold follows the Adaptive Breakpoint
    Detector idea:
 
        D_max = r_a * sin(dphi) / sin(lam - dphi) + 3 * sigma_r
 
    where `lam` is the largest surface incidence angle you want to still count as one
    continuous surface, and `sigma_r` is the range noise.
 
    Args:
        ranges:     Cleaned ranges (NaN for invalid beams).
        angle_inc:  Angular step between beams (rad).
        sigma_r:    Range noise standard deviation (m).
        lam:        Breakpoint angle parameter (rad); must exceed the beam gap dphi.
        min_points: Discard clusters smaller than this (likely noise).
 
    Returns:
        List of arrays, each holding the beam indices of one cluster.
    """
    r = np.asarray(ranges, dtype=float)
    # Indices of valid beams only; NaN beams are skipped, so gaps are measured in index distance
    idx = np.flatnonzero(np.isfinite(r))
    if idx.size == 0:
        return []
    # `clusters` collects finished clusters; `cur` is the cluster being built
    clusters, cur = [], [idx[0]]
    # Walk over each pair of consecutive valid beams (a -> b)
    for a, b in zip(idx[:-1], idx[1:]):
        # Angular gap between the two beams (larger if beams in between were invalid)
        dphi = angle_inc * (b - a)
        if dphi >= lam:
            # Gap too wide: the threshold formula would blow up (sin(lam - dphi) <= 0), so always start a new cluster
            brk = True
        else:
            # Law of cosines: straight-line distance between the two measured points
            d = np.sqrt(r[a] ** 2 + r[b] ** 2 - 2 * r[a] * r[b] * np.cos(dphi))
            # Break if the points are farther apart than the adaptive threshold allows
            brk = d > r[a] * np.sin(dphi) / np.sin(lam - dphi) + 3 * sigma_r
        if brk:
            # Close the current cluster and start a new one at beam b
            clusters.append(np.array(cur))
            cur = []
        cur.append(b)
    # Don't forget the last cluster still being built when the loop ends
    clusters.append(np.array(cur))
    # Keep only clusters with enough points to be considered real objects
    return [c for c in clusters if len(c) >= min_points]

@dataclass(frozen=True)
class ArucoGate:
    """
    Thresholds used to accept or reject an ArUco marker pose estimate.
    Frozen so a gate configuration cannot be changed accidentally after creation.
    """
    max_range: float = 3.0                  # m, camera-frame distance
    min_side_px: float = 20.0               # mean marker side length in pixels
    max_incidence: float = np.deg2rad(70.0) # beyond this the planar pose flips easily
    max_reproj_px: float = 1.5              # RMS corner reprojection error

def incidence_cos(tvec, R_cm) -> float:
    """
    Cosine of the angle between the marker's normal and the line of sight to the camera.
 
    1.0 means the marker is viewed head-on; values near 0 mean it is seen edge-on (grazing).
 
    Args:
        tvec: Marker position in the camera frame (3,).
        R_cm: Rotation matrix of the marker in the camera frame (3x3);
              its third column is the marker's normal (z axis).
    """
    t = np.asarray(tvec, dtype=float).ravel()
    # Unit vector pointing from the marker back to the camera
    to_cam = -t / np.linalg.norm(t)
    # Dot product of the marker normal with that direction = cos(incidence angle)
    return float(np.asarray(R_cm, dtype=float)[:, 2] @ to_cam)

def mean_side_px(corners_px) -> float:
    """
    Average side length of the detected marker in pixels (a proxy for how well it is resolved).
 
    Args:
        corners_px: The 4 marker corners in image coordinates, in order around the marker.
    """
    c = np.asarray(corners_px, dtype=float).reshape(4, 2)
    # Subtract each corner from the next one (np.roll wraps, closing the quadrilateral), 
    # take the length of each edge, then average the four edges
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))

def aruco_check(tvec, R_cm, corners_px, reproj_px: float, gate: ArucoGate = ArucoGate()):
    """
    Decide whether an ArUco pose estimate passes quality gating.
 
    Checks run from cheapest/most fundamental to more specific, returning at the first failure.
 
    Args:
        tvec:       Marker translation in the camera frame.
        R_cm:       Marker rotation in the camera frame.
        corners_px: Detected corner pixels.
        reproj_px:  RMS reprojection error of the pose fit (pixels).
        gate:       Acceptance thresholds.
 
    Returns:
        (accepted, reason) where reason is one of
        "ok", "invalid", "reproj", "small", "range" or "grazing".
    """
    # Gather every numeric input into one vector so a single check catches any NaN/inf
    vals = np.concatenate([np.ravel(tvec), np.ravel(R_cm), np.ravel(corners_px), [reproj_px]])
    if not np.all(np.isfinite(vals)):
        return False, "invalid"
    # Poor corner fit: the pose solution is unreliable
    if reproj_px > gate.max_reproj_px:
        return False, "reproj"
    # Marker is too small in the image, so corner localisation is too coarse
    if mean_side_px(corners_px) < gate.min_side_px:
        return False, "small"
    # Marker is too far away for accurate range
    if np.linalg.norm(tvec) > gate.max_range:
        return False, "range"
    # Viewing angle too oblique; planar pose estimation can flip ambiguously.
    # clip guards arccos against tiny numerical overshoot beyond [-1, 1]
    if np.arccos(np.clip(incidence_cos(tvec, R_cm), -1, 1)) > gate.max_incidence:
        return False, "grazing"
    return True, "ok"

@dataclass(frozen=True)
class ArucoNoiseModel:
    """
    Range/bearing measurement noise model for ArUco observations.
 
    Uncertainty comes from pixel noise on the detected corners:
      - range error grows with the square of distance and shrinks with marker size and focal length,
      - bearing error is roughly pixel noise divided by focal length,
      - both are inflated when the marker is viewed at a steep angle,
      - both are bounded below by constant floors (unmodelled errors such as calibration).
    """
    f_px: float                                  # camera focal length in pixels
    marker_side: float                           # physical marker side length (m)
    sigma_px: float = 0.5                        # corner detection noise (pixels)
    c_r: float = 1.0                             # tuning scale for range noise
    c_phi: float = 1.0                           # tuning scale for bearing noise
    sigma_r_floor: float = 0.005                 # m
    sigma_phi_floor: float = np.deg2rad(0.1)     # rad
 
    def cov(self, r: float, cos_inc: float = 1.0) -> np.ndarray:
        """
        Return the 2x2 diagonal covariance of a [range, bearing] measurement.
 
        Args:
            r:       Measured distance to the marker (m).
            cos_inc: Cosine of the incidence angle (1 = head-on).
        """
        # Inflation factor 1/cos(incidence); cos is clamped at 0.2 so the factor never exceeds 5
        infl = 1.0 / max(cos_inc, 0.2)
        # Range std dev: depth error ~ sigma_px * r^2 / (f * marker size)
        s_r = self.c_r * self.sigma_px * r ** 2 / (self.f_px * self.marker_side) * infl
        # Bearing std dev: pixel error converted to an angle via the focal length
        s_p = self.c_phi * self.sigma_px / self.f_px * infl
        # Combine each model value with its floor in quadrature (sqrt(a^2 + b^2)), then square to get variances on the diagonal
        return np.diag([np.hypot(s_r, self.sigma_r_floor) ** 2,
                        np.hypot(s_p, self.sigma_phi_floor) ** 2])

def wheel_increments_to_motion(dphi_l: float, dphi_r: float, r: float, b: float):
    """
    Convert wheel rotation increments to robot motion (differential-drive kinematics).
 
    Args:
        dphi_l: Left wheel rotation increment (rad).
        dphi_r: Right wheel rotation increment (rad).
        r:      Wheel radius (m).
        b:      Wheel separation / track width (m).
 
    Returns:
        (ds, dtheta): forward distance travelled (m) and heading change (rad).
    """
    # Distance rolled by each wheel (arc length = radius * angle)
    dl, dr = r * dphi_l, r * dphi_r
    # Robot centre moves the average; heading changes with the difference divided by track width
    return (dl + dr) / 2.0, (dr - dl) / b

def odom_cov(dphi_l: float, dphi_r: float, r: float, b: float, k: float = 1e-4) -> np.ndarray:
    """
    Covariance of the (ds, dtheta) odometry increment.
 
    Assumes independent wheel errors whose variance is proportional to the distance each
    wheel travelled, then propagates them through the linear kinematics: Cov = J S J^T.
 
    Args:
        dphi_l, dphi_r: Wheel rotation increments (rad).
        r:              Wheel radius (m).
        b:              Track width (m).
        k:              Noise constant; variance per metre of wheel travel.
 
    Returns:
        2x2 covariance matrix for [ds, dtheta].
    """
    # Wheel-space covariance: independent left/right errors growing with distance travelled
    S = np.diag([k * abs(r * dphi_l), k * abs(r * dphi_r)])
    # Jacobian of (ds, dtheta) with respect to (dl, dr), taken from the kinematics above
    J = np.array([[0.5, 0.5], [-1.0 / b, 1.0 / b]])
    # Linear error propagation to motion space
    return J @ S @ J.T
