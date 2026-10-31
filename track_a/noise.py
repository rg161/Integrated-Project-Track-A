"""Noise handling used by the Track A LiDAR, camera and wheel-encoder pipeline."""

import warnings
from dataclasses import dataclass
import numpy as np


def clean_scan(ranges, r_min=0.12, r_max=3.5, window=3, circular=True):
    """Filter LiDAR hits while preserving no-return beams as positive infinity.

    Finite ranges inside ``[r_min, r_max]`` are median-filtered. Positive
    infinity means the LiDAR saw no obstacle before its maximum range and is
    intentionally preserved so the occupancy mapper can mark that ray as free
    space. Invalid/too-close values remain ``NaN`` and are ignored.
    """
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd number")

    raw = np.asarray(ranges, dtype=float)
    no_return = np.isposinf(raw) | (raw > r_max)
    valid = np.isfinite(raw) & (raw >= r_min) & (raw <= r_max)

    filtered_input = np.full(raw.shape, np.nan, dtype=float)
    filtered_input[valid] = raw[valid]
    if window == 1:
        out = filtered_input
    else:
        h = window // 2
        if circular:
            stack = np.stack([np.roll(filtered_input, s) for s in range(-h, h + 1)])
        else:
            pad = np.pad(filtered_input, h, constant_values=np.nan)
            stack = np.stack([pad[i : i + len(filtered_input)] for i in range(window)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmedian(stack, axis=0)
        # Do not invent a hit at a beam that did not itself contain a valid hit.
        out[~valid] = np.nan

    out = np.asarray(out, dtype=float)
    out[no_return] = np.inf
    return out


@dataclass(frozen=True)
class ArucoGate:
    """Store simple quality limits used to reject unreliable ArUco detections."""

    max_range: float = 3.0
    min_side_px: float = 20.0
    max_incidence: float = np.deg2rad(70.0)
    max_reproj_px: float = 1.5


def incidence_cos(tvec, R_cm):
    """Return cosine of marker viewing incidence, where 1 means face-on."""
    t = np.asarray(tvec, dtype=float).ravel()
    n = float(np.linalg.norm(t))
    if n < 1e-12 or not np.isfinite(n):
        return -1.0
    to_cam = -t / n
    return float(np.asarray(R_cm, dtype=float)[:, 2] @ to_cam)


def mean_side_px(corners_px):
    """Return the mean marker side length in image pixels from four corners."""
    c = np.asarray(corners_px, dtype=float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def aruco_check(tvec, R_cm, corners_px, reproj_px, gate=ArucoGate()):
    """Accept or reject an ArUco pose using range, size, incidence and reprojection error."""
    vals = np.concatenate(
        [np.ravel(tvec), np.ravel(R_cm), np.ravel(corners_px), [reproj_px]]
    )
    if not np.all(np.isfinite(vals)):
        return False, "invalid"
    if reproj_px > gate.max_reproj_px:
        return False, "reproj"
    if mean_side_px(corners_px) < gate.min_side_px:
        return False, "small"
    if np.linalg.norm(tvec) > gate.max_range:
        return False, "range"
    angle = np.arccos(np.clip(incidence_cos(tvec, R_cm), -1.0, 1.0))
    if angle > gate.max_incidence:
        return False, "grazing"
    return True, "ok"


def odom_cov(dphi_l, dphi_r, r, b, k=1e-4):
    """Return covariance of forward/yaw increments from wheel-travel uncertainty."""
    S = np.diag([k * abs(r * dphi_l), k * abs(r * dphi_r)])
    J = np.array([[0.5, 0.5], [-1.0 / b, 1.0 / b]])
    return J @ S @ J.T
