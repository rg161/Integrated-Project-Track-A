"""Sensor cleaning, gating and odometry-noise helpers."""

import warnings
from dataclasses import dataclass
import numpy as np


def clean_scan(ranges, r_min=0.12, r_max=3.5, window=3, circular=True):
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd number")
    r = np.array(ranges, dtype=float)
    r[~np.isfinite(r) | (r < r_min) | (r > r_max)] = np.nan
    if window == 1:
        return r
    h = window // 2
    if circular:
        stack = np.stack([np.roll(r, s) for s in range(-h, h + 1)])
    else:
        pad = np.pad(r, h, constant_values=np.nan)
        stack = np.stack([pad[i : i + len(r)] for i in range(window)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    # A median filter may suppress a spike but must not invent a return where
    # the sensor originally reported no data.
    med[np.isnan(r)] = np.nan
    return med


def scan_to_points(ranges, angle_min, angle_inc):
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r)
    return np.column_stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])


def cluster_scan(ranges, angle_inc, sigma_r=0.01, lam=np.deg2rad(10.0), min_points=3):
    r = np.asarray(ranges, dtype=float)
    idx = np.flatnonzero(np.isfinite(r))
    if idx.size == 0:
        return []
    clusters, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        dphi = angle_inc * (b - a)
        if dphi >= lam:
            brk = True
        else:
            d = np.sqrt(r[a] ** 2 + r[b] ** 2 - 2 * r[a] * r[b] * np.cos(dphi))
            denom = np.sin(lam - dphi)
            threshold = np.inf if abs(denom) < 1e-12 else r[a] * np.sin(dphi) / denom + 3 * sigma_r
            brk = d > threshold
        if brk:
            clusters.append(np.asarray(cur))
            cur = []
        cur.append(b)
    clusters.append(np.asarray(cur))
    return [c for c in clusters if len(c) >= min_points]


@dataclass(frozen=True)
class ArucoGate:
    max_range: float = 3.0
    min_side_px: float = 20.0
    max_incidence: float = np.deg2rad(70.0)
    max_reproj_px: float = 1.5


def incidence_cos(tvec, R_cm):
    t = np.asarray(tvec, dtype=float).ravel()
    n = float(np.linalg.norm(t))
    if n < 1e-12 or not np.isfinite(n):
        return -1.0
    to_cam = -t / n
    return float(np.asarray(R_cm, dtype=float)[:, 2] @ to_cam)


def mean_side_px(corners_px):
    c = np.asarray(corners_px, dtype=float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def aruco_check(tvec, R_cm, corners_px, reproj_px, gate=ArucoGate()):
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


@dataclass(frozen=True)
class ArucoNoiseModel:
    f_px: float
    marker_side: float
    sigma_px: float = 0.5
    c_r: float = 1.0
    c_phi: float = 1.0
    sigma_r_floor: float = 0.005
    sigma_phi_floor: float = np.deg2rad(0.1)

    def cov(self, r, cos_inc=1.0):
        infl = 1.0 / max(float(cos_inc), 0.2)
        s_r = self.c_r * self.sigma_px * r**2 / (self.f_px * self.marker_side) * infl
        s_p = self.c_phi * self.sigma_px / self.f_px * infl
        return np.diag(
            [
                np.hypot(s_r, self.sigma_r_floor) ** 2,
                np.hypot(s_p, self.sigma_phi_floor) ** 2,
            ]
        )


def wheel_increments_to_motion(dphi_l, dphi_r, r, b):
    dl, dr = r * dphi_l, r * dphi_r
    return (dl + dr) / 2.0, (dr - dl) / b


def odom_cov(dphi_l, dphi_r, r, b, k=1e-4):
    """Covariance of [delta_s, delta_theta] from wheel travel uncertainty."""
    S = np.diag([k * abs(r * dphi_l), k * abs(r * dphi_r)])
    J = np.array([[0.5, 0.5], [-1.0 / b, 1.0 / b]])
    return J @ S @ J.T
