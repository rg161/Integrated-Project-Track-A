import warnings
from dataclasses import dataclass
import numpy as np


def clean_scan(
    ranges,
    r_min: float = 0.12,
    r_max: float = 3.5,
    window: int = 3,
    circular: bool = True,
) -> np.ndarray:
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
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
        med = np.nanmedian(stack, axis=0)
    med[np.isnan(r)] = np.nan
    return med


def scan_to_points(ranges, angle_min: float, angle_inc: float) -> np.ndarray:
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r)
    return np.column_stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])])


def cluster_scan(
    ranges,
    angle_inc: float,
    sigma_r: float = 0.01,
    lam: float = np.deg2rad(10.0),
    min_points: int = 3,
) -> list:
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
            brk = d > r[a] * np.sin(dphi) / np.sin(lam - dphi) + 3 * sigma_r
        if brk:
            clusters.append(np.array(cur))
            cur = []
        cur.append(b)
    clusters.append(np.array(cur))
    return [c for c in clusters if len(c) >= min_points]


@dataclass(frozen=True)
class ArucoGate:
    max_range: float = 3.0  # m, camera-frame distance
    min_side_px: float = 20.0  # mean marker side length in pixels
    max_incidence: float = np.deg2rad(70.0)  # beyond this the planar pose flips easily
    max_reproj_px: float = 1.5  # RMS corner reprojection error


def incidence_cos(tvec, R_cm) -> float:
    t = np.asarray(tvec, dtype=float).ravel()
    if np.linalg.norm(t) < 1e-12:
        return -1.0
    to_cam = -t / np.linalg.norm(t)
    return float(np.asarray(R_cm, dtype=float)[:, 2] @ to_cam)


def mean_side_px(corners_px) -> float:
    c = np.asarray(corners_px, dtype=float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def aruco_check(
    tvec, R_cm, corners_px, reproj_px: float, gate: ArucoGate = ArucoGate()
):
    vals = np.concatenate(
        [np.ravel(tvec), np.ravel(R_cm), np.ravel(corners_px), [reproj_px]]
    )
    if not np.all(np.isfinite(vals)):
        return False, "invalid"
    if np.asarray(tvec).ravel()[2] <= 0:
        return False, "behind"
    if reproj_px > gate.max_reproj_px:
        return False, "reproj"
    if mean_side_px(corners_px) < gate.min_side_px:
        return False, "small"
    if np.linalg.norm(tvec) > gate.max_range:
        return False, "range"
    if np.arccos(np.clip(incidence_cos(tvec, R_cm), -1, 1)) > gate.max_incidence:
        return False, "grazing"
    return True, "ok"


@dataclass(frozen=True)
class ArucoNoiseModel:
    f_px: float
    marker_side: float
    sigma_px: float = 0.5
    c_r: float = 1.0
    c_phi: float = 1.0
    sigma_r_floor: float = 0.005  # m
    sigma_phi_floor: float = np.deg2rad(0.1)  # rad

    def cov(self, r: float, cos_inc: float = 1.0) -> np.ndarray:
        infl = 1.0 / max(cos_inc, 0.2)
        s_r = self.c_r * self.sigma_px * r**2 / (self.f_px * self.marker_side) * infl
        s_p = self.c_phi * self.sigma_px / self.f_px * infl
        return np.diag(
            [
                np.hypot(s_r, self.sigma_r_floor) ** 2,
                np.hypot(s_p, self.sigma_phi_floor) ** 2,
            ]
        )


def wheel_increments_to_motion(dphi_l: float, dphi_r: float, r: float, b: float):
    dl, dr = r * dphi_l, r * dphi_r
    return (dl + dr) / 2.0, (dr - dl) / b


def odom_cov(
    dphi_l: float, dphi_r: float, r: float, b: float, k: float = 1e-4
) -> np.ndarray:
    S = np.diag([k * abs(r * dphi_l), k * abs(r * dphi_r)])
    J = np.array([[0.5, 0.5], [-1.0 / b, 1.0 / b]])
    return J @ S @ J.T
