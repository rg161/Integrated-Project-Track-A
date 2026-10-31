"""Shared differential-drive model used by modelling, estimation and control."""

import numpy as np

R_WHEEL = 0.033
L_WHEEL = 0.288


def wrap(angle):
    """Wrap scalar or array angles in radians to the interval ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wheels_to_v_w(wl, wr, r=R_WHEEL, L=L_WHEEL):
    """Convert left/right wheel angular rates to body linear and yaw rates."""
    return r * (wl + wr) / 2.0, r * (wr - wl) / L


def v_w_to_wheels(v, w, r=R_WHEEL, L=L_WHEEL, limit=None):
    """Convert body ``(v, omega)`` to wheel rates while preserving curvature when saturated."""
    wl = (v - 0.5 * L * w) / r
    wr = (v + 0.5 * L * w) / r
    wheels = np.array([wl, wr], dtype=float)
    if limit is not None:
        peak = float(np.max(np.abs(wheels)))
        if peak > limit and peak > 0:
            wheels *= float(limit) / peak
    return wheels


def f(x, u, dt):
    """Propagate ``[x, y, theta]`` one step with the midpoint unicycle model."""
    x = np.asarray(x, dtype=float)
    v, w = np.asarray(u, dtype=float)
    th_mid = x[2] + 0.5 * w * dt
    return np.array(
        [
            x[0] + v * np.cos(th_mid) * dt,
            x[1] + v * np.sin(th_mid) * dt,
            float(wrap(x[2] + w * dt)),
        ]
    )


def F_jac(x, u, dt):
    """Return the state Jacobian ``df/dx`` used to propagate EKF covariance."""
    x = np.asarray(x, dtype=float)
    v, w = np.asarray(u, dtype=float)
    th_mid = x[2] + 0.5 * w * dt
    return np.array(
        [
            [1.0, 0.0, -v * np.sin(th_mid) * dt],
            [0.0, 1.0, v * np.cos(th_mid) * dt],
            [0.0, 0.0, 1.0],
        ]
    )


def G_jac(x, u, dt):
    """Return the input Jacobian ``df/du`` for ``u=[v, omega]``."""
    x = np.asarray(x, dtype=float)
    v, w = np.asarray(u, dtype=float)
    th_mid = x[2] + 0.5 * w * dt
    return np.array(
        [
            [dt * np.cos(th_mid), -0.5 * v * dt**2 * np.sin(th_mid)],
            [dt * np.sin(th_mid), 0.5 * v * dt**2 * np.cos(th_mid)],
            [0.0, dt],
        ]
    )
