"""
Shared discrete differential-drive/unicycle model used by Part A and EKF.

note:
state -> x = [x, y, theta] position and heading 
input -> u = [v, w] speed and turning speed

"""

import numpy as np

R_WHEEL = 0.033 #wheel radius (m) measured, at 3rad/s
"""
XML gave R = 0.033m, L = 0.288m, 
correct calculations found as simulated bot had smaller wheels, larger seperation
"""
L_WHEEL = 0.288 #wheel separation (m) measured
"""
XML gave R = 0.033m, L = 0.288m, 
correct calculations found as simulated bot had smaller wheels, larger seperation
"""


def wrap(a):
    """Wrap radians to [-pi, pi). Works on scalars and arrays."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def wheels_to_v_w(wl, wr, r=R_WHEEL, L=L_WHEEL):
    """Wheel angular rates [rad/s] -> body linear/yaw rates."""
    return r * (wl + wr) / 2.0, r * (wr - wl) / L


def v_w_to_wheels(v, w, r=R_WHEEL, L=L_WHEEL, limit=None):
    """
    Body command -> wheel rates, preserving curvature if saturated.
    convert wheel speeds (rad/s) to linear and angular velocity (m/s, rad/s)
    """
    wl = (v - 0.5 * L * w) / r
    wr = (v + 0.5 * L * w) / r
    wheels = np.array([wl, wr], dtype=float)
    if limit is not None:
        peak = float(np.max(np.abs(wheels)))
        if peak > limit and peak > 0:
            wheels *= float(limit) / peak
    return wheels


def f(x, u, dt):
    """Midpoint discrete unicycle model."""
    x = np.asarray(x, dtype=float)
    v, w = np.asarray(u, dtype=float)
    #input is linear and angular velocity
    th_mid = x[2] + 0.5 * w * dt
    return np.array(
        [
            x[0] + v * np.cos(th_mid) * dt,
            x[1] + v * np.sin(th_mid) * dt,
            float(wrap(x[2] + w * dt)),
        ]
    )

#JACOBIANS

def F_jac(x, u, dt):
    """Jacobian df/dx of :func:`f`."""
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
    """Jacobian df/du of :func:`f`, for u=[v, omega]."""
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


def h_range_bearing(x, m):
    """Range and bearing from pose x to mapped point m."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = dx * dx + dy * dy
    if q < 1e-14:
        raise ValueError("Range/bearing undefined at the landmark position")
    return np.array([np.sqrt(q), float(wrap(np.arctan2(dy, dx) - x[2]))])


def H_jac(x, m):
    """Jacobian dh/dx of :func:`h_range_bearing`."""
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = dx * dx + dy * dy
    if q < 1e-14:
        raise ValueError("Range/bearing Jacobian undefined at zero range")
    rho = np.sqrt(q)
    return np.array([[-dx / rho, -dy / rho, 0.0], [dy / q, -dx / q, -1.0]])
