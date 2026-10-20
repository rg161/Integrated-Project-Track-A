"""Numerical model checks used by tests and project evidence."""

import json
import numpy as np
from track_a.model import f, F_jac, G_jac, H_jac, h_range_bearing


def num_jac(fn, z, eps=1e-6):
    z = np.asarray(z, dtype=float)
    cols = []
    for i in range(len(z)):
        d = np.zeros_like(z); d[i] = eps
        cols.append((fn(z + d) - fn(z - d)) / (2.0 * eps))
    return np.array(cols).T


def controllability(v, dt=0.05):
    x = np.zeros(3); u = np.array([v, 0.0])
    A, B = F_jac(x, u, dt), G_jac(x, u, dt)
    C = np.hstack([B, A @ B, A @ A @ B])
    s = np.linalg.svd(C, compute_uv=False)
    return int(np.linalg.matrix_rank(C)), s


def observability(x0, u, landmarks, dt=0.05, N=30):
    x = np.asarray(x0, dtype=float).copy(); Phi = np.eye(3); rows = []
    for _ in range(N):
        for m in landmarks:
            rows.append(H_jac(x, m) @ Phi)
        Phi = F_jac(x, u, dt) @ Phi
        x = f(x, u, dt)
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int(np.sum(s > 1e-8)), s


def analyse():
    dt = 0.05
    x0 = np.array([0.5, -0.2, 0.5]); u0 = np.array([0.1, 0.3]); m = np.array([2.0, 1.0])
    errors = {
        "F": float(np.max(np.abs(F_jac(x0, u0, dt) - num_jac(lambda x: f(x, u0, dt), x0)))),
        "G": float(np.max(np.abs(G_jac(x0, u0, dt) - num_jac(lambda u: f(x0, u, dt), u0)))),
        "H": float(np.max(np.abs(H_jac(x0, m) - num_jac(lambda x: h_range_bearing(x, m), x0)))),
    }
    ctrb = {}
    for v in [0.0, 0.15]:
        rank, s = controllability(v, dt)
        ctrb[str(v)] = {"rank": rank, "singular_values": s.tolist()}
    cases = {
        "single_point": [np.array([2.0, 1.0])],
        "two_points": [np.array([2.0, 1.0]), np.array([-1.0, 2.0])],
        "two_nearly_aligned": [np.array([2.0, 1.0]), np.array([4.0, 2.0])],
        "two_close": [np.array([2.0, 1.0]), np.array([2.1, 1.0])],
    }
    obs = {}
    for name, landmarks in cases.items():
        rank, s = observability([0, 0, 0.3], [0.15, 0.3], landmarks, dt)
        obs[name] = {"rank": rank, "singular_values": s.tolist()}
    return {"jacobian_max_errors": errors, "controllability": ctrb, "observability": obs}


if __name__ == "__main__":
    print(json.dumps(analyse(), indent=2))
