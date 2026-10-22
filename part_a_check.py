"""
Numerical model checks used by tests and project evidence.

Part A: checks for differential-drive model in model.py

- compare derived and numerical jacobians

- controllability of system (both still and driving)

- observability from range-bearing to a landmark m

- observability with more than 1 landmark

Run from the project root: python part_a_check.py   (plain-text report)
                           python part_a_check.py --json   (JSON, as before)
"""

import json
import sys
import numpy as np
from track_a.model import f, F_jac, G_jac, H_jac, h_range_bearing


def num_jac(fn, z, eps=1e-6):
    """
    find num_jac at point z to check formulas
    measure outputs, record w/ columns
    """
    z = np.asarray(z, dtype=float)
    cols = []
    for i in range(len(z)):
        d = np.zeros_like(z); d[i] = eps
        cols.append((fn(z + d) - fn(z - d)) / (2.0 * eps))
    return np.array(cols).T


def controllability(v, dt=0.05):
    """
    check if robot is controllable
    can v and w get to any pose
    rank 3 if all state directions are achievable
    """
    x = np.zeros(3); u = np.array([v, 0.0])
    A, B = F_jac(x, u, dt), G_jac(x, u, dt)
    C = np.hstack([B, A @ B, A @ A @ B])
    s = np.linalg.svd(C, compute_uv=False)
    return int(np.linalg.matrix_rank(C)), s


def _obs_matrix(x0, u, landmarks, dt=0.05, N=30):
    """Stacked observability matrix: one H*Phi row block per landmark per step.

    simulates N steps of motion from x0 with constant input u. 
    at each step, each landmark's measurement sensitivity H is mapped back to the starting pose
    """
    x = np.asarray(x0, dtype=float).copy(); Phi = np.eye(3); rows = []
    for _ in range(N):
        for m in landmarks:
            rows.append(H_jac(x, m) @ Phi)
        Phi = F_jac(x, u, dt) @ Phi
        x = f(x, u, dt)
    return np.vstack(rows)


def observability(x0, u, landmarks, dt=0.05, N=30):
    """
    check measurements from landmark location
    if observable from m (landmark)

    Returns (rank, singular values) of the stacked observability matrix.
    rank 3 -> recovered full pose
    """
    s = np.linalg.svd(_obs_matrix(x0, u, landmarks, dt, N), compute_uv=False)
    return int(np.sum(s > 1e-8)), s


def null_direction(x0, u, landmarks, dt=0.05, N=30):
    """Direction in state space the measurements cannot see (last row of Vt).
    Only meaningful when the rank is below 3. Defined up to sign."""
    _, _, Vt = np.linalg.svd(_obs_matrix(x0, u, landmarks, dt, N))
    return Vt[-1]


def analyse():
    """
    Run every check and return the results as a dictionary (used by tests,
    and by the --json option).
    """
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


def report():
    """Print the results as plain text, one line per check."""
    result = analyse()

    # 1. Jacobian errors (about 1e-8 or smaller means the formulas match)
    for name, err in result["jacobian_max_errors"].items():
        print(f"{name} error:", err)

    # 2. Controllability: standstill vs driving
    for v, r in result["controllability"].items():
        s = np.round(np.array(r["singular_values"]), 8)
        print(f"v = {v}: controllability rank {r['rank']}, singular values {s}")

    # 3. One landmark: the unobservable direction should be rotation about the
    # landmark, [m_y - y, -(m_x - x), 1] for a robot at the origin (up to sign)
    m = np.array([2.0, 1.0])
    null = null_direction([0, 0, 0.3], [0.15, 0.3], [m])
    pred = np.array([-(0 - m[1]), (0 - m[0]), 1.0]); pred /= np.linalg.norm(pred)
    print("null vector:", np.round(null, 4), "predicted:", np.round(pred, 4))

    # 4. Landmark layouts: rank says whether the pose is recoverable at all
    labels = {
        "single_point": "one landmark",
        "two_points": "two, far apart",
        "two_nearly_aligned": "two, nearly aligned",
        "two_close": "two, close together",
    }
    for key, r in result["observability"].items():
        s = np.round(np.array(r["singular_values"]), 6)
        print(f"{labels[key]}: rank {r['rank']}, singular values {s}")


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(analyse(), indent=2))
    else:
        report()