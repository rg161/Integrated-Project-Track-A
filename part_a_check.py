"""Numerical Jacobian checks and local controllability/observability evidence."""

import argparse
import json
from pathlib import Path
import numpy as np
from track_a.model import f, F_jac, G_jac, H_jac, h_range_bearing


def numerical_jacobian(fun, z, epsilon=1e-6):
    z = np.asarray(z, dtype=float)
    return np.column_stack(
        [
            (
                fun(z + np.eye(len(z))[i] * epsilon)
                - fun(z - np.eye(len(z))[i] * epsilon)
            )
            / (2 * epsilon)
            for i in range(len(z))
        ]
    )


def analyse():
    dt = 0.05
    x = np.array([0.5, -0.2, 0.5])
    u = np.array([0.1, 0.3])
    m = np.array([2.0, 1.0])
    checks = {
        "F": float(
            np.max(
                abs(F_jac(x, u, dt) - numerical_jacobian(lambda xx: f(xx, u, dt), x))
            )
        ),
        "G": float(
            np.max(
                abs(G_jac(x, u, dt) - numerical_jacobian(lambda uu: f(x, uu, dt), u))
            )
        ),
        "H": float(
            np.max(
                abs(
                    H_jac(x, m)
                    - numerical_jacobian(lambda xx: h_range_bearing(xx, m), x)
                )
            )
        ),
    }
    result = {"jacobian_max_errors": checks, "controllability": {}, "observability": {}}
    for v in [0.0, 0.15]:
        A = F_jac(np.zeros(3), [v, 0], dt)
        B = G_jac(np.zeros(3), [v, 0], dt)
        C = np.hstack([B, A @ B, A @ A @ B])
        s = np.linalg.svd(C, compute_uv=False)
        result["controllability"][str(v)] = {
            "rank": int(np.linalg.matrix_rank(C)),
            "singular_values": s.tolist(),
        }
    for name, landmarks in {
        "single_point": [[2.0, 1.0]],
        "two_points": [[2.0, 1.0], [-1.0, 2.0]],
        "two_close_points": [[2.0, 1.0], [2.1, 1.0]],
    }.items():
        xx = np.array([0.0, 0.0, 0.3])
        uu = np.array([0.15, 0.3])
        Phi = np.eye(3)
        rows = []
        for _ in range(30):
            rows.extend([H_jac(xx, m) @ Phi for m in landmarks])
            Phi = F_jac(xx, uu, dt) @ Phi
            xx = f(xx, uu, dt)
        s = np.linalg.svd(np.vstack(rows), compute_uv=False)
        result["observability"][name] = {
            "rank": int(np.sum(s > 1e-8)),
            "singular_values": s.tolist(),
        }
    result["observability"]["oriented_marker_full_pose"] = {
        "rank": 3,
        "measurement_jacobian": "I_3",
    }
    result["interpretation"] = (
        "Ranks describe the stated local linearisation. A point landmark's range/bearing has "
        "a rotation-about-landmark gauge; the integrated camera uses the full pose of an "
        "oriented, mapped marker. At rest the instantaneous linear model has rank 2; "
        "this does not imply the nonlinear vehicle cannot manoeuvre laterally."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results/model_analysis.json",
    )
    args = parser.parse_args()
    result = analyse()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
