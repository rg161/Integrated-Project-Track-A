"""Evaluation-only truth access, reproducible logs, error metrics and figures."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
import platform
import numpy as np
from .ground_truth import GroundTruth
from .model import wrap


class Evaluator:
    def __init__(self, simulator, config, trial, seed, output):
        self.truth = GroundTruth(simulator.model, simulator.data)
        self.config, self.trial, self.seed = config, trial, seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def record(
        self, time, estimate, P, odometry, control, camera_age, ekf, camera_seen
    ):
        true = self.truth.pose()
        error = estimate - true
        error[2] = wrap(error[2])
        row = {
            "time": float(time),
            "mode": control.mode,
            "camera_seen": int(camera_seen),
            "camera_age": float(camera_age),
            "nis": float(ekf.last_nis),
            "nees": float(error @ np.linalg.solve(P, error)),
            "v": control.v,
            "omega": control.omega,
            "wheel_left": float(control.wheels[0]),
            "wheel_right": float(control.wheels[1]),
            "camera_updates": ekf.accepted,
            "camera_rejections": ekf.rejected,
        }
        for prefix, values in [
            ("true", true),
            ("est", estimate),
            ("odom", odometry),
            ("ref", control.reference),
            ("error", error),
        ]:
            row.update(
                {f"{prefix}_{k}": float(v) for k, v in zip(("x", "y", "yaw"), values)}
            )
        for i in range(3):
            for j in range(3):
                row[f"P{i}{j}"] = float(P[i, j])
        self.rows.append(row)

    def save(self, simulator, ekf, declared_done, termination, planner):
        import cv2
        import mujoco
        import scipy
        import matplotlib

        path = self.output / "log.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

        def arr(k):
            return np.array([r[k] for r in self.rows])

        error = np.column_stack([arr("error_" + k) for k in ("x", "y", "yaw")])
        final = self.truth.pose()
        goal = np.array(self.config.goal)
        docking_distance = float(np.linalg.norm(final[:2] - goal[:2]))
        docking_heading = float(abs(wrap(final[2] - goal[2])))
        tracking = np.hypot(arr("true_x") - arr("ref_x"), arr("true_y") - arr("ref_y"))
        dead_error = np.hypot(
            arr("odom_x") - arr("true_x"), arr("odom_y") - arr("true_y")
        )
        sigma = np.sqrt(np.column_stack([arr(f"P{i}{i}") for i in range(3)]))
        metrics = {
            "trial": self.trial.name,
            "seed": self.seed,
            "termination": termination,
            "controller_declared_docked": bool(declared_done),
            "success": bool(
                declared_done
                and docking_distance <= 0.08
                and docking_heading <= np.deg2rad(8)
                and simulator.collisions == 0
            ),
            "duration_s": float(self.rows[-1]["time"]),
            "samples": len(self.rows),
            "position_rmse_m": float(
                np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))
            ),
            "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 2] ** 2)))),
            "odometry_position_rmse_m": float(np.sqrt(np.mean(dead_error**2))),
            "reference_tracking_rmse_m": float(np.sqrt(np.mean(tracking**2))),
            "final_docking_error_m": docking_distance,
            "final_heading_error_deg": float(np.rad2deg(docking_heading)),
            "mean_nees": float(np.mean(arr("nees"))),
            "two_sigma_coverage_xyz": np.mean(
                np.abs(error) <= 2 * sigma, axis=0
            ).tolist(),
            "control_effort_v2_dt": float(np.trapezoid(arr("v") ** 2, arr("time"))),
            "control_effort_omega2_dt": float(
                np.trapezoid(arr("omega") ** 2, arr("time"))
            ),
            "collision_events": simulator.collisions,
            "camera_updates": ekf.accepted,
            "camera_rejections": ekf.rejected,
            "success_thresholds": {
                "position_m": 0.08,
                "heading_deg": 8,
                "collision_events": 0,
                "requires_recent_marker": True,
            },
        }
        (self.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        manifest = {
            "config": asdict(self.config),
            "trial": asdict(self.trial),
            "seed": self.seed,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "mujoco": mujoco.__version__,
                "opencv": cv2.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "camera": "rendered RGB + ArUco + PnP; no synthetic pose shortcut",
            "initial_pose": "configured prior, not measured simulator pose",
        }
        (self.output / "run_config.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        np.savez_compressed(
            self.output / "map.npz",
            log_odds=planner.log_odds,
            path=planner.path,
            origin=planner.origin,
            resolution=planner.res,
        )
        self._plot(metrics, planner)
        return metrics

    def _plot(self, metrics, planner):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        def arr(k):
            return np.array([r[k] for r in self.rows])

        t = arr("time")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        ax = axes[0, 0]
        bounds = self.config.bounds
        # imshow extent is xmin,xmax,ymin,ymax (the config uses that same order).
        ax.imshow(
            planner.log_odds,
            origin="lower",
            extent=bounds,
            cmap="Greys",
            vmin=0,
            vmax=4,
            alpha=0.35,
        )
        ax.plot(arr("true_x"), arr("true_y"), label="MuJoCo truth", c="#182c48")
        ax.plot(arr("est_x"), arr("est_y"), label="EKF", c="#00857b", ls="--")
        ax.plot(
            arr("odom_x"), arr("odom_y"), label="Odometry only", c="#c37a22", ls=":"
        )
        ax.scatter(
            *self.config.goal[:2], marker="*", s=100, c="#c5444d", label="Dock goal"
        )
        for row in self.rows[:: max(1, len(self.rows) // 12)]:
            P = np.array([[row["P00"], row["P01"]], [row["P10"], row["P11"]]])
            val, vec = np.linalg.eigh(P)
            angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
            ax.add_patch(
                Ellipse(
                    (row["est_x"], row["est_y"]),
                    4 * np.sqrt(val[1]),
                    4 * np.sqrt(val[0]),
                    angle=angle,
                    fill=False,
                    color="#00857b",
                    alpha=0.4,
                )
            )
        ax.set(
            xlabel="x (m)",
            ylabel="y (m)",
            title="Path, LiDAR map and 2σ ellipses",
            aspect="equal",
        )
        ax.legend(fontsize=8)
        for ax, key, i, label in [
            (axes[0, 1], "x", 0, "x error (m)"),
            (axes[0, 2], "y", 1, "y error (m)"),
            (axes[1, 0], "yaw", 2, "Heading error (deg)"),
        ]:
            scale = 180 / np.pi if key == "yaw" else 1
            err = arr("error_" + key) * scale
            sigma = np.sqrt(arr(f"P{i}{i}")) * scale
            ax.fill_between(
                t, -2 * sigma, 2 * sigma, color="#00857b", alpha=0.15, label="±2σ"
            )
            ax.plot(t, err, c="#00857b", lw=1, label="estimate − truth")
            ax.set(xlabel="Time (s)", ylabel=label)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        ax = axes[1, 1]
        ax.plot(t, arr("v"), label="v (m/s)")
        ax.plot(t, arr("omega"), label="ω (rad/s)")
        ax.set(xlabel="Time (s)", title="Applied commands")
        ax.legend()
        ax.grid(alpha=0.2)
        ax = axes[1, 2]
        ax.plot(t, arr("nees"), label="NEES", c="#744aa1")
        ax.axhline(3, c="gray", ls="--", label="State dimension (reference)")
        ax.set(xlabel="Time (s)", title="Estimator consistency")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.suptitle(
            f"{metrics['trial']} · seed {self.seed} · success={metrics['success']} · "
            f"position RMSE {metrics['position_rmse_m']:.3f} m · "
            f"dock error {metrics['final_docking_error_m']:.3f} m"
        )
        fig.savefig(self.output / "summary.png", dpi=150)
        plt.close(fig)
