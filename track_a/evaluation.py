"""Evaluation and result export. Ground truth is used here only for scoring."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import numpy as np

from .model import wrap


class Evaluator:
    def __init__(self, config, trial, seed, output, start_pose=None):
        self.c = config
        self.trial = trial
        self.seed = int(seed)
        self.output = Path(output)
        self.start_pose = (
            np.asarray(config.start, dtype=float).copy()
            if start_pose is None
            else np.asarray(start_pose, dtype=float).reshape(3).copy()
        )
        self.rows = []
        self.camera_updates = 0
        self.camera_rejections = 0

    def note_camera(self, accepted):
        if accepted:
            self.camera_updates += 1
        else:
            self.camera_rejections += 1

    def record(self, *, t, estimate, P, truth, odom, reference, wheels, command, mode, nis=np.nan):
        estimate = np.asarray(estimate, dtype=float)
        truth = np.asarray(truth, dtype=float)
        odom = np.asarray(odom, dtype=float)
        reference = np.asarray(reference, dtype=float)
        P = np.asarray(P, dtype=float)
        wheels = np.asarray(wheels, dtype=float)
        err = estimate - truth
        err[2] = float(wrap(err[2]))
        try:
            nees = float(err @ np.linalg.solve(P, err))
        except np.linalg.LinAlgError:
            nees = np.nan
        row = {
            "t": float(t),
            "x_hat": float(estimate[0]), "y_hat": float(estimate[1]), "theta_hat": float(estimate[2]),
            "x_true": float(truth[0]), "y_true": float(truth[1]), "theta_true": float(truth[2]),
            "x_odom": float(odom[0]), "y_odom": float(odom[1]), "theta_odom": float(odom[2]),
            "x_ref": float(reference[0]), "y_ref": float(reference[1]), "theta_ref": float(reference[2]),
            "v": float(command[0]), "omega": float(command[1]),
            "wheel_left": float(wheels[0]), "wheel_right": float(wheels[1]),
            "nis": float(nis) if np.isfinite(nis) else np.nan,
            "nees": nees,
            "mode": str(mode),
            "est_in_dock_zone": int(self.c.in_docking_zone(estimate)),
            "true_in_dock_zone": int(self.c.in_docking_zone(truth)),
        }
        for i in range(3):
            for j in range(3):
                row[f"P{i}{j}"] = float(P[i, j])
        self.rows.append(row)

    def metrics(self, *, controller_done, collisions, termination):
        if not self.rows:
            raise RuntimeError("No evaluation samples were recorded")
        est = np.array([[r["x_hat"], r["y_hat"], r["theta_hat"]] for r in self.rows])
        tru = np.array([[r["x_true"], r["y_true"], r["theta_true"]] for r in self.rows])
        odo = np.array([[r["x_odom"], r["y_odom"], r["theta_odom"]] for r in self.rows])
        ref = np.array([[r["x_ref"], r["y_ref"], r["theta_ref"]] for r in self.rows])
        err = est - tru; err[:, 2] = wrap(err[:, 2])
        od_err = odo - tru; od_err[:, 2] = wrap(od_err[:, 2])
        ref_err = est[:, :2] - ref[:, :2]
        goal = np.asarray(self.c.goal, dtype=float)
        final_pos = float(np.linalg.norm(tru[-1, :2] - goal[:2]))
        final_heading = float(abs(wrap(tru[-1, 2] - goal[2])))
        nees = np.array([r["nees"] for r in self.rows], dtype=float)
        Ps = np.array([[[r[f"P{i}{j}"] for j in range(3)] for i in range(3)] for r in self.rows])
        sig = np.sqrt(np.maximum(np.stack([Ps[:, i, i] for i in range(3)], axis=1), 0.0))
        coverage = np.mean(np.abs(err) <= 2.0 * sig, axis=0)
        v = np.array([r["v"] for r in self.rows]); w = np.array([r["omega"] for r in self.rows])
        final_est_in_zone = self.c.in_docking_zone(est[-1])
        final_true_in_zone = self.c.in_docking_zone(tru[-1])
        true_zone_samples = np.array([r["true_in_dock_zone"] for r in self.rows], dtype=bool)
        est_zone_samples = np.array([r["est_in_dock_zone"] for r in self.rows], dtype=bool)
        success = bool(
            controller_done
            and final_pos <= 0.08
            and final_heading <= np.deg2rad(8.0)
            and collisions == 0
        )
        return {
            "trial": self.trial.name,
            "seed": self.seed,
            "termination": termination,
            "controller_declared_docked": bool(controller_done),
            "success": success,
            "duration_s": float(self.rows[-1]["t"]),
            "samples": len(self.rows),
            "start_pose": self.start_pose.tolist(),
            "docking_zone_bounds": list(self.c.docking_zone_bounds),
            "final_estimate_in_docking_zone": bool(final_est_in_zone),
            "final_true_in_docking_zone": bool(final_true_in_zone),
            "ever_estimate_in_docking_zone": bool(np.any(est_zone_samples)),
            "ever_true_in_docking_zone": bool(np.any(true_zone_samples)),
            "true_time_in_docking_zone_s": float(np.sum(true_zone_samples) * self.c.dt),
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
            "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(err[:, 2] ** 2)))),
            "odometry_position_rmse_m": float(np.sqrt(np.mean(np.sum(od_err[:, :2] ** 2, axis=1)))),
            "reference_tracking_rmse_m": float(np.sqrt(np.mean(np.sum(ref_err**2, axis=1)))),
            "final_docking_error_m": final_pos,
            "final_heading_error_deg": float(np.degrees(final_heading)),
            "mean_nees": float(np.nanmean(nees)),
            "two_sigma_coverage_xyz": coverage.tolist(),
            "control_effort_v2_dt": float(np.sum(v * v) * self.c.dt),
            "control_effort_omega2_dt": float(np.sum(w * w) * self.c.dt),
            "collision_events": int(collisions),
            "camera_updates": int(self.camera_updates),
            "camera_rejections": int(self.camera_rejections),
            "success_thresholds": {
                "position_m": 0.08,
                "heading_deg": 8,
                "collision_events": 0,
                "requires_recent_marker": True,
            },
        }

    def save(self, metrics, planner=None, camera_image=None):
        self.output.mkdir(parents=True, exist_ok=True)
        # CSV
        keys = list(self.rows[0].keys())
        with (self.output / "log.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(self.rows)
        (self.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        config_data = {"config": asdict(self.c), "trial": asdict(self.trial), "seed": self.seed}
        (self.output / "run_config.json").write_text(json.dumps(config_data, indent=2, default=float) + "\n")
        if planner is not None:
            planner.save(self.output / "map.npz")
        if camera_image is not None:
            try:
                import cv2
                cv2.imwrite(str(self.output / "camera.png"), cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR))
            except Exception:
                pass
        self._save_plot(metrics)

    def _save_plot(self, metrics):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        est = np.array([[r["x_hat"], r["y_hat"]] for r in self.rows])
        tru = np.array([[r["x_true"], r["y_true"]] for r in self.rows])
        odo = np.array([[r["x_odom"], r["y_odom"]] for r in self.rows])
        t = np.array([r["t"] for r in self.rows])
        pe = np.linalg.norm(est - tru, axis=1)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(tru[:, 0], tru[:, 1], label="true")
        axs[0].plot(est[:, 0], est[:, 1], label="EKF")
        axs[0].plot(odo[:, 0], odo[:, 1], label="odometry", alpha=0.7)
        axs[0].axis("equal"); axs[0].grid(True); axs[0].legend(); axs[0].set_title("Trajectory")
        axs[1].plot(t, pe)
        axs[1].grid(True); axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("position error [m]")
        axs[1].set_title(f"EKF RMSE {metrics['position_rmse_m']:.3f} m")
        fig.tight_layout(); fig.savefig(self.output / "summary.png", dpi=160); plt.close(fig)
