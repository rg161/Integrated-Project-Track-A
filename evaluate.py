"""Repeated seeded end-to-end robustness trials, with aggregate JSON and CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--scenarios", nargs="+", default=["nominal", "noisy", "dropout", "disturbed"]
    )
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results/robustness",
    )
    args = parser.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from track_a.config import Config, TRIALS
    from track_a.runner import run

    if any(s not in TRIALS for s in args.scenarios):
        parser.error("Unknown scenario")
    config = replace(Config(), max_duration=args.duration)
    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            print(f"Running {scenario}, seed {seed}", flush=True)
            result = run(
                config,
                TRIALS[scenario],
                seed,
                args.output / f"{scenario}_seed{seed}",
                verbose=False,
            )
            rows.append(result)
            print(
                f"  success={result['success']}, dock error={result['final_docking_error_m']:.3f} m, "
                f"collisions={result['collision_events']}",
                flush=True,
            )
    summary = {}
    for scenario in args.scenarios:
        selected = [r for r in rows if r["trial"] == scenario]
        summary[scenario] = {
            "trials": len(selected),
            "successes": sum(r["success"] for r in selected),
            "success_rate": float(np.mean([r["success"] for r in selected])),
            "mean_position_rmse_m": float(
                np.mean([r["position_rmse_m"] for r in selected])
            ),
            "mean_final_docking_error_m": float(
                np.mean([r["final_docking_error_m"] for r in selected])
            ),
            "collision_events": sum(r["collision_events"] for r in selected),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    keys = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with (args.output / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0 if all(r["success"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
