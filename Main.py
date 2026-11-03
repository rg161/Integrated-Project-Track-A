"""Run the Track A TurtleBot3 simulation.

Examples:
    python main.py
    python main.py --layout random --seed 2
    python main.py --headless --scenario noisy
"""

import argparse
import json
import os
import sys
from dataclasses import replace


def main(argv=None):
    """Parse the small command-line interface, run one simulation, and print metrics."""
    parser = argparse.ArgumentParser(description="Track A TurtleBot3 simulation")
    parser.add_argument(
        "--layout",
        choices=["standard", "random"],
        default="standard",
        help="standard = two fixed obstacles; random = six seed-randomised obstacles",
    )
    parser.add_argument(
        "--scenario",
        choices=["nominal", "noisy", "dropout", "disturbed"],
        default="nominal",
        help="sensor/disturbance condition used for robustness testing",
    )
    parser.add_argument("--seed", type=int, default=1, help="reproducible random seed")
    parser.add_argument("--duration", type=float, default=100.0, help="maximum simulated seconds")
    parser.add_argument("--headless", action="store_true", help="run without the MuJoCo viewer")
    args = parser.parse_args(argv)

    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.headless and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    from track_a.config import Config, ROOT, TRIALS
    from track_a.runner import run

    config = replace(Config(), max_duration=args.duration, layout=args.layout)
    output = ROOT / "results" / f"{args.layout}_{args.scenario}_seed{args.seed}"

    try:
        metrics = run(
            config=config,
            trial=TRIALS[args.scenario],
            seed=args.seed,
            output=output,
            viewer=not args.headless,
        )
    except KeyboardInterrupt:
        print("Interrupted; simulator closed safely.")
        return 130

    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {output.resolve()}")
    return 0 if metrics["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
