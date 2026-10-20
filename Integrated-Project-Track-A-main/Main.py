"""Run the integrated project: python main.py (viewer) or --headless."""

import argparse
import json
import os
from pathlib import Path
import sys
from dataclasses import replace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window (camera still renders)",
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Pace headless execution to real time"
    )
    parser.add_argument(
        "--scenario",
        choices=["nominal", "noisy", "dropout", "disturbed", "no_camera"],
        default="nominal",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--duration", type=float, default=100.0, help="Maximum simulated seconds"
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.headless and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from track_a.config import Config, TRIALS, ROOT
    from track_a.runner import run

    config = replace(Config(), max_duration=args.duration)
    output = args.output or ROOT / "results" / f"{args.scenario}_seed{args.seed}"
    try:
        metrics = run(
            config,
            TRIALS[args.scenario],
            args.seed,
            output,
            viewer=not args.headless,
            realtime=not args.headless or args.realtime,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("Interrupted; simulator closed.")
        return 130
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {output.resolve()}")
    return 0 if metrics["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
