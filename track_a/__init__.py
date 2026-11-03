"""Integrated Track A runtime package.

The package deliberately separates simulator truth from sensing, estimation,
planning and control.  Only :mod:`track_a.simulator`, :mod:`track_a.ground_truth`
and :mod:`track_a.evaluation` are allowed to touch MuJoCo physical state.
"""

from .config import Config, Trial, TRIALS

__all__ = ["Config", "Trial", "TRIALS"]
