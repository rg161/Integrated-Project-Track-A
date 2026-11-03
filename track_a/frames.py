"""Minimal coordinate-frame transforms needed by the ArUco sensing pipeline."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Transform:
    """Represent ``T_parent_child`` where ``p_parent = R @ p_child + t``."""

    parent: str
    child: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """Validate and store rotation/translation arrays with fixed shapes."""
        R = np.asarray(self.R, dtype=float).reshape(3, 3)
        t = np.asarray(self.t, dtype=float).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("Transform contains non-finite values")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def inv(self):
        """Return the inverse transform with parent and child frames swapped."""
        return Transform(self.child, self.parent, self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other):
        """Compose two frame-compatible transforms using the ``@`` operator."""
        if self.child != other.parent:
            raise ValueError(
                f"Cannot compose T_{self.parent}_{self.child} with "
                f"T_{other.parent}_{other.child}"
            )
        return Transform(
            self.parent,
            other.child,
            self.R @ other.R,
            self.R @ other.t + self.t,
        )


def transform_to_pose2d(T):
    """Extract planar ``(x, y, yaw)`` from a 3D homogeneous-frame transform."""
    return float(T.t[0]), float(T.t[1]), float(np.arctan2(T.R[1, 0], T.R[0, 0]))
