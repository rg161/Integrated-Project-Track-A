"""Frame transforms for W(world), B(base), C(OpenCV camera), L(LiDAR), M(marker)."""

from dataclasses import dataclass, field
import numpy as np


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quat_wxyz_to_R(q):
    """Convert a finite, nonzero MuJoCo-order quaternion (w,x,y,z)."""
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.all(np.isfinite(q)):
        raise ValueError("Quaternion contains non-finite values")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# MuJoCo camera: +x right, +y up, looks along -z.
# OpenCV optical: +x right, +y down, looks along +z.
R_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Transform:
    """T_parent_child: p_parent = R @ p_child + t."""

    parent: str
    child: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        R = np.asarray(self.R, dtype=float).reshape(3, 3)
        t = np.asarray(self.t, dtype=float).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("Transform contains non-finite values")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    def matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inv(self):
        return Transform(self.child, self.parent, self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other):
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

    def apply(self, pts):
        p = np.asarray(pts, dtype=float)
        return p @ self.R.T + self.t


def pose2d_to_transform(parent, child, x, y, th):
    return Transform(parent, child, rot_z(th), np.array([x, y, 0.0]))


def transform_to_pose2d(T):
    return float(T.t[0]), float(T.t[1]), float(np.arctan2(T.R[1, 0], T.R[0, 0]))


class FrameTree:
    ROOT = "W"

    def __init__(self):
        self._edges = {}

    def set_static(self, T):
        if not isinstance(T, Transform):
            raise TypeError("Expected Transform")
        if T.child == self.ROOT:
            raise ValueError("World/root frame cannot have a parent")
        if T.parent == T.child:
            raise ValueError("A frame cannot be its own parent")
        # Check the proposed parent chain before committing the edge.
        seen = {T.child}
        node = T.parent
        while node != self.ROOT:
            if node in seen:
                raise ValueError("Frame graph cycle detected")
            seen.add(node)
            edge = self._edges.get(node)
            if edge is None:
                break
            node = edge.parent
        self._edges[T.child] = T

    def set_base_pose(self, x, y, th):
        self.set_static(pose2d_to_transform("W", "B", x, y, th))

    def _T_root_frame(self, frame, trail=None):
        if frame == self.ROOT:
            return Transform(self.ROOT, self.ROOT)
        if frame not in self._edges:
            raise KeyError(f"Frame '{frame}' has no transform set")
        trail = set() if trail is None else set(trail)
        if frame in trail:
            raise ValueError("Frame graph cycle detected")
        trail.add(frame)
        e = self._edges[frame]
        return self._T_root_frame(e.parent, trail) @ e

    def get(self, to_frame, from_frame):
        """Return T_to_from."""
        return self._T_root_frame(to_frame).inv() @ self._T_root_frame(from_frame)

    def transform(self, pts, from_frame, to_frame):
        return self.get(to_frame, from_frame).apply(pts)


def build_static_frames(
    tree,
    cam_pos_B,
    cam_quat_B_wxyz,
    lidar_pos_B,
    lidar_quat_B_wxyz=(1, 0, 0, 0),
    marker_pos_W=None,
    marker_R_W=None,
):
    R_B_C = quat_wxyz_to_R(cam_quat_B_wxyz) @ R_MJ_TO_CV
    tree.set_static(Transform("B", "C", R_B_C, np.asarray(cam_pos_B, dtype=float)))
    tree.set_static(
        Transform(
            "B",
            "L",
            quat_wxyz_to_R(lidar_quat_B_wxyz),
            np.asarray(lidar_pos_B, dtype=float),
        )
    )
    if marker_pos_W is not None:
        R = np.eye(3) if marker_R_W is None else np.asarray(marker_R_W, dtype=float)
        tree.set_static(Transform("W", "M", R, np.asarray(marker_pos_W, dtype=float)))
