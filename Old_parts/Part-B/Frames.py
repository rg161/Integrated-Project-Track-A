from dataclasses import dataclass, field
import numpy as np

def wrap_angle(a):
    """Wrap an angle (or array of angles) in radians into the range (-pi, pi]."""
    # sin/cos discard whole turns, and arctan2 rebuilds the angle in the principal range
    return np.arctan2(np.sin(a), np.cos(a))

def rot_x(a: float) -> np.ndarray:
    """3x3 rotation matrix for a right-handed rotation of `a` radians about the x-axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a: float) -> np.ndarray:
    """3x3 rotation matrix for a right-handed rotation of `a` radians about the y-axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a: float) -> np.ndarray:
    """3x3 rotation matrix for a right-handed rotation of `a` radians about the z-axis (yaw)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def quat_wxyz_to_R(q) -> np.ndarray:
    """
    Convert a quaternion in (w, x, y, z) order to a 3x3 rotation matrix.
 
    The quaternion is normalised first, so it does not need to have exactly unit length.
    """
    # MuJoCo quaternion order is (w, x, y, z).
    # Normalise and unpack the components
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    # Standard quaternion-to-rotation-matrix formula
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 
    ])

# MuJoCo camera looks along -z with +y up
# OpenCV camera looks along +z with +y down
# Flipping y and z converts between the two; the matrix is its own inverse
R_MJ_TO_CV = np.diag([1, -1, -1])

@dataclass(frozen=True)
# T_parent_child: p_parent = R @ p_child + t
class Transform:
    """
    Rigid-body transform T_parent_child between two named frames.
 
    Attributes:
        parent: Name of the parent frame (the frame points are mapped into).
        child:  Name of the child frame (the frame points are given in).
        R:      3x3 rotation of the child frame expressed in the parent frame.
        t:      Position of the child frame origin expressed in the parent frame.
    """
    parent: str
    child: str
    # Default to the identity transform (no rotation, no translation)
    R: np.ndarray = field(default_factory = lambda: np.eye(3))
    t: np.ndarray = field(default_factory = lambda: np.zeros(3))

    def matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        # Top-left block is the rotation, right column is the translation
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    def inv(self) -> "Transform":
        """Return the inverse, T_child_parent (parent and child names are swapped)."""
        # For a rigid transform the inverse rotation is R^T and the inverse translation is -R^T t
        return Transform(self.child, self.parent, self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other: "Transform") -> "Transform":
        """
        Chain two transforms with `@`: T_a_b @ T_b_c = T_a_c.
 
        Raises ValueError if the inner frame names do not match, which catches
        mis-ordered compositions early.
        """
        # The child of the left transform must equal the parent of the right one
        if self.child != other.parent:
            raise ValueError(f"Cannot compose T_{self.parent}_{self.child} " f"with T_{other.parent}_{other.child}")
        # Rotations multiply; the right translation is rotated into our frame and added to ours
        return Transform(self.parent, other.child, self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, pts) -> np.ndarray:
        """
        Map points (3,) or (N, 3) from the child frame into the parent frame.
 
        Returns an array with the same shape as the input.
        """
         # Points are stored as rows, so p_parent = p_child @ R^T + t (broadcast over rows)
        return np.asarray(pts, float) @ self.R.T + self.t

def pose2d_to_transform(parent: str, child: str, x: float, y: float, th: float) -> Transform:
    """Build a Transform from a planar pose: yaw `th` about z, translation (x, y, 0)."""
    return Transform(parent, child, rot_z(th), np.array([x, y, 0]))

def transform_to_pose2d(T: Transform):
    """Extract the planar pose (x, y, theta) from a Transform, with theta being the yaw."""
    # Yaw comes from the first column of R (the child x-axis expressed in the parent frame)
    return T.t[0], T.t[1], np.arctan2(T.R[1, 0], T.R[0, 0])

class FrameTree:
    """
    A tree of coordinate frames rooted at W.
 
    Each non-root frame stores a single edge T_parent_child. Any two frames connected
    through the root can then be related with get() or transform().
    """
    # W is the root, every other frame stores one edge T_parent_child
    ROOT = "W"

    def __init__(self):
        # Maps child frame name -> its transform to its parent
        self._edges: dict[str, Transform] = {}

    def set_static(self, T: Transform) -> None:
        """Store (or overwrite) the edge for T.child, e.g. a fixed sensor mounting."""
        self._edges[T.child] = T

    def set_base_pose(self, x: float, y: float, th:float) -> None:
        """Update T_W_B (world -> base) from a planar pose estimate."""
        # Update T_W_B from the EKF estimate not ground truth
        self._edges["B"] = pose2d_to_transform("W", "B", x, y, th)

    def _T_root_frame(self, frame: str) -> Transform:
        """Return T_W_frame by recursively composing edges from the root down to `frame`."""
        # The root relative to itself is the identity
        if frame == self.ROOT:
            return Transform(self.ROOT, self.ROOT)
        # A frame with no stored edge cannot be resolved
        if frame not in self._edges:
            raise KeyError(f"Frame '{frame}' has no transform set")
        e = self._edges[frame]
        # T_W_frame = T_W_parent @ T_parent_frame
        return self._T_root_frame(e.parent) @ e

    def get(self, to_frame: str, from_frame: str) -> Transform:
        """
        Return T_to_from, which maps points in `from_frame` into `to_frame`.
 
        Computed through the root: T_to_from = (T_W_to)^-1 @ T_W_from.
        """
        return self._T_root_frame(to_frame).inv() @ self._T_root_frame(from_frame)

    def transform(self, pts, from_frame: str, to_frame: str) -> np.ndarray:
        """Express points given in `from_frame` in `to_frame`; pts is (3,) or (N, 3)."""
        return self.get(to_frame, from_frame).apply(pts)

def build_static_frames(tree: FrameTree, 
                        cam_pos_B, cam_quat_B_wxyz, 
                        lidar_pos_B, lidar_quat_B_wxyz = (1, 0, 0, 0), 
                        marker_pos_W = None, marker_R_W = None) -> None:
    """
    Populate the tree with the fixed frames: camera (C), lidar (L) and optionally marker (M).
 
    Args:
        tree:              Frame tree to fill.
        cam_pos_B:         Camera position in the base frame (m).
        cam_quat_B_wxyz:   Camera orientation in the base frame, MuJoCo convention, (w, x, y, z).
        lidar_pos_B:       Lidar position in the base frame (m).
        lidar_quat_B_wxyz: Lidar orientation in the base frame; defaults to identity.
        marker_pos_W:      Optional marker position in the world frame (m).
        marker_R_W:        Optional marker rotation in the world frame; identity if omitted.
    """
    # Camera pose is given in MuJoCo convention then converted to optical
    # Right-multiplying by R_MJ_TO_CV re-expresses the camera axes in the OpenCV convention
    R_B_Ccv = quat_wxyz_to_R(cam_quat_B_wxyz) @ R_MJ_TO_CV
    tree.set_static(Transform("B", "C", R_B_Ccv, np.asarray(cam_pos_B, float)))
    # Lidar frame uses its quaternion directly (identity by default, so aligned with the base)
    tree.set_static(Transform("B", "L", quat_wxyz_to_R(lidar_quat_B_wxyz), np.asarray(lidar_pos_B, float)))
    # The marker is optional; only add it when its world position is known
    if marker_pos_W is not None:
        if marker_R_W is None:
            R = np.eye(3)
        else:
            # NOTE: the result of this line is not assigned to R, so R is undefined whenever
            # marker_R_W is given (UnboundLocalError below). It likely should read:
            #     R = np.asarray(marker_R_W, float)
            np.asarray(marker_R_W, float)
        tree.set_static(Transform("W", "M", R, np.asarray(marker_pos_W, float)))
