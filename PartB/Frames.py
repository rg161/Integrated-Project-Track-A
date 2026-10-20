from dataclasses import dataclass, field
import numpy as np

def wrap_angle(a):
    #Wrap the rotation from -pi to pi
    return np.arctan2(np.sin(a), np.cos(a))

def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def quat_wxyz_to_R(q) -> np.ndarray:
    #MuJoCo quaternion order is (w, x, y, z).
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 
    ])

#MuJoCo camera along -z, +y up
#OpenCV camera along +z, +y down
R_MJ_TO_CV = np.diag([1, -1, -1])

@dataclass(frozen=True)
#T_parent_child: p_parent = R @ p_child + t
class Transform:
    parent: str
    child: str
    R: np.ndarray = field(default_factory = lambda: np.eye(3))
    t: np.ndarray = field(default_factory = lambda: np.zeros(3))

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    def inv(self) -> "Transform":
        return Transform(self.child, self.parent, self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other: "Transform") -> "Transform":
        if self.child != other.parent:
            raise ValueError(f"Cannot compose T_{self.parent}_{self.child} "
                             f"with T_{other.parent}_{other.child}")
        return Transform(self.parent, other.child, self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, pts) -> np.ndarray:
        #Map points (3,) or (N,3) from child into parent
        return np.asarray(pts, float) @ self.R.T + self.t

def pose2d_to_transform(parent: str, child: str, x: float, y: float, th: float) -> Transform:
    return Transform(parent, child, rot_z(th), np.array([x, y, 0]))

def transform_to_pose2d(T: Transform):
    return T.t[0], T.t[1], np.arctan2(T.R[1, 0], T.R[0, 0])

class FrameTree:
    #W is the root, every other frame stores one edge T_parent_child
    ROOT = "W"

    def __init__(self):
        self._edges: dict[str, Transform] = {}

    def set_static(self, T: Transform) -> None:
        self._edges[T.child] = T

    def set_base_pose(self, x: float, y: float, th:float) -> None:
        #Update T_W_B from the EKF estimate not ground truth
        self._edges["B"] = pose2d_to_transform("W", "B", x, y, th)

    def _T_root_frame(self, frame: str) -> Transform:
        if frame == self.ROOT:
            return Transform(self.ROOT, self.ROOT)
        if frame not in self._edges:
            raise KeyError(f"Frame '{frame}' has no transform set")
        e = self._edges[frame]
        return self._T_root_frame(e.parent) @ e

    def get(self, to_frame: str, from_frame: str) -> Transform:
        return self._T_root_frame(to_frame).inv() @ self._T_root_frame(from_frame)

    def transform(self, pts, from_frame: str, to_frame: str) -> np.ndarray:
        #Express points in from_frame in to_frame
        return self.get(to_frame, from_frame).apply(pts)

def build_static_frames(tree: FrameTree, 
                        cam_pos_B, cam_quat_B_wxyz, 
                        lidar_pos_B, lidar_quat_B_wxyz = (1, 0, 0, 0), 
                        marker_pos_W = None, marker_R_W = None) -> None:
    #Camera pose is given in MuJoCo convention then converted to optical
    R_B_Ccv = quat_wxyz_to_R(cam_quat_B_wxyz) @ R_MJ_TO_CV
    tree.set_static(Transform("B", "C", R_B_Ccv, np.asarray(cam_pos_B, float)))
    tree.set_static(Transform("B", "L", quat_wxyz_to_R(lidar_quat_B_wxyz), np.asarray(lidar_pos_B, float)))
    if marker_pos_W is not None:
        if marker_R_W is None:
            R = np.eye(3)
        else:
            np.asarray(marker_R_W, float)
        tree.set_static(Transform("W", "M", R, np.asarray(marker_pos_W, float)))
