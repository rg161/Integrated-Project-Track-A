import numpy as np
from frames import wrap_angle

class GroundTruth:
    """
    Read-only view of the true pose of the robot base (and other bodies) in the world frame W.
 
    Args:
        model:     MuJoCo model (MjModel), used to look up body ids by name.
        data:      MuJoCo data (MjData), holding the live simulation state.
        base_body: Name of the body that represents the robot base.
    """
    def __init__(self, model, data, base_body: str = "base"):
        # Keep a reference to the live data, it is updated in place as the simulation steps, so every call below sees the current state
        self._data = data
        # Resolve the base body's integer id once, to avoid a name lookup on every call
        self._id = model.body(base_body).id

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return float(self._data.time)

    def pose(self) -> np.ndarray:
        """
        True planar pose [x, y, theta] of the base in W (theta wrapped).
 
        Returns:
            Array [x (m), y (m), theta (rad)], where theta is the yaw of the base.
        """
        # World position (x, y, z) of the base body
        p = self._data.xpos[self._id]
        # Body orientation as a 3x3 rotation matrix (MuJoCo stores it flattened, row-major)
        R = np.asarray(self._data.xmat[self._id]).reshape(3, 3)
        # Yaw from the first column of R (the body x-axis expressed in W), arctan2 already returns an angle in (-pi, pi]
        return np.array([p[0], p[1], np.arctan2(R[1, 0], R[0, 0])])

    def pose_error(self, est) -> np.ndarray:
        """
        Pose error of an estimate relative to the truth: est - truth for [x, y, theta].
 
        The heading error is wrapped so that, for example, a difference of 359 degrees
        is reported as -1 degree. Intended for evaluation only.
 
        Args:
            est: Estimated pose [x, y, theta].
 
        Returns:
            Error array [ex (m), ey (m), etheta (rad)].
        """
        # Element-wise difference between the estimate and the true pose
        e = np.asarray(est, float) - self.pose()
        # Subtracting angles can leave the (-pi, pi] range, so wrap the heading error
        e[2] = wrap_angle(e[2])
        return e
    def body_position(self, model, name: str) -> np.ndarray:
        """
        True world-frame position of any named body (for example a landmark or marker).
 
        Args:
            model: MuJoCo model used to resolve the body name to an id.
            name:  Name of the body.
 
        Returns:
            Array [x, y, z] in W (metres).
        """
        # Copy into a new array so callers cannot modify MuJoCo's internal buffer
        return np.array(self._data.xpos[model.body(name).id])
