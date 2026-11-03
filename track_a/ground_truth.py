"""Evaluation-only access to MuJoCo truth; this module is never imported by control or estimation."""

import numpy as np
from .model import wrap


class GroundTruth:
    """Read the simulator's true TurtleBot3 base pose strictly for quantitative evaluation."""

    def __init__(self, model, data):
        """Store MuJoCo model/data handles and locate the robot base body."""
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )

    def pose(self):
        """Return the true simulator planar pose ``[x, y, yaw]`` for scoring only."""
        position = np.asarray(self.data.xpos[self.base_id], dtype=float)
        rotation = np.asarray(self.data.xmat[self.base_id], dtype=float).reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return np.array([position[0], position[1], float(wrap(yaw))])
