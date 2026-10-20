"""Evaluation-only access to MuJoCo ground truth."""

import numpy as np
from .model import wrap


class GroundTruth:
    def __init__(self, model, data):
        import mujoco
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def pose(self):
        p = np.asarray(self.data.xpos[self.base_id], dtype=float)
        R = np.asarray(self.data.xmat[self.base_id], dtype=float).reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return np.array([p[0], p[1], float(wrap(yaw))])
