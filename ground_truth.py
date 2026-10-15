import numpy as np
from frames import wrap_angle

class GroundTruth:
    def __init__(self, model, data, turtlebot3_waffle_pi: str = "base_footprint"):
        self._data = data
        self._id = model.body(turtlebot3_waffle_pi).id

    @property
    def time(self) -> float:
        return float(self._data.time)

    def pose(self) -> np.ndarray:
        #true planar pose [x, y, theta] of the base in W (theta wrapped)
        p = self._data.xpos[self._id]
        R = np.asarray(self._data.xmat[self._id]).reshape(3, 3)
        return np.array([p[0], p[1], np.arctan2(R[1, 0], R[0, 0])])

    def pose_error(self, est) -> np.ndarray:
        #est - truth for [x, y, theta], with theta wrapped (evaluation only)
        e = np.asarray(est, float) - self.pose()
        e[2] = wrap_angle(e[2])
        return e
    def body_position(self, model, name: str) -> np.ndarray:
        #true world position for any named body
        return np.array(self._data.xpos[model.body(name).id])