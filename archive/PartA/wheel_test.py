import numpy as np

OFFSET = 2.0          # measured: wheel speed ≈ ctrl - 2*sign(ctrl)
W_MAX = 7.88 - OFFSET

def wheel_ctrl(w_des):
    """Wheel speed target (rad/s) -> actuator ctrl, compensating the deadband."""
    w_des = np.clip(w_des, -W_MAX, W_MAX)
    return np.where(np.abs(w_des) < 1e-3, 0.0, w_des + OFFSET * np.sign(w_des))
