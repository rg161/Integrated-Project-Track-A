"""Small Part C evaluation helper retained for compatibility."""
import numpy as np
from track_a.model import wrap


def calculate_rmse(estimated_states, true_states):
    estimated_states = np.asarray(estimated_states, dtype=float)
    true_states = np.asarray(true_states, dtype=float)
    if estimated_states.shape != true_states.shape or estimated_states.ndim != 2 or estimated_states.shape[1] != 3:
        raise ValueError("estimated_states and true_states must both be N x 3")
    error = estimated_states - true_states
    error[:, 2] = wrap(error[:, 2])
    x_rmse = np.sqrt(np.mean(error[:, 0] ** 2))
    y_rmse = np.sqrt(np.mean(error[:, 1] ** 2))
    heading_rmse = np.sqrt(np.mean(error[:, 2] ** 2))
    position_rmse = np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))
    return {
        "X RMSE": float(x_rmse),
        "y RMSE": float(y_rmse),
        "position RMSE": float(position_rmse),
        "heading RMSE deg": float(np.rad2deg(heading_rmse)),
    }
