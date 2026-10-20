import numpy as np
from helper_Estimation import wrap_angle
import Estimation

# evaluation functions
# ground truth 

def calculate_rmse(estimated_states, true_states):
    """caluculates the RMSE takes in estimated states and true states,
        returns x rmse, y rmse , position RMSE, and heading in deg
    """
    estimated_states = np.array(estimated_states)

    true_states = np.array(true_states)

    error = (estimated_states - true_states)

    # Heading error wrapped
    error[:,2] = np.array([wrap_angle(a) for a in error[:,2]])

    x_rmse = np.sqrt(np.mean(error[:,0]**2))
    y_rmse = np.sqrt(np.mean(error[:,1]**2))

    heading_rmse = np.sqrt(np.mean(error[:,2]**2))

    position_error = np.sqrt(error[:,0]**2 + error[:,1]**2)

    position_rmse = np.sqrt(np.mean(position_error**2))

    return{
        "X rmse": x_rmse,
        "y RMSE": y_rmse,
        "position RMSE": position_rmse,
        "heading RMSE deg": np.rad2deg(heading_rmse)
    }