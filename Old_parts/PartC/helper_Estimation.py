import numpy as np
"""keeps angle between -pi and +pi Takes in agle and returns new agle"""
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wheel_speeds_to_velocity(left_w, right_w, wheel_radius, wheel_base):
    """Convert wheel velocities int0
        v forward
        omega turning
    """
    v_left = wheel_radius * left_w
    v_right = wheel_radius * right_w
    v=(v_right + v_left) / 2.0
    omega = (v_right - v_left) / wheel_base
    return np.array([v,omega])
