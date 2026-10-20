"""Compatibility helpers for the integrated estimator."""
from track_a.model import wrap as wrap_angle, wheels_to_v_w


def wheel_speeds_to_velocity(left_w, right_w, wheel_radius, wheel_base):
    import numpy as np
    return np.array(wheels_to_v_w(left_w, right_w, wheel_radius, wheel_base))
