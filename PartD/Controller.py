"""Compatibility wrapper. ROS is no longer required by the integrated project."""
from track_a.controller import ControlOutput, StateFeedbackController

__all__ = ["ControlOutput", "StateFeedbackController"]
