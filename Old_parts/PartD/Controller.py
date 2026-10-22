#!/usr/bin/env python3
"""ROS controller node for a simple pose-tracking robot controller.

This script subscribes to an EKF-estimated robot pose and a reference pose,
computes the tracking error in the robot's coordinate frame, and generates a
velocity command that drives the robot toward the desired position and heading.
The controller publishes a Twist message to the robot's command topic and logs
tracking data to a CSV file for debugging or later analysis.
"""

import rospy
from geometry_msgs.msg import Twist, PoseStamped
import math
import csv
import time

# Controller gains and limits.
# These values tune the behavior of the feedback controller and can be adjusted
# to improve tracking performance for a specific robot or environment.
k1 = 0.8
k2 = 2.0
k3 = 0.5
max_v = 0.22
max_omega = 2.84
control_hz = 20.0

# ROS topic names.
# EKF_POSE_TOPIC should match the pose output published by your localization system.
# REF_POSE_TOPIC is the target or waypoint pose to track.
EKF_POSE_TOPIC = '/ekf/pose'   # <-- change this to your EKF topic
REF_POSE_TOPIC = '/reference_pose'  # or implement a waypoint provider
CMD_TOPIC = '/cmd_vel'


class ControllerNode:
    """Holds the ROS node logic for receiving pose data and publishing control commands."""

    def __init__(self):
        """Initialize the controller node, ROS subscribers, publisher, and logger."""
        # Start the ROS node with a unique name.
        rospy.init_node('turtlebot_controller')

        # Current estimated robot pose from the EKF.
        self.x = None  # estimated x position
        self.y = None  # estimated y position
        self.theta = None  # estimated heading (yaw)

        # Subscribe to the estimated pose and the reference pose.
        rospy.Subscriber(EKF_POSE_TOPIC, PoseStamped, self.ekf_cb)
        rospy.Subscriber(REF_POSE_TOPIC, PoseStamped, self.ref_cb)

        # Publish velocity commands to the robot.
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)

        # Reference pose (x, y, theta) for tracking.
        self.ref = None

        # Open a log file for debugging the controller output.
        self.logfile = open('/tmp/controller_log.csv', 'w', newline='')
        self.csvw = csv.writer(self.logfile)
        self.csvw.writerow(['t', 'x_hat', 'y_hat', 'theta_hat', 'x_ref', 'y_ref', 'theta_ref', 'v', 'omega'])

        # Rate limiter for the main control loop.
        self.rate = rospy.Rate(control_hz)

    def ekf_cb(self, msg):
        """Store the latest EKF-estimated robot state from the pose topic."""
        # Extract x and y positions from the pose message.
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y

        # Convert quaternion orientation to yaw angle in radians.
        # This is needed because the controller uses heading in the plane.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def ref_cb(self, msg):
        """Store the target pose and target heading from the reference topic."""
        # Save x and y coordinates of the reference pose.
        self.ref = (msg.pose.position.x, msg.pose.position.y)

        # Extract the reference heading from the quaternion orientation.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta_ref = math.atan2(siny_cosp, cosy_cosp)

        # Update the reference to include the heading as the third value.
        self.ref = (self.ref[0], self.ref[1], theta_ref)

    def run(self):
        """Continuously compute the control command and publish it at a fixed rate."""
        while not rospy.is_shutdown():
            # Wait until both the pose estimate and reference target are available.
            if self.x is None or self.ref is None:
                self.rate.sleep()
                continue

            # Current robot state and desired target state.
            x_hat, y_hat, theta_hat = self.x, self.y, self.theta
            x_r, y_r, theta_r = self.ref

            # Position error in the global/world frame.
            dx = x_r - x_hat
            dy = y_r - y_hat

            # Rotate the position error into the robot's local frame.
            # This makes the control law easier to reason about because the robot
            # can track errors along its forward and lateral axes.
            ex = math.cos(theta_hat) * dx + math.sin(theta_hat) * dy
            ey = -math.sin(theta_hat) * dx + math.cos(theta_hat) * dy

            # Shortest angular error between the desired and current heading.
            etheta = math.atan2(math.sin(theta_r - theta_hat), math.cos(theta_r - theta_hat))

            # Reference feedforward terms are currently set to zero, meaning the
            # controller is purely corrective and does not use a desired velocity.
            v_ref = 0.0
            omega_ref = 0.0

            # Control law:
            # - v regulates forward position error
            # - omega regulates heading and lateral error
            v = v_ref * math.cos(etheta) + k1 * ex
            omega = omega_ref + k2 * v_ref * ey + k3 * math.sin(etheta)

            # Clamp the command to the robot's allowed speed and turning rate.
            v = max(-max_v, min(max_v, v))
            omega = max(-max_omega, min(max_omega, omega))

            # Publish the computed motion command.
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # Log the current state, target state, and control output for analysis.
            self.csvw.writerow([time.time(), x_hat, y_hat, theta_hat, x_r, y_r, theta_r, v, omega])
            self.logfile.flush()

            # Sleep to maintain the requested update frequency.
            self.rate.sleep()


if __name__ == '__main__':
    """Script entry point that starts the controller node and keeps it running."""
    try:
        # Construct the controller node and begin the main loop.
        node = ControllerNode()
        node.run()
    except rospy.ROSInterruptException:
        # Clean shutdown if ROS is interrupted or the node is terminated.
        pass
