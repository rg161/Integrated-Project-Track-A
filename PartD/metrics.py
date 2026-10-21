#!/usr/bin/env python3
"""Controller node for a ROS mobile-base tracking task.

This node subscribes to an estimated robot pose and a desired reference pose,
then computes a simple kinematic feedback controller to drive the robot toward
that reference pose. The control output is published as a cmd_vel Twist.
"""
import rospy
from geometry_msgs.msg import Twist, PoseStamped
import math
import csv
import time

# ---------------------------------------------------------------------------
# Controller tuning parameters.
# These values are the gains for the tracking controller. Adjust them to tune
# response speed, damping, and heading convergence for your robot.
# ---------------------------------------------------------------------------
k1 = 0.8
k2 = 2.0
k3 = 0.5
max_v = 0.22
max_omega = 2.84
control_hz = 20.0

# ---------------------------------------------------------------------------
# ROS topic names.
# Replace the EKF topic with the actual pose source used in your system.
# REF_POSE_TOPIC is the target pose (or waypoint) the robot should track.
# CMD_TOPIC is the velocity command topic sent to the robot base.
# ---------------------------------------------------------------------------
EKF_POSE_TOPIC = '/ekf/pose'
REF_POSE_TOPIC = '/reference_pose'
CMD_TOPIC = '/cmd_vel'


class ControllerNode:
    def __init__(self):
        # Initialize the ROS node for this controller.
        rospy.init_node('turtlebot_controller')

        # Store the current robot pose estimate and the desired reference pose.
        self.x = None
        self.y = None
        self.theta = None

        # Subscribe to pose measurements from the EKF and reference/waypoint pose.
        rospy.Subscriber(EKF_POSE_TOPIC, PoseStamped, self.ekf_cb)
        rospy.Subscriber(REF_POSE_TOPIC, PoseStamped, self.ref_cb)

        # Publish velocity commands to the robot.
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)

        # The ref attribute stores the target pose as (x, y, theta).
        self.ref = None

        # Open a CSV log for recording controller performance data.
        self.logfile = open('/tmp/controller_log.csv', 'w', newline='')
        self.csvw = csv.writer(self.logfile)
        self.csvw.writerow(['t', 'x_hat', 'y_hat', 'theta_hat', 'x_ref', 'y_ref', 'theta_ref', 'v', 'omega'])

        # Control loop timing.
        self.rate = rospy.Rate(control_hz)

    def ekf_cb(self, msg):
        # Store the estimated position from the EKF.
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y

        # Convert the quaternion orientation to yaw angle.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def ref_cb(self, msg):
        # Store the target position from the reference message.
        self.ref = (msg.pose.position.x, msg.pose.position.y)

        # Extract the target heading from the quaternion of the reference pose.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta_ref = math.atan2(siny_cosp, cosy_cosp)

        # Store the complete target pose as (x, y, theta).
        self.ref = (self.ref[0], self.ref[1], theta_ref)

    def run(self):
        # Main control loop.
        while not rospy.is_shutdown():
            # Wait until both the robot estimate and the reference pose exist.
            if self.x is None or self.ref is None:
                self.rate.sleep()
                continue

            # Current estimated pose and desired pose.
            x_hat, y_hat, theta_hat = self.x, self.y, self.theta
            x_r, y_r, theta_r = self.ref

            # Compute position error in the global/world frame.
            dx = x_r - x_hat
            dy = y_r - y_hat

            # Rotate the position error into the robot's local frame so the
            # controller can correct longitudinal and lateral error separately.
            ex = math.cos(theta_hat) * dx + math.sin(theta_hat) * dy
            ey = -math.sin(theta_hat) * dx + math.cos(theta_hat) * dy

            # Shortest-angle representation of heading error.
            etheta = math.atan2(math.sin(theta_r - theta_hat), math.cos(theta_r - theta_hat))

            # Reference feedforward terms are currently set to zero.
            # In a more advanced controller these would track a commanded speed
            # and turning rate for smooth path following.
            v_ref = 0.0
            omega_ref = 0.0

            # Kinematic control law:
            # - v controls forward speed toward the target
            # - omega controls heading correction and lateral tracking
            v = v_ref * math.cos(etheta) + k1 * ex
            omega = omega_ref + k2 * v_ref * ey + k3 * math.sin(etheta)

            # Saturate controller output to the hardware limits.
            v = max(-max_v, min(max_v, v))
            omega = max(-max_omega, min(max_omega, omega))

            # Publish the computed twist command.
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # Log pose, target, and command data for later evaluation.
            self.csvw.writerow([time.time(), x_hat, y_hat, theta_hat, x_r, y_r, theta_r, v, omega])
            self.logfile.flush()

            # Maintain the desired control rate.
            self.rate.sleep()


if __name__ == '__main__':
    try:
        # Start the controller node and keep it running until ROS shuts down.
        node = ControllerNode()
        node.run()
    except rospy.ROSInterruptException:
        # ROS interrupt is expected during shutdown.
        pass
