#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped
import math
import csv
import time

# ---------------------------------------------------------------------
# Controller parameters
# ---------------------------------------------------------------------
# These gains determine how strongly the robot reacts to tracking errors.
# - k1: forward tracking error correction
# - k2: lateral tracking error correction
# - k3: heading alignment correction
# max_v and max_omega limit how fast the robot may move.
# control_hz controls the update frequency of the control loop.
k1 = 0.8
k2 = 2.0
k3 = 0.5
max_v = 0.22
max_omega = 2.84
control_hz = 20.0

# ROS topics used by the controller.
# EKF_POSE_TOPIC: the estimated pose of the robot (from localization/odometry fusion)
# REF_POSE_TOPIC: the desired target pose (waypoint / reference trajectory)
# CMD_TOPIC: the velocity command published to the robot base
EKF_POSE_TOPIC = '/ekf/pose'   # <-- change this to your EKF topic
REF_POSE_TOPIC = '/reference_pose'  # or implement a waypoint provider
CMD_TOPIC = '/cmd_vel'


class ControllerNode:
    def __init__(self):
        """Initialize the controller node and all ROS interfaces.

        This method sets up the ROS node, initializes the robot state variables,
        subscribes to the EKF and reference pose topics, creates the command
        publisher, opens the debug CSV logger, and starts the fixed-rate control
        loop timing used by the main controller.
        """
        # Initialize the ROS node. This gives the script a name in the ROS graph.
        rospy.init_node('turtlebot_controller')

        # Current estimated robot state from EKF.
        # These are updated whenever a new message is received.
        self.x = None  # estimated x position
        self.y = None  # estimated y position
        self.theta = None  # estimated yaw angle in radians

        # Subscribe to the EKF pose and reference pose topics.
        # Each callback receives a PoseStamped message.
        rospy.Subscriber(EKF_POSE_TOPIC, PoseStamped, self.ekf_cb)
        rospy.Subscriber(REF_POSE_TOPIC, PoseStamped, self.ref_cb)

        # Publish velocity commands to the robot.
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)

        # Desired pose is stored as a tuple: (x, y, theta)
        self.ref = None

        # Open a CSV log file in /tmp for debugging and analysis.
        # This allows us to inspect the robot's position, target position,
        # and commanded linear/angular velocities over time.
        self.logfile = open('/tmp/controller_log.csv', 'w', newline='')
        self.csvw = csv.writer(self.logfile)
        self.csvw.writerow(['t', 'x_hat', 'y_hat', 'theta_hat', 'x_ref', 'y_ref', 'theta_ref', 'v', 'omega'])

        # Rate object keeps the control loop running at a fixed frequency.
        self.rate = rospy.Rate(control_hz)

    def ekf_cb(self, msg):
        """Update the robot's current pose using the EKF message.

        This callback reads the robot's x/y position and transforms the incoming
        quaternion orientation into a yaw angle in radians so the controller can
        compute heading and position errors against the target pose.
        """
        # Callback for EKF pose updates.
        # Extract x, y from the position and compute yaw from the quaternion orientation.
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y

        # Convert quaternion orientation to yaw angle.
        # For ROS quaternions, yaw is computed from the rotation around z-axis.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def ref_cb(self, msg):
        """Store the desired waypoint pose from the reference topic.

        The callback records the target position and computes the target heading
        from its quaternion orientation so the controller can compare the actual
        robot pose to the desired pose in the same frame.
        """
        # Callback for the reference pose.
        # Save x and y from the target pose, then compute target heading.
        self.ref = (msg.pose.position.x, msg.pose.position.y)

        # Convert target quaternion orientation to yaw angle.
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta_ref = math.atan2(siny_cosp, cosy_cosp)

        # Store full target pose as (x, y, theta).
        self.ref = (self.ref[0], self.ref[1], theta_ref)

    def run(self):
        """Execute the main control loop for robot tracking.

        The loop waits for valid robot and reference states, computes the error
        between the current and desired poses in the robot frame, applies the
        feedback control law, publishes a velocity command, logs the data, and
        repeats at the configured control frequency until shutdown.
        """
        # Main control loop.
        # Keep executing until the ROS node is shutdown.
        while not rospy.is_shutdown():
            # Wait until both the observed state and the desired target are available.
            if self.x is None or self.ref is None:
                self.rate.sleep()
                continue

            # Read current state and target state.
            x_hat, y_hat, theta_hat = self.x, self.y, self.theta
            x_r, y_r, theta_r = self.ref

            # Position error in the global (world) frame.
            # dx and dy tell us how far the robot is from the target point.
            dx = x_r - x_hat
            dy = y_r - y_hat

            # Rotate the position error into the robot's local frame.
            # ex = forward error along the robot's current heading
            # ey = lateral error perpendicular to the robot's heading
            # This makes the control law easier to interpret.
            ex = math.cos(theta_hat) * dx + math.sin(theta_hat) * dy
            ey = -math.sin(theta_hat) * dx + math.cos(theta_hat) * dy

            # Heading error between desired orientation and current orientation.
            # math.atan2(sin(delta), cos(delta)) gives the shortest signed angular difference.
            etheta = math.atan2(math.sin(theta_r - theta_hat), math.cos(theta_r - theta_hat))

            # Reference feedforward terms are set to zero here.
            # In a more advanced controller, these could represent the desired
            # linear and angular velocity of the reference trajectory.
            v_ref = 0.0
            omega_ref = 0.0

            # Control law.
            # v drives the robot forward to reduce distance error.
            # omega aligns the robot heading and corrects lateral offset.
            # This is a simple kinematic feedback controller.
            v = v_ref * math.cos(etheta) + k1 * ex
            omega = omega_ref + k2 * v_ref * ey + k3 * math.sin(etheta)

            # Saturate the computed velocities to the robot's hardware limits.
            v = max(-max_v, min(max_v, v))
            omega = max(-max_omega, min(max_omega, omega))

            # Publish the computed Twist to the robot's command topic.
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)

            # Log the current state and commanded values for later inspection.
            self.csvw.writerow([time.time(), x_hat, y_hat, theta_hat, x_r, y_r, theta_r, v, omega])
            self.logfile.flush()

            # Sleep until the next control cycle to maintain the target rate.
            self.rate.sleep()


if __name__ == '__main__':
    try:
        # Create the controller node and start the main loop.
        node = ControllerNode()
        node.run()
    except rospy.ROSInterruptException:
        # ROS may raise this when shutdown interrupts the loop.
        pass
