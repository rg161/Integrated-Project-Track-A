#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped
import math
import csv
import time

# Parameters (tune these)
k1 = 0.8
k2 = 2.0
k3 = 0.5
max_v = 0.22
max_omega = 2.84
control_hz = 20.0

# Replace with your EKF pose topic
EKF_POSE_TOPIC = '/ekf/pose'   # <-- change this to your EKF topic
REF_POSE_TOPIC = '/reference_pose'  # or implement a waypoint provider
CMD_TOPIC = '/cmd_vel'

class ControllerNode:
    def __init__(self):
        rospy.init_node('turtlebot_controller')
        self.x = None  # estimated pose
        self.y = None
        self.theta = None
        rospy.Subscriber(EKF_POSE_TOPIC, PoseStamped, self.ekf_cb)
        rospy.Subscriber(REF_POSE_TOPIC, PoseStamped, self.ref_cb)
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        self.ref = None
        self.logfile = open('/tmp/controller_log.csv','w', newline='')
        self.csvw = csv.writer(self.logfile)
        self.csvw.writerow(['t','x_hat','y_hat','theta_hat','x_ref','y_ref','theta_ref','v','omega'])
        self.rate = rospy.Rate(control_hz)

    def ekf_cb(self,msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        # get yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def ref_cb(self,msg):
        self.ref = (msg.pose.position.x, msg.pose.position.y)
        q = msg.pose.orientation
        siny_cosp = 2*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
        theta_ref = math.atan2(siny_cosp, cosy_cosp)
        self.ref = (self.ref[0], self.ref[1], theta_ref)

    def run(self):
        while not rospy.is_shutdown():
            if self.x is None or self.ref is None:
                self.rate.sleep()
                continue
            x_hat, y_hat, theta_hat = self.x, self.y, self.theta
            x_r, y_r, theta_r = self.ref
            # errors in global frame
            dx = x_r - x_hat
            dy = y_r - y_hat
            # rotate into robot frame
            ex = math.cos(theta_hat)*dx + math.sin(theta_hat)*dy
            ey = -math.sin(theta_hat)*dx + math.cos(theta_hat)*dy
            etheta = math.atan2(math.sin(theta_r - theta_hat), math.cos(theta_r - theta_hat))
            # reference feedforward (zero here)
            v_ref = 0.0
            omega_ref = 0.0
            # control law
            v = v_ref*math.cos(etheta) + k1*ex
            omega = omega_ref + k2*v_ref*ey + k3*math.sin(etheta)
            # saturate
            v = max(-max_v, min(max_v, v))
            omega = max(-max_omega, min(max_omega, omega))
            # publish
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)
            # log
            self.csvw.writerow([time.time(), x_hat, y_hat, theta_hat, x_r, y_r, theta_r, v, omega])
            self.logfile.flush()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = ControllerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
