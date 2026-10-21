#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

"""
Simple EKF stub node
- Subscribes to /odom and republishes the pose as /ekf/pose (PoseStamped)
- Intended as a placeholder estimator so Controller.py can subscribe to /ekf/pose

WARNING: This is a placeholder for development and testing only. Replace with your real
EKF implementation that fuses odometry and sensor measurements.
"""

class EKFStubNode:
    def __init__(self):
        rospy.init_node('ekf_stub')
        self.pub = rospy.Publisher('/ekf/pose', PoseStamped, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)

    def odom_cb(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        # publish the received odometry pose as the EKF estimate
        self.pub.publish(pose)

    def run(self):
        rospy.loginfo('ekf_stub: started, relaying /odom -> /ekf/pose')
        rospy.spin()

if __name__ == '__main__':
    node = EKFStubNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
