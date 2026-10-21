#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import math

"""
Simple waypoint publisher for testing the controller.
Publishes PoseStamped messages to /reference_pose at a configurable rate.
Waypoints can be set via the ROS parameter /partd_waypoints (list of tuples)
Default waypoints: approach point then docking pose.
"""

DEFAULT_WAYPOINTS = [
    (1.0, 0.0, 0.0),
    (1.8, 0.2, 0.0),
    (2.0, 0.0, 0.0),
]

class WaypointPublisher:
    def __init__(self):
        rospy.init_node('waypoint_publisher')
        self.pub = rospy.Publisher('/reference_pose', PoseStamped, queue_size=1)
        rate_hz = rospy.get_param('~rate', 1.0)
        self.rate = rospy.Rate(rate_hz)
        self.waypoints = rospy.get_param('/partd_waypoints', DEFAULT_WAYPOINTS)
        rospy.loginfo(f'waypoint_publisher: publishing {len(self.waypoints)} waypoints at {rate_hz} Hz')

    def run(self):
        idx = 0
        while not rospy.is_shutdown():
            x, y, theta = self.waypoints[idx % len(self.waypoints)]
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'map'
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            qz = math.sin(theta / 2.0)
            qw = math.cos(theta / 2.0)
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            self.pub.publish(msg)
            idx += 1
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = WaypointPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
