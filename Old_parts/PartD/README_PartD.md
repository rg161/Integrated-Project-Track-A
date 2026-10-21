# PartD helper notes

This branch adds a minimal development scaffold for the Part D controller:

- ekf.py: simple EKF stub that relays /odom -> /ekf/pose
- trajectory.py: publishes /reference_pose waypoints
- run_partd.py: starts the demo nodes together
- metrics.py: computes tracking RMSE from /tmp/controller_log.csv

Dependencies:
- ROS 1 with rospy
- /odom topic should exist from the simulation or odometry source
- Controller.py must be launched with a matching /cmd_vel topic and reference topic

To run:
1. Start roscore
2. Start your simulation or odom publisher
3. python3 run_partd.py
4. python3 metrics.py /tmp/controller_log.csv
