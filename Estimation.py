import numpy as np
from helper_Estimation import wrap_angle,wheel_speeds_to_velocity

class MobileRobotEKF:
    def __init__(self):
        #state:
        # x y heading
        self.x = np.array([0.0,0.0,0.0])
        # initial uncertainty
        self.P = np.diag([0.20**2,0.20**2,np.deg2rad(10)**2])
        #proccess noise
        # represent uncertainty in wheel odometry
        self.Q = np.diag([0.01**2, 0.01**2, np.deg2rad(1.0)**2])
        #camera pose measurment noise
        self.R_pose = np.diag([0.05**2, 0.05**2, np.deg2rad(3.0)**2])

        # Range/bearing noise
        self.R_range_bearing = np.diag([0.05**2,np.deg2rad(2.0)**2])

    # Prediction
    def predict(self, v, omega, dt):
        x,y,theta = self.x

        #predict new robot position

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt

        theta_new = wrap_angle(theta_new)

        self.x = np.array([x_new, y_new, theta_new])

        # jacobian motion model
        F = np.array([
            [1.0,0.0,-v * np.sin(theta) * dt],
            [0.0,1.0,v * np.cos(theta) * dt],
            [0.0,0.0,1.0]
            ])
        #predict covariance
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()


    def update_pose(self, camera_pose):
        """ camera x, y and heading, camera measurments should already be tranformed in coordinite fram as EKF"""

        z = np.array(camera_pose)
        #Camera measures x, y and theta
        H = np.eye(3)
        predicted_measurement = self.x.copy()

        #calculate Difference between measurment and prediction
        innovation = z - predicted_measurement
        innovation[2] = wrap_angle(innovation[2])

        # Innovation covariance
        S = (H @ self.P @ H.T + self.R_pose)

        #Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # correct state estimate
        self.x = (self.x + K @ innovation)
        self.x[2] = wrap_angle(self.x[2])

        # cprrect cpvariance
        I = np.eye(3)
        self.P = ( I - K @ H) @ self.P
        return self.x.copy()

    def update_range_bearing(self, measured_range, measured_bearing, marker_position):
        """takes known x,y position of ArUco 
            camera distance to marker
            angle from robot heading to marker
        """

        x,y, theta = self. x

        marker_x = marker_position[0]
        marker_y = marker_position[1]

        dx = marker_x - x
        dy = marker_y - y

        q = dx**2 + dy**2

        predicted_range = np.sqrt(q)

        predicted_bearing = wrap_angle(np.arctan2(dy,dx) - theta)

        predicted_measuurment = np.array([predicted_range, predicted_bearing])

        measurement = np.array([measured_range, measured_bearing])

        # Measuremnt jacobian
        H = np.array([
            [-dx / predicted_range, -dy / predicted_range, 0.0],
            [dy / q,-dx/q,-1.0]
        ])
        # measurment error
        innovation = ( measurement - predicted_measuurment)

        innovation[1] = wrap_angle(innovation[1])

        #innovation covariance
        S = ( H @ self.P @ H.T + self.R_range_bearing)

        # Kalman gain
        K = (self.P @ H.T @ np.linalg.inv(S))

        # Correct estimate
        self.x[2] = wrap_angle(self.x[2])
        #Correct covariance
        I = np.eye(3)
        self.P = ( I - K @ H) @ self.P
        return self.x.copy()

    def get_state(self):
        """return copy of current estimate"""
        return self.x.copy
    def get_covariance(self):
        """return copy of convariance"""
        return self.P.copy()




