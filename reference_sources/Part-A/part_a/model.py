import numpy as np

"""
note:
state -> x = [x, y, theta] position and heading 
input -> u = [v, w] speed and turning speed
"""

R_wheel = 0.0283 #wheel radius (m) measured, at 3rad/s
L_wheel = 0.353 #wheel separation (m) measured
#XML gave R = 0.033m, L = 0.288m, 
#correct calculations found as simulated bot had smaller wheels, larger seperation

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def wheels_to_v_w(wl, wr, r = R_wheel, L = L_wheel):
#convert wheel speeds (rad/s) to linear and angular velocity (m/s, rad/s)
    return r * (wl + wr) / 2, r * (wr - wl) / L

def f(x, u, dt):
    #discrete motion model 
    v, w = u #input is linear and angular velocity
    th = x[2] + 0.5 * w * dt
    return np.array([x[0] + v * np.cos(th) * dt,
                    x[1] + v * np.sin(th) * dt,
                    wrap(x[2] + w * dt)])

#Jacobians
def F_jac(x, u, dt):
    #df/dx
    v, w, = u
    th = x[2] + 0.5 * w * dt
    return np.array([[1, 0, -v * np.sin(th) * dt],
                    [0, 1, v * np.cos(th) * dt],
                    [0, 0, 1]]
    )

def G_jac(x, u, dt):
    #df/du
    v, w = u
    th = x[2] + 0.5 * w *dt
    return np.array([[dt * np.cos(th), -0.5 * v * dt**2 * np.sin(th)],
                     [dt * np.sin(th), 0.5 * v * dt**2 * np.cos(th)],
                     [0, dt]])

def H_jac(x, m):
    #dh/dx
    dx, dy, = m[0] - x[0], m[1] - x[1]
    q = dx**2 + dy**2
    rho = np.sqrt(q)
    return np.array([[-dx/rho, -dy/rho, 0],
                     [dy/q, -dx/q, -1]])

def h_range_bearing(x, m):
    #range/bearing compare to landmark m (m = [mx, my])
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])
