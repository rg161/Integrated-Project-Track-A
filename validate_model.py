import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from track_a.model import f, wheels_to_v_w, R_WHEEL, L_WHEEL


"""     
Checks the robots kinematic model (model.py) against the simulation (MUJOCO).
Calculate the wheel radius and seperation.
Compare predicted pose with ground truth.
"""


PATH = "models/robotis_tb3/scene_turtlebot3_waffle_pi.xml"
OFFSET = 2.0 #actuator offset from sweep

model = mujoco.MjModel.from_xml_path(PATH)
data = mujoco.MjData(model)
dt = model.opt.timestep

def yaw(d):
    """
    Heading from the base quaternion in qpos[3:7], (w, x, y, z)
    """
    q = d.qpos[3:7]
    #wxyz
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
def wrap(a):
    """
    Wrap angle to (-pi, pi) for rotation past 360deg
    """
    return np.arctan2(np.sin(a), np.cos(a))
def control(w):
    #wheel speed aim
    """
    Turn wheel speed aim into a command for actuator
    adds the offset to get past set range
    """
    return 0.0 if abs(w) < 1e-3 else w + OFFSET * np.sign(w)
def snapshot(d):
     """
     Snapshot specific state at a given moment -> (x, y), yaw, lw angle, rw angle
     """
     return d.qpos[:2].copy(), yaw(d), d.qpos[7], d.qpos[8]
     #xy, yaw, left angle, right angle

#CHECK WHEEL RAD AND SEPERATION (steady)

def steady_test(wl, wr, T = 4.0, t0 = 2.0):
    """
    drive both wheels at fixed wl and wr, return snapshot at T and t0
    difference is steady state motion
    """
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = [control(wl), control(wr)]
    start = None
    while data.time < T:
        mujoco.mj_step(model, data)
        if start is None and data.time >= t0:
            start = snapshot(data)
    return start, snapshot(data)



#straight:
#distance travelled / wheel angle = wheel radius
s, e = steady_test(3.0, 3.0)
dist = np.linalg.norm(e[0] - s[0])
dphi = ((e[2] - s[2]) + (e[3] - s[3])) / 2
r_est = dist / dphi
print(f"radius: measured {r_est:.4f} m vs model {R_WHEEL} m")

#spin:
#wheels spin opp directions
#change in yaw for one wheel rotation = wheel seperation
s, e = steady_test(-3.0, 3.0)
dyaw = wrap(e[1] - s[1])
L_est = r_est * ((e[3] - s[3]) - (e[2] - s[2])) / dyaw
print(f"separation: measured {L_est:.4f} m vs model {L_WHEEL} m")

#compare motion model to ground truth
#wheel speed aims (wl, wr dur)
#convert wheel speeds to (v, w), go through the model f()
#compare predicted pose to true pose
schedule = [(3, 3, 3.0), (2, 4, 3.0), (3, 3, 3.0), (4, 2, 3.0)]
mujoco.mj_resetData(model, data)
x_pred = np.array([data.qpos[0], data.qpos[1], yaw(data)])
t_log, truth, pred = [], [], []

for wl, wr, dur in schedule:
    data.ctrl[:] = [control(wl), control(wr)]
    t_end = data.time + dur
    while data.time < t_end:
        mujoco.mj_step(model, data)
        u = wheels_to_v_w(data.qvel[6], data.qvel[7]) #wheel speed measurements
        x_pred = f(x_pred, u, dt) #prediction
        t_log.append(data.time)
        truth.append([data.qpos[0], data.qpos[1], yaw(data)]) #ground truth
        pred.append(x_pred.copy())

t_log, truth, pred = np.array(t_log), np.array(truth), np.array(pred)
pos_err = np.linalg.norm(truth[:, :2] - pred[:, :2], axis=1)
yaw_err = np.abs(wrap(truth[:, 2] - pred[:, 2]))
print(f"position error: final {pos_err[-1]:.4f} m, max {pos_err.max():.4f} m")
print(f"heading error:  final {np.degrees(yaw_err[-1]):.2f} deg, max {np.degrees(yaw_err.max()):.2f} deg")

#record true and predicted pose
#record data as error numbers in the figure below
#position error found from straight dist between true and predicted
#heading error - wrapped yaw diff


#ERROR FIGURE
#true vs predicted path
#position and heading error over time
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(truth[:, 0], truth[:, 1], label="MuJoCo (ground truth)")
ax[0].plot(pred[:, 0], pred[:, 1], "--", label="Model prediction")
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend()
ax[1].plot(t_log, pos_err, label="position error (m)")
ax[1].plot(t_log, yaw_err, label="heading error (rad)")
ax[1].set_xlabel("time (s)"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/model_validation.png", dpi=150)
print("saved figures/model_validation.png")

