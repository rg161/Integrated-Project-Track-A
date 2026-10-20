import numpy as np
from part_a.model import f, F_jac, G_jac, H_jac, h_range_bearing

"""
Part A: checks for differential-drive model in part_a/model.py

- compare derived and numerical jacobians

- controllability of system (both still and driving)

- observability from range-bearing to a landmark m

- observability with more than 1 landmark
"""


def num_jac(fnct, z, eps = 1e-6):
    """
    find num_jac at point z to check formulas
    measure outputs, record w/ columns
    """
    columns = []
    for i in range(len(z)):
        d = np.zeros(len(z)); d[i] = eps

        columns.append((fnct(z + d) - fnct(z - d)) / (2 * eps))
    return np.array(columns).T


#test point!
dt = 0.05 #seconds
x0 = np.array([0.5, -0.2, 0.5])
u0 = np.array([0.1, 0.3])
m = np.array([2.0, 1.0])

#check if jacobians are correct (1e-8 or less)
#compare model calculations with numerical
print("F error:", np.abs(F_jac(x0, u0, dt) - num_jac(lambda x: f(x, u0, dt), x0)).max())
print("G error:", np.abs(G_jac(x0, u0, dt) - num_jac(lambda u: f(x0, u, dt), u0)).max())
print("H error:", np.abs(H_jac(x0, m) - num_jac(lambda x: h_range_bearing(x, m), x0)).max())

#check if robot is controllable
#can v and w get to any pose
#rank 3 if all state directions are achievable
def ctrb(v):
    xs = np.zeros(3) #origin (0 heading)
    us = np.array([v, 0.0])
    A, B = F_jac(xs, us, dt), G_jac(xs, us, dt)
    C = np.hstack([B, A @ B, A @ A @ B])
    return np.linalg.matrix_rank(C), np.linalg.svd(C, compute_uv = False)

for v in [0.0, 0.15]: #Still/movement
    rank, s = ctrb(v)
    print(f"v = {v}: controllability rank {rank}, singular values {np.round(s, 8)}")

#Check measurements from landmark location
#if observable from m (landmark)
def obs(x0, u, m, N=30):
    """Singular values and right singular vectors of the stacked observability
    matrix for one landmark m, starting at pose x0 with constant input u."""
    x = np.array(x0, float); Phi = np.eye(3); rows = []
    for _ in range(N):
        rows.append(H_jac(x, m) @ Phi)
        Phi = F_jac(x, u, dt) @ Phi
        x = f(x, u, dt)
    _, s, Vt = np.linalg.svd(np.vstack(rows))
    return s, Vt

s, Vt = obs([0, 0, 0.3], [0.15, 0.3], m)
pred = np.array([-(0 - m[1]), (0 - m[0]), 1.0]); pred /= np.linalg.norm(pred)
print("null vector:", np.round(Vt[-1], 4), "predicted:", np.round(pred, 4)) 

#Test 2nd landmark
#same, add measurement to both landmarks per step
def obs_multi(x0, u, landmarks, N = 30):
    x = np.array(x0, float); Phi = np.eye(3); rows = []
    for _ in range(N):
        for mk in landmarks:
            rows.append(H_jac(x, mk) @ Phi)
        Phi = F_jac(x, u, dt) @ Phi
        x = f(x, u, dt)
    _, s, Vt = np.linalg.svd(np.vstack(rows))
    return s, Vt

# Compare landmark layouts. Rank says whether the pose is recoverable at all
cases = {
    "one landmark": [np.array([2.0, 1.0])],
    "two, far apart": [np.array([2.0, 1.0]), np.array([-1.0, 2.0])],
    "two, nearly aligned": [np.array([2.0, 1.0]), np.array([4.0, 2.0])],
    "two, close together": [np.array([2.0, 1.0]), np.array([2.1, 1.0])],
}
for name, lms in cases.items():
    s, Vt = obs_multi([0, 0, 0.3], [0.15, 0.3], lms)
    print(f"{name}: rank {int((s > 1e-8).sum())}, singular values {np.round(s, 6)}")
