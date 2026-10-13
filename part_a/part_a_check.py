import numpy as np
from part_a.model import f, F_jac, G_jac, H_jac, h_range_bearing

def num_jac(fnct, z, eps = 1e-6):
    columns = []
    for i in range(len(z)):
        d = np.zeros(len(z)); d[i] = eps

        columns.append((fnct(z + d) - fnct(z - d)) / (2 * eps))
    return np.array(columns).T

dt = 0.05 #seconds
x0 = np.array([0.5, -0.2, 0.5])
u0 = np.array([0.1, 0.3])
m = np.array([2.0, 1.0])

#check if jacobians are correct (1e-8 or less)
#compare model calculations with numerical
print("F error:", np.abs(F_jac(x0, u0, dt) - num_jac(lambda x: f(x, u0, dt), x0)).max())
print("G error:", np.abs(G_jac(x0, u0, dt) - num_jac(lambda u: f(x0, u, dt), u0)).max())
print("H error:", np.abs(H_jac(x0, m) - num_jac(lambda x: h_range_bearing(x, m), x0)).max())