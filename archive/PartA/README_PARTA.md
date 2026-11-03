PART A:

PARAMETERS
Nominal wheel radius = 0.033m (from geometry)
Nominal wheel seperation = 0.288m
Calculated wheel radius, from XML = 0.0283m
Calculated wheel seperation, from XML = 0.353m
dt = 0.05s

Wheel radius at different wheel speeds.
1.5rad/s -> 0.0313m
3rad/s -> 0.0283m
4.5rad/s -> 0.0281m
At slower speeds, the robot travels a further distance per wheel turn, meaning the wheels have a larger radius.

OBSERVABILITY (starting with [0, 0, 0.3], u = [0.15, 0.3], 30 steps)
Landmarks                   Rank                Singular values
One                         2                   0.00
Two, far away               3                   4.99
Two, nearly aligned         3                   0.85
Two, close together         3                   0.09

CONTROLLABILITY
Speed (v)                   Rank               Singular Values
0m/s                        2                  0.0866, 0.0866, 0
0.15m/s                     3                  0.0866, 0.0866, 5.3e-4

MUJOCO VALIDATION
Final position error:   nominal parameter (0.209m), Calculated parameter (0.044m)
Max heading error:      nominal parameter (14.2 deg), Calculated parameter (4.8 deg)
Final heading error:    nominal parameter(1.1 deg), Calculated parameter (0.7 deg)

