import tinympc
import numpy as np

# Quadrotor system matrices (12 states, 4 inputs)
A = np.eye(12)  # Identity matrix for simplicity
B = np.zeros((12, 4))
# Fill in control effectiveness
B[0:3, 0:4] = 0.01 * np.ones((3, 4))  # Position control
B[3:6, 0:4] = 0.05 * np.ones((3, 4))  # Velocity control
B[6:9, 0:4] = 0.02 * np.ones((3, 4))  # Attitude control
B[9:12, 0:4] = 0.1 * np.ones((3, 4))  # Angular velocity control

# Cost matrices
Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 0.1, 0.1, 0.1])
R = np.diag([0.1, 0.1, 0.1, 0.1])

N = 20

prob = tinympc.TinyMPC()

u_min = -np.ones(4) * 2.0
u_max = np.ones(4) * 2.0
prob.setup(A, B, Q, R, N, rho=1.0, max_iter=100, u_min=u_min, u_max=u_max)

# Enable adaptive rho for quadrotor
if hasattr(prob.settings, 'adaptive_rho'):
    prob.settings.adaptive_rho = 1
    print("Enabled adaptive rho for quadrotor")

# Generate code
prob.codegen("out", verbose=1)