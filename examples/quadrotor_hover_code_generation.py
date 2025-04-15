import tinympc
import numpy as np

# Toggle switch for adaptive rho
ENABLE_ADAPTIVE_RHO = True  # Set to True to enable adaptive rho

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

# Setup with adaptive rho based on toggle
prob.setup(A, B, Q, R, N, rho=1.0, max_iter=100, u_min=u_min, u_max=u_max, 
          adaptive_rho=1 if ENABLE_ADAPTIVE_RHO else 0)

if ENABLE_ADAPTIVE_RHO:
    print("Enabled adaptive rho - generating code with sensitivity matrices...")
    
    # First compute the cache terms (this will compute K, P, etc.)
    Kinf, Pinf, Quu_inv, AmBKt = prob.compute_cache_terms()
    
    # Compute sensitivity matrices using autograd
    # For now, we'll use small perturbations to approximate derivatives
    eps = 1e-4
    rho = 1.0
    
    # Compute perturbed cache terms
    prob.setup(A, B, Q, R, N, rho=rho + eps, max_iter=100, u_min=u_min, u_max=u_max, adaptive_rho=1)
    Kinf_p, Pinf_p, Quu_inv_p, AmBKt_p = prob.compute_cache_terms()
    
    # Compute derivatives with respect to rho using finite differences
    dK = (Kinf_p - Kinf) / eps
    dP = (Pinf_p - Pinf) / eps
    dC1 = (Quu_inv_p - Quu_inv) / eps  # dC1 is derivative of Quu_inv
    dC2 = (AmBKt_p - AmBKt) / eps      # dC2 is derivative of AmBKt
    
    # Generate code with sensitivity matrices
    prob.codegen_with_sensitivity("out", dK, dP, dC1, dC2, verbose=1)   
else:
    print("Running without adaptive rho - generating code without sensitivity matrices...")
    prob.codegen("out", verbose=1)