import tinympc
import numpy as np
import autograd.numpy as anp
from autograd import grad, jacobian

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

# Enable adaptive rho and compute cache terms
if hasattr(prob.settings, 'adaptive_rho'):
    prob.settings.adaptive_rho = 1
    print("Enabled adaptive rho for quadrotor")
    
    # First compute and set cache terms
    print("Computing and setting cache terms...")
    Kinf, Pinf, Quu_inv, AmBKt = prob.compute_cache_terms()
    print("Cache terms set successfully")
    
    def compute_cache_terms_autograd(rho, A, B, Q, R):
        """Compute cache terms as a function of rho for autograd"""
        # Convert inputs to autograd arrays
        A = anp.array(A)
        B = anp.array(B)
        Q = anp.array(Q)
        R = anp.array(R)
        
        # Add rho regularization
        Q_rho = Q + rho * anp.eye(Q.shape[0])
        R_rho = R + rho * anp.eye(R.shape[0])
        
        # Initialize
        Kinf = anp.zeros((B.shape[1], A.shape[0]))
        Pinf = anp.array(Q)
        
        # Compute infinite horizon solution (fixed number of iterations for autograd)
        for _ in range(100):  # Reduced iterations for derivative computation
            Kinf = anp.linalg.solve(
                R_rho + B.T @ Pinf @ B + 1e-8*anp.eye(B.shape[1]),
                B.T @ Pinf @ A
            )
            Pinf = Q_rho + A.T @ Pinf @ (A - B @ Kinf)
        
        AmBKt = (A - B @ Kinf).T
        Quu_inv = anp.linalg.inv(R_rho + B.T @ Pinf @ B)
        
        return Kinf, Pinf, Quu_inv, AmBKt
    
    # Define functions to get each matrix separately for autograd
    def get_Kinf(rho):
        return compute_cache_terms_autograd(rho, A, B, Q, R)[0]
    
    def get_Pinf(rho):
        return compute_cache_terms_autograd(rho, A, B, Q, R)[1]
    
    def get_Quu_inv(rho):
        return compute_cache_terms_autograd(rho, A, B, Q, R)[2]
    
    def get_AmBKt(rho):
        return compute_cache_terms_autograd(rho, A, B, Q, R)[3]
    
    # Now compute derivatives using autograd
    print("Computing sensitivity matrices...")
    rho = 1.0  # Same rho as in setup
    dK = jacobian(get_Kinf)(rho)
    dP = jacobian(get_Pinf)(rho)
    dC1 = jacobian(get_Quu_inv)(rho)
    dC2 = jacobian(get_AmBKt)(rho)
    
    print("Setting sensitivity matrices...")
    prob.set_sensitivity_matrices(dK, dP, dC1, dC2)
    print("Sensitivity matrices set successfully")

# Generate code
prob.codegen("out", verbose=1)