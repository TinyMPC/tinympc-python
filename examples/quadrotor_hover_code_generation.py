import tinympc
import numpy as np
from autograd import jacobian
import autograd.numpy as anp

# Toggle switch for adaptive rho
ENABLE_ADAPTIVE_RHO = True   # Set to True to enable adaptive rho

# Quadrotor system matrices (12 states, 4 inputs)
rho_value = 5.0
Adyn = np.array([
    1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0245250, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000, 0.0002044, 0.0000000,
    0.0000000, 1.0000000, 0.0000000, -0.0245250, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, -0.0002044, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.9810000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0122625, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, -0.9810000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -0.0122625, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000
]).reshape(12, 12)

# Input/control matrix
Bdyn = np.array([
    -0.0007069, 0.0007773, 0.0007091, -0.0007795,
    0.0007034, 0.0007747, -0.0007042, -0.0007739,
    0.0052554, 0.0052554, 0.0052554, 0.0052554,
    -0.1720966, -0.1895213, 0.1722891, 0.1893288,
    -0.1729419, 0.1901740, 0.1734809, -0.1907131,
    0.0123423, -0.0045148, -0.0174024, 0.0095748,
    -0.0565520, 0.0621869, 0.0567283, -0.0623632,
    0.0562756, 0.0619735, -0.0563386, -0.0619105,
    0.2102143, 0.2102143, 0.2102143, 0.2102143,
    -13.7677303, -15.1617018, 13.7831318, 15.1463003,
    -13.8353509, 15.2139209, 13.8784751, -15.2570451,
    0.9873856, -0.3611820, -1.3921880, 0.7659845
]).reshape(12, 4)


Q_diag = np.array([100.0000000, 100.0000000, 100.0000000, 4.0000000, 4.0000000, 400.0000000,
                   4.0000000, 4.0000000, 4.0000000, 2.0408163, 2.0408163, 4.0000000])
R_diag = np.array([4.0000000, 4.0000000, 4.0000000, 4.0000000])
Q = np.diag(Q_diag)
R = np.diag(R_diag)

N = 20

prob = tinympc.TinyMPC()

u_min = -np.ones(4) * 2.0
u_max = np.ones(4) * 2.0

# Setup with adaptive rho based on toggle
prob.setup(Adyn, Bdyn, Q, R, N, rho=rho_value, max_iter=100, u_min=u_min, u_max=u_max, 
          adaptive_rho=1 if ENABLE_ADAPTIVE_RHO else 0)

if ENABLE_ADAPTIVE_RHO:
    print("Enabled adaptive rho - generating code with sensitivity matrices...")
    
    # First compute the cache terms (this will compute K, P, etc.)
    Kinf, Pinf, Quu_inv, AmBKt = prob.compute_cache_terms()
    
    # Compute derivatives with respect to rho via Autograd's Jacobian
    def lqr_direct(rho):
        R_rho = anp.array(R) + rho * anp.eye(4)
        Q_rho = anp.array(Q) + rho * anp.eye(12)
        P = Q_rho
        for _ in range(50):
            K = anp.linalg.solve(
                R_rho + Bdyn.T @ P @ Bdyn + 1e-4*anp.eye(4),
                Bdyn.T @ P @ Adyn
            )
            P = Q_rho + Adyn.T @ P @ (Adyn - Bdyn @ K)
        # Final gain and cache matrices
        K = anp.linalg.solve(
            R_rho + Bdyn.T @ P @ Bdyn + 1e-4*anp.eye(4),
            Bdyn.T @ P @ Adyn
        )
        C1 = anp.linalg.inv(R_rho + Bdyn.T @ P @ Bdyn)
        C2 = (Adyn - Bdyn @ K).T
        return anp.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])

    derivs = jacobian(lqr_direct)(rho_value)
    # Dynamically split the derivative vector based on matrix sizes
    deriv_array = np.array(derivs)
    nu, nx = Bdyn.shape[1], Adyn.shape[0]
    idx = 0
    dK  = deriv_array[idx:idx + nu * nx].reshape(nu, nx); idx += nu * nx
    dP  = deriv_array[idx:idx + nx * nx].reshape(nx, nx); idx += nx * nx
    dC1 = deriv_array[idx:idx + nu * nu].reshape(nu, nu); idx += nu * nu
    dC2 = deriv_array[idx:idx + nx * nx].reshape(nx, nx)
    
    # Generate code with sensitivity matrices
    prob.codegen_with_sensitivity("out", dK, dP, dC1, dC2, verbose=1)   
else:
    print("Running without adaptive rho - generating code without sensitivity matrices...")
    prob.codegen("out", verbose=1)