import tinympc
import numpy as np
import autograd.numpy as anp
from autograd import jacobian

class LQRSensitivity:
    def __init__(self, A, B, Q, R):
        self.NSTATES = A.shape[0]
        self.NINPUTS = B.shape[1]
        self.Adyn = A
        self.Bdyn = B
        self.Q = Q
        self.R = R
        
    def lqr_direct(self, rho):
        R_rho = self.R + rho * anp.eye(self.NINPUTS)
        P = self.Q.copy()
        for _ in range(100):
            K = anp.linalg.solve(R_rho + self.Bdyn.T @ P @ self.Bdyn, 
                               self.Bdyn.T @ P @ self.Adyn)
            P_new = self.Q + self.Adyn.T @ P @ self.Adyn - self.Adyn.T @ P @ self.Bdyn @ K
            if anp.allclose(P, P_new):
                break
            P = P_new
        
        K = anp.linalg.solve(R_rho + self.Bdyn.T @ P @ self.Bdyn, 
                            self.Bdyn.T @ P @ self.Adyn)
        C1 = anp.linalg.inv(R_rho + self.Bdyn.T @ P @ self.Bdyn)
        C2 = self.Adyn - self.Bdyn @ K
        
        return anp.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
    
    def compute_derivatives(self, rho=5.0):
        derivs = jacobian(self.lqr_direct)(rho)
        k_size = self.NINPUTS * self.NSTATES
        p_size = self.NSTATES * self.NSTATES
        c1_size = self.NINPUTS * self.NINPUTS
        
        dK = derivs[:k_size].reshape(self.NINPUTS, self.NSTATES)
        dP = derivs[k_size:k_size+p_size].reshape(self.NSTATES, self.NSTATES)
        dC1 = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(self.NINPUTS, self.NINPUTS)
        dC2 = derivs[k_size+p_size+c1_size:].reshape(self.NSTATES, self.NSTATES)
        
        return dK, dP, dC1, dC2

# Problem setup
A = np.array([[1.0, 0.01, 0.0, 0.0],
              [0.0, 1.0, 0.039, 0.0],
              [0.0, 0.0, 1.002, 0.01],
              [0.0, 0.0, 0.458, 1.002]])
B = np.array([[0.0  ],
              [0.02 ],
              [0.0  ],
              [0.067]])
Q = np.diag([10.0, 1, 10, 1])
R = np.diag([1.0])

N = 20

# Setup TinyMPC problem
prob = tinympc.TinyMPC()

u_min = np.array([-0.5])
u_max = np.array([0.5])
prob.setup(A, B, Q, R, N, rho=1, max_iter=10, u_min=u_min, u_max=u_max)

# Compute sensitivity matrices
lqr = LQRSensitivity(anp.array(A), anp.array(B), anp.array(Q), anp.array(R))
dK, dP, dC1, dC2 = lqr.compute_derivatives(rho=1.0)

# Add sensitivity matrices to prob object (assuming you add this capability to TinyMPC)
prob.set_sensitivity_matrices(dK, dP, dC1, dC2)

# Generate code with sensitivity matrices included
prob.codegen("out", verbose=1)