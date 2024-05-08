import tinympc
import numpy as np

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

prob = tinympc.TinyMPC()

u_min = np.array([-0.5])
u_max = np.array([0.5])
prob.setup(A, B, Q, R, N, rho=1, max_iter=10, u_min=u_min, u_max=u_max)

prob.codegen("out", verbose=1)