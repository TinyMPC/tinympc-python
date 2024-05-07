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

prob.setup(A, B, Q, R, N, rho=1, max_iter=10)

x0 = np.array([0.5, 0, 0, 0])
prob.set_x0(x0)

solution = prob.solve()
print(solution["controls"])