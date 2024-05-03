import src.tinympc.interface as tinympc
import numpy as np

A = np.array([[1.0, 0.01, 0.0, 0.0],
              [0.0, 1.0, 0.039, 0.0],
              [0.0, 0.0, 1.002, 0.01],
              [0.0, 0.0, 0.458, 1.002]])
B = np.array([[0.0  ],
              [0.02 ],
              [0.0  ],
              [0.067]])
Q = np.array([10.0, 1, 10, 1])
R = np.array([1.0])

prob = tinympc.TinyMPC()

prob.setup(A, B, Q, R)

prob.solve()
