import tinympc
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

# Set initial condition
x0 = np.array([0.5, 0, 0, 0])

Nsim = 1000
xs = np.zeros((Nsim-N, 4))
for i in range(Nsim-N):
    prob.set_x0(x0) # Set first state in horizon
    solution = prob.solve()
    x0 = A@x0 + B@solution["controls"] # Simulate forward
    xs[i] = x0

plt.plot(xs, label=["x (meters)", "theta (radians)", "x_dot (m/s)", "theta_dot (rad/s)"])
plt.title("cartpole state over time")
plt.xlabel("time steps (100Hz)")
plt.legend()
plt.show()