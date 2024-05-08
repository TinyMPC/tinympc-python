import numpy as np
import tinympcgen
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# These are only here for simulation purposes
A = np.array([[1.0, 0.01, 0.0, 0.0],
              [0.0, 1.0, 0.039, 0.0],
              [0.0, 0.0, 1.002, 0.01],
              [0.0, 0.0, 0.458, 1.002]])
B = np.array([[0.0  ],
              [0.02 ],
              [0.0  ],
              [0.067]])

x0 = np.array([0.5, 0, 0, 0])

tinympcgen.set_x_ref(np.array([np.array([1.0, 0, 0, 0])]*20).T)

Nsim = 1000
xs = np.zeros((Nsim, 4))
us = np.zeros((Nsim, 1))
for i in range(Nsim):
    tinympcgen.set_x0(x0) # Set first state in horizon
    solution = tinympcgen.solve()
    x0 = A@x0 + B@solution["controls"] # Simulate forward
    xs[i] = x0
    us[i] = solution["controls"]

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(xs, label=["x (meters)", "theta (radians)", "x_dot (m/s)", "theta_dot (rad/s)"])
axs[1].plot(us, label="control (Newtons)")
axs[0].set_title("quadrotor trajectory over time")
axs[1].set_xlabel("time steps (100Hz)")
axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")
plt.show()