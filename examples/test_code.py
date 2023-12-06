import ctypes
import tinympc
import numpy as np
import subprocess
import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    tinympc_dir = "/home/khai/SSD/Code/tinympc-python/generated_code"

    # # Specify the path to your CMakeLists.txt file or the source directory
    # source_directory = tinympc_dir

    # # Specify the path to the build directory (where CMake will generate build files)
    # build_directory = tinympc_dir + "/build"

    # # Make sure the build directory exists
    # os.makedirs(build_directory, exist_ok=True)

    # # Run CMake configuration
    # cmake_configure_cmd = ["cmake", source_directory]
    # subprocess.run(cmake_configure_cmd, cwd=build_directory)

    # # Run the build process (e.g., make)
    # cmake_build_cmd = ["cmake", "--build", "."]
    # subprocess.run(cmake_build_cmd, cwd=build_directory)

    n = 4
    m = 1
    N = 10
    A = [
        1,
        0,
        0,
        0,
        0.01,
        1,
        0,
        0,
        2.2330083403300767e-5,
        0.004466210576510177,
        1.0002605176397052,
        0.05210579005928538,
        7.443037974683548e-8,
        2.2330083403300767e-5,
        0.01000086835443038,
        1.0002605176397052,
    ]
    # Anp is a numpy array 4x4 from A
    Anp = np.array(A).reshape((n, n)).transpose()
    print(Anp)
    B = [
        7.468368562730335e-5,
        0.014936765390161838,
        3.79763323185387e-5,
        0.007595596218554721,
    ]
    Bnp = np.array(B).reshape((n, m))
    print(Bnp)
    
    Q = [10, 1, 10, 1]
    R = [1]
    rho = 0.1

    x_min = [-5] * n * N
    x_max = [5] * n * N
    u_min = [-5] * m * (N - 1)
    u_max = [5] * m * (N - 1)

    abs_pri_tol = 1e-3
    abs_dual_tol = 1e-3
    max_iter = 100
    check_termination = 1

    prob = tinympc.TinyMPC()
    prob.load_lib(tinympc_dir + "/build/tinympc/libtinympcShared.so")  # Load the library
    prob.setup(n, m, N, A, B, Q, R, x_min, x_max, u_min, u_max, rho, abs_pri_tol, abs_dual_tol, max_iter, check_termination)
    x = [0.5, -0.2, 0.1, 0]
    u = [0.0] * m * (N - 1)
    x_all = []

    print("=== START MPC ===")
    NSIM = 300
    for i in range(NSIM):
        prob.set_x0(x, 0)
        prob.solve(0)
        prob.get_u(u, 0)
        # print(np.array(u[0]))
        x = Anp@np.array(x).reshape((n, 1))+ Bnp*np.array(u[0])
        # print(f"X = {x}")
        x = x.reshape(n).tolist()
        # print(f"X = {x}")
        x_all.append(x)
    print(len(x_all))

    # Set up the figure and axis for plotting
    fig, ax = plt.subplots()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1)

    # Initialize the cartpole visualization
    cart, = ax.plot([], [], 'bo', markersize=20)
    pole, = ax.plot([], [], 'r-', linewidth=4)

    def init():
        cart.set_data([], [])
        pole.set_data([], [])
        return cart, pole

    def update(frame):
        x = x_all[frame]
        # Update the cart position
        cart.set_data([x[0]], [0])

        # Update the pole position
        pole.set_data([x[0], x[0] + 0.5*math.sin(x[1])], [0, 0.5 * math.cos(x[1])])
        print(frame)
        if frame==NSIM-1:
            ani.event_source.stop()  # Stop the animation if the episode is 
        return cart, pole

    # Create the animation
    ani = FuncAnimation(fig, update, frames=NSIM, init_func=init, blit=True, interval=10)

    # Display the animation
    plt.show()