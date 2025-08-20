"""
Rocket Landing with Constraints
Based on: https://github.com/TinyMPC/TinyMPC/blob/main/examples/rocket_landing_mpc.cpp
"""

import tinympc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Problem dimensions
NSTATES = 6  # [x, y, z, vx, vy, vz] 
NINPUTS = 3  # [thrust_x, thrust_y, thrust_z]
NHORIZON = 10

# System dynamics (from rocket_landing_params_20hz.hpp)
A = np.array([
    [1.0, 0.0, 0.0, 0.05, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.05, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.05],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])

B = np.array([
    [0.000125, 0.0, 0.0],
    [0.0, 0.000125, 0.0],
    [0.0, 0.0, 0.000125],
    [0.005, 0.0, 0.0],
    [0.0, 0.005, 0.0],
    [0.0, 0.0, 0.005]
])

fdyn = np.array([0.0, 0.0, -0.0122625, 0.0, 0.0, -0.4905]).reshape(-1, 1)
Q = np.diag([101.0, 101.0, 101.0, 101.0, 101.0, 101.0])  # From parameter file
R = np.diag([2.0, 2.0, 2.0])  # From parameter file

# Box constraints
x_min = np.array([-5.0, -5.0, -0.5, -10.0, -10.0, -20.0])
x_max = np.array([5.0, 5.0, 100.0, 10.0, 10.0, 20.0])
u_min = np.array([-10.0, -10.0, -10.0])
u_max = np.array([105.0, 105.0, 105.0])

# SOC constraints 
cx = np.array([0.5])    # coefficients for state cones (mu)
cu = np.array([0.25])   # coefficients for input cones (mu)
Acx = np.array([0])     # start indices for state cones
Acu = np.array([0])     # start indices for input cones  
qcx = np.array([3])     # dimensions for state cones
qcu = np.array([3])     # dimensions for input cones

# Setup solver
solver = tinympc.TinyMPC()
solver.setup(A, B, Q, R, NHORIZON, rho=1.0, fdyn=fdyn,
             x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max,
             max_iter=100, abs_pri_tol=2e-3, verbose=True)

# Set cone constraints (inputs first)
solver.set_cone_constraints(Acu, qcu, cu, Acx, qcx, cx)

# Initial and goal states
xinit = np.array([4.0, 2.0, 20.0, -3.0, 2.0, -4.5])
xgoal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Initial reference trajectory (will be updated each step like C++)
x_ref = np.zeros((NSTATES, NHORIZON))
u_ref = np.zeros((NINPUTS, NHORIZON-1))

# Animation setup - Extended for longer simulation
NTOTAL = 100  # Match C++ 
x_current = xinit * 1.1  # Match C++ (x0 = xinit * 1.1)

# Set initial reference
for i in range(NHORIZON):
    x_ref[:, i] = xinit + (xgoal - xinit) * i / (NTOTAL - 1)  # Use NTOTAL-1 like C++
u_ref[2, :] = 10.0  # Hover thrust

solver.set_x_ref(x_ref)
solver.set_u_ref(u_ref)

# Store trajectory for plotting
trajectory = []
controls = []
constraint_violations = []

print("Starting rocket landing simulation...")
for k in range(NTOTAL - NHORIZON):
    # Calculate tracking error (match C++ exactly: (x0 - Xref.col(1)).norm())
    tracking_error = np.linalg.norm(x_current - x_ref[:, 1])
    print(f"tracking error: {tracking_error:.5f}")
    
    # 1. Update measurement (set current state)
    solver.set_x0(x_current)
    
    # 2. Update reference trajectory (match C++ exactly)
    for i in range(NHORIZON):
        x_ref[:, i] = xinit + (xgoal - xinit) * (i + k) / (NTOTAL - 1)
        if i < NHORIZON - 1:
            u_ref[2, i] = 10.0  # uref stays constant
    
    solver.set_x_ref(x_ref)
    solver.set_u_ref(u_ref)
    
    # 3. Solve MPC problem
    solution = solver.solve()
    
    # 4. Simulate forward (apply first control)
    u_opt = solution["controls"]
    x_current = A @ x_current + B @ u_opt + fdyn.flatten()
    
    # Store data for plotting
    trajectory.append(x_current.copy())
    controls.append(u_opt.copy())
    
    # Check constraint violations
    altitude_violation = x_current[2] < 0  # Ground constraint
    thrust_violation = np.linalg.norm(u_opt[:2]) > 0.25 * abs(u_opt[2])  # Cone constraint
    constraint_violations.append(altitude_violation or thrust_violation)

# Convert to arrays
trajectory = np.array(trajectory)
controls = np.array(controls)

print(f"\nSimulation completed!")
print(f"Initial state was: [{(xinit * 1.1)[0]:.2f}, {(xinit * 1.1)[1]:.2f}, {(xinit * 1.1)[2]:.2f}, {(xinit * 1.1)[3]:.2f}, {(xinit * 1.1)[4]:.2f}, {(xinit * 1.1)[5]:.2f}]")
print(f"Final position: [{x_current[0]:.2f}, {x_current[1]:.2f}, {x_current[2]:.2f}]")
print(f"Final velocity: [{x_current[3]:.2f}, {x_current[4]:.2f}, {x_current[5]:.2f}]")
print(f"Distance to goal: {np.linalg.norm(x_current[:3]):.3f} m")
print(f"Constraint violations: {sum(constraint_violations)}/{len(constraint_violations)}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Rocket Landing with Constraints', fontsize=16)

# 2D trajectory (X-Y view)
ax1 = axes[0, 0]
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
ax1.scatter((xinit * 1.1)[0], (xinit * 1.1)[1], c='red', s=100, label='Start')
ax1.scatter(xgoal[0], xgoal[1], c='green', s=100, label='Goal')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.legend()
ax1.set_title('2D Trajectory (X-Y)')
ax1.grid(True)

# Position vs time
ax2 = axes[0, 1]
time_steps = np.arange(len(trajectory))
ax2.plot(time_steps, trajectory[:, 0], 'r-', label='X')
ax2.plot(time_steps, trajectory[:, 1], 'g-', label='Y')
ax2.plot(time_steps, trajectory[:, 2], 'b-', label='Z')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Ground')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Position (m)')
ax2.legend()
ax2.set_title('Position vs Time')
ax2.grid(True)

# Velocity vs time
ax3 = axes[1, 0]
ax3.plot(time_steps, trajectory[:, 3], 'r-', label='Vx')
ax3.plot(time_steps, trajectory[:, 4], 'g-', label='Vy')
ax3.plot(time_steps, trajectory[:, 5], 'b-', label='Vz')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Velocity (m/s)')
ax3.legend()
ax3.set_title('Velocity vs Time')
ax3.grid(True)

# Thrust vs time
ax4 = axes[1, 1]
ax4.plot(time_steps, controls[:, 0], 'r-', label='Thrust X')
ax4.plot(time_steps, controls[:, 1], 'g-', label='Thrust Y')
ax4.plot(time_steps, controls[:, 2], 'b-', label='Thrust Z')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Thrust (N)')
ax4.legend()
ax4.set_title('Thrust vs Time')
ax4.grid(True)

plt.tight_layout()
plt.show()
