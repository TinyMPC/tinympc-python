import tinympc
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

def log(msg):
    print(f"DEBUG: {msg}")
    sys.stdout.flush()

try:
    log("Creating TinyMPC instance")
    prob = tinympc.TinyMPC()
    
    log("Setting up system matrices")
    # Quadrotor system matrices
    # 12 states, 4 inputs
    
    # Convert the flattened array to a 12x12 matrix
    Adyn_data = [
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
    ]
    A = np.array(Adyn_data).reshape(12, 12)
    
    # Convert the flattened array to a 12x4 matrix
    Bdyn_data = [
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
    ]
    B = np.array(Bdyn_data).reshape(12, 4)
    
    # Cost matrices
    Q_data = [100.0, 100.0, 100.0, 4.0, 4.0, 400.0, 4.0, 4.0, 4.0, 2.0408163, 2.0408163, 4.0]
    Q = np.diag(Q_data)
    
    R_data = [4.0, 4.0, 4.0, 4.0]
    R = np.diag(R_data)
    
    N = 10  # Horizon length
    
    # Set control bounds
    u_min = np.array([-0.2, -0.2, -0.2, -0.2])
    u_max = np.array([0.2, 0.2, 0.2, 0.2])
    
    # Ensure matrices are in the correct format (Fortran-contiguous)
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    Q = np.asfortranarray(Q)
    R = np.asfortranarray(R)
    
    # Define rho values for sensitivity computation
    rho_base = 85.0
    rho_perturbed = rho_base + 0.1  # Small perturbation for finite differences
    
    log("Computing sensitivity matrices")
    
    # Step 1: Compute the infinite horizon LQR solution for base rho
    prob.setup(A, B, Q, R, N, rho=rho_base, u_min=u_min, u_max=u_max)
    
    # Compute the infinite horizon solution using scipy
    P_base = solve_discrete_are(A, B, Q, R)
    
    # Compute K = (R + B'PB)^-1 * B'PA
    temp = B.T @ P_base @ B + R
    K_base = np.linalg.inv(temp) @ B.T @ P_base @ A
    
    # Compute cache matrices
    C1_base = np.linalg.inv(temp)  # Quu_inv
    C2_base = A - B @ K_base  # AmBKt
    
    # Step 2: Compute the infinite horizon LQR solution for perturbed rho
    prob.setup(A, B, Q, R, N, rho=rho_perturbed, u_min=u_min, u_max=u_max)
    
    # Compute the infinite horizon solution using scipy
    P_perturbed = solve_discrete_are(A, B, Q, R)
    
    # Compute K = (R + B'PB)^-1 * B'PA
    temp = B.T @ P_perturbed @ B + R
    K_perturbed = np.linalg.inv(temp) @ B.T @ P_perturbed @ A
    
    # Compute cache matrices
    C1_perturbed = np.linalg.inv(temp)  # Quu_inv
    C2_perturbed = A - B @ K_perturbed  # AmBKt
    
    # Step 3: Compute sensitivity matrices using finite differences
    delta_rho = rho_perturbed - rho_base
    dK = (K_perturbed - K_base) / delta_rho
    dP = (P_perturbed - P_base) / delta_rho
    dC1 = (C1_perturbed - C1_base) / delta_rho
    dC2 = (C2_perturbed - C2_base) / delta_rho
    
    # Ensure matrices are in the correct format
    dK = np.asfortranarray(dK)
    dP = np.asfortranarray(dP)
    dC1 = np.asfortranarray(dC1)
    dC2 = np.asfortranarray(dC2)
    
    log("Sensitivity matrices computed")
    
    # Print some statistics about the sensitivity matrices
    log(f"dK norm: {np.linalg.norm(dK)}")
    log(f"dP norm: {np.linalg.norm(dP)}")
    log(f"dC1 norm: {np.linalg.norm(dC1)}")
    log(f"dC2 norm: {np.linalg.norm(dC2)}")
    
    # Reset the problem with the base rho
    prob.setup(A, B, Q, R, N, rho=rho_base, u_min=u_min, u_max=u_max)
    
    log("Generating code with sensitivity matrices")
    output_dir = "quadrotor_real_sensitivity"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use the codegen_with_sensitivity method directly
    prob._solver.codegen_with_sensitivity(
        output_dir,
        dK, dP, dC1, dC2,
        verbose=1
    )
    
    log(f"Code generation complete. Check the directory: {output_dir}")
    
    # Now we would compile and test the generated code
    log("To use the generated code:")
    log("1. Navigate to the output directory: cd " + output_dir)
    log("2. Compile the code: python setup.py build_ext --inplace")
    log("3. Import and use the generated module in your application")
    
    # For demonstration, let's visualize the sensitivity matrices
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(dK, cmap='viridis')
    plt.colorbar()
    plt.title('dK - Sensitivity of Feedback Gain')
    
    plt.subplot(2, 2, 2)
    plt.imshow(dP, cmap='viridis')
    plt.colorbar()
    plt.title('dP - Sensitivity of Value Function')
    
    plt.subplot(2, 2, 3)
    plt.imshow(dC1, cmap='viridis')
    plt.colorbar()
    plt.title('dC1 - Sensitivity of Quu_inv')
    
    plt.subplot(2, 2, 4)
    plt.imshow(dC2, cmap='viridis')
    plt.colorbar()
    plt.title('dC2 - Sensitivity of AmBKt')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_matrices.png'))
    
    log("Sensitivity matrices visualization saved")
    
except Exception as e:
    log(f"Exception: {e}")
    import traceback
    traceback.print_exc()