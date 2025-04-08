import tinympc
import numpy as np
import sys
import os
from scipy.linalg import solve_discrete_are

def log(msg):
    print(f"DEBUG: {msg}")
    sys.stdout.flush()

try:
    log("Creating TinyMPC instance")
    prob = tinympc.TinyMPC()
    
    log("Setting up system matrices")
    # Cartpole system matrices (simpler than quadrotor)
    A = np.array([[1.0, 0.01, 0.0, 0.0],
                  [0.0, 1.0, 0.039, 0.0],
                  [0.0, 0.0, 1.002, 0.01],
                  [0.0, 0.0, 0.458, 1.002]])
    B = np.array([[0.0],
                  [0.02],
                  [0.0],
                  [0.067]])
    Q = np.diag([10.0, 1, 10, 1])
    R = np.diag([1.0])
    
    N = 20  # Horizon length
    
    # Set control bounds
    u_min = np.array([-0.5])
    u_max = np.array([0.5])
    
    # Ensure matrices are in the correct format (Fortran-contiguous)
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    Q = np.asfortranarray(Q)
    R = np.asfortranarray(R)
    
    # Define rho values for sensitivity computation
    rho_base = 1.0
    rho_perturbed = rho_base + 0.1  # Small perturbation for finite differences
    
    log("Computing sensitivity matrices")
    
    # Step 1: Compute the infinite horizon LQR solution for base rho
    prob.setup(A, B, Q, R, N, rho=rho_base, max_iter=10, u_min=u_min, u_max=u_max)
    
    # Compute the infinite horizon solution using scipy
    P_base = solve_discrete_are(A, B, Q, R)
    
    # Compute K = (R + B'PB)^-1 * B'PA
    temp = B.T @ P_base @ B + R
    K_base = np.linalg.inv(temp) @ B.T @ P_base @ A
    
    # Compute cache matrices
    C1_base = np.linalg.inv(temp)  # Quu_inv
    C2_base = A - B @ K_base  # AmBKt
    
    # Step 2: Compute the infinite horizon LQR solution for perturbed rho
    prob.setup(A, B, Q, R, N, rho=rho_perturbed, max_iter=10, u_min=u_min, u_max=u_max)
    
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
    prob.setup(A, B, Q, R, N, rho=rho_base, max_iter=10, u_min=u_min, u_max=u_max)
    
    log("Generating code with sensitivity matrices")
    output_dir = "out"
    
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
    
except Exception as e:
    log(f"Exception: {e}")
    import traceback
    traceback.print_exc()