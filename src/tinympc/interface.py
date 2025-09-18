import os
import sys
import shutil
import subprocess
import importlib
import importlib.resources
import numpy as np

class TinyMPC:
    def __init__(self):
        self.nx = 0 # number of states
        self.nu = 0 # number of control inputs
        self.N = 0 # number of knotpoints in the horizon
        self.A = [] # state transition matrix
        self.B = [] # control matrix
        self.Q = [] # state cost matrix (diagonal)
        self.R = [] # input cost matrix (digaonal)
        self.rho = 0
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        # Import tinympc pybind extension
        self.ext = importlib.import_module("tinympc.ext_tinympc")

        self._tinytype = np.float32
        self._infty = 1e17 # TODO: make this max system value
        
        self._solver = None # Solver that stores its own settings, cache, and problem vars/workspace
        self.settings = None # Local settings
        

    
    
    def update_settings(self, **kwargs):
        assert self.settings is not None
        
        if 'abs_pri_tol' in kwargs:
            self.settings.abs_pri_tol = kwargs.pop('abs_pri_tol')
        if 'abs_dua_tol' in kwargs:
            self.settings.abs_dua_tol = kwargs.pop('abs_dua_tol')
        if 'max_iter' in kwargs:
            self.settings.max_iter = kwargs.pop('max_iter')
        if 'check_termination' in kwargs:
            self.settings.check_termination = kwargs.pop('check_termination')
        if 'en_state_bound' in kwargs:
            self.settings.en_state_bound = 1 if kwargs.pop('en_state_bound') else 0
        if 'en_input_bound' in kwargs:
            self.settings.en_input_bound = 1 if kwargs.pop('en_input_bound') else 0
        if 'en_state_linear' in kwargs:
            self.settings.en_state_linear = 1 if kwargs.pop('en_state_linear') else 0
        if 'en_input_linear' in kwargs:
            self.settings.en_input_linear = 1 if kwargs.pop('en_input_linear') else 0
        if 'en_state_soc' in kwargs:
            self.settings.en_state_soc = 1 if kwargs.pop('en_state_soc') else 0
        if 'en_input_soc' in kwargs:
            self.settings.en_input_soc = 1 if kwargs.pop('en_input_soc') else 0

        if self._solver is not None:
            self._solver.update_settings(self.settings)        
        

    def expand_ndarray(self, array_, expected_rows, expected_cols, fallback):
        """Takes array_ given by a user, can be of size expected_rows x 1 or expected_rows x expected_cols.
        If of size expected_rows x 1, expands to be of size expected_rows x expected_cols.
        If neither, returns array_ of size expected_rows x expected_cols full of the fallback number.
        """
        if array_ is not None:
            assert array_.shape == (expected_rows, expected_cols) or array_.shape == (expected_rows,), "Expected numpy array to have shape ({},{}) or ({},)".format(expected_rows, expected_cols, expected_rows)
            if len(array_.shape) == 1:
                if len(array_) == expected_rows: # If expected_rows x 1, expand to expected_rows x expected_cols
                    array_ = np.array([array_]*expected_cols).T
            elif len(array_.shape) == 2: # If already expected_rows x expected_cols, do nothing
                if array_.shape == (expected_rows, expected_cols):
                    array_ = array_
            array_[array_ == None] = fallback # Replace all None values with fallback
            assert array_.shape == (expected_rows, expected_cols)
        else:
            array_ = np.ones((expected_rows, expected_cols))*fallback
        return array_

    # Setup the problem data and solver options
    def setup(self, A, B, Q, R, N, rho=1.0, fdyn=None, 
              x_min=None, x_max=None, u_min=None, u_max=None, 
              cone_constraints=None, verbose=False, **settings):
        """Instantiate necessary algorithm variables and parameters
        
        :param A (np.ndarray): State transition matrix of the linear system, size nx x nx
        :param B (np.ndarray): Input matrix of the linear system, size nx x nu
        :param Q (np.ndarray): Stage cost for state, must be diagonal and positive semi-definite, size nx x nx
        :param R (np.ndarray): Stage cost for input, must be diagonal and positive definite, size nu x nu
        :param N (int): Prediction horizon length
        :param rho (float, optional): Penalty term used in ADMM, default 1.0
        :param fdyn (np.ndarray or None, optional): Affine offset vector for dynamics, size nx x 1. If None, defaults to zeros (linear system), default None
        :param x_min (np.ndarray or None, optional): Lower bound state constraints, size nx or nx x N, default None
        :param x_max (np.ndarray or None, optional): Upper bound state constraints, size nx or nx x N, default None  
        :param u_min (np.ndarray or None, optional): Lower bound input constraints, size nu or nu x (N-1), default None
        :param u_max (np.ndarray or None, optional): Upper bound input constraints, size nu or nu x (N-1), default None
        :param cone_constraints (dict or None, optional): Cone constraints dict with keys {Acu, qcu, cu, Acx, qcx, cx}, default None
        :param verbose (bool): Whether or not to print data to console during setup, default False
        :params settings: Dictionary of optional settings
            :param abs_pri_tol (float): Solution tolerance for primal variables
            :param abs_dua_tol (float): Solution tolerance for dual variables
            :param max_iter (int): Maximum number of iterations before returning
            :param check_termination (int): Number of iterations to skip before checking termination
            :param en_state_bound (bool): Enable or disable bound constraints on state
            :param en_input_bound (bool): Enable or disable bound constraints on input
        """
        self.rho = rho
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]
        self.A = np.array(A, order="F") # order=F for compatibility with eigen's column-major storage when using pybind
        self.B = np.array(B, order="F")
        self.Q = np.array(Q, order="F")
        self.R = np.array(R, order="F")

        self.nx = A.shape[0]
        self.nu = B.shape[1]

        # Handle fdyn parameter - default to zeros for linear systems
        if fdyn is None:
            fdyn = np.zeros((self.nx, 1))
        self.fdyn = np.array(fdyn, order="F")

        assert N > 1
        self.N = N


        self.verbose = verbose


        self.settings = self.ext.TinySettings()
        self.ext.tiny_set_default_settings(self.settings)
        # Align with MATLAB/C++: keep all constraints disabled after setup
        self.settings.en_state_bound = 0
        self.settings.en_input_bound = 0
        self.settings.en_state_linear = 0
        self.settings.en_input_linear = 0
        self.settings.en_state_soc = 0
        self.settings.en_input_soc = 0
        self.update_settings(**settings)

        # Add adaptive rho settings
        if 'adaptive_rho' in settings:
            self.settings.adaptive_rho = 1 if settings.pop('adaptive_rho') else 0
        if 'adaptive_rho_min' in settings:
            self.settings.adaptive_rho_min = settings.pop('adaptive_rho_min')
        if 'adaptive_rho_max' in settings:
            self.settings.adaptive_rho_max = settings.pop('adaptive_rho_max')
        if 'adaptive_rho_enable_clipping' in settings:
            self.settings.adaptive_rho_enable_clipping = 1 if settings.pop('adaptive_rho_enable_clipping') else 0

        self._solver = self.ext.TinySolver(self.A, self.B, self.fdyn, self.Q, self.R, self.rho,
                                           self.nx, self.nu, self.N,
                                           self.settings, self.verbose)

        # Handle constraints if provided 
        if any(x is not None for x in [x_min, x_max, u_min, u_max]):
            self.set_bound_constraints(x_min, x_max, u_min, u_max)
        
        if cone_constraints is not None:
            # cone_constraints should be a dict with keys: Acu, qcu, cu, Acx, qcx, cx
            self.set_cone_constraints(**cone_constraints)

    def set_x0(self, x0):
        assert len(x0.shape) == 1
        assert len(x0) == self.nx

        self._solver.set_x0(x0)

    def set_bound_constraints(self, x_min, x_max, u_min, u_max):
        x_min = np.asfortranarray(self.expand_ndarray(x_min, self.nx, self.N, -self._infty))
        x_max = np.asfortranarray(self.expand_ndarray(x_max, self.nx, self.N, self._infty))
        u_min = np.asfortranarray(self.expand_ndarray(u_min, self.nu, self.N-1, -self._infty))
        u_max = np.asfortranarray(self.expand_ndarray(u_max, self.nu, self.N-1, self._infty))
        self._solver.set_bound_constraints(x_min, x_max, u_min, u_max)
        self.update_settings(en_state_bound=True, en_input_bound=True)
    
    def set_x_ref(self, x_ref):
        """Set state reference trajectory
        
        :param x_ref (np.ndarray): State reference trajectory, can be of size nx x 1 or nx x N.
                If of size nx x 1, expands to be of size nx x N
        """
        x_ref = np.array(self.expand_ndarray(x_ref, self.nx, self.N, 0), dtype=float, order="F")
        self._solver.set_x_ref(x_ref)
        
    def set_u_ref(self, u_ref):
        """Set input reference trajectory
        
        :param u_ref (np.ndarray): Input reference trajectory, can be of size nu x 1 or nu x N-1.
                If of size nu x 1, expands to be of size nu x N-1
        """
        u_ref = np.array(self.expand_ndarray(u_ref, self.nu, self.N-1, 0), dtype=float, order="F")
        self._solver.set_u_ref(u_ref)

    def solve(self):
        # self._solver.print_problem_data()
        self._solver.solve()

        solution = self._solver.solution

        if not solution.solved and self.verbose:
            print("Problem not solved after {} iterations".format(solution.iter))

        return {"states_all": solution.x.T, "controls_all": solution.u.T, "controls": solution.u[:,0]}
    
    def codegen(self, codegen_folder, verbose=False):
        codegen_folder_abs = os.path.abspath(codegen_folder)

        # Create codegen files (tiny_data.hpp/cpp and tiny_main.cpp)
        if not codegen_folder_abs.endswith(os.path.sep):
            codegen_folder_abs += os.path.sep
        status = self._solver.codegen(codegen_folder_abs, verbose)
        
        # Copy include/* (Eigen lib) and tinympc/(admm.hpp/cpp, api.hpp/cpp, constants.hpp, and types.hpp)
        # https://github.com/python/importlib_resources/issues/85
        try:
            handle = importlib.resources.files('tinympc.codegen').joinpath('codegen_src')
        except AttributeError:
            handle = importlib.resources.path('tinympc.codegen', 'codegen_src')
        with handle as codegen_src_path:
            shutil.copytree(codegen_src_path, codegen_folder_abs, dirs_exist_ok=True)
        
        # Copy pywrapper files (bindings.cpp, CMakeLists.txt, and setup.py)
        try:
            handle = importlib.resources.files('tinympc.codegen').joinpath('pywrapper')
        except AttributeError:
            handle = importlib.resources.path('tinympc.codegen', 'pywrapper')
        with handle as pywrapper_src_path:
            shutil.copy(pywrapper_src_path.joinpath('bindings.cpp'), codegen_folder_abs)
            shutil.copy(pywrapper_src_path.joinpath('CMakeLists.txt'), codegen_folder_abs)
            shutil.copy(pywrapper_src_path.joinpath('setup.py'), codegen_folder_abs)

        # Compile python module for generated code
        subprocess.check_call(
            [
                sys.executable,
                'setup.py',
                'build_ext',
                '--inplace',
            ],
            cwd=codegen_folder_abs,
        )

        assert status == 0, "Code generation failed"

    def codegen_with_sensitivity(self, codegen_folder, dK, dP, dC1, dC2, verbose=False):
        """Generate code with sensitivity matrices for adaptive rho.
        
        Args:
            codegen_folder (str): Output directory for generated code
            dK (np.ndarray): Derivative of feedback gain w.r.t. rho
            dP (np.ndarray): Derivative of value function w.r.t. rho
            dC1 (np.ndarray): Derivative of first cache matrix w.r.t. rho
            dC2 (np.ndarray): Derivative of second cache matrix w.r.t. rho
            verbose (bool): Whether to print debug information
        """
        codegen_folder_abs = os.path.abspath(codegen_folder)

        # Clean the output directory first
        if os.path.exists(codegen_folder_abs):
            shutil.rmtree(codegen_folder_abs)
        os.makedirs(codegen_folder_abs)

        # Set sensitivity matrices in the solver
        # Convert verbose bool to int for C++
        verbose_int = 1 if verbose else 0
        
        # Ensure matrices are in Fortran-contiguous order
        dK_f = np.asfortranarray(dK)
        dP_f = np.asfortranarray(dP)
        dC1_f = np.asfortranarray(dC1)
        dC2_f = np.asfortranarray(dC2)
        
        # Set sensitivity matrices
        self.set_sensitivity_matrices(dK_f, dP_f, dC1_f, dC2_f)
        
        # Generate code with sensitivity matrices
        status = self._solver.codegen_with_sensitivity(codegen_folder_abs, dK_f, dP_f, dC1_f, dC2_f, verbose_int)
        
        # Copy include/* and tinympc/ files
        try:
            handle = importlib.resources.files('tinympc.codegen').joinpath('codegen_src')
        except AttributeError:
            handle = importlib.resources.path('tinympc.codegen', 'codegen_src')
        with handle as codegen_src_path:
            shutil.copytree(codegen_src_path, codegen_folder_abs, dirs_exist_ok=True)
        
        # Copy pywrapper files
        try:
            handle = importlib.resources.files('tinympc.codegen').joinpath('pywrapper')
        except AttributeError:
            handle = importlib.resources.path('tinympc.codegen', 'pywrapper')
        with handle as pywrapper_src_path:
            shutil.copy(pywrapper_src_path.joinpath('bindings.cpp'), codegen_folder_abs)
            shutil.copy(pywrapper_src_path.joinpath('CMakeLists.txt'), codegen_folder_abs)
            shutil.copy(pywrapper_src_path.joinpath('setup.py'), codegen_folder_abs)

        # Compile python module
        subprocess.check_call(
            [
                sys.executable,
                'setup.py',
                'build_ext',
                '--inplace',
            ],
            cwd=codegen_folder_abs,
        )

        assert status == 0, "Code generation with sensitivity matrices failed"

    def set_sensitivity_matrices(self, dK, dP, dC1, dC2):
        """Set sensitivity matrices for adaptive rho behavior
        
        Args:
            dK (np.ndarray): Derivative of feedback gain w.r.t. rho
            dP (np.ndarray): Derivative of value function w.r.t. rho
            dC1 (np.ndarray): Derivative of first cache matrix w.r.t. rho
            dC2 (np.ndarray): Derivative of second cache matrix w.r.t. rho
        """
        # Validate input dimensions
        assert dK.shape == (self.nu, self.nx), f"dK should have shape ({self.nu}, {self.nx}), got {dK.shape}"
        assert dP.shape == (self.nx, self.nx), f"dP should have shape ({self.nx}, {self.nx}), got {dP.shape}"
        assert dC1.shape == (self.nu, self.nu), f"dC1 should have shape ({self.nu}, {self.nu}), got {dC1.shape}"
        assert dC2.shape == (self.nx, self.nx), f"dC2 should have shape ({self.nx}, {self.nx}), got {dC2.shape}"
        
        # Ensure matrices are in Fortran-contiguous order for C++ compatibility
        dK_f = np.asfortranarray(dK)
        dP_f = np.asfortranarray(dP)
        dC1_f = np.asfortranarray(dC1)
        dC2_f = np.asfortranarray(dC2)
        
        # Pass to the C++ solver with default values for rho and verbose
        self._solver.set_sensitivity_matrices(dK_f, dP_f, dC1_f, dC2_f)
        
        if self.verbose:
            print(f"Sensitivity matrices set with norms: dK={np.linalg.norm(dK):.6f}, dP={np.linalg.norm(dP):.6f}, dC1={np.linalg.norm(dC1):.6f}, dC2={np.linalg.norm(dC2):.6f}")

    def compute_cache_terms(self):
        """Compute cache terms for ADMM solver"""
        if self._solver is None:
            raise RuntimeError("Solver not initialized. Call setup() first.")
        
        # Add rho regularization
        Q_rho = self.Q + self.rho * np.eye(self.nx)
        R_rho = self.R + self.rho * np.eye(self.nu)
        
        # Initialize
        Kinf = np.zeros((self.nu, self.nx))
        Pinf = np.copy(self.Q)
        
        # Compute infinite horizon solution
        for _ in range(5000):
            Kinf_prev = np.copy(Kinf)
            Kinf = np.linalg.solve(
                R_rho + self.B.T @ Pinf @ self.B + 1e-8*np.eye(self.nu),
                self.B.T @ Pinf @ self.A
            )
            Pinf = Q_rho + self.A.T @ Pinf @ (self.A - self.B @ Kinf)
            
            if np.linalg.norm(Kinf - Kinf_prev) < 1e-10:
                break
        
        AmBKt = (self.A - self.B @ Kinf).T
        Quu_inv = np.linalg.inv(R_rho + self.B.T @ Pinf @ self.B)
        
        # Set cache terms in the C++ solver
        self._solver.set_cache_terms(
            np.asfortranarray(Kinf),
            np.asfortranarray(Pinf),
            np.asfortranarray(Quu_inv),
            np.asfortranarray(AmBKt),
            self.verbose
        )
        
        if self.verbose:
            print(f"Cache terms computed with norms: Kinf={np.linalg.norm(Kinf):.6f}, Pinf={np.linalg.norm(Pinf):.6f}")
            print(f"C1={np.linalg.norm(Quu_inv):.6f}, C2={np.linalg.norm(AmBKt):.6f}")
        
        return Kinf, Pinf, Quu_inv, AmBKt

    def compute_sensitivity_autograd(self):
        """Compute dK, dP, dC1, dC2 with respect to rho using Autograd's jacobian."""
        # Local imports to avoid hard dependency unless this method is called
        from autograd import jacobian
        import autograd.numpy as anp

        # Define the vectorized LQR solution mapping rho -> [K,P,C1,C2].flatten()
        def lqr_flat(rho):
            R_rho = anp.array(self.R) + rho * anp.eye(self.nu)
            Q_rho = anp.array(self.Q) + rho * anp.eye(self.nx)
            P = Q_rho
            for _ in range(5000):
                K = anp.linalg.solve(
                    R_rho + self.B.T @ P @ self.B + 1e-8 * anp.eye(self.nu),
                    self.B.T @ P @ self.A
                )
                P = Q_rho + self.A.T @ P @ (self.A - self.B @ K)
            K = anp.linalg.solve(
                R_rho + self.B.T @ P @ self.B + 1e-8 * anp.eye(self.nu),
                self.B.T @ P @ self.A
            )
            C1 = anp.linalg.inv(R_rho + self.B.T @ P @ self.B)
            C2 = (self.A - self.B @ K).T
            return anp.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])

        # Compute the Jacobian w.r.t. rho
        jac = jacobian(lqr_flat)
        vec = jac(self.rho)

        # Split derivative vector into four blocks
        m, n = self.nu, self.nx
        sizes = [m * n, n * n, m * m, n * n]
        parts = np.split(np.array(vec), np.cumsum(sizes)[:-1])
        dK = parts[0].reshape(m, n)
        dP = parts[1].reshape(n, n)
        dC1 = parts[2].reshape(m, m)
        dC2 = parts[3].reshape(n, n)
        return dK, dP, dC1, dC2

    def set_linear_constraints(self, Alin_x, blin_x, Alin_u, blin_u):
        """Set linear constraints: Alin_x * x <= blin_x, Alin_u * u <= blin_u"""
        # Convert to proper format and ensure column vectors
        Alin_x = np.asfortranarray(Alin_x, dtype=np.float64)
        Alin_u = np.asfortranarray(Alin_u, dtype=np.float64)
        blin_x = np.asfortranarray(blin_x, dtype=np.float64).reshape(-1, 1)
        blin_u = np.asfortranarray(blin_u, dtype=np.float64).reshape(-1, 1)
        
        self._solver.set_linear_constraints(Alin_x, blin_x, Alin_u, blin_u)
        self.update_settings(en_state_linear=Alin_x.size>0 and blin_x.size>0,
                             en_input_linear=Alin_u.size>0 and blin_u.size>0)

    def set_cone_constraints(self, Acu, qcu, cu, Acx, qcx, cx):
        """Set second-order cone constraints (inputs first, then states)"""
        # Convert to proper types
        Acu = np.ascontiguousarray(Acu, dtype=np.int32)
        qcu = np.ascontiguousarray(qcu, dtype=np.int32)
        cu = np.asfortranarray(cu, dtype=np.float64)
        Acx = np.ascontiguousarray(Acx, dtype=np.int32)
        qcx = np.ascontiguousarray(qcx, dtype=np.int32)
        cx = np.asfortranarray(cx, dtype=np.float64)
        
        self._solver.set_cone_constraints(Acu, qcu, cu, Acx, qcx, cx)
        self.update_settings(en_input_soc=cu.size>0, en_state_soc=cx.size>0)

    def set_equality_constraints(self, Aeq_x, beq_x, Aeq_u=None, beq_u=None):
        """Set equality constraints: Aeq_x * x == beq_x, Aeq_u * u == beq_u"""
        # Create dual inequalities: Ax <= b and -Ax <= -b
        if Aeq_u is None:
            Aeq_u = np.zeros((0, self.nu))
            beq_u = np.zeros(0)
        
        Alin_x = np.vstack([Aeq_x, -Aeq_x])
        blin_x = np.concatenate([beq_x, -beq_x])
        Alin_u = np.vstack([Aeq_u, -Aeq_u])
        blin_u = np.concatenate([beq_u, -beq_u])
        
        self.set_linear_constraints(Alin_x, blin_x, Alin_u, blin_u)
