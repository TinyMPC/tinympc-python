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
    def setup(self, A, B, Q, R, N, rho=1.0,
        x_min=None, x_max=None, u_min=None, u_max=None, verbose=False, **settings):
        """Instantiate necessary algorithm variables and parameters
        
        :param A (np.ndarray): State transition matrix of the linear system, size nx x nx
        :param B (np.ndarray): Input matrix of the linear system, size nx x nu
        :param Q (np.ndarray): Stage cost for state, must be diagonal and positive semi-definite, size nx x nx
        :param R (np.ndarray): Stage cost for input, must be diagonal and positive definite, size nu x nu
        :param rho (int, optional): Penalty term used in ADMM, default 1
        :param x_min (list[float] or None, optional): Lower bound state constraints of the same length as nx, default None
        :param x_max (list[float] or None, optional): Upper bound state constraints of the same length as nx, default None
        :param u_min (list[float] or None, optional): Lower bound input constraints of the same length as nu, default None
        :param u_max (list[float] or None, optional): Upper bound input constraints of the same length as nu, default None
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

        assert N > 1
        self.N = N


        self.x_min = np.array(self.expand_ndarray(x_min, self.nx, self.N, -self._infty), dtype=float, order="F")
        self.x_max = np.array(self.expand_ndarray(x_max, self.nx, self.N, self._infty), dtype=float, order="F")
        self.u_min = np.array(self.expand_ndarray(u_min, self.nu, self.N-1, -self._infty), dtype=float, order="F")
        self.u_max = np.array(self.expand_ndarray(u_max, self.nu, self.N-1, self._infty), dtype=float, order="F")

        assert len(self.x_min.shape) == 2
        assert len(self.x_max.shape) == 2
        assert len(self.u_min.shape) == 2
        assert len(self.u_max.shape) == 2
        assert self.x_min.shape[0] == self.nx
        assert self.x_max.shape[0] == self.nx
        assert self.u_min.shape[0] == self.nu
        assert self.u_max.shape[0] == self.nu
        assert self.x_min.shape[1] == self.N
        assert self.x_max.shape[1] == self.N
        assert self.u_min.shape[1] == self.N-1
        assert self.u_max.shape[1] == self.N-1

        self.verbose = verbose


        self.settings = self.ext.TinySettings() # instantiate local settings (settings known only to the python interface)
        self.ext.tiny_set_default_settings(self.settings) # set local settings to default defined by C++ implementation
        self.update_settings(**settings) # change local settings based on arguments available to the interface

        self._solver = self.ext.TinySolver(self.A, self.B, self.Q, self.R, self.rho,
                                           self.nx, self.nu, self.N,
                                           self.x_min, self.x_max, self.u_min, self.u_max,
                                           self.settings, self.verbose
        )

    def set_x0(self, x0):
        assert len(x0.shape) == 1
        assert len(x0) == self.nx

        self._solver.set_x0(x0)
    
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
