import os
import sys
import numpy as np
import importlib

class TinyMPC:
    def __init__(self):
        self.nx = 0 # number of states
        self.nu = 0 # number of control inputs
        self.N = 0 # number of knotpoints in the horizon
        self.A = [] # state transition matrix
        self.B = [] # control matrix
        self.Q = [] # state cost matrix (diagonal)
        self.R = [] # input cost matrix (digaonal)
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        # Import tinympc pybind extension
        self.ext = importlib.import_module("tinympc.ext_tinympc")

        self._tinytype = np.float32
        self._infty = 1e17
        
        self._solver = None
        self._settings = None
    
    
    def update_settings(self, **kwargs):
        assert self._settings is not None
        
        if 'abs_pri_tol' in kwargs:
            self._settings.abs_pri_tol = kwargs.pop('abs_pri_tol')
        if 'abs_dua_tol' in kwargs:
            self._settings.abs_dua_tol = kwargs.pop('abs_dua_tol')
        if 'max_iter' in kwargs:
            self._settings.max_iter = kwargs.pop('max_iter')
        if 'check_termination' in kwargs:
            self._settings.check_termination = kwargs.pop('check_termination')
        if 'en_state_bound' in kwargs:
            self._settings.en_state_bound = kwargs.pop('en_state_bound')
        if 'en_input_bound' in kwargs:
            self._settings.en_input_bound = kwargs.pop('en_input_bound')

        if self._solver is not None:
            self._solver.update_settings(self._settings)        
        
        

    # Setup the problem data and solver options
    def setup(self, A, B, Q, R, N,
        x_min=None, x_max=None, u_min=None, u_max=None, **settings):
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]
        self.A = np.array(A, order="F")
        self.B = np.array(B, order="F")
        self.Q = np.array(Q, order="F")
        self.R = np.array(R, order="F")
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.N = N
        self.x_min = x_min if x_min is not None else -self._infty
        self.x_max = x_max if x_max is not None else self._infty
        self.u_min = u_min if u_min is not None else -self._infty
        self.u_max = u_max if u_max is not None else self._infty
        
        self._settings = self.ext.TinySettings()
        self.ext.tiny_set_default_settings(self._settings)
        self.update_settings(**settings)
        
        self._solver = self.ext.TinySolver(self.A, self.B, self.Q, self.R,
                                           self.nx, self.nu, self.N,
                                           self.x_min, self.x_max, self.u_min, self.u_max,
                                           self._settings
        )
        
    # If this function fails, you are already using the generated code 
    # This uses double instead of float
    def tiny_codegen(self, tinympc_dir, output_dir):
        self.lib.tiny_codegen.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
        self.lib.tiny_codegen.restype = ctypes.c_int
        _A = (ctypes.c_double * (self.nx * self.nx))(*self.A)
        _B = (ctypes.c_double * (self.nx * self.nu))(*self.B)
        _Q = (ctypes.c_double * (self.nx * self.nx))(*self.Q)
        _R = (ctypes.c_double * (self.nu * self.nu))(*self.R)
        _x_min = (ctypes.c_double * (self.nx * self.N))(*self.x_min)
        _x_max = (ctypes.c_double * (self.nx * self.N))(*self.x_max)
        _u_min = (ctypes.c_double * (self.nu * (self.N - 1)))(*self.u_min)
        _u_max = (ctypes.c_double * (self.nu * (self.N - 1)))(*self.u_max)

        # TODO: update tiny_codegen to return an error code instead of printing a bunch of stuff
        self.lib.tiny_codegen(
            self.nx,
            self.nu,
            self.N,
            _A,
            _B,
            _Q,
            _R,
            _x_min,
            _x_max,
            _u_min,
            _u_max,
            self.rho,
            self.abs_pri_tol,
            self.abs_dual_tol,
            self.max_iter,
            self.check_termination,
            self._gen_wrapper,
            tinympc_dir.encode("utf-8"),
            output_dir.encode("utf-8"),
        )
        return True


    # Compile the generated code
    def compile_lib(self, src_dir):
        # Specify the path to the build directory (where CMake will generate build files)
        build_directory = src_dir + "/build"

        # Make sure the build directory exists
        os.makedirs(build_directory, exist_ok=True)

        # Run CMake configuration
        cmake_configure_cmd = ["cmake", src_dir]
        subprocess.run(cmake_configure_cmd, cwd=build_directory)

        # Run the build process (e.g., make)
        cmake_build_cmd = ''
        if sys.platform == 'win32' or 'cygwin': # windows system
            cmake_build_cmd = ["cmake", "--build", "."]
        elif sys.platform == 'darwin': # macOS
            cmake_build_cmd = ["make"]
        elif sys.platform == 'linux': # linux
            cmake_build_cmd = ["make"]
        else:
            # error
            error("TinyMPC does not support your operating system: {}".format(sys.platform))
        
        subprocess.run(cmake_build_cmd, cwd=build_directory)
        
        return True

    # TODO: make verbose false by default
    
    # All the functions below are wrappers for the generated code, using float instead of double
    def set_x0(self, x0, verbose=1):
        _x0 = (ctypes.c_float * self.nx)(*x0)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_x0(_x0, _verbose)
        return True

    def solve(self, verbose=1):
        _verbose = ctypes.c_int(verbose)
        self.lib.call_tiny_solve(_verbose)
        return True
    
    def get_u(self, u, verbose=1):
        _verbose = ctypes.c_int(verbose)
        _u = (ctypes.c_float * (self.nu * (self.N - 1)))()
        self.lib.get_u(_u, _verbose)
        for i in range(self.nu * (self.N - 1)):
            u[i] = _u[i]
        return True
    
    def get_x(self, x, verbose=1):
        _verbose = ctypes.c_int(verbose)
        _x = (ctypes.c_float * (self.nx * self.N))()
        self.lib.get_x(_x, _verbose)
        for i in range(self.nx * self.N):
            x[i] = _x[i]
        return True

    def set_xref(self, xref, verbose=1):
        _xref = (ctypes.c_float * (self.nx * self.N))(*xref)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xref(_xref, _verbose)
        return True
    
    def set_umin(self, umin, verbose=1):
        _umin = (ctypes.c_float * (self.nu * (self.N - 1)))(*umin)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_umin(_umin, _verbose)
        return True
    
    def set_umax(self, umax, verbose=1):
        _umax = (ctypes.c_float * (self.nu * (self.N - 1)))(*umax)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_umax(_umax, _verbose)
        return True
    
    def set_xmin(self, xmin, verbose=1):
        _xmin = (ctypes.c_float * (self.nx * self.N))(*xmin)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xmin(_xmin, _verbose)
        return True
    
    def set_xmax(self, xmax, verbose=1):
        _xmax = (ctypes.c_float * (self.nx * self.N))(*xmax)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xmax(_xmax, _verbose)
        return True
    
    def reset_dual_variables(self, verbose=1):
        _verbose = ctypes.c_int(verbose)
        self.lib.reset_dual_variables(_verbose)
        return True