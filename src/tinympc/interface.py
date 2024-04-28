import os
import sys
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
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        self._tinytype = np.float32
        
        self._solver = None

    # # Options
    # rho = 1.0
    # abs_pri_tol = 1e-3
    # abs_dual_tol = 1e-3
    # max_iter = 1000
    # check_termination = 1

    # _gen_wrapper = 1  # always generate wrapper

    # lib = None



    # Setup the problem data and solver options
    def setup(self, A, B, Q, R, N,
        x_min=None, x_max=None, u_min=None, u_max=None,
        rho=1.0, abs_pri_tol=1e-3, abs_dual_tol=1e-3,
        max_iter=1000, check_termination=1,
    ):
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.N = N
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.rho = rho
        self.abs_pri_tol = abs_pri_tol
        self.abs_dual_tol = abs_dual_tol
        self.max_iter = max_iter
        self.check_termination = check_termination
        return True


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