import ctypes
import subprocess
import os
import sys

class TinyMPC:
    # Problem data
    n = 0
    m = 0
    N = 0
    A = []
    B = []
    Q = []
    R = []
    x_min = []
    x_max = []
    u_min = []
    u_max = []

    # Options
    rho = 1.0
    abs_pri_tol = 1e-3
    abs_dual_tol = 1e-3
    max_iter = 1000
    check_termination = 1

    _gen_wrapper = 1  # always generate wrapper

    lib = None


    def __init__(self):
        pass


    # Load the any shared library
    def load_lib(self, lib_dir):
        self.lib = ctypes.CDLL(lib_dir)
        # self.lib.set_x.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_x0.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_xref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_umin.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_umax.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_xmin.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.set_xmax.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.reset_dual_variables.argtypes = [ctypes.c_int]
        # self.lib.call_tiny_solve.argtypes = [ctypes.c_int]
        # self.lib.get_x.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        # self.lib.get_u.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]


    # Setup the problem data and solver options
    def setup(
        self,
        n,
        m,
        N,
        A,
        B,
        Q,
        R,
        x_min,
        x_max,
        u_min,
        u_max,
        rho=1.0,
        abs_pri_tol=1e-3,
        abs_dual_tol=1e-3,
        max_iter=1000,
        check_termination=1,
    ):
        self.n = n
        self.m = m
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
        _A = (ctypes.c_double * (self.n * self.n))(*self.A)
        _B = (ctypes.c_double * (self.n * self.m))(*self.B)
        _Q = (ctypes.c_double * (self.n * self.n))(*self.Q)
        _R = (ctypes.c_double * (self.m * self.m))(*self.R)
        _x_min = (ctypes.c_double * (self.n * self.N))(*self.x_min)
        _x_max = (ctypes.c_double * (self.n * self.N))(*self.x_max)
        _u_min = (ctypes.c_double * (self.m * (self.N - 1)))(*self.u_min)
        _u_max = (ctypes.c_double * (self.m * (self.N - 1)))(*self.u_max)

        self.lib.tiny_codegen(
            self.n,
            self.m,
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


    # All the functions below are wrappers for the generated code, using float instead of double
    def set_x0(self, x0, verbose=1):
        _x0 = (ctypes.c_float * self.n)(*x0)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_x0(_x0, _verbose)
        return True

    def solve(self, verbose=1):
        _verbose = ctypes.c_int(verbose)
        self.lib.call_tiny_solve(_verbose)
        return True
    
    def get_u(self, u, verbose=1):
        _verbose = ctypes.c_int(verbose)
        _u = (ctypes.c_float * (self.m * (self.N - 1)))()
        self.lib.get_u(_u, _verbose)
        for i in range(self.m * (self.N - 1)):
            u[i] = _u[i]
        return True
    
    def get_x(self, x, verbose=1):
        _verbose = ctypes.c_int(verbose)
        _x = (ctypes.c_float * (self.n * self.N))()
        self.lib.get_x(_x, _verbose)
        for i in range(self.n * self.N):
            x[i] = _x[i]
        return True

    def set_xref(self, xref, verbose=1):
        _xref = (ctypes.c_float * (self.n * self.N))(*xref)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xref(_xref, _verbose)
        return True
    
    def set_umin(self, umin, verbose=1):
        _umin = (ctypes.c_float * (self.m * (self.N - 1)))(*umin)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_umin(_umin, _verbose)
        return True
    
    def set_umax(self, umax, verbose=1):
        _umax = (ctypes.c_float * (self.m * (self.N - 1)))(*umax)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_umax(_umax, _verbose)
        return True
    
    def set_xmin(self, xmin, verbose=1):
        _xmin = (ctypes.c_float * (self.n * self.N))(*xmin)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xmin(_xmin, _verbose)
        return True
    
    def set_xmax(self, xmax, verbose=1):
        _xmax = (ctypes.c_float * (self.n * self.N))(*xmax)
        _verbose = ctypes.c_int(verbose)
        self.lib.set_xmax(_xmax, _verbose)
        return True
    
    def reset_dual_variables(self, verbose=1):
        _verbose = ctypes.c_int(verbose)
        self.lib.reset_dual_variables(_verbose)
        return True