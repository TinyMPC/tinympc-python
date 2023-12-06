import ctypes
import pathlib


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

    def __init__(self, lib_dir):
        self.lib = ctypes.CDLL(lib_dir + "/binaries/libtinympc.so")
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

    def tiny_codegen(self, tinympc_dir, output_dir):
        A = (ctypes.c_double * (self.n * self.n))(*self.A)
        B = (ctypes.c_double * (self.n * self.m))(*self.B)
        Q = (ctypes.c_double * (self.n * self.n))(*self.Q)
        R = (ctypes.c_double * (self.m * self.m))(*self.R)
        x_min = (ctypes.c_double * (self.n * self.N))(*self.x_min)
        x_max = (ctypes.c_double * (self.n * self.N))(*self.x_max)
        u_min = (ctypes.c_double * (self.m * (self.N - 1)))(*self.u_min)
        u_max = (ctypes.c_double * (self.m * (self.N - 1)))(*self.u_max)

        self.lib.tiny_codegen(
            self.n,
            self.m,
            self.N,
            A,
            B,
            Q,
            R,
            x_min,
            x_max,
            u_min,
            u_max,
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
