import ctypes
import tinympc.tinympc as tinympc

if __name__ == '__main__':
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
    B = [
        7.468368562730335e-5,
        0.014936765390161838,
        3.79763323185387e-5,
        0.007595596218554721,
    ]
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
    tinympc_dir = "/home/khai/SSD/Code/tinympc-python/src/TinyMPC"
    output_dir = "/home/khai/SSD/Code/tinympc-python/src/TinyMPC"

    prob = tinympc.TinyMPC(tinympc_dir)
    prob.setup(n, m, N, A, B, Q, R, x_min, x_max, u_min, u_max, rho, abs_pri_tol, abs_dual_tol, max_iter, check_termination)
    prob.tiny_codegen(tinympc_dir, output_dir)
