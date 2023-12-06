import ctypes
import pathlib


if __name__ == "__main__":
    # Import the library
    # lib_path = pathlib.Path("../TinyMPC/binaries/libtinympc.so")
    lib_path = "../TinyMPC/binaries/libtinympc.so"
    tinympc = ctypes.CDLL("../TinyMPC/binaries/libtinympc.so")

    n = 4
    m = 1
    N = 10
    # tinytype Adyn_data[n * n] = {1.0, 0.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 2.2330083403300767e-5, 0.004466210576510177, 1.0002605176397052, 0.05210579005928538, 7.443037974683548e-8, 2.2330083403300767e-5, 0.01000086835443038, 1.0002605176397052};
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
    # tinytype Bdyn_data[n * m] = {7.468368562730335e-5, 0.014936765390161838, 3.79763323185387e-5, 0.007595596218554721};
    B = [
        7.468368562730335e-5,
        0.014936765390161838,
        3.79763323185387e-5,
        0.007595596218554721,
    ]
    # tinytype Q_data[n] = {10, 1, 10, 1};
    # tinytype R_data[m] = {1};
    # tinytype rho_value = 0.1;
    Q = [10, 1, 10, 1]
    R = [1]
    rho = 0.1

    # tinytype abs_pri_tol = 1e-3;
    # tinytype abs_dual_tol = 1e-3;
    # int max_iter = 100;
    # int check_termination = 1;
    # int gen_wrapper = 1;
    abs_pri_tol = 1e-3
    abs_dual_tol = 1e-3
    max_iter = 100
    check_termination = 1
    gen_wrapper = 1

    # char tinympc_dir[255] = "/home/sam/Git/tinympc/TinyMPC/";
    # char output_dir[255] = "/generated_code";


    # // Set up constraints (for-loop in main)
    # int i = 0;
    # for (i = 0; i < n * N; i++)
    # {
    #     x_min_data[i] = -5;
    #     x_max_data[i] = 5;
    # }
    # for (i = 0; i < m * (N - 1); i++)
    # {
    #     u_min_data[i] = -5;
    #     u_max_data[i] = 5;
    # }
    # create a 1D array of size (n, N)
    x_min = [-5] * n * N
    x_max = [5] * n * N
    u_min = [-5] * m * (N - 1)
    u_max = [5] * m * (N - 1)

    # print(A)
    # print(B)
    # print(Q)
    # print(R)
    # print(rho)
    # print(abs_pri_tol)
    # print(abs_dual_tol)
    # print(max_iter)
    # print(check_termination)
    # print(gen_wrapper)
    # print(tinympc_dir)
    # print(output_dir)
    # print(x_min)
    # print(x_max)
    # print(u_min)
    # print(u_max)

    #################################################################
    # Call the function with the arguments correctly casted to ctypes
    #################################################################

    # Specify return type and argument types
    # dll.scale.restype = None  # Function returns void
    # dll.scale.argstype = [
    #     ctypes.POINTER(ctypes.c_double),
    #     ctypes.POINTER(ctypes.c_double),
    #     ctypes.c_uint32,
    #     ctypes.c_double,
    # ]

    tinympc.tiny_codegen.restype = ctypes.c_int  # Function returns void
    tinympc.tiny_codegen.argstype = [
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

    # # Create arbitrary input and allocate memory for the output
    # input = [1, 2, 3, 4, 5]
    # scaling_factor = 10
    # output = (ctypes.c_double * len(input))()

    # # Call the function with the arguments correctly casted to ctypes
    # dll.scale(
    #     (ctypes.c_double * len(input))(*input),
    #     output,
    #     len(input),
    #     ctypes.c_double(scaling_factor),
    # )

    tinympc.tiny_codegen(
        ctypes.c_int(n),
        ctypes.c_int(m),
        ctypes.c_int(N),
        (ctypes.c_double * len(A))(*A),
        (ctypes.c_double * len(B))(*B),
        (ctypes.c_double * len(Q))(*Q),
        (ctypes.c_double * len(R))(*R),
        (ctypes.c_double * len(x_min))(*x_min),
        (ctypes.c_double * len(x_max))(*x_max),
        (ctypes.c_double * len(u_min))(*u_min),
        (ctypes.c_double * len(u_max))(*u_max),
        ctypes.c_double(rho),
        ctypes.c_double(abs_pri_tol),
        ctypes.c_double(abs_dual_tol),
        ctypes.c_int(max_iter),
        ctypes.c_int(check_termination),
        ctypes.c_int(gen_wrapper),
        ctypes.c_char_p(tinympc_dir.encode("utf-8")),
        ctypes.c_char_p(output_dir.encode("utf-8")),
    )

    # # Cast the output back to a Python list and print the results
    # output = list(output)
    # print(output)
