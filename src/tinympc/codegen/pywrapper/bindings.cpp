#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "tiny_api.h"
#include "tiny_data.hpp"

int solve() {
    py::gil_scoped_release release;
    int status = tiny_solve(&tiny_solver);
    py::gil_scoped_acquire acquire;

    // if (status != 0) throw std::runtime_error("Solve failed");

    // OSQPInt m;
    // OSQPInt n;
    // osqp_get_dimensions(&tiny_solver, &m, &n);

    // auto x = py::array_t<OSQPFloat>({n}, {sizeof(OSQPFloat)}, (&tiny_solver)->solution->x);
    // auto y = py::array_t<OSQPFloat>({m}, {sizeof(OSQPFloat)}, (&tiny_solver)->solution->y);

    // py::tuple results = py::make_tuple(x, y, status, (&tiny_solver)->info->iter, (&tiny_solver)->info->run_time);
    // return results;
    return status;
}


PYBIND11_MODULE(tiny_codegen_ext, m) {
    m.def("solve", &solve);
    m.def("set_x0", &set_x0);
    m.def("set_x_ref", &set_x_ref);
    m.def("set_u_ref", &set_u_ref);
}
