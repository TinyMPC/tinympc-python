#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include <tinympc/tiny_api.hpp>
#include <tinympc/tiny_data.hpp>

py::dict solve_() {
    py::gil_scoped_release release;
    int status = tiny_solve(&tiny_solver);
    py::gil_scoped_acquire acquire;

    py::dict results("states_all"_a=tiny_solver.solution->x.transpose(),
                     "controls_all"_a=tiny_solver.solution->u.transpose(),
                     "controls"_a=tiny_solver.solution->u.col(0));
    return results;
}

void set_x0(Eigen::Ref<tinyMatrix> x0) {
    if (x0.rows() == tiny_solver.work->nx && x0.cols() == 1) {
        tiny_set_x0(&tiny_solver, x0.replicate(1, tiny_solver.work->N));
    } else {
        throw std::invalid_argument("Check the size of x0. Expected a vector with length equal to the number of states in the system.");
    }
}

void set_x_ref(Eigen::Ref<tinyMatrix> x_ref) {
    if (x_ref.rows() == tiny_solver.work->nx && x_ref.cols() == 1) {
        tiny_set_x_ref(&tiny_solver, x_ref.replicate(1, tiny_solver.work->N));
    } else if (x_ref.cols() == 1) {
        throw std::invalid_argument("Check the size of x_ref. Expected a vector with length equal to the number of states in the system.");
    } else if (x_ref.rows() == tiny_solver.work->nx && x_ref.cols() == tiny_solver.work->N) {
        tiny_set_x_ref(&tiny_solver, x_ref);
    } else {
        throw std::invalid_argument("Check the size of x_ref. Expected a matrix with shape num states x horizon length.");
    }
}

void set_u_ref(Eigen::Ref<tinyMatrix> u_ref) {
    if (u_ref.rows() == tiny_solver.work->nu && u_ref.cols() == 1) {
        tiny_set_u_ref(&tiny_solver, u_ref.replicate(1, tiny_solver.work->N-1));
    } else if (u_ref.cols() == 1) {
        throw std::invalid_argument("Check the size of u_ref. Expected a vector with length equal to the number of inputs in the system.");
    } else if (u_ref.rows() == tiny_solver.work->nu && u_ref.cols() == tiny_solver.work->N-1) {
        tiny_set_u_ref(&tiny_solver, u_ref);
    } else {
        throw std::invalid_argument("Check the size of u_ref. Expected a matrix with shape num inputs x horizon length - 1.");
    }
}

PYBIND11_MODULE(tinympcgen, m) {
    m.def("solve", &solve_);
    m.def("set_x0", &set_x0);
    m.def("set_x_ref", &set_x_ref);
    m.def("set_u_ref", &set_u_ref);
}
