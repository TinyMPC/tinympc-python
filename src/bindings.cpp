#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "tinympc/admm.hpp"
#include "tinympc/tiny_api.hpp"

class PyTinySolver {
    public:
        PyTinySolver(Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // A, B
                     Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // Q, R
                     float, int, int, int,               // rho, nx, nu, N
                     Eigen::Ref<tinyVector>, Eigen::Ref<tinyVector>, // x_min, x_max
                     Eigen::Ref<tinyVector>, Eigen::Ref<tinyVector>, // u_min, u_max
                     TinySettings*); // settings
        int solve();
        TinySolution* get_solution();
    private:
        TinySolver *_solver;
};

PyTinySolver::PyTinySolver(
        Eigen::Ref<tinyMatrix> A,
        Eigen::Ref<tinyMatrix> B,
        Eigen::Ref<tinyMatrix> Q,
        Eigen::Ref<tinyMatrix> R,
        float rho,
        int nx,
        int nu,
        int N,
        Eigen::Ref<tinyVector> x_min,
        Eigen::Ref<tinyVector> x_max,
        Eigen::Ref<tinyVector> u_min,
        Eigen::Ref<tinyVector> u_max,
        TinySettings *settings) {
    TinySolution *solution = new TinySolution();
    TinyCache *cache = new TinyCache();
    TinyWorkspace *work = new TinyWorkspace();
    int status = tiny_setup(cache, work, solution, A, B, Q, R, rho, nx, nu, N, x_min, x_max, u_min, u_max, settings);
    if (status) {
        std::string message = "Error during setup";
        throw py::value_error(message); 
    }
    this->_solver = new TinySolver();
    this->_solver->solution = solution;
    this->_solver->settings = settings;
    this->_solver->cache = cache;
    this->_solver->work = work;
}

int PyTinySolver::solve() {
    py::gil_scoped_release release;
    int status = tiny_solve(this->_solver);
    py::gil_scoped_acquire acquire;
    return status;
}

TinySolution* PyTinySolver::get_solution() {
    return this->_solver->solution;
}

PYBIND11_MODULE(ext_tinympc, m) {
// // PYBIND11_MODULE(@TINYMPC_EXT_MODULE_NAME@, m) {

    // Cache
    py::class_<TinyCache>(m, "TinyCache", py::module_local())
    .def(py::init([]() {
        return new TinyCache();
    }))
    .def_readwrite("rho", &TinyCache::rho)
    .def_readwrite("Kinf", &TinyCache::Kinf)
    .def_readwrite("Pinf", &TinyCache::Pinf)
    .def_readwrite("Quu_inv", &TinyCache::Quu_inv)
    .def_readwrite("AmBKt", &TinyCache::AmBKt);

    // Settings
    py::class_<TinySettings>(m, "TinySettings", py::module_local())
    .def(py::init([]() {
        return new TinySettings(); // only necessary if creating a TinySettings object inside python interface
    }))
    .def_readwrite("abs_pri_tol", &TinySettings::abs_pri_tol)
    .def_readwrite("abs_dua_tol", &TinySettings::abs_dua_tol)
    .def_readwrite("max_iter", &TinySettings::max_iter)
    .def_readwrite("check_termination", &TinySettings::check_termination)
    .def_readwrite("en_state_bound", &TinySettings::en_state_bound)
    .def_readwrite("en_input_bound", &TinySettings::en_input_bound);

    m.def("tiny_set_default_settings", &tiny_set_default_settings);

    // Solver
    py::class_<PyTinySolver>(m, "TinySolver", py::module_local())
    .def(py::init<Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, float, int, int, int, Eigen::Ref<tinyVector>, Eigen::Ref<tinyVector>, Eigen::Ref<tinyVector>, Eigen::Ref<tinyVector>, TinySettings*>(),
            "A"_a.noconvert(), "B"_a.noconvert(), "Q"_a.noconvert(), "R"_a.noconvert(), "rho"_a, "nx"_a, "nu"_a, "N"_a, "x_min"_a.noconvert(), "x_max"_a.noconvert(), "u_min"_a.noconvert(), "u_max"_a.noconvert(), "settings"_a)
    .def_property_readonly("solution", &PyTinySolver::get_solution)
    .def("solve", &PyTinySolver::solve);

}
