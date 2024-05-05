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
        PyTinySolver(const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, // A, B
                     const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, // Q, R
                     const float, const int, const int, const int,               // rho, nx, nu, N
                     const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, // x_min, x_max
                     const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, // u_min, u_max
                     const TinySettings*); // settings
        ~PyTinySolver();
    private:
        TinySolver *_solver;
};

PyTinySolver::PyTinySolver(
        const Eigen::Ref<tinyMatrix> A,
        const Eigen::Ref<tinyMatrix> B,
        const Eigen::Ref<tinyMatrix> Q,
        const Eigen::Ref<tinyMatrix> R,
        const float rho,
        const int nx,
        const int nu,
        const int N,
        const Eigen::Ref<tinyMatrix> x_min,
        const Eigen::Ref<tinyMatrix> x_max,
        const Eigen::Ref<tinyMatrix> u_min,
        const Eigen::Ref<tinyMatrix> u_max,
        const TinySettings *settings) {
    this->_solver = new TinySolver();

    int status = tiny_precompute_and_set_cache(this->_solver->cache, A, B, Q, R, nx, nu, rho);

    if (status) {
        std::string message = "Error when precomputing cache";
        throw py::value_error(message);
    }
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
        return new TinySettings();
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
    .def(py::init<const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const float, const int, const int, const int, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const TinySettings*>(),
            "A"_a.noconvert(), "B"_a.noconvert(), "Q"_a.noconvert(), "R"_a.noconvert(), "rho"_a, "nx"_a, "nu"_a, "N"_a, "x_min"_a.noconvert(), "x_max"_a.noconvert(), "u_min"_a.noconvert(), "u_max"_a.noconvert(), "settings"_a);

}
