#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "tinympc/admm.hpp"


class PyTinySolver {
    public:
        PyTinySolver(const CSC&, const py::array_t<OSQPFloat>, const CSC&, const py::array_t<OSQPFloat>, const py::array_t<OSQPFloat>, OSQPInt, OSQPInt, const OSQPSettings*);
        ~PyTinySolver();

        OSQPSettings* get_settings();
        PyOSQPSolution& get_solution();
        OSQPInfo* get_info();

        OSQPInt update_settings(const OSQPSettings&);
        OSQPInt update_rho(OSQPFloat);
        OSQPInt warm_start(py::object, py::object);
        OSQPInt solve();

        OSQPInt codegen(const char*, const char*, OSQPCodegenDefines&);
    private:
        OSQPInt m;
        OSQPInt n;
        const CSC& _P;
        py::array_t<OSQPFloat> _q;
        py::array_t<OSQPFloat> _l;
        const CSC& _A;
        py::array_t<OSQPFloat> _u;
        OSQPSolver *_solver;
};



.def(py::init<const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const int, const int, const int, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>>(),
        "A"_a.noconvert(), "B"_a.noconvert(), "Q"_a.noconvert(), "R"_a.noconvert(), "nx"_a, "nu"_a, "N"_a, "x_min"_a.noconvert(), "x_max"_a.noconvert(), "u_min"_a.noconvert(), "u_max"_a.noconvert())
        
PyTinySolver::PyTinySolver(
        const Eigen::Ref<tinyMatrix> P,
        const py::array_t<OSQPFloat> q,
        const CSC& A,
        const py::array_t<OSQPFloat> l,
        const py::array_t<OSQPFloat> u,
        OSQPInt m,
        OSQPInt n,
        const OSQPSettings *settings
): m(m), n(n), _P(P), _A(A) {
    this->_solver = new OSQPSolver();
    this->_q = q;
    this->_l = l;
    this->_u = u;

    OSQPInt status = osqp_setup(&this->_solver, &this->_P.getcsc(), (OSQPFloat *)this->_q.data(), &this->_A.getcsc(), (OSQPFloat *)this->_l.data(), (OSQPFloat *)this->_u.data(), m, n, settings);
    if (status) {
        std::string message = "Setup Error (Error Code " + std::to_string(status) + ")";
        throw py::value_error(message);
    }
}

PyTinySolver::~PyTinySolver() {
    osqp_cleanup(this->_solver);
}

OSQPSettings* PyTinySolver::get_settings() {
    return this->_solver->settings;
}

PyOSQPSolution& PyTinySolver::get_solution() {
    PyOSQPSolution* solution = new PyOSQPSolution(*this->_solver->solution, this->m, this->n);
    return *solution;
}

OSQPInfo* PyTinySolver::get_info() {
    return this->_solver->info;
}

OSQPInt PyTinySolver::warm_start(py::object x, py::object y) {
    OSQPFloat* _x;
    OSQPFloat* _y;

    if (x.is_none()) {
        _x = NULL;
    } else {
        _x = (OSQPFloat *)py::array_t<OSQPFloat>(x).data();
    }
    if (y.is_none()) {
        _y = NULL;
    } else {
        _y = (OSQPFloat *)py::array_t<OSQPFloat>(y).data();
    }

    return osqp_warm_start(this->_solver, _x, _y);
}

OSQPInt PyTinySolver::solve() {
    py::gil_scoped_release release;
    OSQPInt results = osqp_solve(this->_solver);
    py::gil_scoped_acquire acquire;
    return results;
}

OSQPInt PyTinySolver::update_settings(const OSQPSettings& new_settings) {
    return osqp_update_settings(this->_solver, &new_settings);
}

OSQPInt PyTinySolver::update_rho(OSQPFloat rho_new) {
    return osqp_update_rho(this->_solver, rho_new);
}

OSQPInt PyTinySolver::update_data_vec(py::object q, py::object l, py::object u) {
    OSQPFloat* _q;
    OSQPFloat* _l;
    OSQPFloat* _u;

    if (q.is_none()) {
        _q = NULL;
    } else {
        _q = (OSQPFloat *)py::array_t<OSQPFloat>(q).data();
    }
    if (l.is_none()) {
        _l = NULL;
    } else {
        _l = (OSQPFloat *)py::array_t<OSQPFloat>(l).data();
    }
    if (u.is_none()) {
        _u = NULL;
    } else {
        _u = (OSQPFloat *)py::array_t<OSQPFloat>(u).data();
    }

    return osqp_update_data_vec(this->_solver, _q, _l, _u);
}

OSQPInt PyTinySolver::update_data_mat(py::object P_x, py::object P_i, py::object A_x, py::object A_i) {
    OSQPFloat* _P_x;
    OSQPInt* _P_i;
    OSQPInt _P_n = 0;
    OSQPFloat* _A_x;
    OSQPInt* _A_i;
    OSQPInt _A_n = 0;

    if (P_x.is_none()) {
        _P_x = NULL;
    } else {
        auto _P_x_array = py::array_t<OSQPFloat>(P_x);
        _P_x = (OSQPFloat *)_P_x_array.data();
        _P_n = _P_x_array.size();
    }

    if (P_i.is_none()) {
        _P_i = NULL;
    } else {
        auto _P_i_array = py::array_t<OSQPInt>(P_i);
        _P_i = (OSQPInt *)_P_i_array.data();
        _P_n = _P_i_array.size();
    }

    if (A_x.is_none()) {
        _A_x = NULL;
    } else {
        auto _A_x_array = py::array_t<OSQPFloat>(A_x);
        _A_x = (OSQPFloat *)_A_x_array.data();
        _A_n = _A_x_array.size();
    }

    if (A_i.is_none()) {
        _A_i = NULL;
    } else {
        auto _A_i_array = py::array_t<OSQPInt>(A_i);
        _A_i = (OSQPInt *)_A_i_array.data();
        _A_n = _A_i_array.size();
    }

    return osqp_update_data_mat(this->_solver, _P_x, _P_i, _P_n, _A_x, _A_i, _A_n);
}


// OSQPInt PyTinySolver::codegen(const char *output_dir, const char *file_prefix, OSQPCodegenDefines& defines) {
//     return osqp_codegen(this->_solver, output_dir, file_prefix, &defines);
// }

PYBIND11_MODULE(ext_tinympc, m) {
// // PYBIND11_MODULE(@TINYMPC_EXT_MODULE_NAME@, m) {

    // Cache
    py::class_<TinyCache>(m, "TinyCache", py::module_local())
    .def(py::init([]() {
        return new TinyCache();
    }))
    .def_readwrite("rho", &TinyCache::rho)
    .def_readwrite("Kinf", &TinyCache:Kinf)
    .def_readwrite("Pinf", &TinyCache:Pinf)
    .def_readwrite("Quu_inv", &TinyCache:Quu_inv)
    .def_readwrite("AmBKt", &TinyCache:AmBKt)

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
    .def_readwrite("en_input_bound", &TinySettings::en_input_bound)

    m.def("tiny_set_default_settings", &tiny_set_default_settings);

    // Codegen Defines
    py::class_<OSQPCodegenDefines>(m, "OSQPCodegenDefines", py::module_local())
    .def(py::init([]() {
        return new OSQPCodegenDefines();
    }))
    .def_readwrite("embedded_mode", &OSQPCodegenDefines::embedded_mode)
    .def_readwrite("float_type", &OSQPCodegenDefines::float_type)
    .def_readwrite("printing_enable", &OSQPCodegenDefines::printing_enable)
    .def_readwrite("profiling_enable", &OSQPCodegenDefines::profiling_enable)
    .def_readwrite("interrupt_enable", &OSQPCodegenDefines::interrupt_enable)
    .def_readwrite("derivatives_enable", &OSQPCodegenDefines::derivatives_enable);

    m.def("osqp_set_default_codegen_defines", &osqp_set_default_codegen_defines);

    // Solution
    py::class_<PyOSQPSolution>(m, "OSQPSolution", py::module_local())
    .def_property_readonly("x", &PyOSQPSolution::get_x)
    .def_property_readonly("y", &PyOSQPSolution::get_y)
    .def_property_readonly("prim_inf_cert", &PyOSQPSolution::get_prim_inf_cert)
    .def_property_readonly("dual_inf_cert", &PyOSQPSolution::get_dual_inf_cert);

    // Info
    py::class_<OSQPInfo>(m, "OSQPInfo", py::module_local())
    .def_readonly("status", &OSQPInfo::status)
    .def_readonly("status_val", &OSQPInfo::status_val)
    .def_readonly("status_polish", &OSQPInfo::status_polish)
    // obj_val is readwrite because Python wrappers may overwrite this value based on status_val
    .def_readwrite("obj_val", &OSQPInfo::obj_val)
    .def_readonly("prim_res", &OSQPInfo::prim_res)
    .def_readonly("dual_res", &OSQPInfo::dual_res)
    .def_readonly("iter", &OSQPInfo::iter)
    .def_readonly("rho_updates", &OSQPInfo::rho_updates)
    .def_readonly("rho_estimate", &OSQPInfo::rho_estimate)
    .def_readonly("setup_time", &OSQPInfo::setup_time)
    .def_readonly("solve_time", &OSQPInfo::solve_time)
    .def_readonly("update_time", &OSQPInfo::update_time)
    .def_readonly("polish_time", &OSQPInfo::polish_time)
    .def_readonly("run_time", &OSQPInfo::run_time);

    // Solver
    py::class_<PyTinySolver>(m, "TinySolver", py::module_local())
    .def(py::init<const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const int, const int, const int, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>, const Eigen::Ref<tinyMatrix>>(),
            "A"_a.noconvert(), "B"_a.noconvert(), "Q"_a.noconvert(), "R"_a.noconvert(), "nx"_a, "nu"_a, "N"_a, "x_min"_a.noconvert(), "x_max"_a.noconvert(), "u_min"_a.noconvert(), "u_max"_a.noconvert())
    // .def(py::init<const CSC&, const py::array_t<OSQPFloat>, const CSC&, const py::array_t<OSQPFloat>, const py::array_t<OSQPFloat>, OSQPInt, OSQPInt, const OSQPSettings*>(),
    //         "P"_a, "q"_a.noconvert(), "A"_a, "l"_a.noconvert(), "u"_a.noconvert(), "m"_a, "n"_a, "settings"_a)
    .def_property_readonly("solution", &PyTinySolver::get_solution, py::return_value_policy::reference)
    .def_property_readonly("info", &PyTinySolver::get_info)
    .def("warm_start", &PyTinySolver::warm_start, "x"_a.none(true), "y"_a.none(true))
    .def("solve", &PyTinySolver::solve)
    .def("update_data_vec", &PyTinySolver::update_data_vec, "q"_a.none(true), "l"_a.none(true), "u"_a.none(true))
    .def("update_data_mat", &PyTinySolver::update_data_mat, "P_x"_a.none(true), "P_i"_a.none(true), "A_x"_a.none(true), "A_i"_a.none(true))
    .def("update_settings", &PyTinySolver::update_settings)
    .def("update_rho", &PyTinySolver::update_rho)
    .def("get_settings", &PyTinySolver::get_settings, py::return_value_policy::reference)

    .def("adjoint_derivative_compute", &PyTinySolver::adjoint_derivative_compute, "dx"_a.none(true), "dy_l"_a.none(true), "dy_u"_a.none(true))
    .def("adjoint_derivative_get_mat", &PyTinySolver::adjoint_derivative_get_mat, "dP"_a, "dA"_a)
    .def("adjoint_derivative_get_vec", &PyTinySolver::adjoint_derivative_get_vec, "dq"_a, "dl"_a, "du"_a)

    .def("codegen", &PyTinySolver::codegen, "output_dir"_a, "file_prefix"_a, "defines"_a);

}
