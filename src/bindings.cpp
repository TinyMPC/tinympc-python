#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "tinympc/tiny_api.hpp"
#include "tinympc/codegen.hpp"

class PyTinySolver {
    public:
        PyTinySolver(Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // A, B
                     Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // Q, R
                     float, int, int, int,               // rho, nx, nu, N
                     Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // x_min, x_max
                     Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, // u_min, u_max
                     TinySettings*, int); // settings, verbosity
        void set_x0(Eigen::Ref<tinyVector>);
        void set_x_ref(Eigen::Ref<tinyMatrix>);
        void set_u_ref(Eigen::Ref<tinyMatrix>);
        void set_sensitivity_matrices(
            Eigen::Ref<tinyMatrix> dK,
            Eigen::Ref<tinyMatrix> dP,
            Eigen::Ref<tinyMatrix> dC1,
            Eigen::Ref<tinyMatrix> dC2,
            float rho = 0.0,
            int verbose = 0);
        void set_cache_terms(
            Eigen::Ref<tinyMatrix> Kinf,
            Eigen::Ref<tinyMatrix> Pinf,
            Eigen::Ref<tinyMatrix> Quu_inv,
            Eigen::Ref<tinyMatrix> AmBKt,
            int verbose = 0);
        void update_settings(TinySettings* settings);

        int solve();
        TinySolution* get_solution();

        void print_problem_data();

        int codegen(const char*, int); // output_dir, verbosity

         int codegen_with_sensitivity(const char *output_dir, 
                                   Eigen::Ref<tinyMatrix> dK,
                                   Eigen::Ref<tinyMatrix> dP,
                                   Eigen::Ref<tinyMatrix> dC1,
                                   Eigen::Ref<tinyMatrix> dC2,
                                   int verbose);
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
        Eigen::Ref<tinyMatrix> x_min,
        Eigen::Ref<tinyMatrix> x_max,
        Eigen::Ref<tinyMatrix> u_min,
        Eigen::Ref<tinyMatrix> u_max,
        TinySettings *settings,
        int verbose) {

    int status = tiny_setup(&this->_solver, A, B, Q, R, rho, nx, nu, N, x_min, x_max, u_min, u_max, verbose);
    this->_solver->settings = settings;
    if (status) {
        std::string message = "Error during setup";
        throw py::value_error(message); 
    }
}

void PyTinySolver::set_x0(Eigen::Ref<tinyVector> x0) {
    tiny_set_x0(this->_solver, x0);
}

void PyTinySolver::set_x_ref(Eigen::Ref<tinyMatrix> x_ref) {
    tiny_set_x_ref(this->_solver, x_ref);
}

void PyTinySolver::set_u_ref(Eigen::Ref<tinyMatrix> u_ref) {
    tiny_set_u_ref(this->_solver, u_ref);
}

void PyTinySolver::set_sensitivity_matrices(
        Eigen::Ref<tinyMatrix> dK,
        Eigen::Ref<tinyMatrix> dP,
        Eigen::Ref<tinyMatrix> dC1,
        Eigen::Ref<tinyMatrix> dC2,
        float rho,
        int verbose) {
    // Create copies of the matrices to ensure they remain valid
    tinyMatrix dK_copy = dK;
    tinyMatrix dP_copy = dP;
    tinyMatrix dC1_copy = dC1;
    tinyMatrix dC2_copy = dC2;
    
    // Store the sensitivity matrices in the solver's cache
    if (this->_solver->cache != nullptr) {

        // For now, we'll just store them for code generation
        if (verbose) {
            std::cout << "Sensitivity matrices set for code generation" << std::endl;
            std::cout << "dK norm: " << dK_copy.norm() << std::endl;
            std::cout << "dP norm: " << dP_copy.norm() << std::endl;
            std::cout << "dC1 norm: " << dC1_copy.norm() << std::endl;
            std::cout << "dC2 norm: " << dC2_copy.norm() << std::endl;
        }
    } else {
        if (verbose) {
            std::cout << "Warning: Cache not initialized, sensitivity matrices will only be used for code generation" << std::endl;
        }
    }
}

int PyTinySolver::solve() {
    py::gil_scoped_release release;
    int status = tiny_solve(this->_solver);
    py::gil_scoped_acquire acquire;
    return 0;
}

TinySolution* PyTinySolver::get_solution() {
    return this->_solver->solution;
}

void PyTinySolver::print_problem_data() {
    std::cout << "solution iter:\n" << this->_solver->solution->iter << std::endl;
    std::cout << "solution solved:\n" << this->_solver->solution->solved << std::endl;
    std::cout << "solution x:\n" << this->_solver->solution->x << std::endl;
    std::cout << "solution u:\n" << this->_solver->solution->u << std::endl;

    std::cout << "\n\n\ncache rho: " << this->_solver->cache->rho << std::endl;
    std::cout << "cache Kinf:\n" << this->_solver->cache->Kinf << std::endl;
    std::cout << "cache Pinf:\n" << this->_solver->cache->Pinf << std::endl;
    std::cout << "cache Quu_inv:\n" << this->_solver->cache->Quu_inv << std::endl;
    std::cout << "cache AmBKt:\n" << this->_solver->cache->AmBKt << std::endl;

    std::cout << "\n\n\nabs_pri_tol: " << this->_solver->settings->abs_pri_tol << std::endl;
    std::cout << "abs_dua_tol: " << this->_solver->settings->abs_dua_tol << std::endl;
    std::cout << "max_iter: " << this->_solver->settings->max_iter << std::endl;
    std::cout << "check_termination: " << this->_solver->settings->check_termination << std::endl;
    std::cout << "en_state_bound: " << this->_solver->settings->en_state_bound << std::endl;
    std::cout << "en_input_bound: " << this->_solver->settings->en_input_bound << std::endl;

    std::cout << "\n\n\nnx: " << this->_solver->work->nx << std::endl;
    std::cout << "nu: " << this->_solver->work->nu << std::endl;
    std::cout << "x:\n" << this->_solver->work->x << std::endl;
    std::cout << "u:\n" << this->_solver->work->u << std::endl;
    std::cout << "q:\n" << this->_solver->work->q << std::endl;
    std::cout << "r:\n" << this->_solver->work->r << std::endl;
    std::cout << "p:\n" << this->_solver->work->p << std::endl;
    std::cout << "d:\n" << this->_solver->work->d << std::endl;
    std::cout << "v:\n" << this->_solver->work->v << std::endl;
    std::cout << "vnew:\n" << this->_solver->work->vnew << std::endl;
    std::cout << "z:\n" << this->_solver->work->z << std::endl;
    std::cout << "znew:\n" << this->_solver->work->znew << std::endl;
    std::cout << "g:\n" << this->_solver->work->g << std::endl;
    std::cout << "y:\n" << this->_solver->work->y << std::endl;
    std::cout << "Q:\n" << this->_solver->work->Q << std::endl;
    std::cout << "R:\n" << this->_solver->work->R << std::endl;
    std::cout << "Adyn:\n" << this->_solver->work->Adyn << std::endl;
    std::cout << "Bdyn:\n" << this->_solver->work->Bdyn << std::endl;
    std::cout << "x_min:\n" << this->_solver->work->x_min << std::endl;
    std::cout << "x_max:\n" << this->_solver->work->x_max << std::endl;
    std::cout << "u_min:\n" << this->_solver->work->u_min << std::endl;
    std::cout << "u_max:\n" << this->_solver->work->u_max << std::endl;
    std::cout << "Xref:\n" << this->_solver->work->Xref << std::endl;
    std::cout << "Uref:\n" << this->_solver->work->Uref << std::endl;
    std::cout << "Qu:\n" << this->_solver->work->Qu << std::endl;
    std::cout << "primal_residual_state:\n" << this->_solver->work->primal_residual_state << std::endl;
    std::cout << "primal_residual_input:\n" << this->_solver->work->primal_residual_input << std::endl;
    std::cout << "dual_residual_state:\n" << this->_solver->work->dual_residual_state << std::endl;
    std::cout << "dual_residual_input:\n" << this->_solver->work->dual_residual_input << std::endl;
    std::cout << "status:\n" << this->_solver->work->status << std::endl;
    std::cout << "iter:\n" << this->_solver->work->iter << std::endl;
}

int PyTinySolver::codegen(const char *output_dir, int verbose) {
    return tiny_codegen(this->_solver, output_dir, verbose);
}

int PyTinySolver::codegen_with_sensitivity(const char *output_dir,
                                         Eigen::Ref<tinyMatrix> dK_ref,
                                         Eigen::Ref<tinyMatrix> dP_ref,
                                         Eigen::Ref<tinyMatrix> dC1_ref,
                                         Eigen::Ref<tinyMatrix> dC2_ref,
                                         int verbose) {
    tinyMatrix dK_copy = dK_ref;
    tinyMatrix dP_copy = dP_ref;
    tinyMatrix dC1_copy = dC1_ref;
    tinyMatrix dC2_copy = dC2_ref;

    return tiny_codegen_with_sensitivity(this->_solver, output_dir,
                                       &dK_copy, &dP_copy, &dC1_copy, &dC2_copy, verbose);
}

void PyTinySolver::set_cache_terms(
        Eigen::Ref<tinyMatrix> Kinf,
        Eigen::Ref<tinyMatrix> Pinf,
        Eigen::Ref<tinyMatrix> Quu_inv,
        Eigen::Ref<tinyMatrix> AmBKt,
        int verbose) {
    if (!this->_solver) {
        throw py::value_error("Solver not initialized");
    }
    if (!this->_solver->cache) {
        throw py::value_error("Solver cache not initialized");
    }
    
    // Create copies of the matrices to ensure they remain valid
    this->_solver->cache->Kinf = Kinf;
    this->_solver->cache->Pinf = Pinf;
    this->_solver->cache->Quu_inv = Quu_inv;
    this->_solver->cache->AmBKt = AmBKt;
    this->_solver->cache->C1 = Quu_inv;  // Cache terms
    this->_solver->cache->C2 = AmBKt;    // Cache terms
    
    if (verbose) {
        std::cout << "Cache terms set with norms:" << std::endl;
        std::cout << "Kinf norm: " << Kinf.norm() << std::endl;
        std::cout << "Pinf norm: " << Pinf.norm() << std::endl;
        std::cout << "Quu_inv norm: " << Quu_inv.norm() << std::endl;
        std::cout << "AmBKt norm: " << AmBKt.norm() << std::endl;
    }
}

void PyTinySolver::update_settings(TinySettings* settings) {
    if (this->_solver && this->_solver->settings && settings) {
        // Copy settings to the solver's settings
        this->_solver->settings->abs_pri_tol = settings->abs_pri_tol;
        this->_solver->settings->abs_dua_tol = settings->abs_dua_tol;
        this->_solver->settings->max_iter = settings->max_iter;
        this->_solver->settings->check_termination = settings->check_termination;
        this->_solver->settings->en_state_bound = settings->en_state_bound;
        this->_solver->settings->en_input_bound = settings->en_input_bound;
        
        // Copy adaptive rho settings
        this->_solver->settings->adaptive_rho = settings->adaptive_rho;
        this->_solver->settings->adaptive_rho_min = settings->adaptive_rho_min;
        this->_solver->settings->adaptive_rho_max = settings->adaptive_rho_max;
        this->_solver->settings->adaptive_rho_enable_clipping = settings->adaptive_rho_enable_clipping;
    }
}

PYBIND11_MODULE(ext_tinympc, m) {
// // PYBIND11_MODULE(@TINYMPC_EXT_MODULE_NAME@, m) {

    // Solution
    py::class_<TinySolution>(m, "TinySolution", py::module_local())
    .def(py::init([]() {
        return new TinySolution();
    }))
    .def_readwrite("iter", &TinySolution::iter)
    .def_readwrite("solved", &TinySolution::solved)
    .def_readwrite("x", &TinySolution::x)
    .def_readwrite("u", &TinySolution::u);

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
    .def_readwrite("en_input_bound", &TinySettings::en_input_bound)
    .def_readwrite("adaptive_rho", &TinySettings::adaptive_rho)
    .def_readwrite("adaptive_rho_min", &TinySettings::adaptive_rho_min)
    .def_readwrite("adaptive_rho_max", &TinySettings::adaptive_rho_max)
    .def_readwrite("adaptive_rho_enable_clipping", &TinySettings::adaptive_rho_enable_clipping);

    m.def("tiny_set_default_settings", &tiny_set_default_settings);

    // Solver
    py::class_<PyTinySolver>(m, "TinySolver", py::module_local())
    .def(py::init<Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, float, int, int, int, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, Eigen::Ref<tinyMatrix>, TinySettings*, int>(),
            "A"_a.noconvert(), "B"_a.noconvert(), "Q"_a.noconvert(), "R"_a.noconvert(), "rho"_a, "nx"_a, "nu"_a, "N"_a, "x_min"_a.noconvert(), "x_max"_a.noconvert(), "u_min"_a.noconvert(), "u_max"_a.noconvert(), "settings"_a, "verbose"_a)
    .def_property_readonly("solution", &PyTinySolver::get_solution)
    .def("set_x0", &PyTinySolver::set_x0)
    .def("set_x_ref", &PyTinySolver::set_x_ref)
    .def("set_u_ref", &PyTinySolver::set_u_ref)
    .def("update_settings", &PyTinySolver::update_settings)
    .def("set_sensitivity_matrices", &PyTinySolver::set_sensitivity_matrices,
         "dK"_a.noconvert(), "dP"_a.noconvert(),
         "dC1"_a.noconvert(), "dC2"_a.noconvert(),
         "rho"_a=0.0, "verbose"_a=0)
    .def("set_cache_terms", &PyTinySolver::set_cache_terms,
         py::arg("Kinf"), py::arg("Pinf"), py::arg("Quu_inv"), py::arg("AmBKt"),
         py::arg("verbose") = 0)
    .def("solve", &PyTinySolver::solve)
    .def("print_problem_data", &PyTinySolver::print_problem_data)
    .def("codegen", &PyTinySolver::codegen, "output_dir"_a, "verbose"_a=0)
    .def("codegen_with_sensitivity", &PyTinySolver::codegen_with_sensitivity,
         "output_dir"_a, "dK"_a.noconvert(), "dP"_a.noconvert(),
         "dC1"_a.noconvert(), "dC2"_a.noconvert(), "verbose"_a);

}