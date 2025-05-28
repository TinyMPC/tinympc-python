import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import tinympc

# Import the example code setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
from quadrotor_hover_code_generation import Adyn, Bdyn, Q, R, N, u_min, u_max, rho_value

class TestQuadrotorCodeGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize MPC using the quadrotor example setup
        self.prob = tinympc.TinyMPC()
        self.prob.setup(Adyn, Bdyn, Q, R, N, rho=rho_value, max_iter=100, u_min=u_min, u_max=u_max)

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_quadrotor_codegen_files_created(self):
        """Test if code generation creates all necessary files for quadrotor"""
        self.prob.codegen(self.test_dir, verbose=1)
        
        # Check essential files
        expected_files = [
            "CMakeLists.txt",
            "setup.py",
            "bindings.cpp",
            "src/tiny_main.cpp",
            "src/tiny_data.cpp",
            "tinympc/tiny_data.hpp"
        ]
        
        for file in expected_files:
            path = os.path.join(self.test_dir, file)
            self.assertTrue(os.path.exists(path), f"Missing file: {file}")

    def test_quadrotor_codegen_build(self):
        """Test if generated code for quadrotor can be built"""
        self.prob.codegen(self.test_dir, verbose=1)
        
        try:
            cwd = os.getcwd()
            os.chdir(self.test_dir)
            
            # Attempt to build using python3 explicitly
            result = os.system("python3 setup.py build_ext --inplace")
            self.assertEqual(result, 0, "Build failed")
            
            # Check for built module
            built_files = os.listdir(".")
            module_exists = any(f.startswith("tinympcgen") and 
                              (f.endswith(".so") or f.endswith(".pyd")) 
                              for f in built_files)
            self.assertTrue(module_exists, "Built module not found")
            
        finally:
            os.chdir(cwd)

    def test_quadrotor_codegen_import(self):
        """Test if generated code for quadrotor can be imported and used"""
        self.prob.codegen(self.test_dir, verbose=1)
        
        try:
            cwd = os.getcwd()
            os.chdir(self.test_dir)
            
            # Build using python3 explicitly
            os.system("python3 setup.py build_ext --inplace")
            
            # Try importing
            sys.path.insert(0, self.test_dir)
            import tinympcgen
            
            # Test basic functionality with a quadrotor initial state
            x0 = np.zeros(12)  # Initial state for quadrotor (all zeros)
            tinympcgen.set_x0(x0)
            solution = tinympcgen.solve()
            
            self.assertIsInstance(solution, dict)
            self.assertIn("controls", solution)
            self.assertIn("states_all", solution)
            
            # Check dimensions of solution
            self.assertEqual(solution["controls"].shape[0], 4)  # 4 control inputs
            self.assertEqual(solution["states_all"].shape[1], 12)  # 12 states
            
        finally:
            os.chdir(cwd)
            if "tinympcgen" in sys.modules:
                del sys.modules["tinympcgen"]
            if self.test_dir in sys.path:
                sys.path.remove(self.test_dir)

    def test_quadrotor_adaptive_rho(self):
        """Test code generation with adaptive rho for quadrotor"""
        # First compute the cache terms
        Kinf, Pinf, Quu_inv, AmBKt = self.prob.compute_cache_terms()
        
        # Create dummy sensitivity matrices (real computation would use autograd)
        nu, nx = Bdyn.shape[1], Adyn.shape[0]
        dK = np.ones((nu, nx)) * 0.001
        dP = np.ones((nx, nx)) * 0.001
        dC1 = np.ones((nu, nu)) * 0.001
        dC2 = np.ones((nx, nx)) * 0.001
        
        # Generate code with sensitivity matrices
        self.prob.codegen_with_sensitivity(self.test_dir, dK, dP, dC1, dC2, verbose=1)
        
        # Check if files were created
        expected_files = [
            "CMakeLists.txt",
            "setup.py",
            "bindings.cpp",
            "src/tiny_main.cpp",
            "src/tiny_data.cpp",
            "tinympc/tiny_data.hpp"
        ]
        
        for file in expected_files:
            path = os.path.join(self.test_dir, file)
            self.assertTrue(os.path.exists(path), f"Missing file: {file}")

if __name__ == '__main__':
    unittest.main()