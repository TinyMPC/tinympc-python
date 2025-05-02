import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import tinympc

# Import the example code setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
from cartpole_example_code_generation import A, B, Q, R, N, u_min, u_max

class TestCodeGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize MPC using the example setup
        self.prob = tinympc.TinyMPC()
        self.prob.setup(A, B, Q, R, N, rho=1, max_iter=10, u_min=u_min, u_max=u_max)

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_codegen_files_created(self):
        """Test if code generation creates all necessary files"""
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

    def test_codegen_build(self):
        """Test if generated code can be built"""
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

    def test_codegen_import(self):
        """Test if generated code can be imported and used"""
        self.prob.codegen(self.test_dir, verbose=1)
        
        try:
            cwd = os.getcwd()
            os.chdir(self.test_dir)
            
            # Build using python3 explicitly
            os.system("python3 setup.py build_ext --inplace")
            
            # Try importing
            sys.path.insert(0, self.test_dir)
            import tinympcgen
            
            # Test basic functionality using same initial state as other examples
            x0 = np.array([0.5, 0, 0, 0])
            tinympcgen.set_x0(x0)
            solution = tinympcgen.solve()
            
            self.assertIsInstance(solution, dict)
            self.assertIn("controls", solution)
            self.assertIn("states_all", solution)
            
        finally:
            os.chdir(cwd)
            if "tinympcgen" in sys.modules:
                del sys.modules["tinympcgen"]
            if self.test_dir in sys.path:
                sys.path.remove(self.test_dir)

if __name__ == '__main__':
    unittest.main() 