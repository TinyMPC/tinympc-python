import ctypes
import tinympc

import subprocess
import os

if __name__ == '__main__':
    tinympc_dir = "/home/khai/SSD/Code/tinympc-python/generated_code"

    # Specify the path to your CMakeLists.txt file or the source directory
    source_directory = tinympc_dir

    # Specify the path to the build directory (where CMake will generate build files)
    build_directory = tinympc_dir + "/build"

    # Make sure the build directory exists
    os.makedirs(build_directory, exist_ok=True)

    # Run CMake configuration
    cmake_configure_cmd = ["cmake", source_directory]
    subprocess.run(cmake_configure_cmd, cwd=build_directory)

    # Run the build process (e.g., make)
    cmake_build_cmd = ["cmake", "--build", "."]
    subprocess.run(cmake_build_cmd, cwd=build_directory)

    # prob = tinympc.TinyMPC(tinympc_dir)

    # prob.set_x0([1, 2, 3, 4])
