cmake_minimum_required(VERSION 3.15)
project(tinympc_codegen_ext)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(PYBIND11_PYTHON_VERSION "" CACHE STRING "")
endif()

# Add Eigen include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc
)

# Instead of using GLOB which might not find any files,
# explicitly include the source files we know should be there
set(TINY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tiny_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/admm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/tiny_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/rho_benchmark.cpp
)

# Create tinympc library
add_library(tinympcstatic STATIC ${TINY_SOURCES})
target_include_directories(tinympcstatic PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc
)

# Create example main executable
add_executable(tiny_codegen_example src/tiny_main.cpp src/tiny_data.cpp)
target_include_directories(tiny_codegen_example PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc
)
target_link_libraries(tiny_codegen_example LINK_PUBLIC tinympcstatic)

# Create python module
set(PYBIND11_FINDPYTHON ON)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git)
FetchContent_MakeAvailable(pybind11)

# Define extension sources explicitly
set(EXT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tiny_data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/admm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/tiny_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/rho_benchmark.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)

pybind11_add_module(tinympcgen ${EXT_SOURCES})
target_include_directories(tinympcgen PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc
)
target_link_libraries(tinympcgen PUBLIC pybind11::module tinympcstatic)
