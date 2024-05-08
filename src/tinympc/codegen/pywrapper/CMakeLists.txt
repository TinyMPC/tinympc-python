cmake_minimum_required(VERSION 3.15)
project(tinympc_tiny_codegen_ext)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

file(
  GLOB
  TINY_SOURCES
  src/*.c)

add_library(tinympcstatic
  STATIC
    ${TINY_SOURCES}
  )

# target_include_directories(tinympcstatic PUBLIC inc/public inc/private .)
# target_include_directories(tinympcstatic PRIVATE inc/private .)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Eigen)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git)
FetchContent_MakeAvailable(pybind11)

file(
  GLOB
  EXT_SOURCES
  tiny_data.cpp
  tiny_data.hpp
  bindings.cpp)

pybind11_add_module(tiny_codegen_ext ${TINY_SOURCES} ${EXT_SOURCES})
target_link_libraries(tiny_codegen_ext PUBLIC pybind11::module tinympcstatic)
