cmake_minimum_required(VERSION 3.15)
project(tinympc_codegen_ext)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

file(GLOB TINY_SOURCES tinympc/*.cpp)

add_library(tinympcstatic STATIC ${TINY_SOURCES})

target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tinympcstatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tiny_codegen_example src/tiny_main.cpp src/tiny_data.cpp)
target_link_libraries(tiny_codegen_example LINK_PUBLIC tinympcstatic)


include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git)
FetchContent_MakeAvailable(pybind11)

file(
  GLOB
  EXT_SOURCES
  src/tiny_data.cpp
  tinympc/tiny_data.hpp
  bindings.cpp)

pybind11_add_module(tinympcgen ${TINY_SOURCES} ${EXT_SOURCES})
target_link_libraries(tinympcgen PUBLIC pybind11::module tinympcstatic)
