cmake_minimum_required(VERSION 3.15...3.26)
project(ext)

set(PYTHON "ON")
set(TINY_CODEGEN_INSTALL_DIR "codegen/codegen_src" CACHE PATH "" FORCE)

include(FetchContent)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Fetching/configuring TinyMPC")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
FetchContent_Declare(
  tinympc
  GIT_REPOSITORY https://github.com/TinyMPC/TinyMPC.git
  GIT_TAG sam-dev-python
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(tinympc)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
#                ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
pybind11_add_module(ext_tinympc src/bindings.cpp)
install(TARGETS ext_tinympc DESTINATION . COMPONENT python)


target_link_libraries(ext_tinympc PUBLIC pybind11::module tinympcstatic)

# # TODO: We shouldn't have to do this once the interfaces are set up correctly
# if(${OSQP_ALGEBRA_BACKEND} STREQUAL "builtin")
#     target_link_libraries(ext_builtin PUBLIC pybind11::module osqpstatic)
# elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "mkl")
#     if(APPLE)
#         target_link_libraries(osqp_mkl PUBLIC pybind11::module osqpstatic)
#     else()
#         target_link_libraries(osqp_mkl PUBLIC pybind11::module osqpstatic $<LINK_ONLY:MKL::MKL>)
#     endif()
# elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "cuda")
#     enable_language(CUDA)
#     find_package(CUDA)
#     include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#     target_link_directories(osqp_cuda PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
#     target_link_libraries(osqp_cuda PUBLIC pybind11::module osqpstatic cublas cusparse)
# endif()
