cmake_minimum_required(VERSION 3.15...3.26)
project(ext)

set(PYTHON "ON")
set(TINY_CODEGEN_INSTALL_DIR "codegen/codegen_src" CACHE PATH "" FORCE)

include(FetchContent)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Fetching/configuring TinyMPC")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
# --- TinyMPC -------------------------------------------------
add_subdirectory(tinympc/TinyMPC tinympc_build)
list(POP_BACK CMAKE_MESSAGE_INDENT)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
#                ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
pybind11_add_module(ext_tinympc src/bindings.cpp)
install(TARGETS ext_tinympc DESTINATION . COMPONENT python)

install(TARGETS ext_tinympc DESTINATION . COMPONENT python)
target_link_libraries(ext_tinympc PUBLIC pybind11::module tinympcstatic)


