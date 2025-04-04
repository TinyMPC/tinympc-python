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
  GIT_TAG main
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(tinympc)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
#                ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
pybind11_add_module(ext_tinympc src/bindings.cpp)
install(TARGETS ext_tinympc DESTINATION . COMPONENT python)

install(TARGETS ext_tinympc DESTINATION . COMPONENT python)
target_link_libraries(ext_tinympc PUBLIC pybind11::module tinympcstatic)


# # --- BEGIN ADDED CODE ---
# # Install codegen source files needed by the interface.py codegen() method
# # These files will be copied into the generated project by interface.py
# install(
#   DIRECTORY ${tinympc_SOURCE_DIR}/include/
#   DESTINATION ${TINY_CODEGEN_INSTALL_DIR}/include
#   COMPONENT codegen
# )
# install(
#   DIRECTORY ${tinympc_SOURCE_DIR}/tinympc/
#   DESTINATION ${TINY_CODEGEN_INSTALL_DIR}/tinympc
#   COMPONENT codegen
#   # Exclude the CMakeLists.txt from the submodule to avoid potential confusion
#   PATTERN "CMakeLists.txt" EXCLUDE
#   # Exclude CMake build artifacts and git files just in case
#   PATTERN "CMakeFiles" EXCLUDE
#   PATTERN "*.cmake" EXCLUDE
#   PATTERN ".git" EXCLUDE
# )
# # --- END ADDED CODE ---