cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dll_test_sln)

find_package(Torch REQUIRED)

add_executable(DLL_Test dll_test.cpp)
target_link_libraries(DLL_Test "${TORCH_LIBRARIES}")
set_property(TARGET DLL_Test PROPERTY CXX_STANDARD 11)
