cmake_minimum_required(VERSION 3.21)
project(hip_vmm_bug LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Point CMAKE_PREFIX_PATH at the ROCm install, e.g. -DCMAKE_PREFIX_PATH=/opt/rocm
find_package(hip REQUIRED)

add_executable(hip_vmm_bug main.cpp)
target_link_libraries(hip_vmm_bug PRIVATE hip::host)
