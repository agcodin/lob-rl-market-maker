cmake_minimum_required(VERSION 3.18)
project(lobrl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(lob_core INTERFACE)
target_include_directories(lob_core INTERFACE ${CMAKE_SOURCE_DIR}/cpp/include)

option(LOBRL_NATIVE "Build with -march/-mcpu=native" ON)
if(LOBRL_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(lob_core INTERFACE -mcpu=native)
  else()
    target_compile_options(lob_core INTERFACE -march=native)
  endif()
endif()

add_executable(bench_latency cpp/bench/bench_latency.cpp)
target_link_libraries(bench_latency PRIVATE lob_core)
target_compile_options(bench_latency PRIVATE -O3 -fno-omit-frame-pointer)

# The Python extension is normally built with `pip install -e .`; this target
# exists so the same sources can be compiled from a pure-CMake tree.
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_lobcore cpp/src/bindings.cpp)
  target_link_libraries(_lobcore PRIVATE lob_core)
endif()
