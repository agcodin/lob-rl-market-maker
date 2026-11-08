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

# Sanitizers are a build option rather than a separate target so the tests
# below run as-is under ASan/UBSan: -DLOBRL_SANITIZE=ON.
option(LOBRL_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(LOBRL_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# Shared by the benchmark and the tests: a naive reference book used both as a
# baseline to measure against and as the other side of the differential fuzz test.
add_library(lob_support INTERFACE)
target_include_directories(lob_support INTERFACE ${CMAKE_SOURCE_DIR}/cpp/support)

set(LOBRL_BENCH_FLAGS "-O3 -fno-omit-frame-pointer")
if(LOBRL_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(LOBRL_BENCH_FLAGS "${LOBRL_BENCH_FLAGS} -mcpu=native")
  else()
    set(LOBRL_BENCH_FLAGS "${LOBRL_BENCH_FLAGS} -march=native")
  endif()
endif()

add_executable(bench_latency cpp/bench/bench_latency.cpp)
target_link_libraries(bench_latency PRIVATE lob_core lob_support)
target_compile_options(bench_latency PRIVATE -O3 -fno-omit-frame-pointer)
# The benchmark prints the flags it was built with, so the numbers carry their
# own provenance instead of relying on the README to describe them.
target_compile_definitions(bench_latency PRIVATE
  LOBRL_BUILD_FLAGS="${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} ${LOBRL_BENCH_FLAGS}")

# The Python extension is normally built with `pip install -e .`; this target
# exists so the same sources can be compiled from a pure-CMake tree.
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_lobcore cpp/src/bindings.cpp)
  target_link_libraries(_lobcore PRIVATE lob_core)
endif()

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(lob_tests cpp/tests/test_book.cpp cpp/tests/test_invariants.cpp)
  target_link_libraries(lob_tests PRIVATE lob_core lob_support Catch2::Catch2WithMain)
  # The arena is a few megabytes of members, so the book is heap-allocated in
  # the tests; keep frame sizes honest anyway.
  target_compile_options(lob_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(lob_tests)
endif()
