cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(CoreVector VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Includes
include(FetchContent)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

# xsimd
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        11.1.0
)
FetchContent_MakeAvailable(xsimd)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
FetchContent_MakeAvailable(benchmark)

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# Include directories
add_library(corevector INTERFACE)
target_include_directories(corevector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(corevector INTERFACE xsimd)

# Main executable
add_executable(corevector_bin src/main.cpp)
target_link_libraries(corevector_bin PRIVATE corevector)

# Tests
add_executable(test_serialization tests/test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE corevector gtest gtest_main)
include(GoogleTest)
gtest_discover_tests(test_serialization)

add_executable(test_mmap tests/test_mmap.cpp)
target_link_libraries(test_mmap PRIVATE corevector gtest gtest_main)
gtest_discover_tests(test_mmap)

# Benchmarks
add_executable(benchmark_math benchmarks/benchmark_math.cpp)
target_link_libraries(benchmark_math PRIVATE corevector benchmark::benchmark benchmark::benchmark_main)

add_executable(benchmark_concurrency benchmarks/benchmark_concurrency.cpp)
target_link_libraries(benchmark_concurrency PRIVATE corevector benchmark::benchmark benchmark::benchmark_main)

# Add compiler optimizations (assuming AVX2 on x86, or NEON on ARM via Native)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(benchmark_math PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(benchmark_concurrency PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(corevector_bin PRIVATE -O3 -march=native -ffast-math)
endif()
