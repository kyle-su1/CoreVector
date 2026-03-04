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

# xsimd (header-only, fast to fetch)
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

# gRPC and Protobuf (system-installed via Homebrew)
# Find gRPC first — it will locate the correct protobuf version automatically
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

# Get the protoc and grpc_cpp_plugin executables
find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin)

# ------------------------------------------------------------------------------
# Proto generation
# ------------------------------------------------------------------------------

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

set(PROTO_FILE "${PROTO_SRC_DIR}/vector_db.proto")
set(PB_CC "${PROTO_GENERATED_DIR}/vector_db.pb.cc")
set(PB_H "${PROTO_GENERATED_DIR}/vector_db.pb.h")
set(GRPC_CC "${PROTO_GENERATED_DIR}/vector_db.grpc.pb.cc")
set(GRPC_H "${PROTO_GENERATED_DIR}/vector_db.grpc.pb.h")

add_custom_command(
    OUTPUT ${PB_CC} ${PB_H} ${GRPC_CC} ${GRPC_H}
    COMMAND ${_PROTOBUF_PROTOC}
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_GENERATED_DIR}
        --grpc_out=${PROTO_GENERATED_DIR}
        --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf and gRPC C++ stubs from vector_db.proto"
)

# Proto library (shared between server and any future clients)
add_library(corevector_proto ${PB_CC} ${GRPC_CC})
target_include_directories(corevector_proto PUBLIC ${PROTO_GENERATED_DIR})
target_link_libraries(corevector_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# Core library (header-only)
add_library(corevector INTERFACE)
target_include_directories(corevector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(corevector INTERFACE xsimd)

# Main executable (basic smoke test)
add_executable(corevector_bin src/main.cpp)
target_link_libraries(corevector_bin PRIVATE corevector)

# gRPC Server
add_executable(corevector_server src/server.cpp)
target_link_libraries(corevector_server PRIVATE corevector corevector_proto)

# Tests
add_executable(test_serialization tests/test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE corevector gtest gtest_main)
include(GoogleTest)
gtest_discover_tests(test_serialization)

add_executable(test_mmap tests/test_mmap.cpp)
target_link_libraries(test_mmap PRIVATE corevector gtest gtest_main)
gtest_discover_tests(test_mmap)

add_executable(test_vector_math tests/test_vector_math.cpp)
target_link_libraries(test_vector_math PRIVATE corevector gtest gtest_main)
gtest_discover_tests(test_vector_math)

add_executable(test_flat_index tests/test_flat_index.cpp)
target_link_libraries(test_flat_index PRIVATE corevector gtest gtest_main)
gtest_discover_tests(test_flat_index)

# Benchmarks
add_executable(benchmark_math benchmarks/benchmark_math.cpp)
target_link_libraries(benchmark_math PRIVATE corevector benchmark::benchmark benchmark::benchmark_main)

add_executable(benchmark_concurrency benchmarks/benchmark_concurrency.cpp)
target_link_libraries(benchmark_concurrency PRIVATE corevector benchmark::benchmark benchmark::benchmark_main)

add_executable(benchmark_network benchmarks/benchmark_network.cpp)
target_link_libraries(benchmark_network PRIVATE corevector corevector_proto benchmark::benchmark benchmark::benchmark_main)

# Compiler optimizations for release builds
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(benchmark_math PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(benchmark_concurrency PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(benchmark_network PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(corevector_bin PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(corevector_server PRIVATE -O3 -march=native -ffast-math)
endif()
