cmake_minimum_required(VERSION 3.12)
project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(benchmark REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES src/matrix_mul.cpp)

# Create static library
add_library(matrix_mul STATIC ${SOURCES})

# Benchmark executable
add_executable(matrix_benchmark benchmarks/benchmark.cpp)
target_link_libraries(matrix_benchmark matrix_mul benchmark::benchmark)

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(matrix_mul PRIVATE -O3 -march=native)
    target_compile_options(matrix_benchmark PRIVATE -O3 -march=native)
endif()

# Enable testing
enable_testing()

# Simple test executable
add_executable(matrix_test tests/test_matrix.cpp)
target_link_libraries(matrix_test matrix_mul)