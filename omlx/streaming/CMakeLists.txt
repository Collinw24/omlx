# SPDX-License-Identifier: Apache-2.0
#
# CMake build for _buffer_access C++ pybind11 extension.
#
# Build (from omlx/streaming/):
#   cmake -B build && cmake --build build
#
# The built module (_buffer_access.cpython-*-darwin.so) is placed directly
# into omlx/streaming/ so Python can import it with:
#   from omlx.streaming._buffer_access import pread_into_array
#
# Notes:
# - Uses the Python buffer protocol to access backing memory, so no MLX
#   headers are required at build time.
# - Works with any Python object supporting the writable buffer protocol
#   (mx.array, bytearray, memoryview, numpy.ndarray).

cmake_minimum_required(VERSION 3.18)
project(_buffer_access LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Python + pybind11 ----
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# ---- Build module ----
pybind11_add_module(_buffer_access MODULE _buffer_access.cpp)

# Install directly into omlx/streaming/ so it's importable.
install(TARGETS _buffer_access
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
