// SPDX-License-Identifier: Apache-2.0
//
// Phase 2b: Zero-copy pread into array backing buffers via Python buffer protocol.
//
// Exposes a single pybind11 function, `pread_into_array`, that reads bytes
// from an open file descriptor (F_NOCACHE) directly into the backing memory
// of any Python object that supports the writable buffer protocol (including
// MLX's `mx.array`).
//
// Why C++?
// --------
// Python's `mx.array` does not support in-place slice assignment for 2D arrays
// (Invariant 0.7).  The Python fallback (`_buffer_access_fallback.py`) allocates
// a temp `bytes` object per read (violates Invariant 0.9: zero-allocation views).
//
// This extension uses the Python buffer protocol to obtain the raw backing
// pointer and writes directly via `pread(2)` -- zero intermediate allocations,
// zero Python heap involvement in the hot path.
//
// Safety
// ------
// - The buffer must be writable, format='B' (uint8), C-contiguous, and have
//   enough capacity for the requested write.
// - The caller must hold a strong reference to the Python object (e.g. the
//   mx.array) for the duration of the call so the backing buffer is not
//   reclaimed.
// - Writes go into the buffer's raw memory at the given byte offset; the caller
//   is responsible for ensuring the region does not overlap with tensors that
//   the GPU is currently reading (see I2: graph gate protocol).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>

namespace py = pybind11;

// -----------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------

/// Validate buffer info for direct pread writes.
///
/// Requirements:
///   1. Buffer must be writable (not read-only).
///   2. Format must be 'B' (unsigned byte / uint8).
///   3. Buffer must be C-contiguous.
///   4. Buffer must have enough capacity for arr_byte_offset + length.
static void validate_buffer(
    const py::buffer_info& info,
    size_t arr_byte_offset,
    size_t length,
    const std::string& arg_name) {

  // --- Writable check ---
  if (info.readonly) {
    throw std::invalid_argument(
        arg_name + " must be a writable buffer (read-only buffer given). "
        "Call mx.eval(arr) before passing to pread_into_array.");
  }

  // --- Format check ---
  // Format 'B' is unsigned char (uint8).  Also accept 'b' (signed char)
  // since some buffer implementations may report it.
  if (info.format != "B" && info.format != "b") {
    throw std::invalid_argument(
        arg_name + " must have dtype=uint8 (format='B'), got format='"
        + info.format + "'");
  }

  // --- Contiguity check ---
  // For a C-contiguous buffer, each stride[i] ==
  //   product(shape[i+1:]) * itemsize.
  // We verify this property explicitly.
  if (info.ndim > 0) {
    ssize_t expected_stride = info.itemsize;
    for (ssize_t i = info.ndim - 1; i >= 0; --i) {
      if (info.strides[i] != expected_stride) {
        std::ostringstream oss;
        oss << arg_name << " must be C-contiguous, but strides=[";
        for (ssize_t d = 0; d < info.ndim; ++d) {
          if (d > 0) oss << ", ";
          oss << info.strides[d];
        }
        oss << "] are not C-contiguous for shape=[";
        for (ssize_t d = 0; d < info.ndim; ++d) {
          if (d > 0) oss << ", ";
          oss << info.shape[d];
        }
        oss << "]";
        throw std::invalid_argument(oss.str());
      }
      expected_stride *= info.shape[i];
    }
  }

  // --- Capacity check ---
  size_t total_bytes = info.size * info.itemsize;
  if (arr_byte_offset + length > total_bytes) {
    std::ostringstream oss;
    oss << arg_name << " capacity (" << total_bytes << " bytes) is too small "
        << "for write starting at byte offset " << arr_byte_offset
        << " of length " << length
        << " (needs " << (arr_byte_offset + length) << " bytes)";
    throw std::invalid_argument(oss.str());
  }
}

// -----------------------------------------------------------------------
// Core function
// -----------------------------------------------------------------------

/// Read *length* bytes from *fd* at *file_offset* directly into a writable
/// buffer at *arr_byte_offset*.
///
/// Uses the POSIX ``pread(2)`` system call which atomically seeks and reads
/// without changing the file offset, making it safe for concurrent access.
///
/// Returns the number of bytes read (always == *length* on success).
/// Raises RuntimeError on short read or I/O error.
/// Raises ValueError if the buffer is not uint8, not contiguous, read-only,
/// or too small.
static py::int_ pread_into_array_impl(
    int fd,
    int64_t file_offset,
    int64_t length,
    py::buffer buf,
    int64_t arr_byte_offset) {

  // Validate input sizes (Python ints may overflow on 32-bit, but we're on
  // 64-bit Apple Silicon).
  if (file_offset < 0) {
    throw std::invalid_argument("file_offset must be >= 0");
  }
  if (length < 0) {
    throw std::invalid_argument("length must be >= 0");
  }
  if (arr_byte_offset < 0) {
    throw std::invalid_argument("arr_byte_offset must be >= 0");
  }

  // For zero-length requests, return 0 immediately (no-op).
  if (length == 0) {
    return py::int_(0);
  }

  size_t ulength = static_cast<size_t>(length);
  size_t uarr_off = static_cast<size_t>(arr_byte_offset);

  // Request buffer info.  This acquires a lock on the buffer protocol;
  // the pointer is valid for the lifetime of the info object.
  py::buffer_info info = buf.request();

  // Validate buffer properties.
  validate_buffer(info, uarr_off, ulength, "arr");

  // Obtain the raw pointer to the buffer data.
  uint8_t* raw_ptr = static_cast<uint8_t*>(info.ptr) + uarr_off;

  // Perform the pread -- direct NVMe -> GPU buffer, zero intermediate copies.
  ssize_t nread = pread(fd, static_cast<void*>(raw_ptr), ulength,
                        static_cast<off_t>(file_offset));

  // Error handling
  if (nread < 0) {
    std::ostringstream oss;
    oss << "pread failed on fd=" << fd
        << " (file_offset=" << file_offset
        << ", length=" << length
        << "): " << strerror(errno);
    throw std::runtime_error(oss.str());
  }

  if (static_cast<size_t>(nread) < ulength) {
    std::ostringstream oss;
    oss << "Short read in pread_into_array: expected " << length
        << " bytes, got " << nread
        << " (fd=" << fd << ", file_offset=" << file_offset << ").";
    throw std::runtime_error(oss.str());
  }

  return py::int_(nread);
}

// -----------------------------------------------------------------------
// pybind11 module registration
// -----------------------------------------------------------------------

PYBIND11_MODULE(_buffer_access, m) {
  m.doc() = "Zero-copy pread into buffer-backed arrays via Python buffer protocol.";

  m.def(
      "pread_into_array",
      &pread_into_array_impl,
      py::arg("fd"),
      py::arg("file_offset"),
      py::arg("length"),
      py::arg("arr").noconvert(),
      py::arg("arr_byte_offset") = 0,
      R"pbdoc(
Read *length* bytes from *fd* at *file_offset* directly into a writable
buffer's backing memory at *arr_byte_offset*.

Uses ``pread(2)`` -- no intermediate allocations, no Python heap involvement.

Parameters
----------
fd : int
    File descriptor (open with O_RDONLY, ideally F_NOCACHE).
file_offset : int
    Byte offset in the file to begin reading.
length : int
    Number of bytes to read.
arr : buffer-like
    A writable Python buffer (e.g. mx.array, bytearray, memoryview,
    numpy.ndarray) of dtype=uint8, C-contiguous.  For ``mx.array``,
    call ``mx.eval(arr)`` before passing.
arr_byte_offset : int
    Byte offset within *arr*'s data to begin writing at.  Default 0.

Returns
-------
int
    Number of bytes read (always equals *length* on success).

Raises
------
ValueError
    If *arr* is not uint8, not writable, not C-contiguous, or too small.
RuntimeError
    If pread fails or returns fewer bytes than *length*.
)pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "0.1.0";
#endif
}
