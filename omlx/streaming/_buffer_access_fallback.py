# SPDX-License-Identifier: Apache-2.0
"""Pure Python fallback for direct I/O into MLX array GPU buffers.

Provides ``pread_into_array()`` which reads bytes from an fd and writes them
directly into the backing storage of a pre-allocated ``mx.array``.

The C++ extension version (``_buffer_access.cpp``) writes via a raw pointer
into the Metal buffer with zero intermediate allocations.  This fallback uses
``os.pread()`` + memoryview slice assignment, which allocates a temporary
Python ``bytes`` object per call.  Correct but not optimal — the C++ version
is preferred for the hot inference path.
"""

from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)


def pread_into_array(
    fd: int,
    file_offset: int,
    length: int,
    arr,  # mx.array, dtype=uint8, contiguous, already eval'd
    arr_byte_offset: int = 0,
) -> int:
    """Read *length* bytes from *fd* into an MLX array's backing buffer.

    This is the pure Python fallback when the C++ pybind11 extension is not
    available.  It allocates a temporary ``bytes`` object for the intermediate
    read, then copies into the array via the Python buffer protocol.

    The C++ version writes directly into the Metal buffer with no intermediate
    allocation, which matters in the per-token hot path where we stream 10+ GB
    of expert weights through NVMe.

    Parameters
    ----------
    fd :
        File descriptor to read from (should be opened with ``F_NOCACHE``).
    file_offset :
        Byte offset within the file to begin reading.
    length :
        Number of bytes to read.
    arr :
        Pre-allocated ``mx.array`` (dtype=uint8, contiguous) whose backing
        buffer will receive the bytes.  Must already be eval'd so the Metal
        buffer exists.
    arr_byte_offset :
        Byte offset within *arr*'s data to begin writing at.  Default 0.

    Returns
    -------
    int
        Number of bytes read (equals *length* on success).

    Raises
    ------
    RuntimeError
        If the read returns fewer bytes than *length* (short read / I/O error).
    """
    # Single pread into a temporary bytes object, then copy into the array's
    # buffer via memoryview.  The memoryview on mx.array is writable and
    # supports the Python buffer protocol (mlx PR #323).
    raw = os.pread(fd, length, file_offset)
    nread = len(raw)

    if nread < length:
        raise RuntimeError(
            f"Short read in pread_into_array: expected {length} bytes, got {nread} "
            f"(fd={fd}, offset={file_offset})."
        )

    # Write into the array's backing buffer via writable memoryview.
    # mx.array supports the Python buffer protocol (mlx PR #323), but slice
    # assignment is only supported on 1D views.  Cast to a flat 1D view so
    # the slot bank can use 2D arrays like (num_slots, expert_bytes).
    mv = memoryview(arr).cast("B", shape=(arr.nbytes,))
    mv[arr_byte_offset : arr_byte_offset + nread] = raw

    return nread


def _emit_once() -> None:
    """Emit a one-shot warning that the C++ extension is not available.

    Note: This warning is also issued by ``__init__.py`` when the C++ extension
    fails to import.  The fallback module avoids duplicating the warning when
    imported directly.
    """
    pass
