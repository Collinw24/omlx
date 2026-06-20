# SPDX-License-Identifier: Apache-2.0
"""
Streaming module — SSD expert streaming for MoE models on Apple Silicon.

Architecture
============
Enables running massive Mixture-of-Experts models (e.g., Qwen3.5-397B-A17B,
~209 GB) on 48-64 GB unified memory by streaming only the top-K active
expert weights from NVMe SSD per token.

Key components:
  - ``StreamingExpertSidecar`` — F_NOCACHE direct I/O sidecar file
  - ``ExpertSlotBank`` — 3-tier GPU memory pool (hot/warm/transient)
  - ``_buffer_access`` — C++ pybind11 extension for zero-copy pread
  - ``patch.py`` — SwitchGLU monkey-patching with quantized gather_qmm
  - ``pipeline.py`` — Orchestrator, model loading, EMA prefetcher
  - ``config.py`` — StreamingConfig dataclass

Invariants
==========
  I1 — Zero dynamic allocation in hot path
  I2 — mx.eval(router_logits) gates slot mutation
  I3 — F_NOCACHE verified, never silently falls back
  I4 — mx.compile() prohibited on streaming path
  I5 — Completely inert when stream_experts=False
  I6 — Original .safetensors never mutated
"""

from __future__ import annotations

import logging
import warnings

from .sidecar import StreamingExpertSidecar, create_sidecar
from .slot_bank import ExpertSlotBank
from .patch import (
    patch_switch_linear,
    unpatch_switch_linear,
    apply_streaming_patches,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C++ buffer extension — zero-copy pread into array backing buffers
# ---------------------------------------------------------------------------
# The C++ extension writes bytes directly into the backing memory of any
# Python buffer-protocol object (mx.array, bytearray, memoryview, numpy
# ndarray) via pread(2), avoiding intermediate Python heap allocations.
#
# When the C++ extension is not available (e.g. CI or build failure), we
# fall back to the pure Python implementation which allocates a temporary
# bytes object per call.

try:
    from ._buffer_access import pread_into_array as _pread_into_array_cpp

    _USE_CPP_EXTENSION = True
except ImportError:
    from ._buffer_access_fallback import pread_into_array as _pread_into_array_cpp

    _USE_CPP_EXTENSION = False
    warnings.warn(
        "omlx streaming: C++ buffer extension not available, "
        "using Python fallback (allocates per-read. "
        "Build _buffer_access.so for zero-copy path).",
        RuntimeWarning,
        stacklevel=2,
    )


def pread_into_array(
    fd: int,
    file_offset: int,
    length: int,
    arr: object,
    arr_byte_offset: int = 0,
) -> int:
    """Read *length* bytes from *fd* at *file_offset* into *arr*.

    Uses the C++ extension when available for zero-copy writes into the
    backing buffer; falls back to pure Python otherwise.
    """
    return _pread_into_array_cpp(fd, file_offset, length, arr, arr_byte_offset)


from .config import StreamingConfig
from .pipeline import (
    EMATrajectoryPrefetcher,
    load_model_with_streaming,
    record_routing,
    prefetch_step,
    resolve_sidecar_path,
    streaming_forward_pass,
    unload_streaming,
)
from .sidecar import register_parser

__all__ = [
    "StreamingExpertSidecar",
    "create_sidecar",
    "pread_into_array",
    "StreamingConfig",
    "ExpertSlotBank",
    "patch_switch_linear",
    "unpatch_switch_linear",
    "apply_streaming_patches",
    "load_model_with_streaming",
    "resolve_sidecar_path",
    "streaming_forward_pass",
    "unload_streaming",
    "EMATrajectoryPrefetcher",
    "register_parser",
    "record_routing",
    "prefetch_step",
]
