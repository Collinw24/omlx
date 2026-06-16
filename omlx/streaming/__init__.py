# SPDX-License-Identifier: Apache-2.0
"""
Streaming module - SSD expert streaming for MoE models.

This package enables running large Mixture-of-Experts models on machines with
limited unified memory by streaming only the top-K active expert weights per
token from NVMe SSD.
"""

from .sidecar import StreamingExpertSidecar, create_sidecar

__all__ = [
    "StreamingExpertSidecar",
    "create_sidecar",
]
