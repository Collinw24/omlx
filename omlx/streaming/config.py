# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for SSD expert streaming.

Enables streaming MoE expert weights from NVMe for models too large to
fit in unified memory.  Feature is completely inert when
``stream_experts`` is ``False`` (default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StreamingConfig:
    """Configuration for SSD expert streaming.

    Parameters
    ----------
    stream_experts : bool
        Master switch.  When ``False`` (default) the entire streaming
        pipeline is disabled — zero overhead in the forward path.
    expert_sidecar_path : str, optional
        Path to the sidecar file.  When ``None`` (default) the pipeline
        auto-detects ``{model_dir}/{model_name}.streaming``.
    expert_hot_count : int
        Number of hot-tier slots (default 13).  These experts are pinned
        and never evicted.
    expert_warm_slots : int
        Number of warm-tier slots (default 64).  LRU + EMA eviction.
    expert_transient_slots : int
        Number of transient streaming slots (default 8).  Circular
        overwrite per token.
    expert_prefetch : bool
        Enable EMA-trajectory prefetching (default True).
    expert_prefetch_window : int
        EMA sliding window for frequency tracking (default 4).
    expert_top_k_override : int, optional
        Override the model's default top-K routing (e.g. for testing).
        ``None`` uses the model's native value.
    calibration_frequencies : dict[int, float], optional
        Pre-computed expert frequency map for hot-tier initialisation.
        When ``None``, slot banks use uniform distribution.
    """

    stream_experts: bool = False
    expert_sidecar_path: Optional[str] = None
    expert_hot_count: int = 13
    expert_warm_slots: int = 64
    expert_transient_slots: int = 8
    expert_prefetch: bool = True
    expert_prefetch_window: int = 4
    expert_top_k_override: Optional[int] = None
    calibration_frequencies: Optional[Dict[int, float]] = None

    @classmethod
    def from_dict(cls, d: dict) -> StreamingConfig:
        """Create config from a dictionary (e.g. parsed JSON config)."""
        return cls(
            stream_experts=d.get("stream_experts", False),
            expert_sidecar_path=d.get("expert_sidecar_path"),
            expert_hot_count=d.get("expert_hot_count", 13),
            expert_warm_slots=d.get("expert_warm_slots", 64),
            expert_transient_slots=d.get("expert_transient_slots", 8),
            expert_prefetch=d.get("expert_prefetch", True),
            expert_prefetch_window=d.get("expert_prefetch_window", 4),
            expert_top_k_override=d.get("expert_top_k_override"),
            calibration_frequencies=d.get("calibration_frequencies"),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary (for logging / status)."""
        return {
            "stream_experts": self.stream_experts,
            "expert_sidecar_path": self.expert_sidecar_path,
            "expert_hot_count": self.expert_hot_count,
            "expert_warm_slots": self.expert_warm_slots,
            "expert_transient_slots": self.expert_transient_slots,
            "expert_prefetch": self.expert_prefetch,
            "expert_prefetch_window": self.expert_prefetch_window,
            "expert_top_k_override": self.expert_top_k_override,
            "calibration_frequencies": self.calibration_frequencies,
        }
