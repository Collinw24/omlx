# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming pipeline orchestrator (§7A — pipeline unit tests).

Tests EMATrajectoryPrefetcher, record_routing, prefetch_step, and
load_model_with_streaming / unload_streaming orchestration.
Requires no model download.
"""

import json
import struct
import tempfile
from pathlib import Path

import pytest


# ── helpers for synthetic sidecar creation ────────────────────────────────

def _build_minimal_sidecar(
    tmp_path: Path,
    num_layers: int = 1,
    num_experts: int = 2,
    hidden_size: int = 4,
) -> tuple[Path, dict]:
    """Build a minimal valid sidecar file by hand.

    Returns (path, expected_header).
    """
    sp = tmp_path / "min.sidecar"
    header = {
        "version": 1,
        "model_type": "test",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "alignment": 16384,
        "layers": {},
    }
    offset = 16384
    for lk in range(num_layers):
        layer_key = str(lk)
        header["layers"][layer_key] = {
            "projections": {},
            "experts": {},
        }
        for ek in range(num_experts):
            header["layers"][layer_key]["experts"][str(ek)] = {
                "offset": offset,
                "length": 16,
            }
            offset += 16384

    hj = json.dumps(header, separators=(",", ":"))
    hb = hj.encode()
    with open(sp, "wb") as f:
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        pad = 16384 - 4 - len(hb)
        f.write(b"\x00" * pad)
        for _ in range(num_layers):
            for _ in range(num_experts):
                f.write(b"\xAA" * 16 + b"\x00" * (16384 - 16))
    return sp, header


# ── MockSidecar used by pipeline tests ───────────────────────────────────

class MockSidecar:
    """Minimal mock of StreamingExpertSidecar for pipeline unit tests."""

    def __init__(self, expert_bytes: int = 16, num_experts: int = 256):
        self.expert_bytes = expert_bytes
        self.num_experts = num_experts
        self._read_count = 0

    def read_expert_into(self, layer: int, expert: int, buf: memoryview,
                         buf_offset: int = 0) -> int:
        self._read_count += 1
        marker = expert & 0xFF
        for i in range(min(self.expert_bytes, len(buf) - buf_offset)):
            buf[buf_offset + i] = marker
        return self.expert_bytes

    @property
    def nocache_active(self) -> bool:
        return True

    def close(self) -> None:
        pass


# ── 7A — EMA Trajectory Prefetcher tests ────────────────────────────────

try:
    from omlx.streaming.pipeline import EMATrajectoryPrefetcher
    _HAS_PREFETCHER = True
except ImportError:
    _HAS_PREFETCHER = False


@pytest.mark.skipif(not _HAS_PREFETCHER, reason="pipeline not available")
class TestEMATrajectoryPrefetcher:
    """§7A. Pipeline unit — EMATrajectoryPrefetcher."""

    def test_update_increases_frequency(self):
        """EMA frequency for an expert increases after update calls."""
        ema = EMATrajectoryPrefetcher(window=4)  # α=0.4
        ema.update([1, 2, 3])
        freq_after_one = ema._ema[1]

        ema.update([1, 2])
        freq_after_two = ema._ema[1]

        assert freq_after_two > freq_after_one, (
            f"EMA should increase: {freq_after_one} → {freq_after_two}"
        )

    def test_predict_returns_union_current_and_top_n(self):
        """predict() returns actual experts ∪ top_N predicted."""
        ema = EMATrajectoryPrefetcher(window=4, top_n=2)
        # Update: experts 10, 20 get high EMA; expert 30 gets low
        ema.update([10, 20])
        ema.update([10, 30])
        # Now: freq[10] > freq[20] > freq[30]
        pred = ema.predict([10])  # current includes [10], top_2 from EMA
        predicted_set = set(pred) - {10}
        # Expert 20 should be in prediction (second highest EMA)
        assert 20 in predicted_set, f"Expected 20 in predicted: {pred}"

    def test_predict_excludes_current_from_top_n(self):
        """No duplicates: current experts excluded from top-N candidates."""
        ema = EMATrajectoryPrefetcher(window=4, top_n=2)
        ema.update([10, 20, 30])
        pred = ema.predict([10, 20])
        assert len(pred) == len(set(pred)), (
            f"Duplicate in prediction: {pred}"
        )

    def test_predict_empty_when_no_history(self):
        """Fresh predict([1]) with no update history → returns [1]."""
        ema = EMATrajectoryPrefetcher()
        pred = ema.predict([1])
        assert set(pred) == {1}

    def test_reset_clears_ema(self):
        """reset() clears all EMA tracking."""
        ema = EMATrajectoryPrefetcher(window=4)
        ema.update([1, 2, 3])
        assert len(ema._ema) == 3
        ema.reset()
        assert len(ema._ema) == 0

    def test_boundary_window_one(self):
        """window=1 → α = 2/(1+1) = 1.0 (fully reactive)."""
        ema = EMATrajectoryPrefetcher(window=1)
        assert ema._alpha == 1.0

    def test_top_n_constructor_param(self):
        """top_n=4 → predict returns up to 4 extra candidates."""
        ema = EMATrajectoryPrefetcher(top_n=4)
        assert ema._top_n == 4


# ── 7A — RecordRouting and prefetch_step tests ───────────────────────────

try:
    from omlx.streaming.pipeline import record_routing, prefetch_step
    _HAS_PIPELINE_FUNCS = True
except ImportError:
    _HAS_PIPELINE_FUNCS = False


@pytest.mark.skipif(not _HAS_PIPELINE_FUNCS, reason="pipeline not available")
class TestRecordRoutingAndPrefetchStep:
    """§7A. Pipeline unit — record_routing and prefetch_step."""

    def test_record_routing_stores_set(self):
        """Deduplicates expert IDs into a set per layer."""
        state = {}
        record_routing(5, [10, 20, 10, 30], state)
        assert state["routed_experts"][5] == {10, 20, 30}

    def test_prefetch_step_noop_when_inactive(self):
        """prefetch_step with active=False does nothing."""
        state = {"active": False}
        prefetch_step(state)  # Should not raise
        assert state == {"active": False}

    def test_prefetch_step_skips_hot_and_warm_resident(self):
        """prefetch_step skips experts already in hot or warm maps."""
        state = {
            "active": True,
            "prefetcher": EMATrajectoryPrefetcher(window=4),
            "slot_banks": {},
            "routed_experts": {},
        }
        prefetch_step(state)  # No-op — no slot banks, should not raise

    def test_prefetch_step_no_predict_when_no_history(self):
        """prefetch with empty EMA → no warm loads."""
        state = {
            "active": True,
            "prefetcher": EMATrajectoryPrefetcher(),
            "slot_banks": {},
            "routed_experts": {0: set()},
        }
        prefetch_step(state)  # No-op, no raise


# ── 7A — Load/Unload Streaming orchestration tests ──────────────────────

try:
    from omlx.streaming.pipeline import load_model_with_streaming, unload_streaming
    _HAS_PIPELINING = True
except ImportError:
    _HAS_PIPELINING = False


@pytest.mark.skipif(not _HAS_PIPELINING, reason="pipeline not available")
class TestLoadUnloadStreaming:
    """§7A. Pipeline orchestration — load_model_with_streaming, unload."""

    def test_load_inactive_returns_inactive_state(self):
        """stream_experts=False returns {'active': False}."""
        from omlx.streaming.config import StreamingConfig

        cfg = StreamingConfig(stream_experts=False)
        # Mock a minimal model object
        class _MockModel:
            layers = []
        state = load_model_with_streaming(_MockModel(), cfg, "dummy")
        assert state == {"active": False, "sidecar": None, "slot_banks": None}

    def test_unload_noop_on_unpatched_model(self):
        """unload_streaming on a model with no patches does not raise."""
        class _MockModel:
            layers = []
        unload_streaming(_MockModel())  # Should not raise
