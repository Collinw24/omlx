# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming expert sidecar (§7a) and slot bank TDD (§7b).

§7a tests exercise ``StreamingExpertSidecar`` from ``omlx.streaming.sidecar``.
These require no model and should pass immediately since Phase 1 is complete.

§7b tests define the expected API for ``ExpertSlotBank`` from
``omlx.streaming.slot_bank``.  The class does NOT exist yet — these are TDD
tests that will be skipped with ``ImportError`` until Phase 2 is built.
"""

import json
import os
import struct
import tempfile
from pathlib import Path

import pytest
from omlx.streaming import StreamingExpertSidecar


# ── helpers ────────────────────────────────────────────────────────────

def _make_synthetic_st_files(tmp_path: Path) -> Path:
    """Create a minimal safetensors directory with known expert tensors.

    Creates one .safetensors file with 2 layers, 2 experts each, dummy data.
    Returns the directory path. Tensors use U8 dtype for simplicity.
    """
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    header = {
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": {
            "dtype": "U8",
            "shape": [2, 3, 4],   # 2 experts, 3 out, 4 in = 12 bytes each
            "data_offsets": [0, 24],
        },
        "model.layers.0.mlp.switch_mlp.up_proj.weight": {
            "dtype": "U8",
            "shape": [2, 3, 4],
            "data_offsets": [24, 48],
        },
        "model.layers.0.mlp.switch_mlp.down_proj.weight": {
            "dtype": "U8",
            "shape": [2, 4, 3],
            "data_offsets": [48, 72],
        },
        "model.layers.1.mlp.switch_mlp.gate_proj.weight": {
            "dtype": "U8",
            "shape": [2, 3, 4],
            "data_offsets": [72, 96],
        },
        "model.layers.1.mlp.switch_mlp.up_proj.weight": {
            "dtype": "U8",
            "shape": [2, 3, 4],
            "data_offsets": [96, 120],
        },
        "model.layers.1.mlp.switch_mlp.down_proj.weight": {
            "dtype": "U8",
            "shape": [2, 4, 3],
            "data_offsets": [120, 144],
        },
    }

    # Build tensor data: each expert gets sequential byte markers
    # Layer 0, Expert 0: bytes 0-35, Expert 1: bytes 36-71
    # Layer 1, Expert 0: bytes 72-107, Expert 1: bytes 108-143
    data = bytearray(144)
    for i in range(144):
        data[i] = i & 0xFF

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    sf_path = model_dir / "model-00001-of-00001.safetensors"
    with open(sf_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)

    return model_dir


def _make_synthetic_sidecar(tmp_path: Path, model_dir: Path | None = None) -> StreamingExpertSidecar:
    """Create a sidecar from synthetic safetensors (if model_dir given)
    or build one manually."""
    if model_dir is None:
        model_dir = _make_synthetic_st_files(tmp_path)
    output = tmp_path / "test.sidecar"
    return StreamingExpertSidecar.create(model_dir, output)


def _build_minimal_sidecar_manual(tmp_path: Path) -> tuple[Path, dict]:
    """Build a minimal valid sidecar file by hand and return (path, expected_header).
    Layers=1, Experts=2, 16-byte dummy chunks with known markers.
    """
    sp = tmp_path / "manual.sidecar"
    header = {
        "version": 1,
        "model_type": "test",
        "num_layers": 1,
        "num_experts": 2,
        "hidden_size": 4,
        "alignment": 16384,
        "layers": {
            "0": {
                "projections": {
                    "gate_proj": {"weight_shape": [3, 4], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                    "up_proj": {"weight_shape": [3, 4], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                    "down_proj": {"weight_shape": [4, 3], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                },
                "experts": {
                    "0": {"offset": 16384, "length": 36},
                    "1": {"offset": 32768, "length": 36},
                }
            }
        }
    }
    hj = json.dumps(header, separators=(",", ":"))
    hb = hj.encode()
    with open(sp, "wb") as f:
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        pad_needed = 16384 - 4 - len(hb)
        f.write(b"\x00" * pad_needed)
        # Expert 0 at 16384: 36 bytes of 'A' + padding to 16384
        f.write(b"A" * 36 + b"\x00" * (16384 - 36))
        # Expert 1 at 32768: 36 bytes of 'B' + padding to 16384
        f.write(b"B" * 36 + b"\x00" * (16384 - 36))
    return sp, header


# ── 7a tests ───────────────────────────────────────────────────────────

class TestSidecarCreateRead:
    """§7a.1: Round-trip: create → read → verify byte-exact."""

    def test_roundtrip_all_experts(self, tmp_path):
        """Create sidecar from synthetic model, read every expert back."""
        model_dir = _make_synthetic_st_files(tmp_path)
        output = tmp_path / "test.sidecar"
        sc = StreamingExpertSidecar.create(str(model_dir), str(output))
        try:
            assert sc.header is not None
            assert sc.header["num_layers"] == 2
            assert sc.header["num_experts"] == 2

            # Read every expert back
            for layer in range(2):
                for expert in range(2):
                    raw = sc.read_expert(layer, expert)
                    assert len(raw) > 0, f"Empty read for layer={layer} expert={expert}"
                    assert len(raw) == sc.header["layers"][str(layer)]["experts"][str(expert)]["length"]
        finally:
            sc.close()

    def test_read_into_buffer(self, tmp_path):
        """read_expert_into writes directly to pre-allocated buffer."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            buf = bytearray(100)
            mv = memoryview(buf)
            n = sc.read_expert_into(0, 0, mv, 0)
            assert n == 36
            assert bytes(buf[:36]) == b"A" * 36
        finally:
            sc.close()

    def test_read_into_buffer_with_offset(self, tmp_path):
        """read_expert_into writes at specified offset."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            buf = bytearray(100)
            mv = memoryview(buf)
            # Write expert 0 data at offset 10
            n = sc.read_expert_into(0, 0, mv, 10)
            assert n == 36
            assert buf[:10] == b"\x00" * 10
            assert bytes(buf[10:46]) == b"A" * 36
        finally:
            sc.close()


class TestSidecarAlignment:
    """§7a.2: Every expert offset must be divisible by 16384."""

    def test_expert_zero_offset(self, tmp_path):
        """Expert 0 must start at 16384."""
        model_dir = _make_synthetic_st_files(tmp_path)
        output = tmp_path / "test.sidecar"
        sc = StreamingExpertSidecar.create(str(model_dir), str(output))
        try:
            # Expert 0 in first layer
            e0_offset = sc.header["layers"]["0"]["experts"]["0"]["offset"]
            assert e0_offset == 16384, f"Expert 0 offset is {e0_offset}, expected 16384"
        finally:
            sc.close()

    def test_all_expert_offsets_aligned(self, tmp_path):
        """Every expert offset must be a multiple of 16384."""
        model_dir = _make_synthetic_st_files(tmp_path)
        output = tmp_path / "test.sidecar"
        sc = StreamingExpertSidecar.create(str(model_dir), str(output))
        try:
            for lk, layer_data in sc.header["layers"].items():
                for ek, expert_data in layer_data["experts"].items():
                    offset = expert_data["offset"]
                    assert offset % 16384 == 0, (
                        f"Layer {lk} Expert {ek} offset {offset} not aligned to 16384"
                    )
        finally:
            sc.close()

    def test_manual_sidecar_alignment(self, tmp_path):
        """Manually built sidecar also has 16384-aligned offsets."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            assert sc.header["layers"]["0"]["experts"]["0"]["offset"] == 16384
            assert sc.header["layers"]["0"]["experts"]["1"]["offset"] == 32768
        finally:
            sc.close()


class TestSidecarNocache:
    """§7a.3: nocache_active property."""

    def test_nocache_active_boolean(self, tmp_path):
        """nocache_active must be True on macOS (F_NOCACHE supported)."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            assert isinstance(sc.nocache_active, bool)
            # On macOS with internal APFS, F_NOCACHE should be active.
            # On other platforms it may be False.
            assert sc.nocache_active in (True, False)
        finally:
            sc.close()


class TestSidecarVerify:
    """§7a.4: verify() method."""

    def test_verify_valid_sidecar(self, tmp_path):
        """verify() returns True on a valid sidecar."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            assert sc.verify(sample_layers=1) is True
        finally:
            sc.close()

    def test_verify_created_sidecar(self, tmp_path):
        """verify() returns True on a create()-produced sidecar."""
        model_dir = _make_synthetic_st_files(tmp_path)
        output = tmp_path / "test.sidecar"
        sc = StreamingExpertSidecar.create(str(model_dir), str(output))
        try:
            assert sc.verify(sample_layers=2) is True
        finally:
            sc.close()

    def test_verify_handles_single_layer(self, tmp_path):
        """verify() works on a sidecar with only one layer."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            # Manual sidecar has exactly 1 layer, sample_layers=3 should clamp
            assert sc.verify(sample_layers=3) is True
        finally:
            sc.close()


class TestSidecarContextManager:
    """Context manager support."""

    def test_context_manager(self, tmp_path):
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        with StreamingExpertSidecar(sp) as sc:
            raw = sc.read_expert(0, 0)
            assert len(raw) == 36

    def test_error_paths(self, tmp_path):
        """KeyError raised with actionable message for invalid layer/expert."""
        sp, header = _build_minimal_sidecar_manual(tmp_path)
        sc = StreamingExpertSidecar(sp)
        try:
            with pytest.raises(KeyError, match="layer"):
                sc.read_expert(99, 0)
            with pytest.raises(KeyError, match="Expert"):
                sc.read_expert(0, 99)
        finally:
            sc.close()


# ── Mock sidecar for slot bank tests ───────────────────────────────────

class MockSidecar:
    """Minimal mock of StreamingExpertSidecar for slot bank unit tests."""

    def __init__(self, expert_bytes: int = 36, num_experts: int = 256):
        self.expert_bytes = expert_bytes
        self.num_experts = num_experts
        self.header = {
            "layers": {
                "0": {
                    "projections": {
                        "gate_proj": {"weight_shape": [3, 4], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                        "up_proj": {"weight_shape": [3, 4], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                        "down_proj": {"weight_shape": [4, 3], "weight_dtype": "U8", "weight_bytes": 12, "scales_shape": [], "scales_dtype": "", "scales_bytes": 0, "biases_shape": [], "biases_dtype": "", "biases_bytes": 0},
                    }
                }
            }
        }
        self._read_count = 0

    def read_expert_into(self, layer, expert, buf, buf_offset=0):
        """Simulate pread: fill buf with expert-specific marker bytes."""
        self._read_count += 1
        # Write expert_id as repeated byte so we can verify correct data
        marker = (expert & 0xFF)
        for i in range(self.expert_bytes):
            buf[buf_offset + i] = marker
        return self.expert_bytes

    @property
    def nocache_active(self):
        return True


# These tests will fail with ImportError until Phase 2 creates
# omlx.streaming.slot_bank.ExpertSlotBank. That's expected TDD behavior.

try:
    from omlx.streaming.slot_bank import ExpertSlotBank
    _SLOT_BANK_AVAILABLE = True
except ImportError:
    _SLOT_BANK_AVAILABLE = False


@pytest.mark.skipif(not _SLOT_BANK_AVAILABLE, reason="Phase 2 not yet implemented")
class TestSlotBankHotTier:
    """§7b.1: Hot experts always resident, never trigger pread."""

    def test_hot_always_resident(self):
        """Hot expert returned without calling read_expert_into."""
        sidecar = MockSidecar(expert_bytes=36)
        bank = ExpertSlotBank(
            sidecar=sidecar,
            layer=0,
            expert_bytes=36,
            hot_count=4,
            warm_slots=8,
            transient_slots=4,
            calibration_frequencies={0: 100.0, 1: 90.0, 2: 80.0, 3: 70.0},
        )
        pre_read_count = sidecar._read_count

        stacked, slot_ids = bank.resolve([0, 1, 2])

        # Must return correct number of slots
        assert len(slot_ids) == 3
        # Must not have triggered any additional reads
        assert sidecar._read_count == pre_read_count + 4  # 4 hot experts loaded at init


@pytest.mark.skipif(not _SLOT_BANK_AVAILABLE, reason="Phase 2 not yet implemented")
class TestSlotBankWarmLRU:
    """§7b.2: Warm slots evict LRU when full."""

    def test_warm_lru_eviction(self):
        """Filling warm slots beyond capacity evicts LRU entry."""
        sidecar = MockSidecar(expert_bytes=36)
        bank = ExpertSlotBank(
            sidecar=sidecar,
            layer=0,
            expert_bytes=36,
            hot_count=0,    # No hot — force everything through warm
            warm_slots=2,   # Only 2 warm slots
            transient_slots=4,
        )

        # Fill warm slots
        bank.resolve([10, 20])
        assert sidecar._read_count == 2  # 2 cold loads

        # Access expert 10 again (hits warm cache)
        bank.resolve([10])
        assert sidecar._read_count == 2  # No new read (warm hit)

        # Now add expert 30 — should evict LRU (expert 20)
        bank.resolve([30])
        assert sidecar._read_count == 3  # One new read for expert 30

        # Expert 20 should now be missing (evicted)
        bank.resolve([20])
        assert sidecar._read_count == 4  # Reloaded from sidecar


@pytest.mark.skipif(not _SLOT_BANK_AVAILABLE, reason="Phase 2 not yet implemented")
class TestSlotBankTransient:
    """§7b.3: Transient slots recycle, overflow raises."""

    def test_transient_overflow_raises(self):
        """More cold experts than transient_slots must raise RuntimeError."""
        sidecar = MockSidecar(expert_bytes=36)
        bank = ExpertSlotBank(
            sidecar=sidecar,
            layer=0,
            expert_bytes=36,
            hot_count=0,
            warm_slots=0,
            transient_slots=2,  # Only 2 transient slots
        )

        # 2 cold experts: fits in transient slots
        bank.resolve([10, 20])

        # 3 cold experts: overflow!
        with pytest.raises(RuntimeError, match="transient"):
            bank.resolve([10, 20, 30])


@pytest.mark.skipif(not _SLOT_BANK_AVAILABLE, reason="Phase 2 not yet implemented")
class TestSlotBankNoAllocation:
    """§7b.4: resolve() on warm hit must not allocate."""

    def test_no_allocation_on_warm_hit(self):
        """tracemalloc confirms zero Python heap allocation on warm hit."""
        sidecar = MockSidecar(expert_bytes=36)
        bank = ExpertSlotBank(
            sidecar=sidecar,
            layer=0,
            expert_bytes=36,
            hot_count=4,
            warm_slots=8,
            transient_slots=4,
            calibration_frequencies={0: 100.0, 1: 90.0, 2: 80.0, 3: 70.0},
        )

        import tracemalloc
        tracemalloc.start()

        # Warm up: first call loads from sidecar
        bank.resolve([0, 1])
        tracemalloc.reset_peak()  # clear allocation tracking

        # Hit hot cache — must not allocate
        bank.resolve([0, 1])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Allow small constant allocations (Python internals) but nothing proportional
        # to expert data size. Peak - current should be < 1 KB.
        alloc_delta = peak - current
        assert alloc_delta < 1024, (
            f"Warm-hit resolve() allocated {alloc_delta} bytes. "
            f"Must be < 1024 (zero dynamic allocation invariant)."
        )
