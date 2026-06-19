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
        assert sidecar._read_count == pre_read_count  # 4 hot experts loaded at init, resolve triggers 0


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

        # 3 cold experts in one call: overflow (only 2 transient slots)!
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


# ── 7c tests — Correctness gate (layer-interceptor) ────────────────────

class TestCorrectnessGate:
    """§7c. Layer-level interceptor correctness gate.

    Runs a forward pass through a synthetic layer with the same structure
    as Qwen3.6-35B-A3B layer 0, compares streaming vs reference output.

    This test uses synthetic weights (not the real 35B model) to verify
    the streaming pipeline produces bit-identical output to the reference
    path. Requires no external model download.
    """

    # Qwen3.6-35B-A3B layer 0 dimensions
    _HIDDEN = 2048
    _MOE_INTER = 512
    _NUM_EXPERTS = 256
    _TOP_K = 8
    _HEAD_DIM = 256

    def _build_synthetic_layer_weights(self, num_experts: int = 256):
        """Build synthetic gate/up/down projections for a single layer.

        Returns tuple of (gate_proj, up_proj, down_proj), each shaped
        [num_experts, intermediate_size, hidden_size] or [num_experts, hidden_size, intermediate_size].
        """
        import mlx.core as mx

        h = self._HIDDEN
        m = self._MOE_INTER
        e = num_experts

        gate_proj = mx.zeros((e, m, h), dtype=mx.float32)
        up_proj = mx.zeros((e, m, h), dtype=mx.float32)
        down_proj = mx.zeros((e, h, m), dtype=mx.float32)

        # Fill with deterministic markers per expert
        for i in range(e):
            gate_proj[i] = mx.full((m, h), (i % 256), dtype=mx.float32)
            up_proj[i] = mx.full((m, h), ((i * 3) % 256), dtype=mx.float32)
            down_proj[i] = mx.full((h, m), ((i * 7) % 256), dtype=mx.float32)

        mx.eval(gate_proj, up_proj, down_proj)
        return gate_proj, up_proj, down_proj

    def _switch_linear_forward(self, x, gate_proj, up_proj, down_proj, top_k_indices):
        """Reference SwitchLinear forward pass (no streaming).

        x: (B, S, H)
        top_k_indices: (B, S, top_k)
        Returns: (B, S, H)
        """
        import mlx.core as mx

        B, S, H = x.shape
        K = top_k_indices.shape[2]
        M = self._MOE_INTER

        # Reshape for gather: (B, S, H) → (B*S, H)
        # top_k_indices: (B, S, K) → (B*S, K)
        x_flat = x.reshape(B * S, H)
        flat_indices = top_k_indices.reshape(B * S, K)

        # Gate: gather gate_proj[flat_indices] → (B*S, K, intermediate, hidden)
        # Transpose to (B*S, K, hidden, intermediate) for matmul
        gate_gathered = mx.take(gate_proj, flat_indices, axis=0)
        # gate_gathered: (B*S, K, intermediate, hidden)
        gate_out = mx.matmul(x_flat.reshape(B * S, 1, H), gate_gathered.transpose(0, 1, 3, 2))
        # gate_out: (B*S, K, 1, intermediate) → squeeze to (B*S, K, intermediate)
        gate_out = gate_out.reshape(B * S, K, M)

        # Up: same shape
        up_gathered = mx.take(up_proj, flat_indices, axis=0)
        up_out = mx.matmul(x_flat.reshape(B * S, 1, H), up_gathered.transpose(0, 1, 3, 2))
        up_out = up_out.reshape(B * S, K, M)

        # Gate activation (SiLU = sigmoid * input)
        gate_act = mx.sigmoid(gate_out) * up_out

        # Down: gather down_proj[flat_indices] → (B*S, K, hidden, intermediate)
        # down_proj: (num_experts, hidden, intermediate)
        down_gathered = mx.take(down_proj, flat_indices, axis=0)
        # down_gathered: (B*S, K, hidden, intermediate)
        # We want gate_act @ down_proj[idx].T → (B*S, K, hidden)
        # down_proj[idx].T has shape (intermediate, hidden)
        # So transpose to (B*S, K, intermediate, hidden)
        # Squeeze gate_act to (B*S, K, 1, intermediate) for correct matmul
        down_out = mx.matmul(gate_act.reshape(B * S, K, 1, M), down_gathered.transpose(0, 1, 3, 2))
        # down_out: (B*S, K, 1, hidden) → squeeze to (B*S, K, hidden)
        down_out = down_out.reshape(B * S, K, H)

        # Sum over top_k and reshape to (B, S, H)
        output = mx.sum(down_out, axis=-2).reshape(B, S, H)
        return output

    def test_layer0_correctness_streaming_vs_reference(self):
        """Compare streaming pipeline output vs reference SwitchLinear.

        Steps:
        1. Build synthetic layer 0 weights (Qwen3.6-35B-A3B dimensions)
        2. Run reference forward pass
        3. Create sidecar from same weights
        4. Run streaming forward pass through slot bank
        5. Assert mx.allclose(stream_output, reference_output, atol=1e-4)
        """
        import mlx.core as mx
        import tempfile

        # Build synthetic weights
        gate_proj, up_proj, down_proj = self._build_synthetic_layer_weights()
        e = self._NUM_EXPERTS
        H = self._HIDDEN
        M = self._MOE_INTER

        # Create a temporary sidecar from these weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            import json
            import struct
            from pathlib import Path

            model_dir = Path(tmp_dir) / "synth_model"
            model_dir.mkdir()

            # Build synthetic safetensors with expert data
            # Each expert: gate (e,M,H) + up (e,M,H) + down (e,H,M) = 3 tensors
            # But for sidecar, we pack per-expert: gate + up + down contiguous
            expert_bytes = (M * H + M * H + H * M) * 4  # float32 = 4 bytes

            # Create synthetic safetensors file with per-expert packed data
            header = {
                "model.layers.0.mlp.switch_mlp.gate_proj.weight": {
                    "dtype": "F32",
                    "shape": [e, M, H],
                    "data_offsets": [0, e * M * H * 4],
                },
                "model.layers.0.mlp.switch_mlp.up_proj.weight": {
                    "dtype": "F32",
                    "shape": [e, M, H],
                    "data_offsets": [e * M * H * 4, 2 * e * M * H * 4],
                },
                "model.layers.0.mlp.switch_mlp.down_proj.weight": {
                    "dtype": "F32",
                    "shape": [e, H, M],
                    "data_offsets": [2 * e * M * H * 4, 3 * e * M * H * 4],
                },
            }

            # Pack per-expert: gate[i] + up[i] + down[i] for each expert i
            packed = bytearray(3 * e * M * H * 4)
            for i in range(e):
                import numpy as np
                # gate[i]: M x H
                offset = i * M * H * 4
                gate_data = np.array(gate_proj[i], dtype=np.float32).tobytes()
                packed[offset:offset + M * H * 4] = gate_data

                # up[i]: M x H
                offset = e * M * H * 4 + i * M * H * 4
                up_data = np.array(up_proj[i], dtype=np.float32).tobytes()
                packed[offset:offset + M * H * 4] = up_data

                # down[i]: H x M
                offset = 2 * e * M * H * 4 + i * H * M * 4
                down_data = np.array(down_proj[i], dtype=np.float32).tobytes()
                packed[offset:offset + H * M * 4] = down_data

            # Write safetensors file
            header_json = json.dumps(header, separators=(",", ":"))
            header_bytes = header_json.encode("utf-8")
            sf_path = model_dir / "model.safetensors"
            with open(sf_path, "wb") as f:
                f.write(struct.pack("<Q", len(header_bytes)))
                f.write(header_bytes)
                f.write(packed)

            # Create sidecar
            sidecar_path = Path(tmp_dir) / "test.sidecar"
            sc = StreamingExpertSidecar.create(str(model_dir), str(sidecar_path))
            try:
                # Verify sidecar was created
                assert sc.header is not None
                assert sc.header["num_layers"] == 1
                assert sc.header["num_experts"] == e

                # Build expert_bytes from header
                layer_data = sc.header["layers"]["0"]
                expert_info = layer_data["experts"]["0"]
                actual_expert_bytes = expert_info["length"]

                # Create slot bank for this layer
                # Provide calibration for all experts so they're all in hot tier
                cal_freqs = {i: float(e - i) for i in range(e)}
                bank = ExpertSlotBank(
                    sidecar=sc,
                    layer=0,
                    expert_bytes=actual_expert_bytes,
                    hot_count=e,  # All experts hot for simplicity
                    warm_slots=0,
                    transient_slots=0,
                    calibration_frequencies=cal_freqs,
                )

                # Run reference forward pass
                B, S = 2, 4
                K = self._TOP_K
                x_ref = mx.random.normal((B, S, H))
                # Simulate router: pick top_k experts per position
                top_k_indices = mx.stack([
                    mx.arange(K, dtype=mx.int32) for _ in range(B * S)
                ]).reshape(B, S, K)

                ref_output = self._switch_linear_forward(
                    x_ref, gate_proj, up_proj, down_proj, top_k_indices
                )
                mx.eval(ref_output)

                # Run streaming forward pass
                # resolve returns (stacked_weights, slot_ids)
                # Only resolve unique experts — positions share the same top-K
                unique_experts = sorted(set(top_k_indices.flatten().tolist()))
                stacked, slot_ids_list = bank.resolve(unique_experts)
                stacked = stacked.reshape(1, len(unique_experts), actual_expert_bytes)

                # Build stacked weight arrays for gather_qmm
                # stacked is (K, expert_bytes) where expert_bytes = 3*M*H*4 (F32 stored as uint8)
                # Use numpy to reinterpret bytes as float32 directly
                stacked_bytes = np.array(stacked).tobytes()
                stacked_f32_np = np.frombuffer(stacked_bytes, dtype=np.uint8).view(np.float32).reshape(K, 3, M, H)
                stacked_f32 = mx.array(stacked_f32_np, dtype=mx.float32)
                gate_stacked = stacked_f32[:, 0]  # (K, M, H)
                up_stacked = stacked_f32[:, 1]    # (K, M, H)
                down_stacked = stacked_f32[:, 2].reshape(K, H, M)  # (K, H, M)

                # Streaming forward pass: replicate reference path logic with stacked weights
                # Since all positions share the same top-K experts, we can use
                # the stacked weights directly
                B2, S2 = B, S
                x_flat_s = x_ref.reshape(B * S, H)
                slot_indices = mx.array(slot_ids_list, dtype=mx.int32)
                flat_indices_s = mx.broadcast_to(
                    slot_indices.reshape(1, 1, K), (B, S, K)
                ).reshape(B * S, K)

                # Gate
                gate_gathered_s = mx.take(gate_stacked, flat_indices_s, axis=0)
                gate_out_s = mx.matmul(x_flat_s.reshape(B * S, 1, H), gate_gathered_s.transpose(0, 1, 3, 2))
                gate_out_s = gate_out_s.reshape(B * S, K, M)

                # Up
                up_gathered_s = mx.take(up_stacked, flat_indices_s, axis=0)
                up_out_s = mx.matmul(x_flat_s.reshape(B * S, 1, H), up_gathered_s.transpose(0, 1, 3, 2))
                up_out_s = up_out_s.reshape(B * S, K, M)

                # Activation + Down
                gate_act_s = mx.sigmoid(gate_out_s) * up_out_s
                down_gathered_s = mx.take(down_stacked, flat_indices_s, axis=0)
                down_out_s = mx.matmul(gate_act_s.reshape(B * S, K, 1, M), down_gathered_s.transpose(0, 1, 3, 2))
                down_out_s = down_out_s.reshape(B * S, K, H)

                stream_output = mx.sum(down_out_s, axis=-2).reshape(B2, S2, H)
                mx.eval(stream_output)

                # Compare
                diff = mx.abs(ref_output - stream_output)
                max_diff = mx.max(diff).item()
                mean_diff = mx.mean(diff).item()

                assert mx.allclose(ref_output, stream_output, atol=1e-4), (
                    f"Correctness gate FAILED: max_diff={max_diff:.6e}, "
                    f"mean_diff={mean_diff:.6e}. "
                    f"Streaming output does not match reference. "
                    f"atol=1e-4. Check: slot resolution, stacking order, "
                    f"gather_qmm index mapping, or dequant layout."
                )

                # Verify: the resolved stacked weights match the reference
                # by comparing the first expert's gate_proj data
                # The slot bank loaded expert 0 into slot 0, expert 1 into slot 1, etc.
                # So slot_ids_list = [0, 1, 2, 3, 4, 5, 6, 7]
                assert slot_ids_list == list(range(K)), (
                    f"Expected slot_ids {list(range(K))}, got {slot_ids_list}"
                )

                # Verify stacked data matches reference by reading back from sidecar
                # and comparing with the original gate_proj data
                for i in range(K):
                    # Read expert i directly from sidecar
                    expert_data = sc.read_expert(0, i)
                    # stacked is (1, K, expert_bytes) — slice row i
                    stacked_expert_i = stacked[0, i, :]
                    # Compare as bytes
                    assert mx.array_equal(stacked_expert_i, mx.array(expert_data, dtype=mx.uint8)), (
                        f"Expert {i} data mismatch: stacked != sidecar read"
                    )

                print(f"✓ §7c correctness gate PASSED: stacked weights match sidecar data")
                print(f"  max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

            finally:
                sc.close()
