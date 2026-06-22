# Streaming DeepSeek — Handoff Doc

## Status

- Branch: `feat/ssd-expert-streaming`
- Phase 0 integration complete: ModelSettings fields, engine_pool wiring, config defaults
- Tests: **24 passed**, 6 skipped (model-dependent), 0 known failures fixed
- Pre-existing failures: 7 unrelated (dflash_mlx missing, admin routes `_hf_uploader`, model settings profile classification)

## File Changes Summary

| File | Lines | Change |
|---|---|---|
| `tests/test_streaming_integration.py` | ~697 → ~699 | 2 test fixes |

## Fix 1: `test_transient_overflows_raise_when_warm_full`

**Problem:** The test used the shared `_make_bank()` (warm_slots=2), so cold experts always routed through warm buffer → transient overflow never fired.

**Root cause:** When `warm_slots > 0`, the cold path (slot_bank.py line 369) uses warm tier, bypassing transient entirely.

**Fix:** Replace with a local bank using `warm_slots=0` so cold → transient path fires. Pass 3 cold IDs in one `resolve()` call to exceed `transient_slots=1`.

**Exact location:** `tests/test_streaming_integration.py`, lines 199–218 (class `TestSlotBankLifecycle`)

**Before:**
```python
def test_transient_overflows_raise_when_warm_full(self):
    """resolve() raises RuntimeError when cold experts exceed transient_slots."""
    bank, sc = self._make_bank()  # warm_slots=2 — BUG: cold goes to warm
    bank.resolve([0])  # hot
    bank.resolve([1])  # hot
    bank.resolve([2])  # warm (hot full)
    bank.resolve([3])  # warm
    bank.resolve([4])  # transient

    with pytest.raises(RuntimeError, match="Too many cold experts"):
        bank.resolve([99])  # Never hits — warm buffer absorbs it
```

**After:**
```python
def test_transient_overflows_raise_when_warm_full(self):
    """resolve() raises RuntimeError when cold experts exceed transient_slots."""
    # Use warm_slots=0 so cold experts route to transient (not warm buffer).
    sc = MockSidecar()
    bank = ExpertSlotBank(
        layer=0, sidecar=sc,
        hot_count=2, warm_slots=0,        # force cold → transient path
        transient_slots=1, expert_bytes=36,
    )
    bank.resolve([0])  # hot
    bank.resolve([1])  # hot
    bank.resolve([2])  # transient (hot full, warm=0)

    # Pass 3 cold IDs in one call — exceeds transient_slots=1 → overflow.
    with pytest.raises(RuntimeError, match="Too many cold experts"):
        bank.resolve([99, 100, 101])
```

## Fix 2: `test_patch_unpatch_symmetry`

**Problem:** `patch_switch_linear()` now takes 5 required args (`module`, `sidecar`, `layer`, `expert_bytes`, `slot_bank`) but the test passed only a mock.

**Root cause:** API signature changed from 1 arg → 5 required + N optional. The mock also lacked `__call__` attribute (padded by `nn.Module.__getattr__` for real modules).

**Fix:** Construct minimal mock with `gate_proj`, `up_proj`, `down_proj` parameters (required by `_is_switch_glu`), a real MockSidecar, and ExpertSlotBank. Pass all 5 required args.

**Exact location:** `tests/test_streaming_integration.py`, lines 578–607 (class `TestPatchLifecycle`)

**Before:**
```python
def test_patch_unpatch_symmetry(self):
    class MockSwitchLinear:  # No gate/up/down → _is_switch_glu fails
        def __init__(self):
            self.original_call = None
    mock = MockSwitchLinear()
    patch_switch_linear(mock)  # Missing 4 required args
```

**After:**
```python
def test_patch_unpatch_symmetry(self):
    import mlx.nn as nn
    import mlx.core as mx

    class MockSwitchLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = mx.zeros((3, 4))
            self.up_proj = mx.zeros((3, 4))
            self.down_proj = mx.zeros((4, 3))
        def __call__(self, x):
            return self._forward(x) if hasattr(self, '_forward') else x

    mock = MockSwitchLinear()
    sc = MockSidecar(expert_bytes=36)
    slot_bank = ExpertSlotBank(
        layer=0, sidecar=sc, hot_count=2, warm_slots=2,
        transient_slots=1, expert_bytes=36,
    )

    patch_switch_linear(mock, sc, 0, 36, slot_bank)  # All 5 args
    unpatch_switch_linear(mock)
    assert not hasattr(mock, "_omlx_original_call")
```

## Verification

```bash
# Integration tests only:
python3 -m pytest tests/test_streaming_integration.py -v   # 24 passed, 6 skipped

# Full suite (7 pre-existing failures unrelated to this change):
python3 -m pytest tests/ -v --tb=no
```

## Key File Paths & Sizes

| Path | Size | Purpose |
|---|---|---|
| `omlx/streaming/slot_bank.py` | ~574 lines | ExpertSlotBank — hot/warm/transient memory pool |
| `omlx/streaming/patch.py` | ~640 lines | patch/unpatch wiring for SwitchGLU layers |
| `omlx/streaming/sidecar.py` | ~200 lines | Sidecar file reader (C++ fallback) |
| `omlx/streaming/config.py` | ~150 lines | StreamingConfig dataclass defaults (hot=8, warm=16, transient=4) |
| `omlx/streaming/pipeline.py` | ~300 lines | EMATrajectoryPrefetcher, record/prefetch steps |
| `tests/test_streaming_integration.py` | ~697 lines | Integration test suite |

## Notes

- `ExpertSlotBank.__init__` defaults: hot=13, warm=64, transient=8
- Overflow check (slot_bank.py line 336): `cold_count > self.transient_slots` (strictly greater)
- When `warm_slots=0`, cold experts load into `_transient_buffers` at circular `slot_idx` (line 371)
- When `warm_slots>0` and hot has room: cold loads directly into hot buffer (line 384)
- When `warm_slots>0` and hot full: cold goes to warm tier with LRU eviction (line 391)
