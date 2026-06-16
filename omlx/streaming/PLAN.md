# SSD Expert Streaming — Implementation Plan

**Branch:** `feat/ssd-expert-streaming` | **Repo:** [jundot/omlx](https://github.com/jundot/omlx)

> Run Mixture-of-Experts models (Qwen3.5-397B-A17B, 209 GB) on Apple Silicon
> with 48–64 GB unified memory by streaming only the top-K active expert weights
> from NVMe SSD per token, bypassing the macOS Unified Buffer Cache (UBC) via
> F_NOCACHE direct I/O.

---

## 0. Invariants (Never Break)

| # | Rule |
|---|------|
| 0.1 | **Zero dynamic `mx.array` allocation in the hot path.** Every GPU expert buffer must be pre-allocated at slot bank init. No `mx.array()`, `np.frombuffer()`, or Python heap allocation per-token. |
| 0.2 | **Strict graph gate before slot mutation.** Never write into a slot buffer while the MLX graph referencing it is still queued on Metal. Gate is `mx.eval(router_logits)` — see §6. |
| 0.3 | **F_NOCACHE must be verified, not assumed.** If `fcntl(fd, F_NOCACHE, 1)` fails or silently returns, log a WARNING and set `self._nocache_active = False`. Expose `nocache_active` property. |
| 0.4 | **Non-expert weights stay native lazy-mmap.** Original `.safetensors` never touched. Only expert projection tensors extracted into the sidecar. |
| 0.5 | **`mx.compile()` prohibited on the streaming path.** Confirmed null-pointer SIGSEGV when `mx.compile()` encounters unmaterialized mmap weights. No `@mx.compile` decorators anywhere in streaming code. |
| 0.6 | **Feature must be completely inert when disabled.** Zero overhead, zero code paths touched when `stream_experts: false`. All gating at `BatchedEngine.start()`, not sprinkled through the forward pass. |

---

## Architecture Overview

```
StreamingExpertSidecar (F_NOCACHE direct I/O, on-disk)
  Byte 0:     [4B: header_len][JSON header][null padding to 16384]
  Byte 16384: ┌──────────┬──────────┬─────┬──────────┐
               │ Expert 0 │ Expert 1 │ ... │ Expert N │
               │(16KB pad)│(16KB pad)│     │(16KB pad)│
               └──────────┴──────────┴─────┴──────────┘
               └── Direct I/O via os.pread() → pre-allocated GPU buffers
         │
         │ F_NOCACHE: bypasses UBC, prevents memory compression
         │
         ▼
ExpertSlotBank (per-layer, in GPU unified memory)
  ┌───────────────────────┬──────────────────────┬──────────────────────────┐
  │ Hot Slots (5%)        │ Warm Slots (25%)     │ Transient Streaming Slots│
  │ Pinned MTLResidencySet│ LRU + EMA frequency  │ Pre-allocated, recycled  │
  │ Never evicted         │ Persistent buffers   │ Overwritten per token    │
  └───────────────────────┴──────────────────────┴──────────────────────────┘
  ALL experts go through slots — zero dynamic mx.array allocation in hot path.
         │
         │ Phase 2b: C++ pread_into_array() writes NVMe bytes → GPU buffer
         │           via array.data<void*>() pointer — zero intermediate copies
         │
         ▼
mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids)
  └─ Single fused Metal dispatch, on-the-fly dequant (FMA registers)
```

---

## Key Technical Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| T1 | **16KB alignment** | Apple Silicon uses 16 KiB virtual memory pages. Pointers wrapped into Metal buffers must be 16 KiB-aligned or the UMA subsystem rejects them. Header padded to 16384B; Expert 0 always at offset 16384. |
| T2 | **F_NOCACHE Direct I/O** | `fcntl(fd, F_NOCACHE, 1)` bypasses the macOS UBC. `os.pread()` goes NVMe→process buffer directly. No UBC bloat, no memory compression from file pages. Eliminates need for `madvise` eviction games. |
| T3 | **Non-mutative sidecar** | Original `.safetensors` untouched. Non-expert weights remain native lazy-mmap. |
| T4 | **Contiguous expert blocks** | gate_proj + up_proj + down_proj (weight, scales, biases) packed into one byte range per expert. One `os.pread()` per expert. |
| T5 | **Three-tier slot bank** | Hot (pinned, never evicted), Warm (LRU with EMA frequency tracking), Transient (recycling, overwritten per token). All experts go through slots. |
| T6 | **C++ pybind11 buffer access** | `pread_into_array()` writes pread bytes directly into pre-allocated `mx.array` GPU buffer via `array.data<void*>()`. Required because Python `mx.array` is immutable. |
| T7 | **No `mx.compile()`** | MLX null-pointer SIGSEGV with unmaterialized mmap weights in compiled kernels. |
| T8 | **Fused dequant in Metal** | `mx.gather_qmm()` dequantizes on-the-fly via FMA in thread registers. Weights stay packed 4-bit in slot buffers. Pre-packed to match MLX's transposed `[N, out, in//2]` layout. |
| T9 | **Strict graph gating** | `mx.eval(router_logits)` blocks until GPU drains. Only then can slot buffers be mutated. Prevents GPU reading corrupted weights mid-matmuls. |
| T10 | **Serial I/O pipeline** | GPU+DMA concurrent overlap causes -73% GPU slowdown on Apple Silicon UMC. Pipeline is: GPU→IDLE→pread→GPU→prefetch during GPU. |
| T11 | **EMA trajectory prefetch** | 4-token sliding window frequency index. Predict: union(current_routing + top_2_ema). Reduces cold-miss rate from ~30% to ~15%. |

---

## Precise Pipeline Execution (per token, per layer)

```
[Token t, Layer L]

  ├── 1.  GPU: attention + router(logits) — build lazy graph
  ├── 2.  HOST: mx.eval(router_logits)    ← BLOCKING sync gate
  │         ▸ GPU drains to IDLE
  │         ▸ MLX computation graph cleared
  │         ▸ SAFE TO MUTATE SLOT BUFFERS
  │
  ├── 3.  CACHE CHECK: identify missing experts (hot→warm→transient)
  ├── 4.  IO WINDOW: pread(F_NOCACHE) cold experts into warm/transient slots
  │         ▸ GPU IDLE during I/O (prevents UMC DMA contention, -73% GPU slowdown)
  │
  ├── 5.  COMPUTE: mx.gather_qmm() for MoE block
  ├── 6.  DEFER: mx.async_eval(moe_output) ← non-blocking submit
  │         ▸ GPU processes MoE matrices in background
  │
  └── 7.  PREFETCH: pread(F_NOCACHE) predicted experts for token t+1
           ▸ EMA union: actual_routing[t] ∪ top_2_ema[t]
           ▸ During GPU MoE compute window — DMA completes before GPU needs data
```

---

## EMA Trajectory Cache (§6)

MoE routers follow a power-law semantic distribution over a sliding token window.

- **Frequency index:** `ema[expert_id] = α × 1.0 + (1-α) × ema[expert_id]` with `α = 2 / (window + 1)`. Default: `window=4`, `α=0.4`.
- **Predict:** for token `t+1`, read `actual_routing[t]` (naive) ∪ `top_2_ema[t]` (highest EMA experts NOT in actual_routing[t]).
- **Promote:** experts with EMA ≥ `warm_promote_threshold` (default 2.0) skip transient on next cold load, go directly to a warm slot.
- **Evict:** experts with EMA=1.0 after 4-token decay are eligible for LRU eviction.

Extends the prefetch set by 2-3 experts per token, reducing cold-miss rate from ~30% to ~15%.

---

## File Structure

```
omlx/
  streaming/
    __init__.py              # exports, fallback import logic
    sidecar.py               # Phase 1: StreamingExpertSidecar
    slot_bank.py             # Phase 2: ExpertSlotBank
    _buffer_access.cpp       # Phase 2b: C++ pybind11 extension
    _buffer_access_fallback.py
    patch.py                 # Phase 3: SwitchGLU monkey-patch
    pipeline.py              # Phase 5+6: execution orchestrator + EMA prefetcher
    hash_router.py           # Phase 8 stub
    CMakeLists.txt           # C++ build
  utils/
    model_loading.py         # Phase 4 integration
  engine/
    batched.py               # Phase 4 integration
tests/
  test_streaming.py          # Phase 7
```

---

## Phase 1: StreamingExpertSidecar (`sidecar.py`) — COMPLETE, HARDENING NEEDED

### 1a. Multi-architecture key detection

Current `_EXPERT_KEY_RE_STR` matches only `switch_mlp` — silently drops DeepSeek, Mixtral, etc.

**Fix:** Pluggable parser registry:

```python
_EXPERT_KEY_PARSERS: list[tuple[str, int, int, str]] = [
    # Qwen3.5-MoE / Qwen2-MoE: switch_mlp
    (r".*\.layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases)$", 2, 3, "qwen_moe"),
    # Mixtral / Mistral-MoE: block_sparse_moe.experts.{idx}
    (r".*\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)$", 2, 3, "mixtral"),
    # DeepSeek-V2/V3/V4: mlp.experts.{idx}
    (r".*\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$", 3, 4, "deepseek_moe"),
]
```

`_parse_expert_key` iterates parsers in order. First-match tag stored in sidecar header as `"model_type"`.

For Mixtral/DeepSeek (expert index in key rather than tensor dimension), `_discover_expert_tensors` must handle the per-expert-per-file layout. Abstract into strategy pattern if needed; binary format stays the same.

### 1b. Header offset recomputation — make it single-pass

Replace the two-pass delta-shift retry with a bounded single-pass:

```python
# Pass 1: compute header size with placeholder MAX_LEN offsets
placeholder_header = {..., "experts": {"0": {"offset": 9999999999, "length": 9999999}}}
upper_bound = len(json.dumps(placeholder_header, separators=(",", ":"))) + 64  # safety margin
first_data_offset = _align_up(4 + upper_bound, alignment)

# Pass 2: compute all real offsets from first_data_offset, serialize once
```

The 64-byte margin + 16KB alignment guarantees no overflow.

### 1d. Validation method

Add `sidecar.verify(sample_layers: int = 3) -> bool` that reads expert 0 + last expert from random layers and confirms byte lengths match header. Replaces inline validation in `create()`; callable independently at startup.

---

## Phase 2: ExpertSlotBank (`slot_bank.py`)

Central GPU memory manager. Every expert access goes through it. Zero dynamic `mx.array` allocation.

### 2a. Structure

```python
class ExpertSlotBank:
    """
    Three-tier GPU memory pool for expert weights.

    Tiers:
      Hot   — top `hot_count` experts by global activation frequency. Populated at
              init from calibration data or first-pass statistics. Never evicted.
      Warm  — `warm_slots` LRU slots with EMA frequency tracking. Persistent
              mx.array buffers. Expert graduates to warm when EMA ≥ threshold.
      Transient — `transient_slots` recycling slots. Overwritten per token.
              Always available; overflow is RuntimeError.
    """
```

Constructor:

```python
def __init__(
    self,
    sidecar: StreamingExpertSidecar,
    layer: int | str,
    expert_bytes: int,
    hot_count: int = 13,
    warm_slots: int = 64,
    transient_slots: int = 8,
    ema_window: int = 4,
    warm_promote_threshold: float = 2.0,
    calibration_frequencies: dict[int, float] | None = None,
) -> None:
```

### 2b. Slot buffer layout

Pre-allocate as one contiguous `mx.array` per tier, indexed by `slot_id`:

```python
self._hot_buffers    = mx.zeros((hot_count, expert_bytes), dtype=mx.uint8)
self._warm_buffers   = mx.zeros((warm_slots, expert_bytes), dtype=mx.uint8)
self._transient_buffers = mx.zeros((transient_slots, expert_bytes), dtype=mx.uint8)
mx.eval(self._hot_buffers, self._warm_buffers, self._transient_buffers)
```

`mx.eval()` forces immediate GPU allocation so backing Metal buffers exist before C++ extension accesses pointers.

### 2c. `resolve()` — hot-path entry

```python
def resolve(self, expert_ids: list[int]) -> tuple[mx.array, list[int]]:
    """
    Returns: (stacked_weights: shape (K, expert_bytes), slot_indices: list[K])

    Must not allocate. Must not call mx.array(). Must not call numpy.
    Cache check order: hot → warm → transient.
    For misses: calls sidecar.read_expert_into() into the appropriate slot.
    Updates EMA frequency counts. Updates warm LRU order.
    Returns stacked view via mx.gather or direct index — NOT a copy.
    """
```

**Gate contract:** `resolve()` must be called AFTER `mx.eval(router_logits)` has returned. The caller owns the sync gate; `resolve()` must not gate itself.

### 2d. Warm slot LRU + EMA

- `OrderedDict[int, int]` mapping `expert_id → warm_slot_index`, ordered by recency.
- Warm miss + cold load: evict LRU entry.
- Warm hit: move to MRU position.
- EMA: `ema[id] = α * 1.0 + (1-α) * ema.get(id, 0.0)` with `α = 2/(window+1)`.
- Experts with EMA ≥ `warm_promote_threshold` skip transient, go directly to warm slot on next cold load.

### 2e. Transient slot recycling

Circular index `self._next_transient`. Each cold load (not qualifying for warm promotion) takes the next transient slot mod `transient_slots`. If `resolve()` is called with more cold experts than transient_slots can satisfy, raise `RuntimeError` — configuration error, not recoverable.

---

## Phase 2b: C++ Buffer Utility (`_buffer_access.cpp`)

Writes pread bytes directly into persistent `mx.array` GPU buffer without Python allocations.

### Required function

```cpp
// pread_into_array(fd, file_offset, length, arr, arr_byte_offset)
// Reads `length` bytes from `fd` at `file_offset` into the Metal buffer
// backing `arr`, starting at `arr_byte_offset` bytes into the array data.
// Returns bytes_read (int). arr must be dtype=uint8, contiguous, already eval'd.
py::int_ pread_into_array(
    int fd,
    off_t file_offset,
    size_t length,
    mlx::core::array& arr,
    size_t arr_byte_offset
);
```

Uses `arr.data<uint8_t>()` for raw pointer. Valid only after `mx.eval(arr)`.

### Build

`CMakeLists.txt` under `omlx/streaming/` depending on `mlx` + `pybind11` via `find_package`. Builds `_buffer_access.cpython-*.so` into `omlx/streaming/`.

### Pure Python fallback (`_buffer_access_fallback.py`)

```python
def pread_into_array(fd, file_offset, length, arr, arr_byte_offset):
    """Fallback: os.pread() → Python bytes → np.frombuffer → slice assignment.
    THIS ALLOCATES. Emits WARNING on first call."""
```

Import logic in `__init__.py`:
```python
try:
    from omlx.streaming._buffer_access import pread_into_array
    _BUFFER_ACCESS_NATIVE = True
except ImportError:
    from omlx.streaming._buffer_access_fallback import pread_into_array
    _BUFFER_ACCESS_NATIVE = False
    warnings.warn("omlx streaming: C++ extension not built. Falling back to allocating path.", RuntimeWarning)
```

---

## Phase 3: SwitchGLU Monkey-Patch (`patch.py`)

Replace `SwitchLinear.__call__` to route through slot bank.

### 3a. Patching

```python
def patch_switch_linear(model, slot_banks: dict, sidecar: StreamingExpertSidecar) -> None:
    """Walk model.layers, replace each SwitchLinear.__call__ with closure over slot bank.
    After patching, free original 3D weight tensors (gate/up/down = None; mx.eval()).
    """
```

### 3b. Weight freeing (memory reclamation)

```python
layer.mlp.switch_mlp.gate_proj = None
layer.mlp.switch_mlp.up_proj   = None
layer.mlp.switch_mlp.down_proj = None
mx.eval()  # flush graph, release Metal buffers
```

### 3c. Patched `__call__`

```python
def _streaming_switch_linear_call(self, x, indices):
    """x: (B,S,H), indices: (B,S,top_k) already eval'd.
    Protocol:
      1. slot_bank.resolve(indices.flatten().tolist()) → (stacked_weights, slot_ids)
      2. mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids, ...)
      3. Return output
    Do not call mx.eval() inside — caller owns the sync gate.
    """
```

---

## Phase 4: Model Loading Integration

### 4a. Config schema

```json
{
  "stream_experts": false,
  "expert_sidecar_path": null,
  "expert_hot_count": 13,
  "expert_warm_slots": 64,
  "expert_transient_slots": 8,
  "expert_prefetch": true,
  "expert_prefetch_window": 4,
  "expert_top_k_override": null
}
```

`stream_experts: false` is default. Must be explicit opt-in per model. No auto-detect-and-enable.

### 4b. Sidecar auto-detection

If `stream_experts: true` and `expert_sidecar_path` is null: check `{model_dir}/{model_name}.streaming`. Log resolved path.

### 4c. BatchedEngine integration

```python
if cfg.stream_experts:
    sidecar = StreamingExpertSidecar(resolve_sidecar_path(cfg))
    slot_banks = build_slot_banks(model, sidecar, cfg)
    patch_switch_linear(model, slot_banks, sidecar)
```

### 4d. CLI command

`omlx create-sidecar <model_path> [--output <path>] [--alignment 16384]`

Progress bar (tqdm) over layers. Print final sidecar size and expert count.

---

## Phase 5+6: Pipeline Execution (`pipeline.py`)

### Sync gate protocol

```python
# Inside per-layer forward pass, after computing router logits:

mx.eval(router_logits)  # STEP 1: flush graph, GPU drains, blocks until complete

# STEP 2 (conditional): if mx.gather_qmm emits multi-segment command buffers:
#   mx.synchronize()  # drain Metal queue fully
#   Only add if empirical testing shows GPU corruption. Start without it.

# STEP 3: SAFE to mutate slot buffers
stacked_weights, slot_ids = slot_bank.resolve(top_k_expert_ids)
```

Do not use `mx.async_eval()` until Phase 7 correctness tests pass. Add as optimization only after bit-identical output confirmed.

### EMA prefetcher

```python
class EMATrajectoryPrefetcher:
    def update(self, routed_experts: list[int]) -> None: ...
    def predict_next(self, current_experts: list[int]) -> list[int]: ...
    def prefetch(self, predicted_experts, slot_bank, sidecar, layer) -> None:
        # Only prefetch into warm slots (not transient) — data must persist to t+1.
        # For each predicted expert not already cached, call sidecar.read_expert_into().
```

### GPU idle window discipline

```
[Token t, Layer L]
  mx.eval(router_logits)              ← GPU drains
  slot_bank.resolve(top_k)            ← I/O window: pread cold experts
  moe_output = mx.gather_qmm(...)     ← GPU resumes
  mx.async_eval(moe_output)           ← defer (only after correctness confirmed)
  prefetcher.prefetch(predicted, ...) ← prefetch: pread during GPU MoE compute
```

---

## Phase 7: Testing (`tests/test_streaming.py`)

Tests run in exact order. Do not skip ahead.

### 7a. Sidecar unit tests (no model required)

- `test_sidecar_create_read_roundtrip` — synthetic safetensors, byte-exact readback
- `test_sidecar_alignment` — every expert offset divisible by 16384
- `test_sidecar_nocache_property` — `nocache_active` is bool, doesn't raise
- `test_sidecar_verify` — `verify()` returns True on valid sidecar

### 7b. Slot bank unit tests (no model, mock sidecar)

- `test_slot_bank_hot_always_resident` — hot expert returned without pread
- `test_slot_bank_warm_lru_eviction` — beyond capacity evicts LRU
- `test_slot_bank_transient_overflow_raises` — RuntimeError on cold > transient_slots
- `test_slot_bank_no_allocation_in_hot_path` — tracemalloc confirms zero allocation on warm-hit resolve()

### 7c. Layer-level interceptor test (requires Qwen3.5-35B-A3B)

**Correctness gate.** Do not proceed to full model until this passes.

```
1. Load unpatched model (full RAM)
2. Run prompt, capture layer 0: input states, router logits, expert indices, MoE output
3. Load patched model, enable streaming on layer 0 only
4. Run same prompt, inject same inputs at layer 0
5. Assert mx.allclose(stream_output, reference_output, atol=1e-4)
6. Only after passing, enable streaming on all layers
```

### 7d. Full model tests

- `test_full_model_output_matches_baseline` — perplexity within 0.5% of unpatched
- `test_gpu_memory_reduction` — peak GPU memory ≤ 60% of without streaming
- `test_ubc_stability` — 100-token gen, RSS growth < 50 MB (only meaningful if nocache_active)

---

## Phase 8: DeepSeek Hash-Routing Stub (`hash_router.py`)

Stub only — do not implement full DeepSeek support.

```python
class HashRouterPrefetcher:
    """For DeepSeek-V4 layers 0-2: deterministic expert from token ID.
    Stub: expert_id = token_id % num_experts_per_layer.
    Placeholder for future DeepSeek integration.
    """
    def predict(self, token_id: int, layer: int) -> int:
        raise NotImplementedError("DeepSeek hash routing not yet implemented.")
```

Provides hook point in architecture without requiring DeepSeek model support in this PR.

---

## Open Questions (Require Empirical Verification)

These are architectural risks. Do not paper over with assumptions.

| # | Question | Test Method |
|---|----------|-------------|
| Q1 | **`mx.gather_qmm` dispatch count** — single cmd buffer or multi-segment? | Instrument MLX Metal backend; count segments per call. If >1, add fast path for top-K=1 routing decisions. |
| Q2 | **`mx.eval()` drain guarantee** — does it fully wait for Metal completion? | Submit gather_qmm, mx.eval() its inputs (not output), immediately overwrite input buffer, run gather_qmm again. Corrupted outputs → need mx.synchronize(). |
| Q3 | **16KB vs 4KB Metal alignment** — what does `newBufferWithBytesNoCopy` actually require? | Experiment with both alignments on target hardware. If 4KB sufficient, change `_DEFAULT_ALIGNMENT` to 4096 (saves 75% padding). |
| Q4 | **F_NOCACHE under memory pressure** — does `os.pread()` latency stay < 2ms per chunk under concurrent memory compression? | Test with 48 GB RAM + 209 GB model + active memory compressor. If latency spikes, GPU starvation is worse than estimated. |

---

## Coding Standards

- Every function in `streaming/` must have a docstring explaining **WHY**, not just what. The Metal memory model is non-obvious.
- All public APIs must have type annotations.
- Use `logging.getLogger(__name__)`. Never `print()`.
- Error messages must be **actionable**: "Expert 5 not found in layer '3'. Available experts: 0–63. Sidecar may have been created from a different model checkpoint."
- All new config keys documented in omlx README under "Expert Streaming" section.
- Feature must be completely inert when `stream_experts: false`. Gating at `BatchedEngine.start()` only.
