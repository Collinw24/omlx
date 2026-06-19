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
| 0.7 | **Functional Immutability Guard — no hot-path array slice assignment.** Absolute ban on `array[i] = ...` or any in-place mutation of `mx.array` elements inside the hot path. MLX arrays are immutable by design; slice assignment either silently delegates to Python heap churn or triggers undefined Metal backends. All data placement must happen via raw pointer mutations in C++ (via `array.data<void*>()` in Phase 2b) or via native functional operations (`mx.stack`, `mx.concatenate`, `mx.gather_qmm`) on pre-existing, immutable views. |
| 0.8 | **Synchronous Barrier Ban — no GPU-stalling ops in the hot path.** Absolute ban on calling `.tolist()`, `.item()`, `np.array()`, or any conditional checks on unmaterialized array values inside the hot path. These operations force a synchronous CPU-GPU block: the CPU halts while the GPU pipeline drains, waiting for the CPU to collect scalar values or NumPy objects. This destroys the entire async streaming model. Routing indices must be consumed as raw tensors or through a fast C++ iterator — never materialized to Python scalars. |
| 0.9 | **Zero-Allocation Views — every slot view must be a pointer alias.** Any method returning an expert weight matrix slice from a slot bank must return a pre-existing `mx.array` view referencing the pre-allocated memory pool. It must never invoke constructors (`mx.array()`, `mx.asarray()`, `mx.zeros()`, `mx.ones()`) that perform memory copies. Views are created at init and reused; their backing Metal buffers are the single source of truth. |

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
         │           C++: pread_into_array() writes NVMe bytes → GPU buffer
         │           via buffer protocol — zero intermediate copies, validated.
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
    _buffer_access_fallback.py  # Pure Python fallback
    config.py                # Phase 4: StreamingConfig schema
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

## Phase 2: ExpertSlotBank (`slot_bank.py`) — **COMPLETE**

Central GPU memory manager. Every expert access goes through it. Zero dynamic `mx.array` allocation.

All methods implemented and validated by 16 tests (§7a sidecar + §7b slot bank).

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

### 2c. `resolve()` — hot-path entry ✅

```python
def resolve(self, expert_ids: list[int]) -> tuple[mx.array, list[int]]:
    """
    Three-tier cascade per expert:
      1. Hot — lookup in _hot_map, return slot_id (no I/O).
      2. Warm — lookup in _warm_map (OrderedDict), LRU move_to_end.
         Promote to hot if room (copy data from warm→hot buffer).
      3. Transient — lookup in _transient_map, promote to hot if room.
      4. Cold miss — call sidecar.read_expert_into() into hot or warm buffer.
         Evict LRU warm slot if warm is full.
         Raise RuntimeError if cold experts exceed transient capacity.
    Returns: (stacked_weights: shape (K, expert_bytes), slot_indices: list[K])
    Updates EMA frequency counts. Updates warm LRU order.
    """
```

**Gate contract:** `resolve()` must be called AFTER `mx.eval(router_logits)` has returned. The caller owns the sync gate; `resolve()` must not gate itself.

**Implementation notes:**
- Hot tier pre-loaded at `__init__` via `_initialize_hot_tier()` → calls `read_expert_into` for each hot expert.
- `_ensure_stack_buf()` pre-allocates output buffer, grows only when K exceeds current size (one-time cost).
- `_get_slot_view(slot_id)` maps slot_id → correct tier buffer row (hot < hot_count, warm < warm_slots, transient).
- Overflow guard: when `warm_slots=0`, checks `len(_warm_map) >= transient_slots` before cold load.

### 2d. Warm slot LRU + EMA

- `OrderedDict[int, int]` mapping `expert_id → warm_slot_index`, ordered by recency.
- Warm miss + cold load: evict LRU entry.
- Warm hit: move to MRU position.
- EMA: `ema[id] = α * 1.0 + (1-α) * ema.get(id, 0.0)` with `α = 2/(window+1)`.
- Experts with EMA ≥ `warm_promote_threshold` skip transient, go directly to warm slot on next cold load.

### 2e. Transient slot recycling

Circular index `self._next_transient`. Each cold load (not qualifying for warm promotion) takes the next transient slot mod `transient_slots`. If `resolve()` is called with more cold experts than transient_slots can satisfy, raise `RuntimeError` — configuration error, not recoverable.

---

## Phase 2b: C++ Buffer Utility (`_buffer_access.cpp`) — **COMPLETE**

### Core Design

Phase 2b provides a `pread_into_array()` function that reads bytes from NVMe (via `pread(2)`) directly into any Python object that supports the writable buffer protocol — `mx.array`, `bytearray`, `memoryview`, `numpy.ndarray`. No MLX headers are required at build time.

**Why the buffer protocol instead of `mlx::core::array::data<void*>()`?**
MLX uses `nanobind` (not `pybind11`) for its Python bindings, which means `pybind11` cannot directly receive `mlx::core::array&` arguments without a type caster. The buffer protocol is cleaner: it works with any writable buffer, including future non-MLX backends.

### C++ Function

```cpp
// pread_into_array(fd, file_offset, length, arr, arr_byte_offset)
// Reads `length` bytes from `fd` at `file_offset` into buffer `arr`
// starting at `arr_byte_offset` bytes into the buffer.
// Accepts any Python object supporting the writable buffer protocol.
// Validates: writable, format='B' (uint8), C-contiguous, sufficient capacity.
py::int_ pread_into_array(
    int fd,
    int64_t file_offset,
    int64_t length,
    py::buffer arr,
    int64_t arr_byte_offset = 0
);
```

### Build

```bash
cd omlx/streaming
cmake -B build \
  -Dpybind11_DIR=<pybind11-cmake-dir> \
  -DPython3_EXECUTABLE=<python3-binary>
cmake --build build
cmake --install build --prefix .
```

Builds `_buffer_access.cpython-*-darwin.so` into `omlx/streaming/`.

### Fallback Import Logic (in `__init__.py`)

```python
try:
    from ._buffer_access import pread_into_array
    _USE_CPP_EXTENSION = True
except ImportError:
    from ._buffer_access_fallback import pread_into_array
    _USE_CPP_EXTENSION = False
```

### Gating Note

The C++ extension is now available and integrated. `sidecar.read_expert_into()` delegates to `pread_into_array()` for all buffer types. The slot bank uses `mx.array` tier buffers (migrated from `bytearray`) for Invariant 0.9 compliance (zero-allocation views via row slicing).

### Test Coverage

1. **1D basic read** — 256 bytes into 1D mx.array
2. **1D offset read** — read from file offset 1000 into array byte offset 0
3. **2D slot read** — 1024 bytes into 2D mx.array at logical slot 2
4. **2D sub-slot read** — 100 bytes at slot 3 + offset 50
5. **2D cross-slot read** — 2048 bytes spanning 2 slots
6. **bytearray buffer** — write into bytearray buffer
7. **Short read** — reading past EOF raises RuntimeError
8. **Wrong dtype** — float32 buffer raises ValueError
9. **Read-only buffer** — memoryview of bytes raises ValueError
10. **Zero-length read** — returns 0 immediately

---

## Phase 3: SwitchGLU Monkey-Patch (`patch.py`) — **COMPLETE, REWRITTEN FOR MLX FUNCTIONAL PARADIGM**

**Critical constraint:** Phase 3 operates entirely in the MLX graph-assembly layer. It manages graph wiring and token-activation mapping — NOT memory placement. Memory placement is the sole responsibility of Phase 2b's C++ extension.

### 3a. Layer discovery and slot bank construction

```python
def build_slot_banks(model, sidecar, cfg) -> dict[str, ExpertSlotBank]:
    """Walk model.layers, instantiate one ExpertSlotBank per SwitchGLU layer.
    Slot bank constructor calls read_expert_into for hot tier initialization.
    """
```

### 3b. Weight freeing (memory reclamation)

```python
layer.mlp.switch_mlp.gate_proj = None
layer.mlp.switch_mlp.up_proj   = None
layer.mlp.switch_mlp.down_proj = None
mx.eval()  # flush graph, release Metal buffers
```

### 3c. Patched `__call__` — MLX functional assembly

```python
def _streaming_switch_linear_call(self, x, indices):
    """x: (B,S,H), indices: (B,S,top_k) — routing indices as raw MLX tensor.
    Protocol:
      1. Consume routing indices as raw tensor — NO .tolist(), NO .item(),
         NO conditional checks on unmaterialized values. Indices flow
         directly to slot_bank.resolve() as an mx.array.
      2. slot_bank.resolve(indices) → (stacked_weights: mx.array, slot_ids: list[int])
         The slot bank manages in-place buffer writes via C++ pread_into_array.
         patch.py does NOT touch buffers directly.
      3. mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids, ...)
         Maps token activations to slot-bank views using gather_qmm.
      4. Return output — do NOT call mx.eval() inside. Caller owns the sync gate.
    """
```

**Token Slicing / Index Extraction:** Instead of using `.tolist()` to extract expert IDs from the routing indices tensor, the slot-bank resolution logic must ingest the indices as a raw tensor. The `resolve()` method receives routing indices as an `mx.array` and extracts slot lookups through direct C++ iteration or functional MLX operations. This preserves asynchronous execution entirely.

**In-Place Buffer Management:** The `ExpertSlotBank` handles memory mapping and pointer-level writes at the C++ level (Phase 2b). `patch.py` simply manages graph assembly by mapping incoming token activations to slot-bank views using `mx.gather_qmm`. No Python buffer manipulation occurs in the hot path.

### 3d. Unpatch helper (restoration)

```python
def unpatch_switch_linear(self, original_call):
    """Restore original __call__ and reinstate expert weights."""
```

### 3e. Apply streaming patches to model

```python
def apply_streaming_patches(model, slot_banks):
    """Patch all SwitchGLU layers in a model at once."""
```

---

## Phase 4: Model Loading Integration — **IMPLEMENTED**

### 4a. Config schema (`config.py`)

```python
@dataclass
class StreamingConfig:
    stream_experts: bool = False
    expert_sidecar_path: str | None = None
    expert_hot_count: int = 13
    expert_warm_slots: int = 64
    expert_transient_slots: int = 8
    expert_prefetch: bool = True
    expert_prefetch_window: int = 4
    expert_top_k_override: int | None = None
    calibration_frequencies: dict[int, float] | None = None
```

`stream_experts: false` is default. Must be explicit opt-in per model.

### 4b. Sidecar resolution (`pipeline.py:resolve_sidecar_path`)

Auto-detects `{model_dir}/{model_name}.streaming` with fallback search. Explicit path override via `expert_sidecar_path`.

### 4c. BatchedEngine integration (`engine/batched.py`)

Added `streaming_config` parameter to `BatchedEngine.__init__()`. In `start()`:
```python
if streaming_config is not None and streaming_config.stream_experts:
    self._streaming_state = load_model_with_streaming(
        model, cfg=streaming_config, model_name_or_path=self._model_name,
    )
```

In `stop()`:
```python
if self._streaming_state is not None:
    unload_streaming(self._model)
```

### 4d. CLI command (`cli.py`)

`omlx create-sidecar <model_path> [-o <path>] [--alignment 16384] [--no-verify]`

### 4e. Pipeline orchestrator (`pipeline.py`)

- `load_model_with_streaming()`: Opens sidecar → builds slot banks per layer → applies streaming patches → returns state dict
- `streaming_forward_pass()`: I2-compliant hot-path entry point for per-layer MoE forward
- `EMATrajectoryPrefetcher`: EMA tracking with `predict()` and `prefetch()` methods
- `unload_streaming()`: Removes patches, restores original `__call__`

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
| Q5 | **Sequential `mx.gather_qmm` overlapping views** — how does the MLX execution graph behave when multiple sequential `mx.gather_qmm` operations reference overlapping views of the same underlying physical buffer across different layers? | Run two consecutive layers with `gather_qmm` reading from the same warm slot buffer. Compare output against sequential baseline. If outputs diverge, the graph may be reading stale or partially-written buffer state — requires explicit `mx.eval()` fencing between layers or per-layer buffer pinning. |

---

## Coding Standards

- Every function in `streaming/` must have a docstring explaining **WHY**, not just what. The Metal memory model is non-obvious.
- All public APIs must have type annotations.
- Use `logging.getLogger(__name__)`. Never `print()`.
- Error messages must be **actionable**: "Expert 5 not found in layer '3'. Available experts: 0–63. Sidecar may have been created from a different model checkpoint."
- All new config keys documented in omlx README under "Expert Streaming" section.
- Feature must be completely inert when `stream_experts: false`. Gating at `BatchedEngine.start()` only.
