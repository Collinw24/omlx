# SSD Expert Streaming for oMLX

Branch: `feat/ssd-expert-streaming` | Fork: [Collinw24/omlx](https://github.com/Collinw24/omlx)

## Goal

Enable running large Mixture-of-Experts models (Qwen3.5-397B-A17B, 209 GB) on Apple Silicon machines with limited unified memory (48-64 GB) by streaming only the top-K active expert weights per token from NVMe SSD, using kernel-level zero-copy paging.

Initial target: **Qwen3.5-35B-A3B** (fits in RAM for correctness verification).
Aspirational target: **Qwen3.5-397B-A17B** (209 GB on disk).

## Architecture

```
StreamingExpertSidecar (mapped via mmap, read-only)
  Byte 0:   [4B: header_len][JSON header][null padding to 4096]
  Byte 4096:┌──────────┬──────────┬─────┬──────────┐
             │ Expert 0 │ Expert 1 │ ... │ Expert N │
             │ (4KB pad)│ (4KB pad)│     │ (4KB pad)│
             └──────────┴──────────┴─────┴──────────┘
             └── mmap'd, accessed via pointer + offset
         │
         │ madvise(MADV_WILLNEED / MADV_DONTNEED) controls UBC
         │
         ▼
ExpertSlotBank (per-layer, in GPU unified memory)
  ┌────────────────┬─────────────────┬──────────────────────────────┐
  │ Hot Slots (5%) │ Warm Slots (25%)│ Transient Streaming Slots    │
  │ Pinned         │ LRU-managed     │ Pre-allocated, overwritten   │
  │ Never evicted  │ Evicted on miss │ per token, no allocation     │
  └────────────────┴─────────────────┴──────────────────────────────┘
  Cold experts read directly into transient slots from mmap ptr.
  ALL experts go through the slot bank — zero dynamic allocation.
         │
         │ pybind11 C++ utility: array.data<void*>() → pread/memcpy → GPU buffer
         │
         ▼
mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids)
  └─ Single fused Metal dispatch, on-the-fly dequant (FMA)
```

## Key Technical Decisions

1. **Non-mutative sidecar**: Original `.safetensors` untouched. Non-expert weights remain native lazy-mmap.

2. **Fixed 4096-byte header**: JSON header padded to 4096. Expert 0 always at byte offset 4096. Hardware-aligned for `F_NOCACHE` direct I/O.

3. **Contiguous expert blocks**: gate_proj + up_proj + down_proj (weight, scales, biases) packed into one byte range per expert. Single offset per expert.

4. **mmap + madvise zero-copy I/O**: Entire sidecar mapped via `mmap.mmap(PROT_READ, MAP_SHARED)`. No `os.pread()` in hot path.
   - **Prefetch**: `madvise(ptr + offset, length, MADV_WILLNEED)` — kernel async page-in, no Python I/O, bypasses GIL
   - **Evict**: `madvise(ptr + offset, length, MADV_DONTNEED)` — instantly free UBC pages, prevent memory compression/swap
   - **Access**: `mx.array` wraps mmap'd pointer region zero-copy, or memcpy into slot bank GPU buffer

5. **Three-tier slot bank** (ALL experts go through slots, no dynamic allocation):
   - **Hot** (top 5%): Pinned `MTLResidencySet`, never evicted
   - **Warm** (middle 25%): LRU-managed persistent slots
   - **Transient Streaming** (cold 70%): Pre-allocated recycling slots, overwritten per token
   - Eliminates `mx.array` allocation churn in hot path

6. **C++ pybind11 buffer access**: Python `mx.array` objects are immutable. A minimal C++ extension extracts `array.data<void*>()` for direct `pread()`/`memcpy` into persistent slot bank GPU buffers without allocating new arrays.

7. **Clustered packing for fine-grained experts** (DeepSeek-V4): Co-activated experts grouped into larger 4KB-aligned blocks to reduce syscall/madvise overhead for small experts (<1MB).

8. **Serial I/O pipeline** (hardware-optimal on unified memory):
   See "Precise Pipeline Execution" section below.

9. **No `mx.compile()`** on streaming path due to MLX mmap null-pointer SIGSEGV bug.

10. **`mx.gather_qmm`** dequantizes on-the-fly in Metal kernel registers. Weights stay packed 4-bit in persistent slot buffers.

## Precise Pipeline Execution (per token, per layer)

```
[Token Step t]
  │
  ├── 1. GPU: attention + router logits
  ├── 2. Host: mx.eval(router_logits)  ← blocking, GPU drops to IDLE
  │      ▼
  ├── 3. Cache Check: identify which experts are missing from Hot/Warm slots
  ├── 4. IO Window: CPU reads missing experts into Transient/Warm slots
  │      (GPU IDLE during I/O — prevents UMC DMA contention)
  │      ▼
  ├── 5. Compute: mx.gather_qmm() for MoE block
  ├── 6. Defer: mx.async_eval(MoE_output)  ← non-blocking submit
  │      ▼ GPU processes MoE matrices in background
  │
  └── 7. Prefetch: madvise(MADV_WILLNEED) for token t+1's predicted experts
         └─ During GPU MoE compute window, populates UBC asynchronously
```

Step 7 refinement: after MoE output is consumed and LRU eviction fires, immediately call `madvise(MADV_DONTNEED)` on evicted/cold expert pages to prevent UBC bloat.

## Implementation Phases

### Phase 1: Sidecar Format + mmap Access ← CURRENTLY IMPLEMENTING
**Files:** `omlx/streaming/sidecar.py`, `omlx/streaming/__init__.py`

StreamingExpertSidecar:
- Fixed 4096-byte header: `[4B: len][JSON][null-pad to 4096]`
- Expert 0 always at byte offset 4096
- `__init__()`: mmap entire file, parse header, build `{(layer, expert): (mmap_ptr, length)}` LUT
- `get_ptr(layer, expert) -> int`: memory offset into mmap region
- `madvise_willneed(layer, expert)`: call `madvise(MADV_WILLNEED)` on expert's pages
- `madvise_dontneed(layer, expert)`: call `madvise(MADV_DONTNEED)` on expert's pages
- `create()`: scan safetensors, extract expert bytes, write sidecar with fixed header alignment

### Phase 2: ExpertSlotBank (Three-Tier)
**File:** `omlx/streaming/slot_bank.py`

- Hot slots: pinned in GPU, populated at init, never evicted
- Warm slots: LRU-managed, populated on miss via memcpy from mmap
- Transient slots: fixed set of pre-allocated recycling buffers, overwritten per token
- `resolve(expert_ids) → (stacked_weights, slot_per_expert)`
- Eviction hook: `madvise(MADV_DONTNEED)` on warm→cold transitions

### Phase 2b: C++ pybind11 Buffer Utility
**File:** `omlx/streaming/_buffer_access.cpp` (or similar)

- Extracts `void*` raw pointer from `mx.array` for zero-copy slot bank writes
- `pread_into_buffer(fd, offset, len, mx_array, slot_idx)`: direct pread into slot bank GPU buffer
- `memcpy_expert_to_slot(mmap_ptr, length, mx_array, slot_idx)`: memcpy from mmap to GPU buffer
- Builds as optional extension; falls back to `mx.array(np.frombuffer(...))` if not compiled

### Phase 3: SwitchGLU Monkey-Patch
**File:** `omlx/streaming/patch.py`

- Replace `SwitchLinear.__call__` to route through slot bank
- Gate/up fusion: single slot bank read for both projections
- Free original 3D weight tensors from GPU memory after patching

### Phase 4: Model Loading Integration
**Files:** `omlx/utils/model_loading.py`, `omlx/engine/batched.py`

- Per-model config: `stream_experts`, `expert_sidecar_path`, `expert_warm_slots`
- Auto-detect sidecar in model directory
- Wire mmap + slot bank init into BatchedEngine.start()
- Free non-expert GPU memory after slot bank initialized

### Phase 5 + 6 (Fused): Precise Pipeline Execution
**Files:** `omlx/patches/mlx_lm_mtp/qwen35_model.py`, `omlx/streaming/pipeline.py`

Per-token per-layer execution gating:
1. `mx.eval(x_attn, router_logits)` — blocking flush, GPU→IDLE
2. Cache check + I/O window (memcpy from mmap into transient/warm slots)
3. `mx.gather_qmm()` — MoE compute
4. `mx.async_eval(y)` — deferred submission
5. `madvise(MADV_WILLNEED)` — prefetch next-token predicted experts
6. `madvise(MADV_DONTNEED)` — evict consumed cold experts

Temporal prefetch uses `madvise(MADV_WILLNEED)` instead of ThreadPoolExecutor + `os.pread()`.
This is kernel-level async page-in — no Python overhead, no GIL, no user-space buffer allocation.

### Phase 7: Testing
**File:** `tests/test_streaming.py`

- Bit-identical output vs full-RAM baseline
- GPU memory reduction verification (should see ~50%+ reduction for A3B)
- UBC memory stability: verify resident memory doesn't grow unbounded
- Throughput benchmarks at multiple context lengths

### Phase 8: DeepSeek-V4 Support
**File:** `omlx/patches/deepseek_v4/deepseek_v4_model.py`

- Hash-routed layers (100% predictable, deterministic expert selection)
- Fine-grained experts: clustered packing in sidecar for amortized madvise
- Different router interface, same slot bank architecture

## Configuration

Per-model settings (in `model_settings.json` or equivalent):
```json
{
  "stream_experts": true,
  "expert_sidecar_path": "~/.omlx/sidecars/Qwen3.5-35B-A3B-4bit.streaming",
  "expert_warm_slots": 64,
  "expert_hot_experts": 13,
  "expert_transient_slots": 8,
  "expert_prefetch": true,
  "expert_top_k_override": null
}
```

## macOS UBC Protection via madvise

Without explicit UBC control, streaming a 209 GB model causes the Darwin kernel to
consume all free unified RAM for the Unified Buffer Cache (file pages), triggering
memory compression of active anonymous allocations (Python heap, MLX tensors).

The mmap + madvise strategy prevents this:
- **MADV_WILLNEED**: Kernel prefetches pages from NVMe asynchronously. No Python I/O.
- **MADV_DONTNEED**: Instantly frees physical pages after expert consumption. File
  mapping is read-only (backed by SSD), so no write-back needed.
- **Zero-copy access**: `mx.array` wraps the mmap'd region directly, or memcpy
  into pre-allocated GPU slot buffers.

This shifts the entire I/O and cache management workload to the Darwin kernel,
keeping the Python runtime lightweight and UBC-safe.
