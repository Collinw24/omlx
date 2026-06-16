# SSD Expert Streaming for oMLX

Branch: `feat/ssd-expert-streaming` | Fork: [Collinw24/omlx](https://github.com/Collinw24/omlx)

## Goal

Enable running large Mixture-of-Experts models (Qwen3.5-397B-A17B, 209 GB) on Apple Silicon machines with limited unified memory (48-64 GB) by streaming only the top-K active expert weights per token from NVMe SSD.

Initial target: **Qwen3.5-35B-A3B** (fits in RAM for correctness verification).
Aspirational target: **Qwen3.5-397B-A17B** (209 GB on disk).

## Architecture

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
  │ Pinned MTLResidencySet│ LRU with EMA freq    │ Pre-allocated, recycled  │
  │ Never evicted         │ Persistent buffers   │ Overwritten per token    │
  └───────────────────────┴──────────────────────┴──────────────────────────┘
  ALL experts go through slots — zero dynamic mx.array allocation in hot path.
         │
         │ Phase 2b: C++ pybind11 reads raw pread bytes into slot GPU buffers
         │           via array.data<void*>() pointer — no intermediate copies
         │
         ▼
mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids)
  └─ Single fused Metal dispatch, on-the-fly dequant (FMA registers)
```

## Key Technical Decisions

1. **16KB alignment**: Apple Silicon uses 16 KiB virtual memory pages. Any pointer
   wrapped into a Metal buffer must be 16 KiB-aligned. The header is padded to
   16384 bytes; Expert 0 always at offset 16384.

2. **F_NOCACHE Direct I/O**: `fcntl(fd, F_NOCACHE, 1)` on the sidecar file
   descriptor bypasses the macOS Unified Buffer Cache entirely. `os.pread()`
   goes directly from NVMe into process buffers. No UBC bloat, no memory
   compression from file pages, no need for `madvise(DONTNEED)`.

3. **Non-mutative sidecar**: Original `.safetensors` untouched. Non-expert weights remain native lazy-mmap.

4. **Contiguous expert blocks**: gate_proj + up_proj + down_proj (weight, scales, biases) packed into one byte range per expert. Single `os.pread()` per expert.

5. **Three-tier slot bank** (ALL experts go through slots, zero dynamic allocation):
   - **Hot** (top 5%): Pinned via `MTLResidencySet`, never evicted
   - **Warm** (middle 25%): LRU-managed with EMA frequency tracking; experts
     appearing multiple times in trailing 4-token window get priority pinning
   - **Transient Streaming** (cold ~70%): Pre-allocated recycling slots,
     overwritten per token. Direct `pread()` into pre-allocated GPU buffer
     memory via C++ extension

6. **C++ pybind11 buffer access**: Python `mx.array` objects are immutable. A
   minimal C++ extension extracts `array.data<void*>()` for direct `pread()`
   into persistent slot bank GPU buffers without allocating new array objects.

7. **No `mx.compile()`** on streaming path due to MLX mmap null-pointer SIGSEGV bug.

8. **`mx.gather_qmm`**: Dequantizes on-the-fly in Metal kernel registers via FMA.
   Weights stay packed 4-bit in persistent slot buffers. Pre-packed during Phase 1
   to match MLX's internal transposed layout `[num_experts, out, in//2]`.

9. **Strict graph gating**: Synchronous `mx.eval()` on router logits clears the
   active MLX computation graph BEFORE any background thread writes into
   transient slot memory. Prevents GPU from reading corrupted expert weights
   mid-matrix-multiplication.

## Precise Pipeline Execution (per token, per layer)

```
[Token Step t]
  │
  ├── 1. GPU: attention + router logits
  ├── 2. Host: mx.eval(router_logits)  ← BLOCKING sync gate
  │      ▼ GPU drops to IDLE.  MLX graph cleared.  Safe to mutate slots.
  │
  ├── 3. Cache Check: identify missing experts (hot miss + warm miss + cold).
  ├── 4. IO Window: pread(F_NOCACHE) missing experts into Transient/Warm slots.
  │      (GPU IDLE during I/O — prevents UMC DMA contention, -73% GPU slowdown)
  │      ▼
  ├── 5. Compute: mx.gather_qmm() for MoE block
  ├── 6. Defer: mx.async_eval(MoE_output)  ← non-blocking submit
  │      ▼ GPU processes MoE matrices in background
  │
  └── 7. Prefetch: pread(F_NOCACHE) for token t+1's predicted experts.
         └─ EMA trajectory cache: union of (current prediction + top 2
            historical high-frequency experts from trailing 4-token window).
            During GPU MoE compute window — DMA completes before GPU needs data.
```

## EMA Trajectory Cache (Phase 6)

MoE routers follow a power-law semantic distribution over a sliding token window.
The prefetcher maintains:

- **Frequency index**: count of each expert's activations over trailing 4 tokens.
- **Predict**: for token `t+1`, read `actual_routing[t]` (naive) UNION `top_2_ema[t]`
  (highest-frequency experts from the trailing window, excluding already-read).
- **Pin**: experts with frequency ≥ 2 in the window get extended warm-slot retention.
- **Evict**: experts with frequency = 1 in the window are eligible for LRU eviction.

Extensions the prefetch set by 2-3 experts per token reduces cold-miss rate from
~30% to ~15%, keeping the GPU fed during the I/O window.

## Implementation Phases

### Phase 1: Sidecar Format + Direct I/O Reader ← COMPLETE
**Files:** `omlx/streaming/sidecar.py`, `omlx/streaming/__init__.py`

- 16KB-aligned header (Expert 0 at offset 16384)
- `create()`: scans safetensors, extracts expert weights, writes 16KB-aligned sidecar
- `__init__()`: opens with `O_RDONLY`, applies `F_NOCACHE`
- `read_expert()`: `os.pread()` direct from NVMe (no UBC)

### Phase 2: ExpertSlotBank (Three-Tier)
**File:** `omlx/streaming/slot_bank.py`

- Hot slots: pinned in GPU, populated at init, never evicted
- Warm slots: LRU-managed with EMA frequency tracking, persistent Metal buffers
- Transient slots: fixed set of pre-allocated recycling buffers, overwritten per token
- `resolve(expert_ids) → (stacked_weights, slot_per_expert)`
- Strict gate: only mutate slots AFTER `mx.eval(router_logits)` has returned

### Phase 2b: C++ pybind11 Buffer Utility
**File:** `omlx/streaming/_buffer_access.cpp` (minimal extension)

- `pread_into_slot(fd, offset, len, mx_array, slot_idx)`: direct `pread()` into
  persistent GPU buffer via `array.data<uint8_t>() + slot_offset`
- Falls back to `mx.array(np.frombuffer(raw))` + copy if extension not built
- Required because Python `mx.array` objects are immutable

### Phase 3: SwitchGLU Monkey-Patch
**File:** `omlx/streaming/patch.py`

- Replace `SwitchLinear.__call__` to route through slot bank
- Gate/up fusion: single slot bank read for both projections
- Free original 3D weight tensors from GPU memory after patching

### Phase 4: Model Loading Integration
**Files:** `omlx/utils/model_loading.py`, `omlx/engine/batched.py`

- Per-model config: `stream_experts`, `expert_sidecar_path`, etc.
- Auto-detect sidecar in model directory
- Wire sidecar + slot bank init into BatchedEngine.start()
- Free non-expert GPU memory after slot bank initialized

### Phase 5 + 6 (Fused): Precise Pipeline Execution
**Files:** `omlx/patches/mlx_lm_mtp/qwen35_model.py`, `omlx/streaming/pipeline.py`

Per-token per-layer execution gating:
1. `mx.eval(x_attn, router_logits)` — blocking sync gate, clears MLX graph
2. Cache check + I/O window (pread F_NOCACHE into transient/warm slots)
3. `mx.gather_qmm()` — MoE compute
4. `mx.async_eval(y)` — deferred submission
5. EMA trajectory prefetch (union: current + top-2 historical from 4-token window)
6. Eviction: free cold slot pages (no-op with F_NOCACHE — no page cache to flush)

### Phase 7: Testing
**File:** `tests/test_streaming.py`

**Layer-by-layer interceptor test harness (FIRST, before full model tests):**
- Load Qwen3.5-35B-A3B (fits in RAM for baseline)
- Run single prompt through unpatched native full-RAM model
- Use forward hook to save Layer 0: input hidden states, routing decisions, output tensor
- Run same prompt through patched streaming architecture
- Assert: `mx.allclose(stream_output, reference_output, atol=1e-5)`
- Only proceed to full model after Layer 0 achieves bit-identical convergence

**Full model tests:**
- Bit-identical output vs full-RAM baseline (full model)
- GPU memory reduction verification (~50%+ for A3B)
- UBC memory stability: verify no unbounded growth with F_NOCACHE
- Throughput benchmarks at multiple context lengths

### Phase 8: DeepSeek-V4 Support
**File:** `omlx/patches/deepseek_v4/deepseek_v4_model.py`

- Hash-routed layers (100% predictable, deterministic expert selection)
- Fine-grained experts: clustered packing in sidecar for amortized pread overhead
- Different router interface, same slot bank infrastructure

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
  "expert_prefetch_window": 4,
  "expert_top_k_override": null
}
```

Activation: configurable per-model, integrated like any other omlx feature.
