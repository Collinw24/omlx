# SSD Expert Streaming for oMLX

Branch: `feat/ssd-expert-streaming` | Fork: [Collinw24/omlx](https://github.com/Collinw24/omlx)

## Goal

Enable running large Mixture-of-Experts models (Qwen3.5-397B-A17B, 209 GB) on Apple Silicon machines with limited unified memory (48-64 GB) by streaming only the top-K active expert weights per token from NVMe SSD.

Initial target: **Qwen3.5-35B-A3B** (fits in RAM for correctness verification).
Aspirational target: **Qwen3.5-397B-A17B** (209 GB on disk).

## Architecture

```
StreamingExpertSidecar (on-disk, one .sidecar file)
  ┌─────────┬──────────┬──────────┬─────┬──────────┐
  │ Header  │ Expert 0 │ Expert 1 │ ... │ Expert N │
  │ (JSON)  │ (4KB pad)│ (4KB pad)│     │ (4KB pad)│
  └─────────┴──────────┴──────────┴─────┴──────────┘
  Each expert chunk: gate_proj + up_proj + down_proj contiguous
  4KB sector-aligned → single pread() per expert
         │
         ▼
ExpertSlotBank (per-layer, in GPU unified memory)
  ┌────────────────┬──────────────────────────────────┐
  │ Hot Slots (13) │ Warm Slots (64, LRU-managed)     │
  │ MTLResidencySet│ Persistent Metal buffers (4-bit) │
  │ Never evicted  │                                  │
  └────────────────┴──────────────────────────────────┘
  Cold experts (bottom ~70%): stream directly, no caching
         │
         ▼
mx.gather_qmm(x, stacked_weights, rhs_indices=slot_ids)
  └─ Single fused Metal dispatch, on-the-fly dequant (FMA)
```

## Key Technical Decisions

1. **Non-mutative sidecar**: Original `.safetensors` untouched. Non-expert weights remain native lazy-mmap.

2. **4KB sector alignment**: Every expert chunk padded to 4096 bytes for F_NOCACHE direct I/O support.

3. **Contiguous expert blocks**: gate_proj + up_proj + down_proj packed into one byte range. One `pread()` per expert instead of three.

4. **Tiered caching** (power-law routing distribution):
   - Hot: top 5% experts, pinned, never evicted
   - Warm: middle 25%, LRU slot bank
   - Cold: bottom 70%, stream-thru per use

5. **Serial I/O pipeline** (hardware-optimal on unified memory):
   ```
   GPU: attention+routing → IDLE → expert gather_qmm → IDLE
   CPU:                   pread()         submit async
   ```
   GPU+DMA concurrent overlap causes -73% GPU slowdown on Apple Silicon UMC.

6. **No `mx.compile()`** on streaming path due to MLX mmap null-pointer SIGSEGV bug.

7. **`mx.gather_qmm`** dequantizes on-the-fly in Metal kernel registers. Weights stay packed 4-bit in persistent slot buffers.

8. **Temporal prefetch**: During `mx.eval()` GIL-release window, `ThreadPoolExecutor` preads next-token predicted experts into OS page cache (~70% hit rate).

## Implementation Phases

### Phase 1: Sidecar Format + Repacker ← CURRENT
**Files:** `omlx/streaming/sidecar.py`, `omlx/streaming/__init__.py`
**CLI:** `omlx repack-experts` (later)

StreamingExpertSidecar class:
- JSON header with byte-offset lookup table per (layer, expert)
- `create()`: scan safetensors, extract expert weights, write contiguously
- `read_expert(layer, expert) → bytes`: single `os.pread()` call

### Phase 2: ExpertSlotBank
**File:** `omlx/streaming/slot_bank.py`

- Persistent Metal buffers for hot+warm slots (4-bit packed uint8)
- Tiered LRU eviction (hot slots pinned, warm slots evicted)
- `resolve(expert_ids) → (stacked_weights, slot_per_expert)`

### Phase 3: SwitchGLU Monkey-Patch
**File:** `omlx/streaming/patch.py`

- Replace `SwitchLinear.__call__` and `SwitchGLU.__call__`
- Route weight reads through slot bank
- Free original 3D weight tensors from GPU memory

### Phase 4: Model Loading Integration
**Files:** `omlx/utils/model_loading.py`, `omlx/engine/batched.py`

- Per-model config: `stream_experts`, `expert_sidecar_path`, `expert_warm_slots`
- Auto-detect sidecar in model directory
- Wire into BatchedEngine.start()

### Phase 5: Scheduler Flush-Load-Execute Gate
**File:** `omlx/patches/mlx_lm_mtp/qwen35_model.py`

- `mx.eval(x)` before MoE block (Metal flush)
- `mx.async_eval(y)` after expert compute (deferred submit)

### Phase 6: Temporal Prefetch
**File:** `omlx/streaming/prefetch.py`

- ThreadPoolExecutor for parallel `os.pread()` during GIL window
- Record current routing, prefetch same experts for next token
- ~70% hit rate in steady-state generation

### Phase 7: Testing
**File:** `tests/test_streaming.py`

- Bit-identical output vs full-RAM baseline
- GPU memory reduction verification
- Throughput benchmarks at multiple context lengths

### Phase 8: DeepSeek-V4 Support
**File:** `omlx/patches/deepseek_v4/deepseek_v4_model.py`

- Hash-routed layers (100% predictable)
- Different expert layout, same slot bank infrastructure

## Configuration

Per-model settings (in `model_settings.json` or equivalent):
```json
{
  "stream_experts": true,
  "expert_sidecar_path": "~/.omlx/sidecars/Qwen3.5-35B-A3B-4bit.streaming",
  "expert_warm_slots": 64,
  "expert_hot_experts": 13,
  "expert_prefetch": true,
  "expert_top_k_override": null
}
```

Activation: configurable per-model, integrated like any other omlx feature.
