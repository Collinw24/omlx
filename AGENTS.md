# AGENTS.md | omlx: feat/ssd-expert-streaming

## 1. Context Architecture & 96k Window Hygiene
Operating under a strict local 96k token context window. Voluminous data ingest causes tool execution timeout/OOM interrupts.
* **No Raw Ingestion:** Never use broad file reads or text searches to parse codebase structures.
* **Sandboxed Code Execution:** Use `context-mode:ctx_execute` to run local analysis scripts (Node/Python/Shell) for grep or AST parsing. Output only target line references or micro-JSON strings via console logs.
* **Compaction Triggering:** Let `pi-ultra-compact` handle auto-compression at 80%. Explicitly invoke the `/ultracompact` macro after heavy turns to checkpoint state before starting new complex code blocks.

## 2. System Objectives & Architecture
Streams Qwen3.5-397B-A17B MoE weights (209 GB) from NVMe to pre-allocated GPU buffers per token. Target: 48-64 GB Apple Silicon UMA.
* **Core Mechanism:** `F_NOCACHE` direct I/O bypasses macOS UBC; C++ `pybind11` writes `pread` bytes directly into Metal-backed `mx.array` buffers (zero Python heap allocation in hot path).
* **Disk Layout:** Byte 0: `[4B: header_len][JSON header][null padding to 16384B]`. Byte 16384+: Contiguous blocks of packed gate+up+down weights. 16KB-aligned for Metal UMA buffer wrapping. One `os.pread` per expert/token.
* **Quantized Layout (--quant 4):** Each projection stores `[qw(uint32) | scales(float32) | biases(float32)]` in MLX's packed 4-bit layout, enabling fused `mx.gather_qmm` dispatch with on-the-fly dequant in Metal registers.

### Slot Bank Tiers & Pipeline
| Tier | Count | Eviction | Rule / Use Case |
| :--- | :--- | :--- | :--- |
| **Hot** | 13 | Never | Pinned at init from calibration data |
| **Warm** | 64 | LRU + EMA | Promote from transient if EMA ≥ 2.0 |
| **Transient** | 8 | Circular | Overwrite cold misses below EMA threshold |

* **EMA Prefetch:** $\alpha = 2/(\text{window}+1)$ (Default window=4, $\alpha=0.4$). `ema[id] = \alpha * 1.0 + (1-\alpha) * ema.get(id, 0.0)`. Predict $t+1$: `actual_routing[t] ∪ top_2_ema[t]`. Prefetch to warm slots only.
* **Per-Token Pipeline:** (1) Router logits (lazy graph) → (2) `mx.eval(router_logits)` *BLOCKING* → (3) `slot_bank.resolve(top_k_ids)` *pread cold experts* → (4) `mx.gather_qmm` (quantized) or `mx.take+mx.matmul` (float32) → (5) Prefetch during MoE compute. **Note:** Do not overlap GPU and DMA execution concurrently (-73% UMC throughput drop).

## 3. Implementation Status & Phase Gates
* **Active Phase:** Phase 4 (hardening) / Phase 7d (blocked). Track progress strictly by sub-phase. Do not implement across phases concurrently.
* **Current State:**
  * *Phase 1 (sidecar.py):* COMPLETE. Core missing hooks: 1a (pluggable parser), 1b (single-pass header), 1d (`verify()`).
  * *Phase 2 (slot_bank.py):* **COMPLETE.** 3-tier cascade, **mx.array buffers** (migrated from bytearray), circular transient, stacked returns, zero-allocation views (I9). 17/17 tests.
  * *Phase 2b (_buffer_access.cpp):* **COMPLETE.** C++ pybind11 extension (buffer protocol) reads bytes from NVMe directly into any writable buffer. 10 unit tests, integrated into sidecar.
  * *Phase 3 (patch.py):* **COMPLETE.** SwitchGLU monkey-patching, weight reclamation, **quantized `gather_qmm` dispatch** (bits=4, group_size=32-128), float32 matmul fallback.
  * *Phase 3b (unpatch):* Implemented — restores `__call__`, cleans metadata.
  * *Phase 3c (apply_streaming_patches):* Implemented — batch-patches all SwitchGLU layers.
  * *Phase 4 (config, pipeline, integration):* **IMPLEMENTED.** `StreamingConfig` schema, `load_model_with_streaming`, `BatchedEngine` wiring, `omlx create-sidecar` CLI command, `EMATrajectoryPrefetcher`. End-to-end synthetic pipeline test passes.
  * *Phase 5/6 (runtime, prefetch):* Partially implemented (prefetcher class exists; not yet wired into token loop).
  * *Phase 8 (hash_router.py):* DeepSeek stub. Hard block: Raise `NotImplementedError`, do not write.
* **Testing Gate Sequence:** 7a (Pass) → 7b (Pass) → 7c (Pass) → 2b (unit tests + integration) → 7d (pending).
  * *Gate 7c Rule:* Must match `mx.allclose(stream_output, ref_output, atol=1e-4)` on Layer 0 using synthetic weights. **PASSED** with 17/17 tests.
  * *Gate 2b Unit:* C++ extension passes buffer protocol smoke test (10 tests: basic read, offset, 2D slot, sub-slot, cross-slot, bytearray, short read, dtype validation, readonly validation, zero-length).
  * *Gate 2b Integration:* `read_expert_into()` delegates to C++ extension when available; falls back to Python fallback.
  * *Gate 7d Rule:* Full model test with Qwen3.5-35B-A3B — pending GPU resources (full model triggers Metal timeout).

## 4. Engineering Invariants & Guardrails
* **I1 (Hot Path):** Zero dynamic allocations inside `resolve()` or patched `__call__`. No `mx.array()`, `np.frombuffer()`, or heap churn.
* **I2 (Graph Gate):** `mx.eval(router_logits)` must complete before slot mutation. GPU must be idle. Caller owns the gate.
* **I3 (Direct I/O):** `F_NOCACHE` must verify via `fcntl`. If failed: log warning, flag `self._nocache_active = False`. Never silently fall back.
* **I4 (No Compile):** Absolute ban on `@mx.compile` or `mx.compile()` anywhere in streaming code. Avoids SIGSEGV on unmaterialized mmap weights.
* **I5 (Inert Default):** `stream_experts: false` must remain completely inert. Zero execution overhead inside `BatchedEngine.start()`.
* **I6 (Immutability):** Original `.safetensors` are never mutated. Non-expert weights remain lazy-mmap.

## 5. Empirical Questions & Coding Rules
* **Q1-Q4 Flags:** Flag explicitly if coding around: `mx.gather_qmm` command buffer segments (Q1), `mx.eval()` completion guarantees (Q2), 4KB vs 16KB Metal alignment boundary verification (Q3), or `F_NOCACHE` pread latency scaling under heavy memory pressure (Q4).
* **Standards:** Docstrings must explain *WHY* (Metal memory anomalies). No `print()`, use `logging.getLogger(__name__)`. Explicit types required.
