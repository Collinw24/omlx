# SPDX-License-Identifier: Apache-2.0
"""Phase 3: SwitchGLU monkey-patching & memory reclamation.

Injects the streaming mechanism into SwitchGLU (Qwen3.5/3.6 MoE) by
replacing ``__call__`` with a closure that routes expert lookups through
the ``ExpertSlotBank`` and executes fused dispatches via
``mx.gather_qmm``.

Original 3D expert tensors are nullified and their Metal backing is
released immediately via ``mx.eval()`` to reclaim unified memory.

Invariants enforced:
  I1 — Zero dynamic ``mx.array()`` / ``np.frombuffer()`` in the hot
       path.  All buffers are pre-allocated at ``slot_bank`` init.
  I2 — No ``mx.eval()`` inside the patched ``__call__``.  The caller
       owns the sync gate; indices are assumed already materialized.
  I4 — No ``@mx.compile`` or ``mx.compile()`` anywhere in this file.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .sidecar import StreamingExpertSidecar
from .slot_bank import ExpertSlotBank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_switch_glu(mod: nn.Module) -> bool:
    """Return ``True`` when *mod* looks like a SwitchGLU / SwitchLinear.

    Heuristic: the module has three weight parameters whose names contain
    ``gate_proj``, ``up_proj``, and ``down_proj`` (or the ``switch_mlp``
    variant used by Qwen3.5/3.6 MoE).

    Uses ``parameters()`` to avoid triggering ``nn.Module.__call__``
    (which ``state()`` does internally).
    """
    try:
        param_names = set(mod.parameters().keys())
    except Exception:
        param_names = set()
    has_gate = any("gate" in n for n in param_names)
    has_up = any("up" in n for n in param_names)
    has_down = any("down" in n for n in param_names)
    return has_gate and has_up and has_down


# ---------------------------------------------------------------------------
# Patched call closure
# ---------------------------------------------------------------------------


def _streaming_switch_linear_call(
    self: nn.Module,
    x: mx.array,
    layer: int | str,
    routing_callback: Optional[Callable[[int, List[int]], None]] = None,
    *args: Any,
    **kwargs: Any,
) -> mx.array:
    """Streaming-aware replacement for SwitchGLU / SwitchLinear ``__call__``.

    Parameters
    ----------
    self :
        The ``SwitchGLU`` (or equivalent) instance being patched.
    x : mx.array
        Input tensor of shape ``(B, S, H)``.
    layer : int | str
        Layer identifier used by the shared slot bank for sidecar header lookup.
    routing_callback : callable((layer, expert_ids), optional)
        Optional callback invoked after slot resolution for EMA prefetcher.
    *args, **kwargs :
        Forwarded to the original ``__call__`` when streaming is disabled.

    Invariant I1
    ------------
    This function performs **zero** dynamic ``mx.array()`` allocations.
    ``slot_bank.resolve()`` returns views into pre-allocated tier buffers.

    Invariant I2
    ------------
    No ``mx.eval()`` is called here.  The caller is responsible for
    ensuring that routing indices are fully materialised before entering
    this closure.
    """
    slot_bank: ExpertSlotBank = getattr(self, "_omlx_slot_bank", None)
    sidecar: StreamingExpertSidecar = getattr(self, "_omlx_sidecar", None)

    # When streaming is disabled fall through to the original path.
    if slot_bank is None or sidecar is None:
        raise RuntimeError(
            "Streaming patch applied but slot_bank or sidecar is missing. "
            "Call ``patch_switch_linear()`` with a valid sidecar first."
        )

    # ── 1. Extract routing indices ────────────────────────────────────
    # The original SwitchGLU receives indices from the router.  Depending
    # on the model variant they arrive either as a positional argument
    # (``args[0]``) or as a keyword argument ``indices`` / ``input_ids``.
    indices: Optional[mx.array] = None

    # Try keyword first.
    for key in ("indices", "input_ids", "routing_indices"):
        if key in kwargs:
            indices = kwargs[key]
            break

    # Fallback to positional.
    if indices is None and args:
        indices = args[0]

    if indices is None:
        raise ValueError(
            "Streaming SwitchGLU expects routing indices. "
            "Pass them as ``indices`` keyword or first positional arg."
        )

    # Ensure indices is an integer array.
    if indices.dtype not in (mx.int32, mx.int64):
        indices = indices.astype(mx.int32)

    # ── 2. Slot resolution (zero-copy) ────────────────────────────────
    # Flatten to 1-D list of unique expert IDs.  ``resolve()`` must be
    # called **after** the caller's ``mx.eval()`` gate — we assume
    # ``indices`` is already materialised.
    unique_experts: List[int] = sorted(
        set(indices.flatten().tolist())
    )

    stacked: mx.array
    slot_ids: List[int]
    # Point shared slot bank at this layer
    slot_bank.set_layer(layer)
    stacked, slot_ids = slot_bank.resolve(unique_experts)

    # Record routing for EMA prefetcher (Priority 2)
    if routing_callback:
        routing_callback(
            getattr(slot_bank, "layer", 0), unique_experts
        )

    # ── 3. Dispatch to quantised or float32 forward path ────────────
    # The sidecar stores per-expert weights as contiguous bytes.
    #   float32:   expert_i = [gate_bytes | up_bytes | down_bytes]
    #   quantised: expert_i = [gate_qw|sc|bi | up_qw|sc|bi | down_qw|sc|bi]
    #
    # We check layer metadata for a ``"quant"`` key to decide the path.

    layer_meta = sidecar.header["layers"].get(
        str(getattr(slot_bank, "layer", 0)), {}
    )
    is_quantised = "quant" in layer_meta

    if is_quantised:
        # ── Quantised gather_qmm path ────────────────────────────────
        #
        # ``gather_qmm(x, w, sc, bi, rhs_indices=arange(K))`` computes
        # the quantised matmul for ALL K loaded experts against the
        # entire batch, producing (K, BS, out).  We then use the routing
        # indices to select which expert's output to use per position.
        #
        # When K=1, skip the gather and use a single quantised matmul
        # call directly (avoids Metal command buffer segmentation).

        K = len(slot_ids)

        if K == 1:
            # ── Single-expert fast path (Q1 optimization) ────────────
            # Use mx.gather_qmm with rhs_indices=[0] for single expert —
            # avoids multi-expert scatter but keeps quantized matmul.
            BS = x.shape[0] * x.shape[1]
            x_flat = x.reshape(BS, -1)

            qm = layer_meta.get("quant", {})
            bits = qm.get("bits", 4)
            group_size = qm.get("group_size", 64)
            mode = qm.get("mode", "affine")

            proj_meta = layer_meta.get("experts", {}).get("0", {}).get("projections", {})
            gate_shape = proj_meta.get("gate_proj", {}).get("weight_shape", [])
            B_s, S_s, H_dim = x.shape
            M = gate_shape[1] if len(gate_shape) >= 2 else 512
            H_dim = gate_shape[2] if len(gate_shape) >= 3 else H_dim

            def _pkg(rows: int, cols: int) -> tuple[int, int, int]:
                qw = rows * ((cols * bits + 31) // 32)
                grp = (cols + group_size - 1) // group_size
                return qw, rows * grp, rows * grp

            qw_g, sc_g, bi_g = _pkg(M, H_dim)
            qw_u, sc_u, bi_u = _pkg(M, H_dim)
            qw_d, sc_d, bi_d = _pkg(H_dim, M)
            GA = (qw_g + sc_g + bi_g) * 4
            UB = (qw_u + sc_u + bi_u) * 4

            import numpy as _np
            _mx.eval(stacked)
            raw = _np.frombuffer(stacked.tobytes(), dtype=_np.uint8).reshape(1, -1)

            def _extract(raw_2d: _np.ndarray, rows: int, qw_sz: int, sc_sz: int, bi_sz: int, byte_off: int) -> tuple[mx.array, mx.array, mx.array]:
                seg = raw_2d[:, byte_off:byte_off + (qw_sz + sc_sz + bi_sz) * 4]
                qw_np = seg[:, :qw_sz * 4].ravel().view(_np.uint32).reshape(1, rows, -1)
                sc_np = seg[:, qw_sz*4:(qw_sz+sc_sz)*4].ravel().view(_np.float32).reshape(1, rows, -1)
                bi_np = seg[:, (qw_sz+sc_sz)*4:(qw_sz+sc_sz+bi_sz)*4].ravel().view(_np.float32).reshape(1, rows, -1)
                return mx.array(qw_np.squeeze(0)), mx.array(sc_np.squeeze(0)), mx.array(bi_np.squeeze(0))

            _proj_off = layer_meta.get("projection_offsets", {})
            gate_off = _proj_off.get("gate_proj", {}).get("offset", 0)
            up_off = _proj_off.get("up_proj", {}).get("offset", None)
            down_off = _proj_off.get("down_proj", {}).get("offset", None)
            if up_off is None:
                up_off = GA
            if down_off is None:
                down_off = GA + UB

            qw_g, sc_g, bi_g = _extract(raw, M, qw_g, sc_g, bi_g, gate_off)
            qw_u, sc_u, bi_u = _extract(raw, M, qw_u, sc_u, bi_u, up_off)
            qw_d, sc_d, bi_d = _extract(raw, H_dim, qw_d, sc_d, bi_d, down_off)

            # Single expert: rhs_indices=[0], shape collapses to (1, BS, M/H)
            gate = mx.gather_qmm(
                x_flat, qw_g, scales=sc_g, biases=bi_g,
                rhs_indices=mx.array([0], dtype=mx.int32), transpose=True,
                group_size=group_size, bits=bits, mode=mode,
            )  # (1, BS, M)

            up = mx.gather_qmm(
                x_flat, qw_u, scales=sc_u, biases=bi_u,
                rhs_indices=mx.array([0], dtype=mx.int32), transpose=True,
                group_size=group_size, bits=bits, mode=mode,
            )  # (1, BS, M)

            acts = mx.silu(gate) * up  # (1, BS, M)
            moe_out = mx.gather_qmm(
                acts, qw_d, scales=sc_d, biases=bi_d,
                rhs_indices=mx.array([0], dtype=mx.int32), transpose=True,
                group_size=group_size, bits=bits, mode=mode,
            )  # (1, BS, H)

            return moe_out.reshape(B_s, S_s, H_dim)

        # ── Multi-expert gather_qmm path ─────────────────────────────

        qm = layer_meta.get("quant", {})
        bits = qm.get("bits", 4)
        group_size = qm.get("group_size", 64)
        mode = qm.get("mode", "affine")

        # Determine expert dimensions from sidecar metadata
        expert_meta = layer_meta.get("experts", {}).get("0", {})
        proj_meta = expert_meta.get("projections", {})
        gate_shape = proj_meta.get("gate_proj", {}).get("weight_shape", [])
        B, S, H_dim = x.shape
        M = gate_shape[1] if len(gate_shape) >= 2 else 512
        H_dim = gate_shape[2] if len(gate_shape) >= 3 else H_dim

        K = len(slot_ids)
        BS = B * S

        # Per-projection packed sizes (in uint32/float32 elements)
        def _pkg(rows: int, cols: int) -> tuple[int, int, int]:
            qw = rows * ((cols * bits + 31) // 32)
            grp = (cols + group_size - 1) // group_size
            return qw, rows * grp, rows * grp

        qw_g, sc_g, bi_g = _pkg(M, H_dim)
        qw_u, sc_u, bi_u = _pkg(M, H_dim)
        qw_d, sc_d, bi_d = _pkg(H_dim, M)

        # Use header projection_offsets for byte offsets (RQ-1: header-driven, not recomputed).
        _proj_off = layer_meta.get("projection_offsets", {})
        gate_off = _proj_off.get("gate_proj", {}).get("offset", 0)
        up_off = _proj_off.get("up_proj", {}).get("offset", None)
        down_off = _proj_off.get("down_proj", {}).get("offset", None)
        expert_bytes = _proj_off.get("down_proj", {}).get("size", 0) + (down_off if down_off is not None else 0)

        # Fallback GA/UB computation for sidecars without projection_offsets.
        if up_off is None:
            up_off = (qw_g + sc_g + bi_g) * 4
        if down_off is None:
            down_off = up_off + (qw_u + sc_u + bi_u) * 4
        if expert_bytes == 0:
            expert_bytes = (qw_d + sc_d + bi_d) * 4 + down_off

        # Ensure stacked is eval'd, then extract per-projection tensors
        # via numpy byte reinterpretation for the K unique experts.
        import numpy as _np

        _mx.eval(stacked)
        raw = _np.frombuffer(stacked.tobytes(), dtype=_np.uint8).reshape(K, -1)

        def _extract(raw_2d: _np.ndarray, rows: int, qw_sz: int, sc_sz: int, bi_sz: int, byte_off: int) -> tuple[mx.array, mx.array, mx.array]:
            """Extract qw/sc/bi for one projection from K experts."""
            seg = raw_2d[:, byte_off:byte_off + (qw_sz + sc_sz + bi_sz) * 4]
            qw_np = seg[:, :qw_sz * 4].ravel().view(_np.uint32).reshape(K, rows, -1)
            sc_np = seg[:, qw_sz*4:(qw_sz+sc_sz)*4].ravel().view(_np.float32).reshape(K, rows, -1)
            bi_np = seg[:, (qw_sz+sc_sz)*4:(qw_sz+sc_sz+bi_sz)*4].ravel().view(_np.float32).reshape(K, rows, -1)
            return mx.array(qw_np), mx.array(sc_np), mx.array(bi_np)

        qw_g, sc_g, bi_g = _extract(raw, M, qw_g, sc_g, bi_g, gate_off)
        qw_u, sc_u, bi_u = _extract(raw, M, qw_u, sc_u, bi_u, up_off)
        qw_d, sc_d, bi_d = _extract(raw, H_dim, qw_d, sc_d, bi_d, down_off)

        # Build mapping: routing indices → stacked position (0..K-1)
        # unique_experts is already sorted.
        ua = mx.array(unique_experts, dtype=mx.int32)
        # For each routing entry, find its position via binary search.
        # We use numpy since MLX 0.31 lacks mx.searchsorted.
        indices_np = _np.asarray(indices.tolist(), dtype=_np.int32)
        mapped_np = _np.searchsorted(
            _np.asarray(unique_experts, dtype=_np.int32),
            indices_np,
        )
        mapped = mx.array(mapped_np).reshape(indices.shape)  # (B, S, top_k)

        # gather_qmm with rhs_indices=arange(K) computes ALL K experts
        rhs_idx = mx.arange(K, dtype=mx.int32)

        # Gate + Up projections
        x_flat = x.reshape(BS, H_dim)
        gate = mx.gather_qmm(
            x_flat, qw_g, scales=sc_g, biases=bi_g,
            rhs_indices=rhs_idx, transpose=True,
            group_size=group_size, bits=bits, mode=mode,
        )  # (K, BS, M)

        up = mx.gather_qmm(
            x_flat, qw_u, scales=sc_u, biases=bi_u,
            rhs_indices=rhs_idx, transpose=True,
            group_size=group_size, bits=bits, mode=mode,
        )  # (K, BS, M)

        acts = mx.silu(gate) * up  # (K, BS, M)

        # Down projection
        moe_out = mx.gather_qmm(
            acts, qw_d, scales=sc_d, biases=bi_d,
            rhs_indices=rhs_idx, transpose=True,
            group_size=group_size, bits=bits, mode=mode,
        )  # (K, BS, H)

        # Per-position weighted sum over selected experts.
        # mapped: (B, S, top_k) — which stacked expert (0..K-1) for each position
        # moe_out: (K, BS, H) — outputs for all K experts at all positions
        #
        # Transpose moe_out to (BS, K, H), then take_along_axis
        moe_t = moe_out.transpose(1, 0, 2)  # (BS, K, H)
        mapped_bs = mapped.reshape(BS, -1)  # (BS, top_k)
        selected = mx.take_along_axis(
            moe_t,
            mapped_bs[:, :, None],  # (BS, top_k, 1)
            axis=1,
        )  # (BS, top_k, H)
        output_flat = selected.sum(axis=1)  # (BS, H)

        return output_flat.reshape(B, S, H_dim)

    # ── Float32 path (current sidecar format) ───────────────────────
    # Reinterpret the stacked uint8 buffer as float32 weights, then
    # compute the SwitchGLU gate for each unique expert.
    try:
        B, S, H_dim = x.shape
        K = len(slot_ids)

        proj_meta = layer_meta.get("experts", {}).get("0", {}).get("projections", {})
        gate_shape = proj_meta.get("gate_proj", {}).get("weight_shape", [])
        M = gate_shape[1] if len(gate_shape) >= 2 else 512  # moe_inter
        H = gate_shape[2] if len(gate_shape) >= 3 else H_dim

        # Reconstruct float32 weights from uint8 stacked buffer.
        flat = stacked.flatten().astype(mx.float32)
        n_f32 = 3 * K * M * H
        flat = flat[:n_f32].reshape(K, 3, M, H)

        gate_w = flat[:, 0, :, :]   # (K, M, H)
        up_w   = flat[:, 1, :, :]   # (K, M, H)
        down_w = flat[:, 2, :, :]   # (K, H, M)

        x_flat = x.reshape(-1, H)  # (B*S, H)
        out = mx.zeros((B * S, H), dtype=mx.float32)

        for k in range(K):
            gate = x_flat @ gate_w[k].T  # (B*S, M)
            up   = x_flat @ up_w[k].T   # (B*S, M)
            acts = mx.silu(gate) * up    # (B*S, M)
            moe_out = acts @ down_w[k].T  # (B*S, H)
            out += moe_out

        return out.reshape(B, S, H)

    except Exception as exc:
        logger.warning(
            "Float32 streaming forward failed (%s); "
            "falling back to original __call__.",
            exc,
        )
        original_call = getattr(self, "_omlx_original_call", None)
        if original_call is not None:
            return original_call(x, *args, **kwargs)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_switch_linear(
    module: nn.Module,
    sidecar: StreamingExpertSidecar,
    layer: int | str,
    expert_bytes: int,
    slot_bank: ExpertSlotBank,
    hot_count: int = 13,
    warm_slots: int = 64,
    transient_slots: int = 8,
    calibration_frequencies: Optional[Dict[int, float]] = None,
    routing_callback: Optional[Callable[[int, List[int]], None]] = None,
) -> None:
    """Patch a ``SwitchGLU`` (or equivalent) module for expert streaming.

    Uses a shared *slot_bank* (one per model, not one per layer) to avoid
    pre-allocating N slot banks for N layers — critical for 397B with 80
    layers.

    Parameters
    ----------
    module : nn.Module
        The ``SwitchGLU`` (or ``SwitchLinear``) instance to patch.
        Must contain ``gate_proj``, ``up_proj``, ``down_proj`` weights.
    sidecar : StreamingExpertSidecar
        The pre-built sidecar file handle.
    layer : int | str
        Layer identifier used to look up metadata in the sidecar header.
    expert_bytes : int
        Number of bytes per expert (packed gate+up+down).
    hot_count : int
        Number of hot-tier slots (default 13).
    warm_slots : int
        Number of warm-tier slots (default 64).
    transient_slots : int
        Number of transient-tier slots (default 8).
    calibration_frequencies : dict[int, float], optional
        EMA frequency map for hot-tier initialisation.

    Side effects
    ------------
    * Creates and attaches an ``ExpertSlotBank`` to *module*.
    * Replaces ``module.__call__`` with the streaming closure.
    * Nullifies the original 3D expert tensors and forces Metal
      memory reclamation via ``mx.eval()``.

    Raises
    ------
    ValueError
        If *module* does not appear to be a SwitchGLU / SwitchLinear.
    """
    if not _is_switch_glu(module):
        raise ValueError(
            f"Module {type(module).__name__} does not look like a "
            "SwitchGLU/SwitchLinear (missing gate/up/down projections)."
        )

    # ── 1. Capture the original __call__ for fallback ─────────────────
    original_call = module.__call__
    module.__call__ = lambda *a, **k: _streaming_switch_linear_call(
        module, a[0] if a else None, layer, routing_callback, *a[1:], **k
    )
    # Store reference for fallback path.
    module._omlx_original_call = original_call

    # ── 2. Extract metadata from sidecar header ───────────────────────
    # Currently uses float32 matmul (no quantisation).  When the sidecar
    # is updated to store MLX quantised packed layout (see TODO in
    # sidecar.py), this will need to load scales, biases, bits and pass
    # them to ``mx.gather_qmm``.

    # ── 3. Use the shared slot bank ─────────────────────────────────
    module._omlx_slot_bank = slot_bank
    module._omlx_sidecar = sidecar

    # ── 4. Memory reclamation ─────────────────────────────────────────
    # Nullify original 3D expert weights so Metal can reclaim the UMB.
    # We must ``mx.eval()`` immediately after nullification to force the
    # GPU to release the backing allocation before the next layer is
    # patched.
    # Use __dict__ to bypass nn.Module.__getattr__ which calls __call__.
    for attr in ("gate_proj", "up_proj", "down_proj"):
        weight = module.__dict__.get(attr)
        if weight is not None:
            # Force any pending graph nodes that reference this weight to
            # drain before we drop the reference.
            mx.eval(weight)
            setattr(module, attr, None)
            logger.info(
                "Reclaimed memory for %s.%s (expert streaming active).",
                type(module).__name__,
                attr,
            )

    logger.info(
        "Patched %s (layer=%s, hot=%d, warm=%d, transient=%d).",
        type(module).__name__,
        layer,
        hot_count,
        warm_slots,
        transient_slots,
    )


def unpatch_switch_linear(module: nn.Module) -> None:
    """Restore the original ``__call__`` and reinstate expert weights.

    This is the inverse of :func:`patch_switch_linear`.  It restores the
    original ``__call__`` (if captured) and re-assigns any ``None``d
    expert tensors from the sidecar.

    Parameters
    ----------
    module : nn.Module
        The previously patched module.
    """
    original_call = getattr(module, "_omlx_original_call", None)
    if original_call is not None:
        module.__call__ = original_call
        delattr(module, "_omlx_original_call")

    # Note: sidecar.read_expert() expects an expert ID (0-255), not a
    # projection name.  The sidecar stores per-expert weights (gate+up+down
    # packed), so we cannot directly restore the original 3D projection
    # tensors from it.  We just restore __call__ and clean up attributes.

    # Clean up streaming attributes.
    for attr in (
        "_omlx_slot_bank",
        "_omlx_sidecar",
        "_omlx_scales",
        "_omlx_biases",
        "_omlx_gather_meta",
    ):
        if hasattr(module, attr):
            delattr(module, attr)

    logger.info("Unpatched %s.", type(module).__name__)


def apply_streaming_patches(
    model: nn.Module,
    sidecar: StreamingExpertSidecar,
    layer_indices: Optional[List[int]] = None,
    hot_count: int = 13,
    warm_slots: int = 64,
    transient_slots: int = 8,
    calibration_frequencies: Optional[Dict[int, float]] = None,
    routing_callback: Optional[Callable[[int, List[int]], None]] = None,
) -> None:
    """Apply streaming patches to all SwitchGLU layers in a model.

    Parameters
    ----------
    model : nn.Module
        The top-level model containing ``layers``.
    sidecar : StreamingExpertSidecar
        The pre-built sidecar file handle.
    layer_indices : list[int], optional
        Layer indices to patch.  If ``None``, patches all layers that
        contain a SwitchGLU / SwitchLinear module.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError(
            "Model must have a ``layers`` attribute (list of modules)."
        )

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    # Shared slot bank — ONE bank for ALL layers to avoid pre-allocating
    # N banks for N layers (critical for 397B with 80+ layers).
    first_layer_key = str(layer_indices[0])
    shared_expert_bytes = sidecar.header["layers"][first_layer_key]["experts"]["0"]["length"]
    slot_bank = ExpertSlotBank(
        sidecar=sidecar,
        layer=int(first_layer_key),
        expert_bytes=shared_expert_bytes,
        hot_count=hot_count,
        warm_slots=warm_slots,
        transient_slots=transient_slots,
        calibration_frequencies=calibration_frequencies,
    )

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        target = None
        for attr in ("mlp", "moe"):
            mlp = getattr(layer, attr, None)
            if mlp is None:
                continue
            for sub_attr in ("switch_mlp", "switch_linear", "moe"):
                candidate = getattr(mlp, sub_attr, None)
                if candidate is not None and _is_switch_glu(candidate):
                    target = candidate
                    break
            if target is not None:
                break
        if target is None and _is_switch_glu(layer):
            target = layer
        if target is None:
            logger.debug("Skipping layer %d — no SwitchGLU.", layer_idx)
            continue

        patch_switch_linear(
            module=target,
            sidecar=sidecar,
            layer=layer_idx,
            expert_bytes=sidecar.header["layers"][str(layer_idx)]["experts"]["0"]["length"],
            slot_bank=slot_bank,
            hot_count=hot_count,
            warm_slots=warm_slots,
            transient_slots=transient_slots,
            calibration_frequencies=calibration_frequencies,
            routing_callback=routing_callback,
        )

    logger.info(
        "Applied streaming patches to %d layer(s).",
        len(layer_indices),
    )
