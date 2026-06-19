# SPDX-License-Identifier: Apache-2.0
"""Pipeline orchestrator for SSD expert streaming.

Wires together:
  1. Sidecar creation / loading
  2. Slot bank instantiation per SwitchGLU layer
  3. Monkey-patching via ``apply_streaming_patches``
  4. Token-level prefetch via EMA trajectory prediction

Invariants enforced:
  I2 — The caller owns the sync gate (``mx.eval(router_logits)``).
  I5 — Completely inert when ``stream_experts=False``.
  I6 — Original ``.safetensors`` are never mutated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import StreamingConfig
from .sidecar import StreamingExpertSidecar, create_sidecar
from .patch import (
    apply_streaming_patches,
    patch_switch_linear,
    unpatch_switch_linear,
)
from .slot_bank import ExpertSlotBank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sidecar resolution
# ---------------------------------------------------------------------------


def resolve_sidecar_path(
    model_name_or_path: str,
    sidecar_path: Optional[str] = None,
) -> str:
    """Resolve the sidecar file path for a given model.

    If *sidecar_path* is provided, use it directly.
    Otherwise auto-detect: ``{model_dir}/{model_name}.streaming``.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name or local path.
    sidecar_path : str, optional
        Explicit sidecar path override.

    Returns
    -------
    str
        Absolute path to the sidecar file.
    """
    if sidecar_path is not None:
        return str(Path(sidecar_path).expanduser().resolve())

    # Auto-detect from model directory
    model_dir = _resolve_model_dir(model_name_or_path)
    model_name = Path(model_name_or_path).name
    candidates = [
        model_dir / f"{model_name}.streaming",
        model_dir / "model.streaming",
        model_dir / "sidecar.streaming",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Auto-detected sidecar: %s", candidate)
            return str(candidate)

    raise FileNotFoundError(
        f"No sidecar found for model {model_name_or_path!r}. "
        f"Searched: {[str(c) for c in candidates]}. "
        f"Create one with: omlx create-sidecar <model_path>"
    )


def _resolve_model_dir(model_name_or_path: str) -> Path:
    """Resolve the model directory from a name or path."""
    path = Path(model_name_or_path)
    if path.exists():
        return path.resolve()
    # Check HF cache
    hf_home = Path(os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    ))
    hub = hf_home / "hub"
    snapshot = hub / f"models--{model_name_or_path.replace('/', '--')}" / "snapshots"
    if snapshot.exists():
        snapshots = sorted(snapshot.iterdir(), reverse=True)
        if snapshots:
            return snapshots[0]
    raise FileNotFoundError(
        f"Model {model_name_or_path!r} not found locally or in HF cache."
    )


# ---------------------------------------------------------------------------
# Model loading with streaming
# ---------------------------------------------------------------------------


def load_model_with_streaming(
    model: nn.Module,
    cfg: StreamingConfig,
    model_name_or_path: str = "",
) -> Dict[str, Any]:
    """Wrap a loaded model with streaming infrastructure.

    When ``cfg.stream_experts`` is ``False``, this is a no-op (returns
    empty dict).  When ``True``, it creates/opens the sidecar, builds
    slot banks for each SwitchGLU layer, applies patches, and reclaims
    weight memory.

    Parameters
    ----------
    model : nn.Module
        The loaded MLX model (must have a ``layers`` attribute).
    cfg : StreamingConfig
        Streaming configuration.
    model_name_or_path : str
        Model name or path for sidecar auto-detection.

    Returns
    -------
    dict
        Streaming state: ``{"sidecar": ..., "slot_banks": {...},
        "active": bool}``.  Pass this to the forward pass orchestrator.
    """
    if not cfg.stream_experts:
        logger.debug("Expert streaming disabled — no-op.")
        return {"active": False}

    # ── 1. Open / create sidecar ──────────────────────────────────────
    sc_path = resolve_sidecar_path(
        model_name_or_path, cfg.expert_sidecar_path
    )
    sidecar = StreamingExpertSidecar(sc_path)

    if not sidecar.nocache_active:
        logger.warning(
            "F_NOCACHE is not active on sidecar %s. "
            "Streaming will use buffered I/O (UBC may bloat).",
            sc_path,
        )

    # ── 2. Build slot banks for each layer ────────────────────────────
    slot_banks: Dict[int, ExpertSlotBank] = {}
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model must have a ``layers`` attribute.")

    for layer_idx in range(len(layers)):
        layer_meta = sidecar.header["layers"].get(str(layer_idx))
        if layer_meta is None:
            logger.debug("Layer %d has no sidecar metadata — skipping.", layer_idx)
            continue
        expert_info = layer_meta["experts"].get("0")
        if expert_info is None:
            continue
        expert_bytes = expert_info["length"]

        bank = ExpertSlotBank(
            sidecar=sidecar,
            layer=layer_idx,
            expert_bytes=expert_bytes,
            hot_count=cfg.expert_hot_count,
            warm_slots=cfg.expert_warm_slots,
            transient_slots=cfg.expert_transient_slots,
            ema_window=cfg.expert_prefetch_window,
            calibration_frequencies=cfg.calibration_frequencies,
        )
        slot_banks[layer_idx] = bank

    logger.info("Built %d slot banks from sidecar.", len(slot_banks))

    # ── 3. Apply streaming patches ────────────────────────────────────
    apply_streaming_patches(
        model=model,
        sidecar=sidecar,
        layer_indices=list(slot_banks.keys()),
        hot_count=cfg.expert_hot_count,
        warm_slots=cfg.expert_warm_slots,
        transient_slots=cfg.expert_transient_slots,
        calibration_frequencies=cfg.calibration_frequencies,
    )

    return {
        "active": True,
        "sidecar": sidecar,
        "slot_banks": slot_banks,
    }


def unload_streaming(model: nn.Module) -> None:
    """Remove streaming patches and release sidecar resources.

    Inverse of :func:`load_model_with_streaming`.
    """
    layers = getattr(model, "layers", [])
    for layer in layers:
        # Walk submodules to find patched SwitchGLU modules
        for mod_name in ("mlp", "moe"):
            mlp = getattr(layer, mod_name, None)
            if mlp is None:
                continue
            for sub_name in ("switch_mlp", "switch_linear"):
                target = getattr(mlp, sub_name, None)
                if target is not None and hasattr(target, "_omlx_original_call"):
                    unpatch_switch_linear(target)

    logger.info("Streaming patches removed for all layers.")


# ---------------------------------------------------------------------------
# Per-token forward pass orchestrator
# ---------------------------------------------------------------------------


def streaming_forward_pass(
    x: mx.array,
    indices: mx.array,
    slot_bank: ExpertSlotBank,
    top_k: int,
) -> Tuple[mx.array, List[int]]:
    """Execute a single streaming MoE forward pass for one layer.

    This is the hot-path entry point called from the patched
    ``__call__`` (or directly by ``BatchedEngine``).

    **I2 compliance:** The caller MUST have called
    ``mx.eval(router_logits)`` *before* calling this function.
    This function does NOT call ``mx.eval()``.

    Parameters
    ----------
    x : mx.array, shape ``(B, S, H)``
        Input to the MoE block.
    indices : mx.array, shape ``(B, S, top_k)`` or ``(*, top_k)``
        Routing indices (expert IDs) from the router.
    slot_bank : ExpertSlotBank
        The pre-configured slot bank for this layer.
    top_k : int
        Number of active experts per token.

    Returns
    -------
    (output, slot_ids)
        output : mx.array, shape ``(B, S, H)``
        slot_ids : list[int] — resolved slot indices for debugging.
    """
    # Flatten and deduplicate expert IDs for the slot bank.
    # Note: This materializes indices to Python (I8 caveat).
    # The caller has already called mx.eval(router_logits), so the
    # indices tensor is in CPU-accessible memory — the tolist() cost
    # is negligible compared to the eval barrier that preceded it.
    flat_ids = indices.flatten().tolist()
    unique_ids = sorted(set(flat_ids))

    stacked, slot_ids = slot_bank.resolve(unique_ids)
    return stacked, slot_ids


# ---------------------------------------------------------------------------
# EMA trajectory prefetcher
# ---------------------------------------------------------------------------


class EMATrajectoryPrefetcher:
    """Exponential Moving Average trajectory prefetcher.

    Tracks expert activation frequencies over a sliding window and
    predicts which experts will be needed on the next token.

    The prefetcher:
    - Updates EMA for each routed expert: ``ema[id] = α * 1.0 + (1-α) * ema[id]``
    - Predicts next token's experts: ``current_routing ∪ top_N_ema_high``
    - Prefetches predicted experts into warm slots (not transient — data
      must persist to the next token).

    Reference: PLAN.md §6 "EMA Trajectory Cache"
    """

    def __init__(self, window: int = 4, top_n: int = 2):
        self._alpha = 2.0 / (window + 1)
        self._top_n = top_n
        self._ema: Dict[int, float] = {}

    def update(self, routed_experts: List[int]) -> None:
        """Update EMA for all experts routed on the current token."""
        for exp_id in routed_experts:
            current = self._ema.get(exp_id, 0.0)
            self._ema[exp_id] = self._alpha * 1.0 + (1.0 - self._alpha) * current

    def predict(self, current_experts: List[int]) -> List[int]:
        """Predict experts needed for the next token.

        Returns ``current_routing ∪ top_N_high_ema_not_in_current``.
        """
        # Sort by EMA descending, exclude already-routed experts
        candidates = sorted(
            [(eid, freq) for eid, freq in self._ema.items()
             if eid not in current_experts],
            key=lambda x: -x[1],
        )
        predicted = [eid for eid, _ in candidates[:self._top_n]]
        return list(set(current_experts) | set(predicted))

    def prefetch(
        self,
        predicted: List[int],
        slot_bank: ExpertSlotBank,
        sidecar: StreamingExpertSidecar,
        layer: int,
    ) -> None:
        """Prefetch predicted experts into warm slots.

        Only prefetches experts not already cached in hot or warm tiers.
        Runs during the GPU MoE compute window (post ``async_eval``).
        """
        # Check which predicted experts are already cached
        stack_buf = getattr(slot_bank, "_stack_buf", None)
        warm_map = getattr(slot_bank, "_warm_map", {})
        hot_map = getattr(slot_bank, "_hot_map", {})

        for exp_id in predicted:
            if exp_id in hot_map or exp_id in warm_map:
                continue  # Already cached

            # Find a warm slot to prefetch into
            warm_slots = getattr(slot_bank, "warm_slots", 0)
            warm_bufs = getattr(slot_bank, "_warm_buffers", None)
            if warm_slots > 0 and warm_bufs is not None and len(warm_map) < warm_slots:
                # Use an empty warm slot
                slot_idx = len(warm_map)
                slot_bank._load_expert_into_buffer(exp_id, warm_bufs, slot_idx)
                warm_map[exp_id] = slot_idx
                slot_bank._warm_map[exp_id] = slot_idx
                logger.debug("Prefetched expert %d into warm slot %d.", exp_id, slot_idx)

    def reset(self) -> None:
        """Clear all EMA state."""
        self._ema.clear()
