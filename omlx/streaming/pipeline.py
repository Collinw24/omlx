# SPDX-License-Identifier: Apache-2.0
"""Pipeline orchestrator for SSD expert streaming.

Wires together:
  1. Sidecar creation / loading
  2. Shared slot bank (ONE bank per model, not per layer — critical for 397B)
  3. Monkey-patching via ``apply_streaming_patches``
  4. Token-level prefetch via EMA trajectory prediction

Invariants:
  I2 — Caller owns the sync gate (``mx.eval(router_logits)``).
  I5 — Completely inert when ``stream_experts=False``.
  I6 — Original ``.safetensors`` never mutated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Resolve the sidecar file path for a given model."""
    if sidecar_path is not None:
        return str(Path(sidecar_path).expanduser().resolve())

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

    Creates ONE shared slot bank for all SwitchGLU layers (not one per
    layer) to avoid N× pre-allocation.  The ``set_layer()`` method on the
    slot bank is called per-token per-layer before ``resolve()``.

    When ``cfg.stream_experts`` is ``False``, this is a no-op.
    """
    if not cfg.stream_experts:
        return {"active": False, "sidecar": None, "slot_banks": None}

    # 1. Open sidecar
    sc_path = resolve_sidecar_path(model_name_or_path, cfg.expert_sidecar_path)
    sidecar = StreamingExpertSidecar(sc_path)

    if not sidecar.nocache_active:
        logger.warning(
            "F_NOCACHE not active — UBC may bloat under sustained streaming.",
        )

    # 2. Apply patches — creates one shared slot bank internally
    routing_callback: Any = lambda layer, experts: record_routing(
        layer, experts, {},
    )
    apply_streaming_patches(
        model=model,
        sidecar=sidecar,
        hot_count=cfg.expert_hot_count,
        warm_slots=cfg.expert_warm_slots,
        transient_slots=cfg.expert_transient_slots,
        calibration_frequencies=cfg.calibration_frequencies,
        routing_callback=routing_callback,
    )

    # 3. Extract the shared slot bank from the first patched module
    shared: Optional[ExpertSlotBank] = None
    layers = getattr(model, "layers", [])
    for layer in layers:
        for attr in ("mlp", "moe"):
            mlp = getattr(layer, attr, None)
            if mlp is None:
                continue
            for sub_attr in ("switch_mlp", "switch_linear", "moe"):
                cand = getattr(mlp, sub_attr, None)
                if cand is not None:
                    shared = getattr(cand, "_omlx_slot_bank", None)
                if shared is not None:
                    break
            if shared is not None:
                break
        if shared is not None:
            break

    logger.info(
        "Streaming enabled: sidecar=%s, shared_slot_bank=%s, "
        "hot=%d warm=%d transient=%d",
        sc_path,
        "present" if shared else "missing",
        cfg.expert_hot_count,
        cfg.expert_warm_slots,
        cfg.expert_transient_slots,
    )

    return {
        "active": True,
        "sidecar": sidecar,
        "slot_bank": shared,
        "slot_banks": {0: shared} if shared else {},
        "prefetcher": EMATrajectoryPrefetcher(window=cfg.expert_prefetch_window),
        "routed_experts": {},
    }


def unload_streaming(model: nn.Module) -> None:
    """Remove streaming patches and release sidecar resources."""
    layers = getattr(model, "layers", [])
    for layer in layers:
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
# Per-token prefetch wiring
# ---------------------------------------------------------------------------


def record_routing(
    layer_key: int,
    expert_ids: list[int],
    streaming_state: dict,
) -> None:
    """Record which experts were routed on this token for prefetching."""
    routed = streaming_state.setdefault("routed_experts", {})
    routed[layer_key] = set(expert_ids)


def prefetch_step(streaming_state: dict) -> None:
    """Run one round of EMA trajectory prefetch after a token step."""
    if not streaming_state.get("active"):
        return
    prefetcher = streaming_state.get("prefetcher")
    if prefetcher is None:
        return
    slot_banks = streaming_state.get("slot_banks", {})
    routed = streaming_state.get("routed_experts", {})
    for layer_key, current_experts in routed.items():
        slot_bank = slot_banks.get(layer_key)
        if slot_bank is None:
            continue
        prefetcher.update(current_experts)
        predicted = prefetcher.predict(current_experts)
        warm_map = getattr(slot_bank, "_warm_map", {})
        hot_map = getattr(slot_bank, "_hot_map", {})
        exclude = set(warm_map.keys()) | set(hot_map.keys()) | set(current_experts)
        safe = [e for e in predicted if e not in exclude]
        for exp_id in safe:
            ws = getattr(slot_bank, "warm_slots", 0)
            if ws > 0 and len(warm_map) < ws:
                idx = len(warm_map)
                slot_bank._load_expert_into_buffer(exp_id, slot_bank._warm_buffers, idx)
                warm_map[exp_id] = idx
                slot_bank._warm_map[exp_id] = idx


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

    **I2 compliance:** The caller MUST have called ``mx.eval(router_logits)``
    *before* calling this function.
    """
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
    """

    def __init__(self, window: int = 4, top_n: int = 2):
        self._alpha = 2.0 / (window + 1)
        self._top_n = top_n
        self._ema: Dict[int, float] = {}

    def update(self, routed_experts: List[int]) -> None:
        for exp_id in routed_experts:
            current = self._ema.get(exp_id, 0.0)
            self._ema[exp_id] = self._alpha * 1.0 + (1.0 - self._alpha) * current

    def predict(self, current_experts: List[int]) -> List[int]:
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
        warm_map = getattr(slot_bank, "_warm_map", {})
        hot_map = getattr(slot_bank, "_hot_map", {})
        for exp_id in predicted:
            if exp_id in hot_map or exp_id in warm_map:
                continue
            warm_slots = getattr(slot_bank, "warm_slots", 0)
            warm_bufs = getattr(slot_bank, "_warm_buffers", None)
            if warm_slots > 0 and warm_bufs is not None and len(warm_map) < warm_slots:
                slot_idx = len(warm_map)
                slot_bank._load_expert_into_buffer(exp_id, warm_bufs, slot_idx)
                warm_map[exp_id] = slot_idx
                slot_bank._warm_map[exp_id] = slot_idx

    def reset(self) -> None:
        self._ema.clear()
