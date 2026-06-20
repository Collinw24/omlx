# SPDX-License-Identifier: Apache-2.0
"""ExpertSlotBank: Three-tier GPU memory pool for expert weights.

This class manages pre-allocated `mx.array` buffers for experts, organized into
three tiers to optimize memory residency based on usage frequency:

1. Hot Tier: A small set of experts that are frequently used (e.g., top 5% 
   experts). These are pinned in memory and never evicted.
2. Warm Tier: A larger set of experts managed by an LRU policy and 
   frequency tracking (EMA). Experts that are used often but are not in the
   hot tier stay here.
3. Transient Tier: A set of "streaming" slots that are overwritten every token. 
   If a cold expert is needed and we have no warm slots, it goes here.

This implementation ensures zero dynamic `mx.array` allocation in the 
inference hot path.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import mlx.core as mx
from .sidecar import StreamingExpertSidecar

import os as _sys_os  # for OMLX_STREAMING_STRICT check

# ---------------------------------------------------------------------------
# Sync gate debug enforcement
# ---------------------------------------------------------------------------
# When OMLX_STREAMING_STRICT=1, resolve() runs a heuristic drain check.


def _assert_graph_drained() -> None:
    """Heuristic that the MLX graph is drained before slot mutation.

    Forces a CPU-side round-trip by evaluating a trivial scalar.  If the
    GPU is still executing prior commands, this stalls until completion.
    Not a true Metal fence, but catches the I2 violation pattern where
    resolve() is called before mx.eval(router_logits).

    Only active when ``OMLX_STREAMING_STRICT=1``.
    """
    _ = mx.array(0).item()


logger = logging.getLogger(__name__)

# EMA smoothing factor: α = 2 / (window + 1)
# Default window=4 → α=0.4
_EMA_ALPHA_BASE = 2.0


class ExpertSlotBank:
    """
    GPU memory manager for MoE expert weights.
    """

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
        calibration_frequencies: Dict[int, float] | None = None,
    ) -> None:
        self.sidecar = sidecar
        self.layer = layer
        self.expert_bytes = expert_bytes
        self.hot_count = hot_count
        self.warm_slots = warm_slots
        self.transient_slots = transient_slots
        self.ema_window = ema_window
        self.warm_promote_threshold = warm_promote_threshold
        self.calibration_frequencies = calibration_frequencies or {}

        # 1. Pre-allocate tiers as contiguous mx.array uint8 buffers.
        # Shape: (num_slots, expert_bytes) — Metal-backed via mx.eval().
        # The C++ buffer extension (pread_into_array) writes directly into
        # the Metal backing buffer via Python buffer protocol — zero
        # intermediate allocations.  2D indexing (buf[slot, offset]) works
        # for row views; raw byte writes go through the C++ extension.
        self._hot_buffers = mx.zeros((hot_count, expert_bytes), dtype=mx.uint8)
        mx.eval(self._hot_buffers)
        # When warm_slots=0, cold experts are stored in warm buffer
        # using transient slot indices — allocate transient_slots capacity.
        warm_buf_size = warm_slots if warm_slots > 0 else transient_slots
        self._warm_buffers = mx.zeros((warm_buf_size, expert_bytes), dtype=mx.uint8)
        mx.eval(self._warm_buffers)
        self._transient_buffers = mx.zeros((transient_slots, expert_bytes), dtype=mx.uint8)
        mx.eval(self._transient_buffers)

        # 3. Initialize tracking structures
        # Hot experts: {expert_id: slot_index}
        self._hot_map: Dict[int, int] = {}
        # Warm experts: {expert_id: slot_index} (Ordered for LRU)
        self._warm_map: OrderedDict[int, int] = OrderedDict()
        # Transient experts: {expert_id: slot_index}
        self._transient_map: Dict[int, int] = {}

        # EMA frequency tracking
        self._ema: Dict[int, float] = {}
        if calibration_frequencies:
            self._ema = {k: v for k, v in calibration_frequencies.items()}

        # Determine Hot Tier experts from calibration
        self._initialize_hot_tier()

        # Track the next available transient slot (circular)
        self._next_transient = 0

        # Pre-allocate output stacking buffer: (max_experts, expert_bytes)
        # Reused across resolve() calls to avoid dynamic allocation.
        self._stack_buf: mx.array | None = None

    def _ensure_stack_buf(self, k: int) -> mx.array:
        """Ensure pre-allocated stacking buffer is large enough for K experts.

        Allocates only when K exceeds the current buffer size — a one-time
        cost, not per-call.  Returns a view of the first K rows.
        """
        if self._stack_buf is None or k > self._stack_buf.shape[0]:
            # Allocate larger buffer (2x growth to amortize).  Old data
            # is stale (previous resolve call with different indices),
            # so no need to copy.
            new_size = k if self._stack_buf is None else max(k, self._stack_buf.shape[0] * 2)
            self._stack_buf = mx.zeros((new_size, self.expert_bytes), dtype=mx.uint8)
            mx.eval(self._stack_buf)
        return self._stack_buf[:k]

    def _update_ema(self, expert_id: int) -> None:
        """Update exponential moving average frequency for an expert.

        α = 2 / (window + 1).  Update: ema[id] = α * 1.0 + (1-α) * ema.get(id, 0.0).
        """
        alpha = _EMA_ALPHA_BASE / (self.ema_window + 1)
        current = self._ema.get(expert_id, 0.0)
        self._ema[expert_id] = alpha * 1.0 + (1.0 - alpha) * current

    def _load_expert_into_buffer(self, expert_id: int, buf: mx.array, slot_idx: int) -> None:
        """Load an expert from the sidecar into a tier buffer slot.

        Uses read_expert_into to write directly into the buffer at the
        correct byte offset for this slot.
        """
        offset = slot_idx * self.expert_bytes
        # Pass mx.array directly — sidecar handles pread fallback for
        # mx.array buffers (memoryview of 2D mx.array blocks slice assign).
        self.sidecar.read_expert_into(self.layer, expert_id, buf, offset)

    def _initialize_hot_tier(self) -> None:
        """Populate the hot tier based on calibration frequencies.

        Sorts experts by frequency descending, assigns the top `hot_count` to
        hot slots, and loads their data from the sidecar into the hot buffers.
        """
        sorted_freqs = sorted(
            self.calibration_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        for i, (exp_id, _) in enumerate(sorted_freqs):
            if i < self.hot_count:
                self._hot_map[exp_id] = i
                # Load the expert's data into the hot buffer
                self._load_expert_into_buffer(exp_id, self._hot_buffers, i)

        logger.info(
            "Initialized Hot Tier with %d experts based on calibration.", len(self._hot_map)
        )

    def set_layer(self, layer: int | str) -> None:
        """Point this slot bank at a different layer in the sidecar.

        When one slot bank is shared across all layers (the default),
        this is called per-token per-layer before ``resolve()`` to
        ensure ``_load_expert_into_buffer`` reads from the correct
        layer in the sidecar file.
        """
        self.layer = layer

    def resolve(self, expert_ids: List[int]) -> Tuple[mx.array, List[int]]:
        """
        Resolves expert IDs to slot indices and returns stacked weights.

        .. note::
           **I2 compliance (sync gate):** The caller MUST have called
           ``mx.eval(router_logits)`` *before* calling this method.
           This function does not call ``mx.eval()``.  Set
           ``OMLX_STREAMING_STRICT=1`` to enable a heuristic drain check.

        Three-tier cascade per expert:

        1. **Hot** — all hot experts are pinned at init, never evicted.
           A hot hit returns immediately with no I/O.
        2. **Warm** — LRU-ordered; a warm hit promotes the entry to the end
           of the order (most-recently-used).  If hot has spare capacity,
           the warm expert is promoted into hot; otherwise it stays warm.
        3. **Transient** — circular overwrite on cold miss.  If transient is
           full and more cold experts arrive, raises ``RuntimeError``.

        **I2 compliance:** This method does NOT call ``mx.eval()`` or block
        on GPU.  The caller must have already completed
        ``mx.eval(router_logits)`` before calling ``resolve()``.

        Returns:
            ``stacked_weights``: ``(K, expert_bytes)`` uint8 array — a view
            into the pre-allocated tier buffers.
            ``slot_ids``: list of K slot indices (into the returned stacked
            weights, or equivalently into the tier buffers).

        Raises:
            RuntimeError: When the number of cold experts in a single
            ``resolve()`` call exceeds ``transient_slots``.
        """
        # Update the sidecar layer pointer for shared slot bank usage.
        # In a multi-layer model, all layers share one slot bank;
        # set_layer is called each token per layer from the patched
        # ``__call__``.

        # Debug sync gate check (I2)
        if _sys_os.environ.get("OMLX_STREAMING_STRICT") == "1":
            _assert_graph_drained()

        K = len(expert_ids)
        slot_ids: List[int] = []
        cold_count = 0

        for exp_id in expert_ids:
            # ── 1. Hot path (no I/O) ──────────────────────────────────
            if exp_id in self._hot_map:
                slot_ids.append(self._hot_map[exp_id])
                self._update_ema(exp_id)
                continue

            # ── 2. Warm cascade (warm hit — no I/O) ──────────────────
            if exp_id in self._warm_map:
                warm_slot = self._warm_map[exp_id]
                if len(self._hot_map) < self.hot_count and self.hot_count > 0:
                    # Promote to hot: reload from sidecar into hot buffer.
                    # mx.array 2D slice assignment is not supported, so we
                    # reload via read_expert_into (C++ extension writes
                    # directly into the Metal buffer).  This I/O cost is
                    # acceptable because promotion is rare.
                    hot_slot = len(self._hot_map)
                    self._load_expert_into_buffer(exp_id, self._hot_buffers, hot_slot)
                    self._hot_map[exp_id] = hot_slot
                    del self._warm_map[exp_id]
                    slot_ids.append(hot_slot)
                else:
                    # Stay warm — move to end of LRU order (no I/O)
                    self._warm_map.move_to_end(exp_id)
                    slot_ids.append(warm_slot)
                self._update_ema(exp_id)
                continue

            # ── 3. Transient check (transient hit — no I/O) ──────────
            if exp_id in self._transient_map:
                trans_slot = self._transient_map[exp_id]
                if len(self._hot_map) < self.hot_count and self.hot_count > 0:
                    # Promote to hot: reload from sidecar into hot buffer.
                    hot_slot = len(self._hot_map)
                    self._load_expert_into_buffer(exp_id, self._hot_buffers, hot_slot)
                    self._hot_map[exp_id] = hot_slot
                    del self._transient_map[exp_id]
                    slot_ids.append(hot_slot)
                else:
                    slot_ids.append(trans_slot)
                self._update_ema(exp_id)
                continue

            # ── 4. Cold miss — need to load from sidecar ──────────────
            cold_count += 1

            # Check transient overflow BEFORE loading.
            # When warm_slots=0, cold experts go to _transient_buffers
            # (circular reuse).  If a single resolve() call exceeds
            # transient_slots, we can't satisfy all the requests
            # without evicting experts that were just loaded.
            if cold_count > self.transient_slots:
                cold_ids = [
                    eid for eid in expert_ids
                    if eid not in self._hot_map
                    and eid not in self._warm_map
                    and eid not in self._transient_map
                ]
                raise RuntimeError(
                    f"Too many cold experts ({cold_count}) for "
                    f"transient slots ({self.transient_slots}). "
                    f"Missing IDs: {cold_ids}. "
                    f"Increase transient_slots or add more warm capacity."
                )

            # --- 4a. Find a slot to load into ---
            #
            # Slot assignment paths:
            #
            # warm_slots=0:
            #   Cold experts load into _transient_buffers at circular slot_idx.
            #   _transient_map tracks occupancy.  slot_id encoding:
            #   slot_id = hot_count + slot_idx (transient slots are right
            #   after hot in _get_slot_view indexing).
            #
            # warm_slots>0, hot has room:
            #   Load directly into hot.  Simple append.
            #
            # warm_slots>0, hot full:
            #   If warm is full: evict LRU warm entry, reuse its slot.
            #   Otherwise: use the next free warm slot.
            #   Data goes into _warm_buffers, tracked by _warm_map.
            #   slot_id = hot_count + slot_idx.

            use_transient = self.warm_slots == 0

            if use_transient:
                # ── Transient-only: load into _transient_buffers ─────
                slot_idx = self._next_transient
                self._next_transient = (
                    self._next_transient + 1
                ) % self.transient_slots
                old_exp_id = self._transient_map.pop(slot_idx, None)
                self._load_expert_into_buffer(
                    exp_id, self._transient_buffers, slot_idx
                )
                self._transient_map[exp_id] = slot_idx
                slot_ids.append(self.hot_count + slot_idx)

            elif len(self._hot_map) < self.hot_count and self.hot_count > 0:
                # ── Hot has room: load directly into hot buffer ───────
                hot_slot = len(self._hot_map)
                self._load_expert_into_buffer(exp_id, self._hot_buffers, hot_slot)
                self._hot_map[exp_id] = hot_slot
                slot_ids.append(hot_slot)

            else:
                # ── Warm tier (warm_slots > 0) ────────────────────────
                evict_lru_slot: int | None = None
                if len(self._warm_map) >= self.warm_slots:
                    evicted_id, evicted_slot = next(
                        iter(self._warm_map.items())
                    )
                    del self._warm_map[evicted_id]
                    evict_lru_slot = evicted_slot

                slot_idx = (
                    evict_lru_slot if evict_lru_slot is not None
                    else len(self._warm_map)
                )
                # Clear transient mapping if overwriting a transient slot
                old_exp_id = self._transient_map.pop(slot_idx, None)

                self._load_expert_into_buffer(
                    exp_id, self._warm_buffers, slot_idx
                )
                self._warm_map[exp_id] = slot_idx
                slot_ids.append(self.hot_count + slot_idx)

            self._update_ema(exp_id)

        # ── 5. Stack results into pre-allocated buffer ──────────────
        stacked = self._ensure_stack_buf(K)
        for i, slot_id in enumerate(slot_ids):
            stacked[i] = self._get_slot_view(slot_id)

        return stacked, slot_ids

    def resolve_tensor(
        self,
        indices: mx.array,
    ) -> Tuple[mx.array, List[int]]:
        """Resolve expert IDs from an ``mx.array`` tensor (I0.8-aware).

        Extracts unique expert IDs from *indices* and dispatches to
        ``resolve()``.  The caller MUST have already called
        ``mx.eval(indices)`` — this method does NOT call ``mx.eval()``
        (I2 compliance).

        The ``.tolist()`` call here operates on already-materialised
        CPU data (post-eval), so it is NOT a sync barrier — it is a
        plain Python list conversion with no GPU interaction.

        Parameters
        ----------
        indices : mx.array
            Routing indices of shape ``(*)`` or ``(*, top_k)``.

        Returns
        -------
        (stacked, slot_ids)
            Same as :meth:`resolve`.
        """
        flat = indices.flatten().tolist()
        unique = sorted(set(flat))
        return self.resolve(unique)

    def _get_slot_view(self, slot_id: int) -> mx.array:
        """Return a view into the tier buffer row for a given slot ID.

        Determines which tier owns the slot and returns a pre-existing
        ``mx.array`` row view — zero allocation, zero copy (I9).

        Slot ID encoding:
          0..hot_count-1            → hot tier
          hot_count..hot_count+warm-1 → warm tier (only when warm_slots > 0)
          everything else             → transient tier
        """
        if slot_id < self.hot_count:
            return self._hot_buffers[slot_id]
        hot_offset = self.hot_count
        if self.warm_slots > 0 and slot_id < hot_offset + self.warm_slots:
            return self._warm_buffers[slot_id - hot_offset]
        return self._transient_buffers[slot_id - hot_offset - max(self.warm_slots, 0)]

    # ── Public introspection (for tests and monitoring) ────────────

    @property
    def hot_size(self) -> int:
        """Number of experts currently in the hot tier."""
        return len(self._hot_map)

    @property
    def warm_size(self) -> int:
        """Number of experts currently in the warm tier."""
        return len(self._warm_map)

    @property
    def transient_size(self) -> int:
        """Number of experts currently in the transient tier."""
        return len(self._transient_map)

    @property
    def next_transient(self) -> int:
        """Circular index of the next transient slot to be recycled."""
        return self._next_transient
