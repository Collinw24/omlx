# SPDX-License-Identifier: Apache-2.0
"""Sync-before-clear primitive for the Metal buffer cache.

``mx.clear_cache()`` releases buffers from MLX's Metal buffer pool. If work
that references those buffers is still in flight, the driver can hit a
kernel panic, so every cache clear in oMLX has to drain the stream that
carried the work first. This module is the single home for that primitive
plus the lock that keeps it from racing the async store-cache worker.

Callers on an inference thread pass the stream their work rode on (the
per-engine stream for scheduler paths, ``BatchGenerator._stream`` inside
mlx-lm patches, the dependency's own stream where the dependency dispatched
the work). An mlx ``ThreadLocalStream`` resolves to a different concrete
``mx.Stream`` per calling thread, so the drain only covers the stream it is
given, resolved on the calling thread.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import mlx.core as mx
from mlx_lm.generate import generation_stream

# Module-level alias so callers can fall back to mlx-lm's default stream
# when no per-engine stream is provided.
_default_generation_stream = generation_stream

# Serializes Metal buffer-protocol access from the async store-cache worker
# against inference-thread mx.clear_cache / mx.synchronize calls that can
# invalidate the underlying buffer pool. Closes a SIGABRT path where
# _async_store_cache_worker reads tensor bytes via memoryview while the
# inference thread concurrently issues a reclaim-triggering mx op.
# See: https://github.com/jundot/omlx/issues/1106
_mx_buffer_access_lock = threading.RLock()


class _ConversionCoordinator:
    """Process-wide reader/writer gate and conversion peak reservation.

    Prefill forward/eval regions participate as shared readers. A waiting
    conversion prevents later readers from entering, drains active readers,
    and then owns the exclusive gate until conversion cleanup completes.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._active_prefills = 0
        self._waiting_conversions = 0
        self._conversion_owner: object | None = None
        self._reservation_owner: object | None = None
        self._outstanding_bytes = 0

    @contextmanager
    def prefill_memory_operation(self) -> Iterator[None]:
        """Join a concurrent prefill memory operation."""
        with self._condition:
            while self._conversion_owner is not None or self._waiting_conversions > 0:
                self._condition.wait()
            self._active_prefills += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_prefills -= 1
                if self._active_prefills == 0:
                    self._condition.notify_all()

    @contextmanager
    def conversion(self) -> Iterator[object]:
        """Own exclusive conversion admission until the context exits."""
        owner = object()
        acquired = False
        with self._condition:
            self._waiting_conversions += 1
            self._condition.notify_all()
            try:
                while self._conversion_owner is not None or self._active_prefills > 0:
                    self._condition.wait()
                self._conversion_owner = owner
                acquired = True
            finally:
                self._waiting_conversions -= 1
                if not acquired:
                    self._condition.notify_all()
        try:
            yield owner
        finally:
            with self._condition:
                if self._reservation_owner is owner:
                    self._reservation_owner = None
                    self._outstanding_bytes = 0
                if self._conversion_owner is not owner:
                    raise RuntimeError("TurboQuant conversion gate ownership was lost")
                self._conversion_owner = None
                self._condition.notify_all()

    def try_reserve(
        self,
        owner: object,
        *,
        current_bytes: int,
        peak_bytes: int,
        limit_bytes: int,
    ) -> tuple[bool, int]:
        """Atomically check headroom and publish an accepted conversion peak."""
        if current_bytes < 0 or peak_bytes < 0 or limit_bytes < 0:
            raise ValueError("conversion memory values must be non-negative")
        with self._condition:
            if self._conversion_owner is not owner:
                raise RuntimeError(
                    "conversion reservation requires exclusive ownership"
                )
            if (
                self._reservation_owner is not None
                and self._reservation_owner is not owner
            ):
                raise RuntimeError("another conversion reservation is active")
            prior_outstanding = (
                0 if self._reservation_owner is owner else self._outstanding_bytes
            )
            estimated_bytes = current_bytes + prior_outstanding + peak_bytes
            if limit_bytes > 0 and estimated_bytes > limit_bytes:
                return False, estimated_bytes
            self._reservation_owner = owner
            self._outstanding_bytes = peak_bytes
            self._condition.notify_all()
            return True, estimated_bytes

    def release_reservation(self, owner: object) -> None:
        """Release the holder's peak before its post-conversion sample."""
        with self._condition:
            if self._conversion_owner is not owner:
                raise RuntimeError("conversion reservation owner is not active")
            if self._reservation_owner is None:
                return
            if self._reservation_owner is not owner:
                raise RuntimeError("conversion reservation ownership was lost")
            self._reservation_owner = None
            self._outstanding_bytes = 0
            self._condition.notify_all()

    def outstanding_bytes(self, *, exclude_owner: object | None = None) -> int:
        """Return bytes reserved by another conversion holder."""
        with self._condition:
            if self._reservation_owner is exclude_owner:
                return 0
            return self._outstanding_bytes

    def snapshot(self) -> tuple[int, int, bool, int]:
        """Return reader, waiting-writer, active-writer, and reservation state."""
        with self._condition:
            return (
                self._active_prefills,
                self._waiting_conversions,
                self._conversion_owner is not None,
                self._outstanding_bytes,
            )


_conversion_coordinator = _ConversionCoordinator()


def _sync_and_clear_cache(stream=None):
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Held under _mx_buffer_access_lock so the async store-cache worker cannot
    observe a half-reclaimed Metal buffer pool while it is in the middle of
    reading tensor bytes via the Python buffer protocol (#1106).

    See: https://github.com/jundot/omlx/issues/300, #888, #1106
    """
    with _mx_buffer_access_lock:
        # The engine stream may not have in-flight work on the current thread
        # (for example, during teardown before that thread submits work). On
        # some MLX builds mx.synchronize raises "There is no Stream(gpu, 0) in
        # current thread" in that case; swallow it since there is nothing to
        # drain.
        target = stream if stream is not None else _default_generation_stream
        try:
            mx.synchronize(target)
        except RuntimeError:
            pass
        mx.synchronize()  # default stream
        mx.clear_cache()
