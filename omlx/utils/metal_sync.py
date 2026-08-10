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
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generation_stream

from ..exceptions import TurboQuantProcessExclusiveError

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
    """Guard process-exclusive mid-prefill Metal access.

    A mid-prefill engine must be the process's sole ``EngineCore`` and claims
    the capability only after the global MLX executor has drained. While that
    engine lives, new engines and independent Metal workers fail closed. This
    keeps cache-clearing conversion away from streams it cannot drain.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registered_engines: weakref.WeakSet[Any] = weakref.WeakSet()
        self._exclusive_owner: weakref.ReferenceType[Any] | None = None
        self._background_metal_operations = 0

    def _exclusive_owner_unlocked(self) -> Any | None:
        owner_ref = self._exclusive_owner
        if owner_ref is None:
            return None
        owner = owner_ref()
        if owner is None:
            self._exclusive_owner = None
        return owner

    def register_engine(self, owner: Any) -> None:
        """Register an EngineCore before it creates a Metal executor."""
        with self._lock:
            exclusive_owner = self._exclusive_owner_unlocked()
            if exclusive_owner is not None and exclusive_owner is not owner:
                raise TurboQuantProcessExclusiveError(
                    "TurboQuant mid-prefill requires process-exclusive Metal "
                    "access; unload the mid-prefill model before loading "
                    "another engine"
                )
            self._registered_engines.add(owner)

    def unregister_engine(self, owner: Any) -> None:
        """Release an engine after its owning-thread stream has drained."""
        with self._lock:
            if self._exclusive_owner_unlocked() is owner:
                self._exclusive_owner = None
            self._registered_engines.discard(owner)

    def claim_process_exclusive(self, owner: Any) -> None:
        """Claim the mid-prefill capability for the process's sole engine."""
        with self._lock:
            if owner not in self._registered_engines:
                raise RuntimeError("mid-prefill owner is not a registered engine")
            exclusive_owner = self._exclusive_owner_unlocked()
            if exclusive_owner is not None and exclusive_owner is not owner:
                raise TurboQuantProcessExclusiveError(
                    "another engine already owns process-exclusive Metal access"
                )
            if any(
                engine is not owner for engine in self._registered_engines
            ):
                raise TurboQuantProcessExclusiveError(
                    "TurboQuant mid-prefill requires process-exclusive Metal "
                    "access; unload all other engines before enabling it"
                )
            if self._background_metal_operations:
                raise TurboQuantProcessExclusiveError(
                    "TurboQuant mid-prefill cannot start while an independent "
                    "Metal operation is active"
                )
            self._exclusive_owner = weakref.ref(owner)

    def process_exclusive(self, owner: Any | None) -> bool:
        """Return whether a registered ``owner`` holds the process capability."""
        if owner is None:
            return False
        with self._lock:
            return (
                owner in self._registered_engines
                and self._exclusive_owner_unlocked() is owner
            )

    def assert_process_exclusive(self, owner: Any | None) -> None:
        """Require ``owner`` to hold the process capability."""
        if not self.process_exclusive(owner):
            raise RuntimeError(
                "TurboQuant mid-prefill conversion lacks process-exclusive "
                "Metal ownership"
            )

    def assert_background_metal_allowed(self) -> None:
        """Reject a global-executor task while a mid-prefill engine is live."""
        with self._lock:
            if self._exclusive_owner_unlocked() is not None:
                raise TurboQuantProcessExclusiveError(
                    "Independent Metal work is unavailable while a "
                    "TurboQuant mid-prefill engine owns the process"
                )

    @contextmanager
    def background_metal_operation(self) -> Iterator[None]:
        """Track a non-executor Metal worker such as oQ quantization."""
        with self._lock:
            if self._exclusive_owner_unlocked() is not None:
                raise TurboQuantProcessExclusiveError(
                    "Independent Metal work is unavailable while a "
                    "TurboQuant mid-prefill engine owns the process"
                )
            self._background_metal_operations += 1
        try:
            yield
        finally:
            with self._lock:
                self._background_metal_operations -= 1


_conversion_coordinator = _ConversionCoordinator()


def _sync_and_clear_cache(stream: Any | None = None) -> None:
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
