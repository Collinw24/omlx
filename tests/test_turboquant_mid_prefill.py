# SPDX-License-Identifier: Apache-2.0
"""Contracts for bounded TurboQuant conversion during prefill."""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_vlm.turboquant import TurboQuantKVCache, _slice_state

from omlx.exceptions import PrefillMemoryExceededError
from omlx.patches.turboquant_attention import _patch_update_eval_policy
from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillKVPhase,
)
from omlx.turboquant_kv import (
    _state_length,
    convert_kv_cache_sliced,
    estimate_turboquant_capacity_growth_bytes,
    estimate_turboquant_conversion_peak_bytes,
    estimate_turboquant_prefill_attention_workspace_bytes,
    reserve_turboquant_prefill_capacity,
    turboquant_mse_bytes_per_element,
)
from omlx.utils.metal_sync import _conversion_coordinator


class _AppendModel:
    """Small cache-mutating model stand-in for both scheduler prefill paths."""

    def __init__(self) -> None:
        self.layers: list[Any] = []
        self.dtype = mx.float16
        self.config = SimpleNamespace(
            model_type="unit",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            head_dim=32,
        )
        self.calls = 0

    def make_cache(self) -> list[Any]:
        return [KVCache(), KVCache()]

    def __call__(
        self,
        tokens: mx.array,
        cache: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.calls += 1
        if cache is None:
            return
        n_tokens = int(tokens.shape[1])
        value = float(self.calls)
        for cache_obj in cache:
            if not isinstance(cache_obj, (KVCache, TurboQuantKVCache)):
                continue
            keys = mx.full((1, 2, n_tokens, 32), value, dtype=mx.float16)
            values = mx.full(
                (1, 2, n_tokens, 32),
                value + 0.25,
                dtype=mx.float16,
            )
            cache_obj.update_and_fetch(keys, values)


class _TestProcessOwner:
    pass


_active_test_process_owner: _TestProcessOwner | None = None


@pytest.fixture(autouse=True)
def _claim_test_process_owner() -> Any:
    """Give direct Scheduler fixtures the same ownership as EnginePool."""
    global _active_test_process_owner
    owner = _TestProcessOwner()
    _conversion_coordinator.register_engine(owner)
    _conversion_coordinator.claim_process_exclusive(owner)
    _active_test_process_owner = owner
    try:
        yield
    finally:
        _active_test_process_owner = None
        _conversion_coordinator.unregister_engine(owner)


def _make_scheduler(*, step_size: int = 4) -> Scheduler:
    model = _AppendModel()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=step_size,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler._turboquant_kv_bits = 4.0
    scheduler._turboquant_skip_last = True
    scheduler._turboquant_mid_prefill = True
    scheduler._set_model_info_for_monitor()
    assert _active_test_process_owner is not None
    scheduler._metal_process_owner = _active_test_process_owner
    return scheduler


def _make_request(
    request_id: str,
    tokens: list[int],
    cache: list[Any] | None = None,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(tokens)
    request.remaining_tokens = list(tokens)
    request.num_prompt_tokens = len(tokens)
    request.prompt_cache = cache
    return request


def _append_dense(
    cache_obj: KVCache,
    *,
    tokens: int,
    value: float = 1.0,
    head_dim: int = 32,
) -> None:
    cache_obj.update_and_fetch(
        mx.full((1, 2, tokens, head_dim), value, dtype=mx.float16),
        mx.full((1, 2, tokens, head_dim), value + 0.5, dtype=mx.float16),
    )


def _dense_cache(*, tokens: int = 0, layers: int = 2) -> list[Any]:
    cache: list[Any] = [KVCache() for _ in range(layers)]
    if tokens > 0:
        for index, cache_obj in enumerate(cache):
            _append_dense(cache_obj, tokens=tokens, value=float(index + 1))
        mx.eval([cache_obj.state for cache_obj in cache])
    return cache


def _configure_pressure(
    scheduler: Scheduler,
    *,
    pressure_after_tokens: int = 4,
) -> int:
    cap = 64 * 1024**2
    usage = 1024**2
    scheduler._memory_limit_bytes = 0
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = cap
    scheduler._prefill_abort_margin = 1.0
    scheduler._prefill_min_chunk_tokens = 1

    def _current(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    def _reclaim(self: Scheduler) -> int:
        del self
        return usage

    def _bound(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self, n_tokens
        if phase is _PrefillKVPhase.DENSE and kv_len >= pressure_after_tokens:
            return float(cap * 2)
        return 1024.0

    scheduler._current_usage_bytes = MethodType(_current, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)
    scheduler._admission_transient_bound = MethodType(_bound, scheduler)
    return cap


def _state_equal(left: Any, right: Any, *, tokens: int) -> bool:
    left = _slice_state(left, tokens)
    right = _slice_state(right, tokens)
    return bool(
        mx.all(left.norms == right.norms).item()
        and mx.all(left.indices == right.indices).item()
    )


@pytest.mark.parametrize("bits", [4.0, 8.0])
def test_sliced_conversion_matches_whole_conversion_and_appends(bits: float) -> None:
    source = KVCache()
    _append_dense(source, tokens=17)
    mx.eval(source.state)
    whole = TurboQuantKVCache.from_cache(source, bits=bits)

    sliced_source = KVCache()
    sliced_source.update_and_fetch(*source.state)
    cache: list[Any] = [sliced_source]
    stats = convert_kv_cache_sliced(
        cache,
        bits=bits,
        skip_last=False,
        slice_tokens=5,
        reserve_tokens=32,
    )
    sliced = cache[0]

    assert isinstance(sliced, TurboQuantKVCache)
    assert stats.converted_layers == 1
    assert stats.slices == math.ceil(17 / 5)
    assert sliced.offset == 17
    assert sliced.keys.norms.shape[2] == 32
    assert sliced.values.norms.shape[2] == 32
    assert _state_equal(whole.keys, sliced.keys, tokens=17)
    assert _state_equal(whole.values, sliced.values, tokens=17)

    append_keys = mx.full((1, 2, 2, 32), 3.0, dtype=mx.float16)
    append_values = mx.full((1, 2, 2, 32), 4.0, dtype=mx.float16)
    whole.update_and_fetch(append_keys, append_values)
    sliced.update_and_fetch(append_keys, append_values)
    mx.eval(whole.keys, whole.values, sliced.keys, sliced.values)
    assert whole.offset == sliced.offset == 19
    assert _state_equal(whole.keys, sliced.keys, tokens=19)
    assert _state_equal(whole.values, sliced.values, tokens=19)

    queries = mx.ones((1, 2, 1, 32), dtype=mx.float16)
    whole_attention = whole.decode_attention(queries, scale=32**-0.5)
    sliced_attention = sliced.decode_attention(queries, scale=32**-0.5)
    mx.eval(whole_attention, sliced_attention)
    assert mx.allclose(whole_attention, sliced_attention).item()

    second = convert_kv_cache_sliced(
        cache,
        bits=bits,
        skip_last=False,
        slice_tokens=5,
    )
    assert cache[0] is sliced
    assert second.converted_layers == 0
    assert second.slices == 0


def test_prefill_append_evals_only_geometric_capacity_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_update_eval_policy()
    cache = TurboQuantKVCache(bits=8.0)
    initial_keys = mx.ones((1, 2, 2048, 32), dtype=mx.float16)
    initial_values = mx.ones((1, 2, 2048, 32), dtype=mx.float16)
    append_keys = mx.ones((1, 2, 16, 32), dtype=mx.float16)
    append_values = mx.ones((1, 2, 16, 32), dtype=mx.float16)
    cache.update_and_fetch(initial_keys, initial_values)
    mx.eval(cache.keys, cache.values)

    current_state_bytes = sum(
        int(array.nbytes) for state in (cache.keys, cache.values) for array in state
    )
    expected_growth = current_state_bytes * 2560 // 2048
    assert estimate_turboquant_capacity_growth_bytes([cache], 16) == expected_growth

    eval_calls = 0

    def _record_eval(*args: Any, **kwargs: Any) -> None:
        nonlocal eval_calls
        del args, kwargs
        eval_calls += 1

    monkeypatch.setattr(mx, "eval", _record_eval)
    cache.update_and_fetch(append_keys, append_values)

    assert eval_calls == 1
    assert _state_length(cache.keys) == 2560
    assert estimate_turboquant_capacity_growth_bytes([cache], 16) == 0

    cache.update_and_fetch(append_keys, append_values)
    assert eval_calls == 1
    assert _state_length(cache.keys) == 2560


def test_prefill_capacity_growth_clamps_to_request_limit() -> None:
    _patch_update_eval_policy()
    cache = TurboQuantKVCache(bits=8.0)
    cache.update_and_fetch(
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
    )
    mx.eval(cache.keys, cache.values)

    current_state_bytes = sum(
        int(array.nbytes) for state in (cache.keys, cache.values) for array in state
    )
    unbounded_growth = estimate_turboquant_capacity_growth_bytes([cache], 16)
    limited_growth = estimate_turboquant_capacity_growth_bytes(
        [cache],
        16,
        capacity_limit=2304,
    )

    assert unbounded_growth == current_state_bytes * 2560 // 2048
    assert limited_growth == current_state_bytes * 2304 // 2048
    assert limited_growth < unbounded_growth
    assert (
        reserve_turboquant_prefill_capacity(
            [cache],
            16,
            capacity_limit=2304,
        )
        == 1
    )
    assert _state_length(cache.keys) == 2304
    with pytest.raises(
        ValueError,
        match="below the requested cache end",
    ):
        estimate_turboquant_capacity_growth_bytes(
            [cache],
            16,
            capacity_limit=2048,
        )


def test_prefill_capacity_reserve_clears_each_grown_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_update_eval_policy()
    prompt_cache: list[Any] = []
    for _ in range(3):
        cache = TurboQuantKVCache(bits=8.0)
        cache.update_and_fetch(
            mx.ones((1, 2, 2048, 32), dtype=mx.float16),
            mx.ones((1, 2, 2048, 32), dtype=mx.float16),
        )
        mx.eval(cache.keys, cache.values)
        prompt_cache.append(cache)

    cleared_streams: list[Any | None] = []

    def _record_clear(stream: Any | None = None) -> None:
        cleared_streams.append(stream)

    monkeypatch.setattr(
        "omlx.utils.metal_sync._sync_and_clear_cache",
        _record_clear,
    )
    current_state_bytes = sum(
        int(array.nbytes)
        for state in (prompt_cache[0].keys, prompt_cache[0].values)
        for array in state
    )
    single_layer_peak = estimate_turboquant_capacity_growth_bytes(
        [prompt_cache[0]],
        16,
    )
    persistent_delta = single_layer_peak - current_state_bytes
    assert persistent_delta > 0
    assert (
        estimate_turboquant_capacity_growth_bytes(
            prompt_cache,
            16,
        )
        == single_layer_peak + 2 * persistent_delta
    )

    grown_layers = reserve_turboquant_prefill_capacity(prompt_cache, 16)

    assert grown_layers == 3
    assert cleared_streams == [None, None, None]
    assert all(_state_length(cache.keys) == 2560 for cache in prompt_cache)
    assert estimate_turboquant_capacity_growth_bytes(prompt_cache, 16) == 0


def test_scheduler_prereserves_only_after_mid_prefill_trigger() -> None:
    _patch_update_eval_policy()
    scheduler = _make_scheduler(step_size=16)
    cache = TurboQuantKVCache(bits=8.0)
    cache.update_and_fetch(
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
    )
    mx.eval(cache.keys, cache.values)
    prompt_cache: list[Any] = [cache]
    context = SimpleNamespace(
        phase=_PrefillKVPhase.TURBOQUANT,
        mid_triggered=False,
        turboquant_capacity_limit=2304,
    )

    scheduler._prepare_mid_prefill_turboquant_capacity(
        prompt_cache,
        16,
        context,
    )
    assert _state_length(cache.keys) == 2048

    context.mid_triggered = True
    scheduler._prepare_mid_prefill_turboquant_capacity(
        prompt_cache,
        16,
        context,
    )
    assert _state_length(cache.keys) == 2304


def test_prefill_guard_charges_geometric_capacity_replacement() -> None:
    _patch_update_eval_policy()
    scheduler = _make_scheduler(step_size=16)
    cache = TurboQuantKVCache(bits=8.0)
    initial_keys = mx.ones((1, 2, 2048, 32), dtype=mx.float16)
    initial_values = mx.ones((1, 2, 2048, 32), dtype=mx.float16)
    cache.update_and_fetch(initial_keys, initial_values)
    mx.eval(cache.keys, cache.values)
    prompt_cache: list[Any] = [cache]

    growth_for_full = estimate_turboquant_capacity_growth_bytes(prompt_cache, 16)
    growth_for_floor = estimate_turboquant_capacity_growth_bytes(prompt_cache, 15)
    assert growth_for_full > growth_for_floor > 0

    usage = 1024**2
    ordinary_transient = 1024.0
    cap = usage + int(ordinary_transient) + growth_for_floor
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = cap
    scheduler._prefill_abort_margin = 1.0
    scheduler._prefill_headroom_safety = 1.0
    scheduler._prefill_min_chunk_tokens = 1

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    def _reclaim(self: Scheduler) -> int:
        del self
        return usage

    def _ordinary_bound(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self, n_tokens, kv_len, phase
        return ordinary_transient

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)
    scheduler._admission_transient_bound = MethodType(_ordinary_bound, scheduler)
    context = SimpleNamespace(phase=_PrefillKVPhase.TURBOQUANT)

    guarded = scheduler._guard_prefill_chunk(
        16,
        kv_len=2048,
        progress=2048,
        loop_label="test",
        prompt_cache=prompt_cache,
        prefill_context=context,
    )

    assert usage + ordinary_transient < cap
    assert usage + ordinary_transient + growth_for_full > cap
    assert 1 <= guarded < 16
    assert (
        usage
        + ordinary_transient
        + estimate_turboquant_capacity_growth_bytes(prompt_cache, guarded)
        <= cap
    )


def test_prefill_guard_uses_request_capacity_limit() -> None:
    _patch_update_eval_policy()
    scheduler = _make_scheduler(step_size=16)
    cache = TurboQuantKVCache(bits=8.0)
    cache.update_and_fetch(
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
        mx.ones((1, 2, 2048, 32), dtype=mx.float16),
    )
    mx.eval(cache.keys, cache.values)
    prompt_cache: list[Any] = [cache]
    limited_growth = estimate_turboquant_capacity_growth_bytes(
        prompt_cache,
        16,
        capacity_limit=2304,
    )
    unbounded_growth = estimate_turboquant_capacity_growth_bytes(prompt_cache, 16)
    usage = 1024**2
    ordinary_transient = 1024.0
    cap = usage + int(ordinary_transient) + limited_growth
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = cap
    scheduler._prefill_abort_margin = 1.0
    scheduler._prefill_headroom_safety = 1.0

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    def _reclaim(self: Scheduler) -> int:
        del self
        return usage

    def _ordinary_bound(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self, n_tokens, kv_len, phase
        return ordinary_transient

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)
    scheduler._admission_transient_bound = MethodType(_ordinary_bound, scheduler)
    context = SimpleNamespace(
        phase=_PrefillKVPhase.TURBOQUANT,
        turboquant_capacity_limit=2304,
    )

    guarded = scheduler._guard_prefill_chunk(
        16,
        kv_len=2048,
        progress=2048,
        loop_label="test",
        prompt_cache=prompt_cache,
        prefill_context=context,
    )

    assert limited_growth < unbounded_growth
    assert guarded == 16
    assert usage + ordinary_transient + limited_growth <= cap
    assert usage + ordinary_transient + unbounded_growth > cap


def test_conversion_peak_carries_reserved_destinations_across_layers() -> None:
    one_layer = _dense_cache(tokens=4, layers=1)
    three_layers = _dense_cache(tokens=4, layers=3)

    one_layer_peak = estimate_turboquant_conversion_peak_bytes(
        one_layer,
        bits=4.0,
        skip_last=False,
        slice_tokens=4,
        reserve_tokens=32,
    )
    three_layer_peak = estimate_turboquant_conversion_peak_bytes(
        three_layers,
        bits=4.0,
        skip_last=False,
        slice_tokens=4,
        reserve_tokens=32,
    )

    assert three_layer_peak > one_layer_peak


def test_prefill_reserve_stages_mid_prefill_growth() -> None:
    scheduler = _make_scheduler(step_size=4)
    request = _make_request("reserve", list(range(9)))

    assert request.max_tokens == 4
    assert scheduler._turboquant_prefill_reserve_tokens(request, 4) == 13
    assert scheduler._turboquant_prefill_reserve_tokens(request, 4, staged=True) == 8
    context = scheduler._new_prefill_context(
        request,
        _dense_cache(tokens=4, layers=3),
        loop_label="external",
    )
    assert context.turboquant_capacity_limit == 256
    assert scheduler._turboquant_prefill_reserve_tokens(request, 10, staged=True) == 13

    cache = _dense_cache(tokens=4, layers=3)
    full_peak = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=4,
        reserve_tokens=13,
    )
    staged_peak = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=4,
        reserve_tokens=8,
    )
    assert staged_peak < full_peak


@pytest.mark.parametrize("bits", [4.0, 8.0])
def test_packed_width_matches_state_and_phase_accounting(bits: float) -> None:
    source = KVCache()
    _append_dense(source, tokens=5)
    mx.eval(source.state)
    converted = TurboQuantKVCache.from_cache(source, bits=bits)
    mx.eval(converted.keys, converted.values)
    actual_state_bytes = sum(
        int(array.nbytes)
        for state in converted.state
        for array in (state.norms, state.indices)
    )
    actual_width = actual_state_bytes / (2 * 1 * 2 * 5 * 32)
    assert turboquant_mse_bytes_per_element(32, bits) == actual_width

    scheduler = _make_scheduler()
    scheduler._turboquant_kv_bits = bits
    scheduler._turboquant_skip_last = False
    scheduler._set_model_info_for_monitor()
    assert scheduler._prefill_dense_kv_dtype_size == 2.0
    assert scheduler._prefill_tq_kv_dtype_size == actual_width

    dense = scheduler._admission_estimate(
        num_prompt_tokens=1024,
        cached_tokens=0,
        current=0,
        phase=_PrefillKVPhase.DENSE,
    )
    turboquant = scheduler._admission_estimate(
        num_prompt_tokens=1024,
        cached_tokens=0,
        current=0,
        phase=_PrefillKVPhase.TURBOQUANT,
    )
    assert dense is not None and turboquant is not None
    assert turboquant.kv_exact < dense.kv_exact


def test_sliced_converter_is_layer_atomic() -> None:
    cache = _dense_cache(tokens=4, layers=2)
    callbacks = 0

    def _cancel_after_first_layer() -> None:
        nonlocal callbacks
        callbacks += 1
        if callbacks == 3:
            raise _PrefillAbortedError([], 4)

    with pytest.raises(_PrefillAbortedError):
        convert_kv_cache_sliced(
            cache,
            bits=4.0,
            skip_last=False,
            slice_tokens=4,
            check_cancelled=_cancel_after_first_layer,
        )

    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_phase_classifier_rejects_partial_or_wrong_width_state() -> None:
    scheduler = _make_scheduler()
    dense = _dense_cache(tokens=4, layers=3)
    assert scheduler._classify_prefill_cache(dense) == (_PrefillKVPhase.DENSE, True)

    converted = _dense_cache(tokens=4, layers=3)
    scheduler._apply_turboquant_kv_convert_sliced(converted, log_result=False)
    assert scheduler._classify_prefill_cache(converted) == (
        _PrefillKVPhase.TURBOQUANT,
        False,
    )

    partial = _dense_cache(tokens=4, layers=3)
    partial[0] = TurboQuantKVCache.from_cache(partial[0], bits=4.0)
    assert (
        scheduler._classify_prefill_cache(partial)[0] is _PrefillKVPhase.INVALID_PARTIAL
    )

    wrong_bits = _dense_cache(tokens=4, layers=2)
    wrong_bits[0] = TurboQuantKVCache.from_cache(wrong_bits[0], bits=8.0)
    assert (
        scheduler._classify_prefill_cache(wrong_bits)[0]
        is _PrefillKVPhase.INVALID_PARTIAL
    )


def test_conversion_safety_cap_is_checked_before_mutation() -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("cap-rejected", list(range(9)), cache)
    context = scheduler._new_prefill_context(request, cache, loop_label="external")
    usage = 4096

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    conversion_peak = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=True,
        reserve_tokens=scheduler._turboquant_prefill_reserve_tokens(
            request, 4, staged=True
        ),
    )
    converter_entered = False

    def _unexpected_convert(
        prompt_cache: list[Any],
        *,
        reserve_tokens: int | None = None,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> Any:
        nonlocal converter_entered
        del prompt_cache, reserve_tokens, check_cancelled, log_result
        converter_entered = True
        raise AssertionError("converter entered before safety-cap rejection")

    scheduler._apply_turboquant_kv_convert_sliced = _unexpected_convert
    with pytest.raises(PrefillMemoryExceededError) as raised:
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=usage + conversion_peak - 1,
        )

    assert converter_entered is False
    assert raised.value.estimated_bytes == usage + conversion_peak
    assert raised.value.limit_bytes == usage + conversion_peak - 1
    assert cache == []
    assert request.prompt_cache is None
    assert request.turboquant_mid_prefill_attempted is True
    assert raised.value.__context__ is None


def test_pressure_guard_converts_once_without_eviction() -> None:
    scheduler = _make_scheduler()
    cap = _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4)
    request = _make_request("guard", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(request, cache, loop_label="external")
    converter = MagicMock(wraps=scheduler._apply_turboquant_kv_convert_sliced)
    scheduler._apply_turboquant_kv_convert_sliced = converter

    result = scheduler._guard_prefill_chunk(
        4,
        kv_len=4,
        progress=4,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )
    second = scheduler._guard_prefill_chunk(
        4,
        kv_len=8,
        progress=8,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )

    assert result == second == 4
    assert converter.call_count == 1
    assert converter.call_args.kwargs["reserve_tokens"] == 8
    assert request.prefill_eviction_retries == 0
    assert request.turboquant_mid_prefill_attempted is True
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert context.trigger_tokens == 4
    assert context.memory_after_bytes < cap
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_process_ownership_is_required_before_conversion() -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("wrong-owner", list(range(9)), cache)
    context = scheduler._new_prefill_context(request, cache, loop_label="external")
    scheduler._metal_process_owner = object()

    with pytest.raises(RuntimeError, match="process-exclusive"):
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=0,
        )

    assert all(isinstance(cache_obj, KVCache) for cache_obj in cache)


def test_feature_off_uses_ordinary_final_conversion() -> None:
    scheduler = _make_scheduler()
    scheduler._turboquant_mid_prefill = False
    cache = _dense_cache(tokens=4)
    request = _make_request("feature-off", list(range(5)), cache)
    context = scheduler._new_prefill_context(request, cache, loop_label="external")

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("mid-prefill-only path ran while disabled")

    scheduler._classify_prefill_cache = _unexpected
    scheduler._apply_turboquant_kv_convert_sliced = _unexpected
    scheduler._run_guarded_turboquant_conversion = _unexpected
    result = scheduler._finalize_turboquant_prefill_cache(
        request,
        cache,
        processed_tokens=4,
        context=context,
    )

    assert result is None
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)
    assert request.turboquant_mid_prefill_attempted is False
    assert context.phase is _PrefillKVPhase.DENSE


def test_dense_preflight_defers_only_confirmed_conversion() -> None:
    scheduler = _make_scheduler(step_size=4)
    cap = _configure_pressure(scheduler, pressure_after_tokens=0)
    scheduler._memory_hard_watermark_bytes = cap
    scheduler._prefill_memory_guard = True
    request = _make_request("deferred-preflight", list(range(9)))
    assert scheduler._turboquant_preflight_conversion_eligible is True

    dense_estimate = scheduler._admission_estimate(
        num_prompt_tokens=request.num_prompt_tokens,
        cached_tokens=0,
        current=scheduler._current_usage_bytes(),
        phase=_PrefillKVPhase.DENSE,
    )
    assert dense_estimate is not None
    assert dense_estimate.estimated > cap
    assert scheduler._preflight_memory_check(request) is None
    scheduler.preflight_or_raise(
        num_prompt_tokens=request.num_prompt_tokens,
        request_id=request.request_id,
    )

    request.turboquant_mid_prefill_attempted = True
    assert scheduler._preflight_memory_check(request) is not None
    request.turboquant_mid_prefill_attempted = False
    scheduler._turboquant_preflight_conversion_eligible = False
    assert scheduler._preflight_memory_check(request) is not None
    with pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(
            num_prompt_tokens=request.num_prompt_tokens,
            request_id=request.request_id,
        )


@pytest.mark.parametrize("path", ["external", "chunked"])
def test_prefill_paths_surface_conversion_failure_and_clear_cache(path: str) -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_pressure(scheduler)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request(f"{path}-failure", tokens, cache)
    scheduler.requests[request.request_id] = request

    def _partially_fail(
        prompt_cache: list[Any],
        *,
        reserve_tokens: int | None = None,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> Any:
        del reserve_tokens, check_cancelled, log_result
        prompt_cache[0] = TurboQuantKVCache.from_cache(prompt_cache[0], bits=4.0)
        raise RuntimeError("injected path failure")

    scheduler._apply_turboquant_kv_convert_sliced = _partially_fail
    if path == "external":
        with pytest.raises(PrefillMemoryExceededError, match="cache was discarded"):
            scheduler._do_external_prefill(request, tokens, cache)
    else:
        state = scheduler._begin_prefill(request, tokens, cache)
        assert scheduler._step_prefill_chunk(state) is False
        with pytest.raises(PrefillMemoryExceededError, match="cache was discarded"):
            scheduler._step_prefill_chunk(state)

    assert cache == []
    assert request.prompt_cache is None


def test_external_pressure_conversion_matches_final_conversion(caplog: Any) -> None:
    pressured = _make_scheduler(step_size=4)
    normal = _make_scheduler(step_size=4)
    _configure_pressure(pressured)
    tokens = list(range(9))
    pressured_cache = _dense_cache()
    normal_cache = _dense_cache()
    pressured_request = _make_request("quality-pressured", tokens, pressured_cache)
    normal_request = _make_request("quality-normal", tokens, normal_cache)
    pressured.requests[pressured_request.request_id] = pressured_request
    normal.requests[normal_request.request_id] = normal_request

    with caplog.at_level("INFO", logger="omlx.scheduler"):
        pressured_result, pressured_last = pressured._do_external_prefill(
            pressured_request,
            tokens,
            pressured_cache,
        )
    normal_result, normal_last = normal._do_external_prefill(
        normal_request,
        tokens,
        normal_cache,
    )

    assert pressured_result is pressured_cache
    assert normal_result is normal_cache
    assert pressured_last == normal_last == [8]
    assert pressured_request.turboquant_mid_prefill_attempted is True
    assert normal_request.turboquant_mid_prefill_attempted is False
    assert isinstance(pressured_cache[0], TurboQuantKVCache)
    assert isinstance(normal_cache[0], TurboQuantKVCache)
    pressured_keys, pressured_values = pressured_cache[0].state
    normal_keys, normal_values = normal_cache[0].state
    assert pressured_cache[0].offset == normal_cache[0].offset
    logical_tokens = pressured_cache[0].offset
    assert _state_equal(pressured_keys, normal_keys, tokens=logical_tokens)
    assert _state_equal(pressured_values, normal_values, tokens=logical_tokens)
    assert mx.array_equal(pressured_cache[1].state[0], normal_cache[1].state[0]).item()
    assert mx.array_equal(pressured_cache[1].state[1], normal_cache[1].state[1]).item()

    queries = mx.ones((1, 2, 1, 32), dtype=mx.float16)
    pressured_attention = pressured_cache[0].decode_attention(
        queries,
        scale=32**-0.5,
    )
    normal_attention = normal_cache[0].decode_attention(
        queries,
        scale=32**-0.5,
    )
    mx.eval(pressured_attention, normal_attention)
    assert mx.allclose(pressured_attention, normal_attention).item()

    trigger_logs = [
        message for message in caplog.messages if "mid-prefill trigger" in message
    ]
    summary_logs = [
        message for message in caplog.messages if "mid-prefill complete" in message
    ]
    assert len(trigger_logs) == 1
    assert len(summary_logs) == 1
    assert "post_trigger_tokens=4" in summary_logs[0]
    assert "conversion_pause=" in summary_logs[0]
    assert "post_trigger_chunks=1" in summary_logs[0]
    assert "min_post_trigger_chunk=4" in summary_logs[0]


def test_below_trigger_feature_is_output_equivalent() -> None:
    enabled = _make_scheduler(step_size=4)
    disabled = _make_scheduler(step_size=4)
    disabled._turboquant_mid_prefill = False
    tokens = list(range(9))
    enabled_cache = _dense_cache()
    disabled_cache = _dense_cache()
    enabled_request = _make_request("below-enabled", tokens, enabled_cache)
    disabled_request = _make_request("below-disabled", tokens, disabled_cache)

    enabled._do_external_prefill(enabled_request, tokens, enabled_cache)
    disabled._do_external_prefill(disabled_request, tokens, disabled_cache)

    assert isinstance(enabled_cache[0], TurboQuantKVCache)
    assert isinstance(disabled_cache[0], TurboQuantKVCache)
    enabled_keys, enabled_values = enabled_cache[0].state
    disabled_keys, disabled_values = disabled_cache[0].state
    assert enabled_cache[0].offset == disabled_cache[0].offset
    logical_tokens = enabled_cache[0].offset
    assert _state_equal(enabled_keys, disabled_keys, tokens=logical_tokens)
    assert _state_equal(enabled_values, disabled_values, tokens=logical_tokens)
    assert mx.array_equal(enabled_cache[1].state[0], disabled_cache[1].state[0]).item()
    assert mx.array_equal(enabled_cache[1].state[1], disabled_cache[1].state[1]).item()
    assert enabled_request.turboquant_mid_prefill_attempted is False
    assert disabled_request.turboquant_mid_prefill_attempted is False


def test_chunked_prefill_converts_once_and_inserts() -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_pressure(scheduler)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request("chunked", tokens, cache)
    scheduler.requests[request.request_id] = request
    converter = MagicMock(wraps=scheduler._apply_turboquant_kv_convert_sliced)
    scheduler._apply_turboquant_kv_convert_sliced = converter
    state = scheduler._begin_prefill(request, tokens, cache)

    assert scheduler._step_prefill_chunk(state) is False
    assert scheduler._step_prefill_chunk(state) is True
    assert converter.call_count == 1
    assert state.prefill_context is not None
    assert state.prefill_context.phase is _PrefillKVPhase.TURBOQUANT
    assert state.prefill_context.post_trigger_tokens == 4
    assert state.prefill_context.post_trigger_chunks == 1
    assert state.prefill_context.minimum_post_trigger_chunk_tokens == 4

    batch_generator = MagicMock()
    batch_generator.insert.return_value = [77]
    scheduler.batch_generator = batch_generator
    scheduled: list[Request] = []
    scheduler._insert_prefilled_request(request, state, scheduled)
    assert scheduled == [request]
    assert scheduler.running[request.request_id] is request
    assert state.prefill_context.summary_logged is True


def test_complete_hybrid_prefix_extends_without_reconversion() -> None:
    scheduler = _make_scheduler(step_size=4)
    scheduler._memory_hard_limit_bytes = 0
    scheduler._memory_abort_limit_bytes = 0
    first = KVCache()
    recurrent = ArraysCache(1)
    last = KVCache()
    _append_dense(first, tokens=4, value=1.0)
    _append_dense(last, tokens=4, value=2.0)
    recurrent[0] = mx.ones((1, 2, 32), dtype=mx.float16)
    cache: list[Any] = [first, recurrent, last]
    scheduler._apply_turboquant_kv_convert(cache)
    converted_first = cache[0]
    recurrent_state = recurrent[0]
    request = _make_request("hybrid-prefix", [4, 5, 6], cache)
    request.cached_tokens = 4

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("complete TurboQuant prefix was reconverted")

    scheduler._apply_turboquant_kv_convert_sliced = _unexpected
    result, last_token = scheduler._do_external_prefill(
        request,
        [4, 5, 6],
        cache,
    )

    assert result is cache
    assert last_token == [6]
    assert cache[0] is converted_first
    assert cache[1] is recurrent
    assert recurrent[0] is recurrent_state
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[2], KVCache)
    assert cache[0].offset == cache[2].offset == 6
    assert scheduler._classify_prefill_cache(cache)[0] is _PrefillKVPhase.TURBOQUANT


def test_attempt_flag_blocks_retry_trigger_but_not_final_conversion() -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("retry", list(range(5)), cache)
    request.turboquant_mid_prefill_attempted = True
    context = scheduler._new_prefill_context(request, cache, loop_label="retry")

    assert context.conversion_attempted is True
    assert scheduler._mid_prefill_conversion_available(context, cache, request) is False
    scheduler._finalize_turboquant_prefill_cache(
        request,
        cache,
        processed_tokens=4,
        context=context,
    )
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_phase_transient_histories_remain_separate() -> None:
    scheduler = _make_scheduler()
    scheduler._prefill_min_chunk_tokens = 1
    scheduler._record_chunk_transient(
        1,
        0,
        100,
        request_id="dense",
        loop_label="unit",
        phase=_PrefillKVPhase.DENSE,
    )
    scheduler._record_chunk_transient(
        1,
        0,
        200,
        request_id="tq",
        loop_label="unit",
        phase=_PrefillKVPhase.TURBOQUANT,
    )

    assert scheduler._prefill_transient_tracker.samples == 1
    assert scheduler._prefill_tq_transient_tracker.samples == 1
    assert scheduler._prefill_transient_tracker.last_delta_bytes == 100
    assert scheduler._prefill_tq_transient_tracker.last_delta_bytes == 200


def test_qwen_q8_suffix_uses_structural_workspace_bound() -> None:
    workspace = estimate_turboquant_prefill_attention_workspace_bytes(
        query_tokens=2048,
        kv_len=131071,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=256,
        bits=8.0,
        compute_dtype_size=2,
        causal=True,
    )
    assert workspace == 1_408_012_288

    scheduler = _make_scheduler(step_size=2048)
    scheduler._turboquant_kv_bits = 8.0
    assert scheduler.memory_monitor is not None
    scheduler.memory_monitor.set_model_info(
        num_layers=64,
        num_kv_heads=4,
        head_dim=256,
        dtype_size=turboquant_mse_bytes_per_element(256, 8.0),
        num_attention_heads=24,
        num_kv_cache_layers=8,
        compute_dtype_size=2,
    )
    predicted = scheduler._predicted_chunk_transient(
        2048,
        131071 - 2048,
        phase=_PrefillKVPhase.TURBOQUANT,
    )
    admission = scheduler._admission_transient_bound(
        2048,
        131071 - 2048,
        phase=_PrefillKVPhase.TURBOQUANT,
    )

    assert predicted >= workspace * scheduler._PREFILL_TRANSIENT_SAFETY
    assert admission >= predicted
