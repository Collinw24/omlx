# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for the TurboQuant mid-prefill validation tool."""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest

from scripts import validate_turboquant_mid_prefill as validation


def _fake_manifest(digest: str = "a" * 64) -> list[dict[str, Any]]:
    """Build one valid manifest entry for result-shape tests."""
    return [
        {
            "path": "model-00001-of-00001.safetensors",
            "size_bytes": 4,
            "sha256": digest,
        }
    ]


def _matrix_result(
    mode: str,
    metadata: dict[str, Any],
    *,
    prompt_hash: str = "prompt",
) -> dict[str, Any]:
    """Build the minimum successful matrix result used by consistency tests."""
    return {
        "status": "ok",
        "kind": "matrix",
        "mode": mode,
        "metadata": metadata,
        "metrics": {
            "dataset_prompt_sha256": "dataset-prompt",
            "rendered_prompt_sha256": prompt_hash,
            "answer_sha256": "answer",
            "prompt_token_ids_sha256": "prompt-tokens",
            "answer_token_ids_sha256": "answer-tokens",
            "prompt_token_count": 32000,
            "answer_token_count": 512,
        },
    }


def test_mode_expansion_is_exact_and_ordered() -> None:
    """The public matrix contains only the five requested validation modes."""
    modes = validation.matrix_modes()

    assert [mode.name for mode in modes] == [
        "dense",
        "ordinary-q8",
        "ordinary-q4",
        "mid-q8",
        "mid-q4",
    ]
    assert [(mode.conversion, mode.bits) for mode in modes] == [
        ("dense", None),
        ("ordinary", 8.0),
        ("ordinary", 4.0),
        ("mid", 8.0),
        ("mid", 4.0),
    ]


def test_direct_loader_releases_transient_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matrix cell clears unused loader buffers before supervised prefill."""
    from omlx.utils import model_loading, tokenizer

    events: list[str] = []
    model = object()
    loaded_tokenizer = object()
    monkeypatch.setattr(
        model_loading,
        "maybe_apply_pre_load_patches",
        lambda *_args, **_kwargs: events.append("pre"),
    )
    monkeypatch.setattr(
        tokenizer,
        "get_tokenizer_config",
        lambda *_args, **_kwargs: {"local_files_only": True},
    )
    monkeypatch.setattr(
        model_loading,
        "lm_load_compat",
        lambda *_args, **_kwargs: (model, loaded_tokenizer),
    )
    monkeypatch.setattr(
        model_loading,
        "apply_post_load_transforms",
        lambda loaded, _settings: events.append("transform") or loaded,
    )
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        lambda loaded: events.append("materialize") if loaded is model else None,
    )
    monkeypatch.setattr(validation.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(mx, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear"))

    result = validation._load_direct_model(
        "/model",
        validation.ValidationMode("dense", "dense", None),
        False,
    )

    assert result == (model, loaded_tokenizer)
    assert events == ["pre", "transform", "materialize", "gc", "synchronize", "clear"]


def test_pinned_dataset_selection_and_cell_configuration() -> None:
    """Dataset provenance, rows, and primary performance cells stay pinned."""
    config = validation.pinned_dataset_config()
    rows = [{"value": index} for index in range(400)]
    selected = validation.select_rows(rows, validation.DEFAULT_MATRIX_ROWS)

    assert config == {
        "repo_id": "openai/mrcr",
        "revision": "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
        "filename": "2needle/2needle_0.parquet",
        "matrix_row_indices": [104, 109, 136, 301, 311, 328],
        "organic_row_index": 35,
    }
    assert [row["row_index"] for row in selected] == [104, 109, 136, 301, 311, 328]
    assert (
        validation.matrix_shared_metadata(
            row_index=104,
            chunk_size=2048,
            max_prompt_tokens=131072,
            greedy_token_limit=1024,
        )["performance_cell"]
        == "primary_32k"
    )
    assert (
        validation.matrix_shared_metadata(
            row_index=311,
            chunk_size=2048,
            max_prompt_tokens=131072,
            greedy_token_limit=1024,
        )["performance_cell"]
        == "primary_64k"
    )
    assert (
        validation.matrix_shared_metadata(
            row_index=109,
            chunk_size=2048,
            max_prompt_tokens=131072,
            greedy_token_limit=1024,
        )["quality_cell"]
        is True
    )


def test_row_selection_rejects_duplicates_and_out_of_range() -> None:
    """Positional row selection cannot silently substitute a sample."""
    rows = [{"value": index} for index in range(3)]

    with pytest.raises(ValueError, match="duplicate"):
        validation.select_rows(rows, [1, 1])
    with pytest.raises(IndexError, match="outside"):
        validation.select_rows(rows, [3])


@pytest.mark.parametrize(
    ("response", "answer", "prefix", "expected"),
    [
        ("abcneedle", "abcneedle", "abc", 1.0),
        ("abcneed", "abcneedle", "abc", 0.8),
        (" needle", "abcneedle", "abc", 0.0),
        ("wrongneedle", "abcneedle", "abc", 0.0),
    ],
)
def test_official_retrieval_scoring(
    response: str, answer: str, prefix: str, expected: float
) -> None:
    """MRCR scoring gates on the exact prefix before SequenceMatcher."""
    assert validation.official_retrieval_score(
        response, answer, prefix
    ) == pytest.approx(expected)


def test_host_memory_reading_uses_free_plus_inactive() -> None:
    """The supervisor derives headroom from supported host VM counters."""
    reading = validation.host_memory_reading(
        {
            "free": 2 * validation.GIB,
            "inactive": 4 * validation.GIB,
            "active": 42 * validation.GIB,
        }
    )

    assert reading.free_bytes == 2 * validation.GIB
    assert reading.inactive_bytes == 4 * validation.GIB
    assert reading.active_bytes == 42 * validation.GIB
    assert reading.headroom_bytes == 6 * validation.GIB


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"free": 1, "inactive": 2},
        {"free": True, "inactive": 2, "active": 3},
        {"free": -1, "inactive": 2, "active": 3},
        {"free": 1, "inactive": "2", "active": 3},
    ],
)
def test_host_memory_invalid_cases_fail_closed(stats: dict[str, object]) -> None:
    """Missing or invalid host VM counters are never interpreted as safe."""
    with pytest.raises(validation.TelemetryError):
        validation.host_memory_reading(stats)


@pytest.mark.parametrize(
    ("child", "headroom", "child_limit", "host_minimum", "fragment"),
    [
        (
            35 * validation.GIB,
            6 * validation.GIB,
            36 * validation.GIB,
            6 * validation.GIB,
            None,
        ),
        (
            36 * validation.GIB,
            10 * validation.GIB,
            36 * validation.GIB,
            6 * validation.GIB,
            "footprint",
        ),
        (
            validation.GIB,
            6 * validation.GIB - 1,
            36 * validation.GIB,
            6 * validation.GIB,
            "headroom",
        ),
        (0, 10 * validation.GIB, 36 * validation.GIB, 6 * validation.GIB, "invalid"),
    ],
)
def test_safety_threshold_decisions(
    child: int,
    headroom: int,
    child_limit: int,
    host_minimum: int,
    fragment: str | None,
) -> None:
    """Threshold boundaries and invalid telemetry produce explicit decisions."""
    violation = validation.safety_violation(
        child,
        headroom,
        child_limit_bytes=child_limit,
        host_minimum_bytes=host_minimum,
    )

    if fragment is None:
        assert violation is None
    else:
        assert fragment in str(violation)


def test_stable_provenance_and_model_manifest_comparison(tmp_path: Path) -> None:
    """Canonical provenance and source hashes ignore mapping order, not content."""
    assert validation.stable_digest({"b": 2, "a": 1}) == validation.stable_digest(
        {"a": 1, "b": 2}
    )

    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model.safetensors"
    shard.write_bytes(b"safe")
    before = validation.build_model_manifest(model)
    reordered_keys = [
        {
            "sha256": before[0]["sha256"],
            "size_bytes": before[0]["size_bytes"],
            "path": before[0]["path"],
        }
    ]
    assert validation.manifests_equal(before, reordered_keys)

    shard.write_bytes(b"changed")
    after = validation.build_model_manifest(model)
    assert not validation.manifests_equal(before, after)


def test_hash_file_requires_verified_darwin_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tensor hashing proves F_NOCACHE success instead of silently falling back."""
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"safe")
    calls: list[tuple[int, int, int]] = []

    def _verified_fcntl(file_descriptor: int, command: int, value: int) -> int:
        calls.append((file_descriptor, command, value))
        return 0

    monkeypatch.setattr(validation.fcntl, "fcntl", _verified_fcntl)
    digest = validation.hash_file(source)

    assert len(digest) == 64
    assert len(calls) == 1
    assert calls[0][0] > 2
    assert calls[0][1:] == (validation.fcntl.F_NOCACHE, 1)

    def _unverified_fcntl(file_descriptor: int, command: int, value: int) -> int:
        del file_descriptor, command, value
        return 1

    monkeypatch.setattr(validation.fcntl, "fcntl", _unverified_fcntl)
    with pytest.raises(validation.ValidationError, match="unverified result"):
        validation.hash_file(source)


def test_atomic_result_shape_and_source_change_rejection(tmp_path: Path) -> None:
    """Atomic output has a stable schema and turns tensor drift into failure."""
    manifest = _fake_manifest()
    payload = validation.shape_validation_result(
        kind="matrix",
        provenance={"git": {"head": "abc"}},
        config={"chunk_size": 2048},
        before_manifest=manifest,
        after_manifest=manifest,
        results=[{"mode": "dense"}],
        error=None,
    )
    output = tmp_path / "nested" / "result.json"
    validation.atomic_write_json(output, payload)

    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == 1
    assert decoded["status"] == "ok"
    assert decoded["source_tensors"]["unchanged"] is True
    assert list(output.parent.glob(".result.json.*.tmp")) == []

    changed = validation.shape_validation_result(
        kind="matrix",
        provenance={},
        config={},
        before_manifest=manifest,
        after_manifest=_fake_manifest("b" * 64),
        results=[],
        error=None,
    )
    assert changed["status"] == "failed"
    assert changed["source_tensors"]["unchanged"] is False
    assert "safetensors" in changed["error"]


def test_running_checkpoint_never_claims_final_tensor_verification() -> None:
    """Partial matrix data stays useful without claiming a final source hash."""
    checkpoint = validation.shape_validation_checkpoint(
        kind="matrix",
        provenance={"git": {"head": "abc"}},
        config={"chunk_size": 2048},
        before_manifest=_fake_manifest(),
        results=[{"mode": "dense"}],
    )

    assert checkpoint["status"] == "running"
    assert checkpoint["results"] == [{"mode": "dense"}]
    assert checkpoint["source_tensors"]["after"] is None
    assert checkpoint["source_tensors"]["unchanged"] is None


def test_matrix_results_require_identical_metadata_and_inputs() -> None:
    """Every sample's five cells carry the exact same dataset/chunk/limit data."""
    metadata = validation.matrix_shared_metadata(
        row_index=104,
        chunk_size=2048,
        max_prompt_tokens=131072,
        greedy_token_limit=1024,
    )
    results = [
        _matrix_result(mode.name, dict(metadata)) for mode in validation.matrix_modes()
    ]

    validation.validate_matrix_results(results)

    results[0]["metadata"] = {**metadata, "chunk_size": 1024}
    with pytest.raises(validation.ValidationError, match="metadata drifted"):
        validation.validate_matrix_results(results)


def test_matrix_results_reject_missing_mode_and_input_drift() -> None:
    """A lost child or a different rendered prompt cannot pass matrix shaping."""
    metadata = validation.matrix_shared_metadata(
        row_index=311,
        chunk_size=2048,
        max_prompt_tokens=131072,
        greedy_token_limit=1024,
    )
    missing = [
        _matrix_result(mode.name, metadata) for mode in validation.matrix_modes()[:-1]
    ]
    with pytest.raises(validation.ValidationError, match="has modes"):
        validation.validate_matrix_results(missing)

    drifted = [
        _matrix_result(mode.name, metadata) for mode in validation.matrix_modes()
    ]
    drifted[-1]["metrics"]["rendered_prompt_sha256"] = "different"
    with pytest.raises(validation.ValidationError, match="inputs drifted"):
        validation.validate_matrix_results(drifted)


def test_child_command_and_environment_remove_all_force_hooks(tmp_path: Path) -> None:
    """Neither child argv nor inherited environment can force a conversion."""
    environment = validation.child_environment(
        {
            "PATH": "/usr/bin",
            "OMLX_FORCE_TURBOQUANT_TRIGGER": "1",
            "OMLX_FORCE_PREFILL_PRESSURE": "1",
            "OMLX_LOG_LEVEL": "debug",
        }
    )
    command = validation.build_child_command(
        tmp_path / "validate.py",
        tmp_path / "spec.json",
        tmp_path / "result.json",
    )

    assert environment == {"PATH": "/usr/bin", "OMLX_LOG_LEVEL": "debug"}
    assert all("OMLX_FORCE_" not in argument for argument in command)
    assert command[2] == "__child"
    assert command[3:] == [
        "--spec",
        str((tmp_path / "spec.json").resolve()),
        "--result",
        str((tmp_path / "result.json").resolve()),
    ]


def test_direct_ordinary_conversion_matches_production_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary TQ converts N-1 cache state before the held prompt token."""
    import mlx_lm.models.cache as cache_module

    events: list[str] = []

    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def read(self) -> float:
            self.value += 1.0
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    clock = FakeClock()

    class FakeMx:
        def synchronize(self) -> None:
            return None

        def clear_cache(self) -> None:
            events.append("clear")
            clock.advance(10.0)

        def array(self, value: list[list[int]]) -> list[list[int]]:
            return value

        def eval(self, *values: Any) -> None:
            del values

    class FakeModel:
        def __call__(
            self,
            tokens: list[list[int]],
            *,
            cache: list[Any],
        ) -> str:
            del cache
            events.append(f"forward:{','.join(str(token) for token in tokens[0])}")
            return "logits"

    cache = [SimpleNamespace(state=None)]

    def _make_prompt_cache(model: Any) -> list[Any]:
        del model
        return cache

    def _ordinary_convert(
        mx: Any, prompt_cache: list[Any], bits: float
    ) -> dict[str, Any]:
        del mx, prompt_cache, bits
        events.append("convert")
        return {"path": "ordinary_from_cache"}

    monkeypatch.setattr(cache_module, "make_prompt_cache", _make_prompt_cache)
    monkeypatch.setattr(validation, "_ordinary_convert", _ordinary_convert)
    monkeypatch.setattr(validation.time, "perf_counter", clock.read)

    _, _, metrics = validation._run_direct_prefill(
        FakeMx(),
        FakeModel(),
        [1, 2, 3, 4, 5],
        validation.ValidationMode("ordinary-q8", "ordinary", 8.0),
        chunk_size=2,
    )

    assert events == [
        "forward:1,2",
        "clear",
        "forward:3,4",
        "clear",
        "convert",
        "forward:5",
    ]
    assert (
        metrics["same_boundary_suffix_seconds"]
        > metrics["same_boundary_suffix_forward_seconds"]
    )
    assert metrics["same_boundary_suffix_tokens_per_second"] == pytest.approx(
        metrics["same_boundary_suffix_tokens"] / metrics["same_boundary_suffix_seconds"]
    )


def test_teacher_forced_replay_scores_logits_in_float32() -> None:
    """High-confidence float16 logits retain their small positive NLL."""
    logits = mx.array([[[7.0, 0.0]]], dtype=mx.float16)

    class FakeModel:
        """Return fixed logits for the one replay forward."""

        def __call__(self, tokens: Any, *, cache: list[Any]) -> Any:
            del tokens, cache
            return logits

    metrics = validation._teacher_forced_replay(
        mx,
        FakeModel(),
        [],
        logits,
        [0],
    )

    assert metrics["mean_nll"] == pytest.approx(
        math.log1p(math.exp(-7.0)),
        rel=1e-3,
    )
    assert metrics["perplexity"] > 1.0


def test_supervisor_telemetry_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed mandatory probe terminates the child and remains in output data."""

    class FakeProcess:
        """Minimal live process double for fail-closed supervision."""

        pid = 123

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("child", timeout)
            return self.returncode

    process = FakeProcess()

    def _popen(command: list[str], env: dict[str, str]) -> FakeProcess:
        del command, env
        return process

    def _invalid_probe(pid: int) -> int:
        del pid
        raise validation.TelemetryError("no footprint")

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(validation, "probe_child_footprint", _invalid_probe)

    outcome = validation.supervise_child(
        command=["child"],
        environment={},
        result_path=tmp_path / "missing.json",
        poll_interval_seconds=0.01,
        child_limit_bytes=36 * validation.GIB,
        host_minimum_bytes=6 * validation.GIB,
    )

    assert process.terminated is True
    assert outcome.error == "no footprint"
    assert outcome.telemetry["samples"] == 0
    assert outcome.telemetry["peak_child_phys_footprint_bytes"] == 0


def test_supervisor_accepts_exit_race_after_valid_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A normal child exit between poll and probe preserves a valid result."""

    class FakeProcess:
        """Process double that exits while its second footprint probe runs."""

        pid = 321

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False
            self.exiting = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is not None:
                return self.returncode
            if self.exiting:
                self.returncode = 0
                return self.returncode
            raise subprocess.TimeoutExpired("child", timeout)

    process = FakeProcess()
    probe_calls = 0
    result_path = tmp_path / "result.json"
    result_path.write_text('{"status":"ok"}', encoding="utf-8")

    def _popen(command: list[str], env: dict[str, str]) -> FakeProcess:
        del command, env
        return process

    def _racing_probe(pid: int) -> int:
        nonlocal probe_calls
        del pid
        probe_calls += 1
        if probe_calls == 1:
            return validation.GIB
        process.exiting = True
        raise validation.TelemetryError("no footprint")

    def _safe_host_memory() -> validation.HostMemoryReading:
        return validation.HostMemoryReading(
            free_bytes=8 * validation.GIB,
            inactive_bytes=8 * validation.GIB,
            active_bytes=48 * validation.GIB,
            headroom_bytes=16 * validation.GIB,
        )

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(validation, "probe_child_footprint", _racing_probe)
    monkeypatch.setattr(validation, "probe_host_memory", _safe_host_memory)
    monkeypatch.setattr(validation.time, "sleep", lambda seconds: None)

    outcome = validation.supervise_child(
        command=["child"],
        environment={},
        result_path=result_path,
        poll_interval_seconds=0.01,
        child_limit_bytes=36 * validation.GIB,
        host_minimum_bytes=6 * validation.GIB,
    )

    assert outcome.error is None
    assert outcome.result == {"status": "ok"}
    assert outcome.returncode == 0
    assert outcome.telemetry["samples"] == 1
    assert outcome.telemetry["peak_child_phys_footprint_bytes"] == validation.GIB
    assert process.terminated is False


def test_supervisor_interrupt_still_terminates_its_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interrupted parent cannot leave its currently supervised child alive."""

    class FakeProcess:
        pid = 456

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return -15 if self.returncode is None else self.returncode

    process = FakeProcess()

    def _popen(command: list[str], env: dict[str, str]) -> FakeProcess:
        del command, env
        return process

    def _interrupt_probe(pid: int) -> int:
        del pid
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(validation, "probe_child_footprint", _interrupt_probe)

    with pytest.raises(KeyboardInterrupt):
        validation.supervise_child(
            command=["child"],
            environment={},
            result_path=tmp_path / "missing.json",
            poll_interval_seconds=0.01,
            child_limit_bytes=36 * validation.GIB,
            host_minimum_bytes=6 * validation.GIB,
        )

    assert process.terminated is True


@pytest.mark.parametrize("row", validation.DEFAULT_MATRIX_ROWS)
def test_matrix_cli_accepts_single_pinned_row(row: int) -> None:
    """Final-head spot checks may select one row without weakening provenance."""
    parser = validation.build_parser()
    args = parser.parse_args(
        [
            "matrix",
            "--model",
            "/models/example",
            "--output",
            "/tmp/result.json",
            "--rows",
            str(row),
        ]
    )

    validation._validate_cli_arguments(args)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["104", "104"], "unique"),
        (["105"], "selected from pinned rows"),
    ],
)
def test_matrix_cli_rejects_unpinned_or_duplicate_rows(
    rows: list[str],
    message: str,
) -> None:
    """Subset support remains bounded to unique pinned dataset rows."""
    parser = validation.build_parser()
    args = parser.parse_args(
        [
            "matrix",
            "--model",
            "/models/example",
            "--output",
            "/tmp/result.json",
            "--rows",
            *rows,
        ]
    )

    with pytest.raises(validation.ValidationError, match=message):
        validation._validate_cli_arguments(args)


def test_organic_contract_uses_natural_fixed_prompt_without_force_controls() -> None:
    """Organic constants bind to pinned natural row 35 and exact contiguous spans."""
    parser = validation.build_parser()
    args = parser.parse_args(
        [
            "organic",
            "--model",
            "/models/example",
            "--output",
            "/tmp/result.json",
        ]
    )
    config = validation._organic_config(args)

    assert config["dataset"]["row_index"] == 35
    assert config["prompt_tokens"] == 131071
    assert config["teacher_forced_replay_tokens"] == 256
    assert config["turboquant_bits"] == 8.0
    assert config["turboquant_skip_last"] is True
    assert config["exclusive_ownership"] is True
    assert config["no_cache"] is True
    assert config["forced_trigger"] is False
    assert config["scheduler"]["prefill_abort_margin"] == 0.95
    assert config["scheduler"]["prefill_min_chunk_tokens"] == 32
    assert config["prefill_abort_margin"] == 0.95
    assert config["prefill_min_chunk_tokens"] == 32
    assert not any(key.startswith("OMLX_FORCE_") for key in config)

    assert config["child_memory_limit_bytes"] == 37 * validation.GIB
    assert config["host_headroom_minimum_bytes"] == 6 * validation.GIB


def test_organic_pressure_uses_production_custom_tier_controls() -> None:
    """Manual limits retain custom-tier abort and minimum-chunk controls."""
    scheduler = SimpleNamespace()

    validation._configure_organic_pressure(
        scheduler,
        soft_limit_bytes=32 * validation.GIB,
        hard_limit_bytes=37 * validation.GIB,
        prefill_abort_margin=validation.ORGANIC_PREFILL_ABORT_MARGIN,
        prefill_min_chunk_tokens=validation.ORGANIC_PREFILL_MIN_CHUNK_TOKENS,
    )

    assert scheduler._memory_limit_bytes == 32 * validation.GIB
    assert scheduler._memory_abort_limit_bytes == 37 * validation.GIB
    assert scheduler._memory_guard_tier == "custom"
    assert scheduler._prefill_abort_margin == 0.95
    assert scheduler._prefill_memory_guard is True
    assert scheduler._prefill_min_chunk_tokens == 32
    assert scheduler._memory_limits_propagated is True


def test_organic_sole_model_requires_one_attempt_without_eviction_pause() -> None:
    """The organic helper proves conversion without an external-victim restart."""
    context = SimpleNamespace(
        mid_triggered=True,
        conversion_attempted=True,
        trigger_tokens=2,
        conversion_seconds=0.5,
        converted_layers=1,
        conversion_slices=1,
        skipped_dense_layers=0,
        conversion_completed_at=0.0,
        post_trigger_tokens=1,
    )
    request = SimpleNamespace(
        request_id="organic",
        prefill_eviction_retries=0,
        turboquant_mid_prefill_attempted=True,
    )

    class FakeMx:
        def __init__(self) -> None:
            self.clear_count = 0

        def synchronize(self) -> None:
            return None

        def clear_cache(self) -> None:
            self.clear_count += 1

        def get_active_memory(self) -> int:
            return 10

        def get_cache_memory(self) -> int:
            return 5

        def reset_peak_memory(self) -> None:
            return None

        def get_peak_memory(self) -> int:
            return 20

        def stream(self, stream: Any) -> contextlib.AbstractContextManager[None]:
            del stream
            return contextlib.nullcontext()

    class FakeScheduler:
        _stream = SimpleNamespace()
        model = SimpleNamespace()

        def _new_prefill_context(
            self,
            inner_request: Any,
            prompt_cache: list[Any],
            *,
            loop_label: str,
        ) -> Any:
            del inner_request, prompt_cache, loop_label
            return context

        def _do_external_prefill(
            self,
            inner_request: Any,
            tokens: list[int],
            existing_cache: list[Any] | None,
        ) -> tuple[list[Any], list[int]]:
            self._new_prefill_context(
                inner_request,
                [SimpleNamespace(state=None)],
                loop_label="external",
            )
            del tokens, existing_cache
            return [SimpleNamespace(state=None)], [3]

    mx = FakeMx()
    with (
        patch.object(
            validation,
            "_model_chunk",
            return_value=SimpleNamespace(),
        ),
        patch.object(
            validation,
            "_teacher_forced_replay",
            return_value={"mean_nll": 0.0},
        ),
    ):
        metrics = validation._capture_no_victim_organic_prefill(
            mx,
            FakeScheduler(),
            request,
            [1, 2, 3],
            [4],
        )

    assert metrics["actual_mid_prefill_trigger"] is True
    assert metrics["mid_prefill_conversion_attempt_count"] == 1
    assert metrics["prefill_attempt_count"] == 1
    assert metrics["eviction_pause_count"] == 0
    assert metrics["prefill_eviction_retries"] == 0
    assert metrics["external_prefill_attempt_seconds"] == [
        metrics["external_prefill_seconds"]
    ]
    assert mx.clear_count == 2


def test_write_locations_cannot_resolve_under_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Result and cache destinations beneath source tensors are rejected."""
    model = tmp_path / "model"
    model.mkdir()
    safe_output = tmp_path / "result.json"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    validation._validate_runtime_write_locations(
        model,
        output_path=safe_output,
        dataset_cache_dir=None,
    )
    with pytest.raises(validation.ValidationError, match="output"):
        validation._validate_runtime_write_locations(
            model,
            output_path=model / "result.json",
            dataset_cache_dir=None,
        )


def test_help_describes_public_supervised_modes() -> None:
    """Top-level help exposes both useful public validation workflows."""
    parser = validation.build_parser()
    help_text = parser.format_help()

    assert "matrix" in help_text
    assert "organic" in help_text
    assert "quality, throughput, and memory" in help_text
