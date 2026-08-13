#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible, parent-supervised TurboQuant mid-prefill validation."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

DATASET_REPO = "openai/mrcr"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
DATASET_FILE = "2needle/2needle_0.parquet"
DEFAULT_MATRIX_ROWS: tuple[int, ...] = (104, 109, 136, 301, 311, 328)
PRIMARY_PERFORMANCE_ROWS: Mapping[int, str] = {
    104: "primary_32k",
    311: "primary_64k",
}
ORGANIC_ROW_INDEX = 35
ORGANIC_PROMPT_TOKENS = 131_071
FIXED_REPLAY_TOKENS = 256
ORGANIC_PREFILL_ABORT_MARGIN = 0.95
ORGANIC_PREFILL_MIN_CHUNK_TOKENS = 32
SCHEMA_VERSION = 1
GIB = 1024**3
TELEMETRY_EXIT_GRACE_SECONDS = 1.0
_FORCE_ENV_PREFIX = "OMLX_FORCE_"
_T = TypeVar("_T")


class ValidationError(RuntimeError):
    """Raised when the validation contract cannot be established."""


class TelemetryError(ValidationError):
    """Raised when mandatory supervisor telemetry is invalid."""


@dataclass(frozen=True, slots=True)
class ValidationMode:
    """One fixed matrix validation mode."""

    name: str
    conversion: str
    bits: float | None


@dataclass(frozen=True, slots=True)
class HostMemoryReading:
    """Supported macOS host-memory counters and conservative headroom."""

    free_bytes: int
    inactive_bytes: int
    active_bytes: int
    headroom_bytes: int


@dataclass(slots=True)
class SupervisedOutcome:
    """One child result plus mandatory parent-side safety observations."""

    result: dict[str, Any] | None
    telemetry: dict[str, Any]
    returncode: int
    error: str | None




def matrix_modes() -> tuple[ValidationMode, ...]:
    """Return the complete, ordered five-mode validation matrix."""
    return (
        ValidationMode("dense", "dense", None),
        ValidationMode("ordinary-q8", "ordinary", 8.0),
        ValidationMode("ordinary-q4", "ordinary", 4.0),
        ValidationMode("mid-q8", "mid", 8.0),
        ValidationMode("mid-q4", "mid", 4.0),
    )


def pinned_dataset_config() -> dict[str, Any]:
    """Return immutable MRCR retrieval provenance as JSON-compatible data."""
    return {
        "repo_id": DATASET_REPO,
        "revision": DATASET_REVISION,
        "filename": DATASET_FILE,
        "matrix_row_indices": list(DEFAULT_MATRIX_ROWS),
        "organic_row_index": ORGANIC_ROW_INDEX,
    }


def select_rows(
    rows: Sequence[Mapping[str, Any]], row_indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Select rows by stable positional index and annotate each selected row."""
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_index in row_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("row indices must be integers")
        if raw_index < 0 or raw_index >= len(rows):
            raise IndexError(f"row index {raw_index} is outside 0..{len(rows) - 1}")
        if raw_index in seen:
            raise ValueError(f"duplicate row index {raw_index}")
        seen.add(raw_index)
        row = dict(rows[raw_index])
        row["row_index"] = raw_index
        selected.append(row)
    return selected


def official_retrieval_score(response: str, answer: str, prefix: str) -> float:
    """Apply MRCR's official prefix gate and SequenceMatcher ratio."""
    if not response.startswith(prefix):
        return 0.0
    sampled = response.removeprefix(prefix)
    expected = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, sampled, expected).ratio())


def host_memory_reading(stats: Mapping[str, Any]) -> HostMemoryReading:
    """Build conservative headroom from supported host VM counters."""
    values = tuple(stats.get(name) for name in ("free", "inactive", "active"))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise TelemetryError(
            "host_statistics64 omitted valid free, inactive, or active bytes"
        )
    free_bytes, inactive_bytes, active_bytes = values
    return HostMemoryReading(
        free_bytes=free_bytes,
        inactive_bytes=inactive_bytes,
        active_bytes=active_bytes,
        headroom_bytes=free_bytes + inactive_bytes,
    )


def safety_violation(
    child_phys_bytes: int,
    host_headroom_bytes: int,
    *,
    child_limit_bytes: int,
    host_minimum_bytes: int,
) -> str | None:
    """Return the fail-closed threshold violation, if any."""
    values = (
        child_phys_bytes,
        host_headroom_bytes,
        child_limit_bytes,
        host_minimum_bytes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return "invalid safety telemetry type"
    if child_phys_bytes <= 0 or host_headroom_bytes < 0:
        return "invalid safety telemetry value"
    if child_limit_bytes <= 0 or host_minimum_bytes < 0:
        return "invalid safety threshold"
    if child_phys_bytes >= child_limit_bytes:
        return "child physical footprint reached its safety limit"
    if host_headroom_bytes < host_minimum_bytes:
        return "host headroom fell below its safety minimum"
    return None


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data deterministically for hashing and atomic output."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON data."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_text(value: str) -> str:
    """Return the exact UTF-8 SHA-256 of a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Return a stable SHA-256 over an exact token-id sequence."""
    return stable_digest(list(token_ids))


def _enable_verified_no_cache(file_descriptor: int) -> None:
    """Enable Darwin F_NOCACHE and require the setter's verified success result."""
    if sys.platform != "darwin":
        raise ValidationError("F_NOCACHE validation requires Darwin")
    command = getattr(fcntl, "F_NOCACHE", None)
    if isinstance(command, bool) or not isinstance(command, int):
        raise ValidationError("Python does not expose Darwin F_NOCACHE")
    try:
        result = fcntl.fcntl(file_descriptor, command, 1)
    except OSError as exc:
        raise ValidationError(
            f"F_NOCACHE failed for source tensor read: {exc}"
        ) from exc
    if result != 0:
        raise ValidationError(f"F_NOCACHE setter returned unverified result {result!r}")


def hash_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one file through a verified no-cache descriptor."""
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        _enable_verified_no_cache(handle.fileno())
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_manifest_path(path: Path, *, root: Path) -> str:
    """Return one manifest path relative to its model root."""
    return path.relative_to(root).as_posix()


def build_model_manifest(model_path: Path) -> list[dict[str, Any]]:
    """Hash every source safetensors file below a local model directory."""
    root = model_path.expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"model path is not a directory: {root}")
    files = sorted(
        (path for path in root.rglob("*.safetensors") if path.is_file()),
        key=partial(_relative_manifest_path, root=root),
    )
    if not files:
        raise ValidationError(f"no source .safetensors files found under {root}")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        }
        for path in files
    ]


def manifests_equal(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> bool:
    """Compare source tensor manifests independent of mapping key order."""
    return canonical_json_bytes(list(before)) == canonical_json_bytes(list(after))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON result and fsync both file and directory."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def shape_validation_result(
    *,
    kind: str,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    before_manifest: Sequence[Mapping[str, Any]],
    after_manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    """Build the stable top-level result shape used by both public modes."""
    unchanged = manifests_equal(before_manifest, after_manifest)
    effective_error = error
    if not unchanged:
        mutation_error = "source .safetensors manifest changed during validation"
        effective_error = (
            mutation_error
            if effective_error is None
            else f"{effective_error}; {mutation_error}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if effective_error is None else "failed",
        "kind": kind,
        "provenance": dict(provenance),
        "config": dict(config),
        "source_tensors": {
            "before": list(before_manifest),
            "after": list(after_manifest),
            "unchanged": unchanged,
        },
        "results": [dict(result) for result in results],
        "error": effective_error,
    }


def shape_validation_checkpoint(
    *,
    kind: str,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    before_manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a restart-safe partial result without claiming final tensor proof."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "kind": kind,
        "provenance": dict(provenance),
        "config": dict(config),
        "source_tensors": {
            "before": list(before_manifest),
            "after": None,
            "unchanged": None,
        },
        "results": [dict(result) for result in results],
        "error": None,
    }


def child_environment(base_environment: Mapping[str, str]) -> dict[str, str]:
    """Build a child environment with every forced diagnostic hook removed."""
    return {
        key: value
        for key, value in base_environment.items()
        if not key.startswith(_FORCE_ENV_PREFIX)
    }


def build_child_command(
    script_path: Path, spec_path: Path, result_path: Path
) -> list[str]:
    """Build the private child invocation without any trigger controls."""
    return [
        sys.executable,
        str(script_path.expanduser().resolve()),
        "__child",
        "--spec",
        str(spec_path.expanduser().resolve()),
        "--result",
        str(result_path.expanduser().resolve()),
    ]


def matrix_shared_metadata(
    *,
    row_index: int,
    chunk_size: int,
    max_prompt_tokens: int,
    greedy_token_limit: int,
) -> dict[str, Any]:
    """Build metadata that must be byte-identical across a sample's modes."""
    return {
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "filename": DATASET_FILE,
            "row_index": row_index,
        },
        "quality_cell": True,
        "performance_cell": PRIMARY_PERFORMANCE_ROWS.get(row_index),
        "thinking_enabled": False,
        "chunk_size": chunk_size,
        "max_prompt_tokens": max_prompt_tokens,
        "greedy_token_limit": greedy_token_limit,
        "teacher_forced_replay_token_limit": FIXED_REPLAY_TOKENS,
    }


def validate_matrix_results(
    results: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[int] | None = None,
) -> None:
    """Reject incomplete modes, rows, or cross-mode metadata/input drift."""
    expected_modes = {mode.name for mode in matrix_modes()}
    by_row: dict[int, list[Mapping[str, Any]]] = {}
    for result in results:
        metadata = result.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValidationError("matrix child result omitted metadata")
        dataset = metadata.get("dataset")
        if not isinstance(dataset, Mapping) or not isinstance(
            dataset.get("row_index"), int
        ):
            raise ValidationError("matrix child result omitted its row index")
        by_row.setdefault(int(dataset["row_index"]), []).append(result)
    if expected_rows is not None and set(by_row) != set(expected_rows):
        raise ValidationError(
            f"matrix rows {sorted(by_row)} do not match {sorted(expected_rows)}"
        )
    for row_index, group in by_row.items():
        modes = {str(item.get("mode")) for item in group}
        if len(group) != len(expected_modes) or modes != expected_modes:
            raise ValidationError(
                f"row {row_index} has modes {sorted(modes)}, expected {sorted(expected_modes)}"
            )
        metadata_digests = {stable_digest(item.get("metadata")) for item in group}
        if len(metadata_digests) != 1:
            raise ValidationError(f"row {row_index} metadata drifted across modes")
        input_rows: list[tuple[Any, ...]] = []
        for item in group:
            metrics = item.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValidationError(f"row {row_index} omitted input metrics")
            values = (
                metrics.get("dataset_prompt_sha256"),
                metrics.get("rendered_prompt_sha256"),
                metrics.get("answer_sha256"),
                metrics.get("prompt_token_ids_sha256"),
                metrics.get("answer_token_ids_sha256"),
                metrics.get("prompt_token_count"),
                metrics.get("answer_token_count"),
            )
            if any(value is None for value in values):
                raise ValidationError(f"row {row_index} omitted exact input identity")
            input_rows.append(values)
        if len(set(input_rows)) != 1:
            raise ValidationError(f"row {row_index} inputs drifted across modes")


def _run_text_command(command: Sequence[str], cwd: Path | None = None) -> str | None:
    """Run a short provenance command and return stripped stdout on success."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer provenance value."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def collect_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect git, package, hardware, and exact invocation configuration."""
    repo_root = Path(__file__).resolve().parents[1]
    packages: dict[str, str | None] = {}
    for package in (
        "omlx",
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "transformers",
        "huggingface-hub",
        "pyarrow",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    git_head = _run_text_command(("git", "rev-parse", "HEAD"), repo_root)
    git_status = _run_text_command(
        ("git", "status", "--porcelain", "--untracked-files=no"), repo_root
    )
    physical_memory = _run_text_command(("/usr/sbin/sysctl", "-n", "hw.memsize"))
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "macos_version": platform.mac_ver()[0],
        "chip": _run_text_command(
            ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
        ),
        "physical_memory_bytes": _parse_optional_int(physical_memory),
    }
    return {
        "git": {
            "head": git_head,
            "tracked_worktree_clean": (
                git_status == "" if git_status is not None else None
            ),
        },
        "packages": packages,
        "hardware": hardware,
        "config_sha256": stable_digest(config),
    }


def fetch_mrcr_rows(
    row_indices: Sequence[int], cache_dir: Path | None
) -> list[dict[str, Any]]:
    """Fetch the pinned parquet revision and return validated positional rows."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        revision=DATASET_REVISION,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    table = parquet.read_table(
        parquet_path,
        columns=["prompt", "answer", "random_string_to_prepend"],
    )
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in row_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError("MRCR row indices must be integers")
        if index < 0 or index >= table.num_rows:
            raise ValidationError(
                f"MRCR row {index} is outside pinned file length {table.num_rows}"
            )
        if index in seen:
            raise ValidationError(f"duplicate MRCR row {index}")
        seen.add(index)
        selected.append(
            {
                "row_index": index,
                "prompt": table["prompt"][index].as_py(),
                "answer": table["answer"][index].as_py(),
                "random_string_to_prepend": table["random_string_to_prepend"][
                    index
                ].as_py(),
            }
        )
    for row in selected:
        prompt = row.get("prompt")
        answer = row.get("answer")
        prefix = row.get("random_string_to_prepend")
        if (
            not isinstance(prompt, str)
            or not isinstance(answer, str)
            or not isinstance(prefix, str)
        ):
            raise ValidationError(f"MRCR row {row['row_index']} has invalid fields")
        messages = json.loads(prompt)
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValidationError(
                f"MRCR row {row['row_index']} prompt is not a message list"
            )
        row["messages"] = messages
        row["dataset_prompt_sha256"] = hash_text(prompt)
        del row["prompt"]
    return selected


def probe_host_memory() -> HostMemoryReading:
    """Read supported host_statistics64 counters or fail closed."""
    from omlx.utils.psutil_compat import get_macos_vm_stats

    try:
        stats = get_macos_vm_stats()
    except Exception as exc:
        raise TelemetryError(f"host_statistics64 probe failed: {exc}") from exc
    if stats is None:
        raise TelemetryError("host_statistics64 returned no valid data")
    return host_memory_reading(stats)


def probe_child_footprint(pid: int) -> int:
    """Read a child's kernel phys_footprint or fail closed."""
    from omlx.utils.proc_memory import get_phys_footprint

    try:
        value = get_phys_footprint(pid)
    except Exception as exc:
        raise TelemetryError(f"phys_footprint probe failed: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TelemetryError("phys_footprint probe returned no valid data")
    return value


def _terminate_child(
    process: subprocess.Popen[bytes], grace_seconds: float = 5.0
) -> None:
    """Terminate a supervised child and escalate only after a bounded grace."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def supervise_child(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    result_path: Path,
    poll_interval_seconds: float,
    child_limit_bytes: int,
    host_minimum_bytes: int,
) -> SupervisedOutcome:
    """Run one child while requiring valid safety telemetry on every poll."""
    if poll_interval_seconds <= 0:
        raise ValueError("poll interval must be positive")
    process = subprocess.Popen(list(command), env=dict(environment))
    peak_child = 0
    minimum_headroom: int | None = None
    samples = 0
    error: str | None = None
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                if samples == 0:
                    error = "child exited before mandatory safety telemetry was sampled"
                break
            try:
                child_phys = probe_child_footprint(process.pid)
                pressure = probe_host_memory()
                samples += 1
                peak_child = max(peak_child, child_phys)
                minimum_headroom = (
                    pressure.headroom_bytes
                    if minimum_headroom is None
                    else min(minimum_headroom, pressure.headroom_bytes)
                )
                violation = safety_violation(
                    child_phys,
                    pressure.headroom_bytes,
                    child_limit_bytes=child_limit_bytes,
                    host_minimum_bytes=host_minimum_bytes,
                )
                if violation is not None:
                    error = violation
                    _terminate_child(process)
                    break
            except TelemetryError as exc:
                # Darwin can stop serving proc_pid_rusage just before waitpid
                # observes a normal exit. Allow only a bounded teardown grace.
                try:
                    process.wait(timeout=TELEMETRY_EXIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    error = str(exc)
                    _terminate_child(process)
                else:
                    if samples == 0:
                        error = (
                            "child exited before mandatory safety telemetry was sampled"
                        )
                break
            time.sleep(poll_interval_seconds)

        returncode = process.wait()
        result: dict[str, Any] | None = None
        if error is None:
            if not result_path.is_file():
                error = "child exited without an atomic result"
            else:
                try:
                    decoded = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    error = f"child result is invalid: {exc}"
                else:
                    if not isinstance(decoded, dict):
                        error = "child result is not an object"
                    else:
                        result = decoded
                        if decoded.get("status") != "ok":
                            error = str(
                                decoded.get("error") or "child reported failure"
                            )
        if returncode != 0 and error is None:
            error = f"child exited with status {returncode}"
        telemetry = {
            "samples": samples,
            "peak_child_phys_footprint_bytes": peak_child,
            "minimum_host_headroom_bytes": minimum_headroom,
            "host_headroom_metric": "host_statistics64.free+inactive",
            "child_limit_bytes": child_limit_bytes,
            "host_minimum_bytes": host_minimum_bytes,
            "poll_interval_seconds": poll_interval_seconds,
        }
        return SupervisedOutcome(result, telemetry, returncode, error)
    finally:
        if process.poll() is None:
            _terminate_child(process)


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Render a model chat template while explicitly disabling thinking."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValidationError("model tokenizer has no chat template")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise ValidationError("chat template did not return text")
    return rendered


def _encode_prompt(tokenizer: Any, text: str) -> list[int]:
    """Encode already-templated prompt text using the model tokenizer."""
    encoded = tokenizer.encode(text)
    return [int(token) for token in encoded]


def _encode_answer(tokenizer: Any, text: str) -> list[int]:
    """Encode answer text without introducing a fresh sequence BOS token."""
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    return [int(token) for token in encoded]


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode generated token IDs without retaining model outputs."""
    try:
        decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        decoded = tokenizer.decode(list(token_ids))
    if not isinstance(decoded, str):
        raise ValidationError("tokenizer.decode did not return text")
    return decoded


def _extract_logits(output: Any) -> Any:
    """Normalize mlx-lm array and model-output return forms."""
    return output.logits if hasattr(output, "logits") else output


def _timed_synchronized(mx: Any, operation: Callable[[], _T]) -> tuple[_T, float]:
    """Fence one timed MLX phase on both sides."""
    mx.synchronize()
    started = time.perf_counter()
    result = operation()
    mx.synchronize()
    return result, time.perf_counter() - started


def _evaluate_cache(mx: Any, cache: list[Any]) -> None:
    """Materialize cache state for one completed fixed chunk."""
    mx.eval([cache_object.state for cache_object in cache])


def _prefill_chunk(
    mx: Any, model: Any, cache: list[Any], token_ids: Sequence[int]
) -> None:
    """Advance one prompt chunk while materializing only bounded cache state."""
    model(mx.array([list(token_ids)]), cache=cache)
    _evaluate_cache(mx, cache)


def _model_chunk(
    mx: Any, model: Any, cache: list[Any], token_ids: Sequence[int]
) -> Any:
    """Run one bounded final/decode forward and materialize current logits."""
    output = model(mx.array([list(token_ids)]), cache=cache)
    logits = _extract_logits(output)
    mx.eval(logits, [cache_object.state for cache_object in cache])
    return logits


def _ordinary_convert(mx: Any, cache: list[Any], bits: float) -> dict[str, Any]:
    """Use the production ordinary post-prefill conversion implementation."""
    from omlx.scheduler import Scheduler

    owner = SimpleNamespace(_turboquant_kv_bits=bits, _turboquant_skip_last=True)
    Scheduler._apply_turboquant_kv_convert(owner, cache)
    _evaluate_cache(mx, cache)
    return {"path": "ordinary_from_cache"}


def _mid_convert(mx: Any, cache: list[Any], bits: float) -> dict[str, Any]:
    """Use the bounded existing sliced conversion at the fixed midpoint."""
    from omlx.turboquant_kv import convert_kv_cache_sliced

    stats = convert_kv_cache_sliced(cache, bits=bits, skip_last=True)
    _evaluate_cache(mx, cache)
    return {"path": "sliced", **asdict(stats)}


def _run_direct_prefill(
    mx: Any,
    model: Any,
    prompt_ids: Sequence[int],
    mode: ValidationMode,
    chunk_size: int,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Run the direct fixed-chunk prompt loop with its exact conversion boundary."""
    from mlx_lm.models.cache import make_prompt_cache

    if len(prompt_ids) < 2:
        raise ValidationError("rendered prompt must contain at least two tokens")
    cache = make_prompt_cache(model)
    cacheable = list(prompt_ids[:-1])
    midpoint = len(cacheable) // 2
    prefix_forward_seconds = 0.0
    suffix_forward_seconds = 0.0
    conversion_seconds = 0.0
    conversion: dict[str, Any] | None = None

    mx.synchronize()
    total_started = time.perf_counter()
    for start in range(0, midpoint, chunk_size):
        end = min(start + chunk_size, midpoint)
        _, elapsed = _timed_synchronized(
            mx,
            partial(_prefill_chunk, mx, model, cache, cacheable[start:end]),
        )
        prefix_forward_seconds += elapsed
        mx.clear_cache()

    if mode.conversion == "mid":
        if mode.bits is None:
            raise ValidationError("mid mode omitted its bit width")
        conversion, conversion_seconds = _timed_synchronized(
            mx, partial(_mid_convert, mx, cache, mode.bits)
        )

    mx.synchronize()
    suffix_wall_started = time.perf_counter()
    for start in range(midpoint, len(cacheable), chunk_size):
        end = min(start + chunk_size, len(cacheable))
        _, elapsed = _timed_synchronized(
            mx,
            partial(_prefill_chunk, mx, model, cache, cacheable[start:end]),
        )
        suffix_forward_seconds += elapsed
        mx.clear_cache()
    mx.synchronize()
    suffix_wall_seconds = time.perf_counter() - suffix_wall_started

    # Production external prefill converts the resident N-1 prompt cache
    # before BatchGenerator processes the held final prompt token.
    if mode.conversion == "ordinary":
        if mode.bits is None:
            raise ValidationError("ordinary mode omitted its bit width")
        conversion, conversion_seconds = _timed_synchronized(
            mx, partial(_ordinary_convert, mx, cache, mode.bits)
        )

    logits, final_seconds = _timed_synchronized(
        mx, partial(_model_chunk, mx, model, cache, prompt_ids[-1:])
    )
    suffix_forward_seconds += final_seconds
    suffix_wall_seconds += final_seconds
    mx.synchronize()
    total_seconds = time.perf_counter() - total_started
    suffix_tokens = len(prompt_ids) - midpoint
    return (
        cache,
        logits,
        {
            "prompt_tokens": len(prompt_ids),
            "held_final_prompt_tokens": 1,
            "midpoint_token_index": midpoint,
            "same_boundary_suffix_tokens": suffix_tokens,
            "prefix_forward_seconds": prefix_forward_seconds,
            "same_boundary_suffix_forward_seconds": suffix_forward_seconds,
            "same_boundary_suffix_seconds": suffix_wall_seconds,
            "same_boundary_suffix_tokens_per_second": (
                suffix_tokens / suffix_wall_seconds if suffix_wall_seconds > 0 else 0.0
            ),
            "conversion_seconds": conversion_seconds,
            "conversion": conversion,
            "total_prefill_seconds": total_seconds,
            "total_prefill_tokens_per_second": (
                len(prompt_ids) / total_seconds if total_seconds > 0 else 0.0
            ),
        },
    )


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _model_token_chunk(
    mx: Any,
    model: Any,
    cache: list[Any],
    token_id: int,
    stream: Any | None,
) -> Any:
    """Run one teacher-forced token on the scheduler stream when provided."""
    if stream is None:
        return _model_chunk(mx, model, cache, (token_id,))
    with mx.stream(stream):
        return _model_chunk(mx, model, cache, (token_id,))


def _teacher_forced_replay(
    mx: Any,
    model: Any,
    cache: list[Any],
    initial_logits: Any,
    replay_ids: Sequence[int],
    *,
    stream: Any | None = None,
) -> dict[str, Any]:
    """Score and time bounded one-token teacher-forced forwards."""
    if not replay_ids:
        raise ValidationError("teacher-forced replay is empty")
    logits = initial_logits
    latencies: list[float] = []
    nll_values: list[float] = []
    for token_id in replay_ids:
        row = logits[0, -1].astype(mx.float32)
        nll = mx.logsumexp(row) - row[int(token_id)]
        mx.eval(nll)
        nll_values.append(float(nll.item()))

        logits, elapsed = _timed_synchronized(
            mx,
            partial(
                _model_token_chunk,
                mx,
                model,
                cache,
                int(token_id),
                stream,
            ),
        )
        latencies.append(elapsed)
    total_nll = math.fsum(nll_values)
    mean_nll = total_nll / len(nll_values)
    remaining = nll_values[1:]
    remaining_mean = math.fsum(remaining) / len(remaining) if remaining else None
    total_seconds = math.fsum(latencies)
    return {
        "token_count": len(replay_ids),
        "token_ids_sha256": hash_token_ids(replay_ids),
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "first_token_nll": nll_values[0],
        "tokens_2_to_n_count": len(remaining),
        "tokens_2_to_n_mean_nll": remaining_mean,
        "tokens_2_to_n_perplexity": (
            math.exp(remaining_mean) if remaining_mean is not None else None
        ),
        "decode_seconds": total_seconds,
        "decode_tokens_per_second": (
            len(replay_ids) / total_seconds if total_seconds > 0 else 0.0
        ),
        "latency_median_seconds": float(statistics.median(latencies)),
        "latency_p95_seconds": _percentile_nearest_rank(latencies, 0.95),
    }


def _eos_token_ids(tokenizer: Any) -> set[int]:
    """Normalize a tokenizer's EOS declaration."""
    raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, int) and not isinstance(raw, bool):
        return {raw}
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
        return {int(value) for value in raw}
    return set()


def _greedy_response(
    mx: Any,
    model: Any,
    cache: list[Any],
    initial_logits: Any,
    tokenizer: Any,
    token_limit: int,
) -> tuple[list[int], str]:
    """Generate a separate untimed greedy response for official MRCR grading."""
    logits = initial_logits
    generated: list[int] = []
    eos_ids = _eos_token_ids(tokenizer)
    for _ in range(token_limit):
        token = mx.argmax(logits[0, -1])
        mx.eval(token)
        token_id = int(token.item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        logits = _model_chunk(mx, model, cache, (token_id,))
    return generated, _decode_tokens(tokenizer, generated)


def _reset_mlx_measurement(mx: Any) -> dict[str, int]:
    """Clear transients, capture separate baselines, and reset allocator peak."""
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()
    baseline = {
        "mlx_active_baseline_bytes": int(mx.get_active_memory()),
        "mlx_cache_baseline_bytes": int(mx.get_cache_memory()),
    }
    mx.reset_peak_memory()
    return baseline


def _finish_mlx_measurement(mx: Any, baseline: Mapping[str, int]) -> dict[str, int]:
    """Capture MLX peak/final counters without conflating phys_footprint."""
    mx.synchronize()
    return {
        **dict(baseline),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_active_final_bytes": int(mx.get_active_memory()),
        "mlx_cache_final_bytes": int(mx.get_cache_memory()),
    }


def _load_direct_model(
    model_path: str, mode: ValidationMode, trust: bool
) -> tuple[Any, Any]:
    """Load one direct-loop model with production compatibility transforms."""
    import mlx.core as mx

    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import (
        apply_post_load_transforms,
        lm_load_compat,
        materialize_lazy_state,
        maybe_apply_pre_load_patches,
    )
    from omlx.utils.tokenizer import get_tokenizer_config

    settings = ModelSettings(
        enable_thinking=False,
        turboquant_kv_enabled=mode.bits is not None,
        turboquant_mid_prefill=mode.conversion == "mid",
        turboquant_kv_bits=mode.bits or 4.0,
        turboquant_skip_last=True,
    )
    maybe_apply_pre_load_patches(model_path, model_settings=settings)
    tokenizer_config = get_tokenizer_config(model_path, trust_remote_code=trust)
    model, tokenizer = lm_load_compat(
        model_path,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust,
    )
    model = apply_post_load_transforms(model, settings)
    materialize_lazy_state(model)
    # Match the production EnginePool handoff: source loading and lazy
    # materialization can leave unreferenced Metal buffers in the allocator
    # pool. Release them before the supervised prefill starts so the 6 GiB
    # host floor measures the model request, not loader residue.
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    if mode.bits is not None:
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

        apply_turboquant_attention_patch()
    return model, tokenizer


def run_matrix_child(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one fresh real-model matrix cell."""
    import mlx.core as mx

    mode_data = spec["mode"]
    mode = ValidationMode(
        str(mode_data["name"]),
        str(mode_data["conversion"]),
        float(mode_data["bits"]) if mode_data["bits"] is not None else None,
    )
    sample = spec["sample"]
    metadata = dict(spec["metadata"])
    model: Any | None = None
    tokenizer: Any | None = None
    primary_cache: list[Any] | None = None
    greedy_cache: list[Any] | None = None
    try:
        model, tokenizer = _load_direct_model(
            str(spec["model_path"]), mode, bool(spec.get("trust_remote_code", False))
        )
        rendered = _apply_chat_template(tokenizer, list(sample["messages"]))
        prompt_ids = _encode_prompt(tokenizer, rendered)
        if len(prompt_ids) > int(metadata["max_prompt_tokens"]):
            raise ValidationError(
                f"rendered prompt has {len(prompt_ids)} tokens, above safety limit "
                f"{metadata['max_prompt_tokens']}"
            )
        answer = str(sample["answer"])
        answer_ids = _encode_answer(tokenizer, answer)
        replay_ids = answer_ids[:FIXED_REPLAY_TOKENS]
        baseline = _reset_mlx_measurement(mx)

        primary_cache, primary_logits, prefill = _run_direct_prefill(
            mx, model, prompt_ids, mode, int(metadata["chunk_size"])
        )
        replay = _teacher_forced_replay(
            mx, model, primary_cache, primary_logits, replay_ids
        )
        primary_cache = None
        primary_logits = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        greedy_cache, greedy_logits, _ = _run_direct_prefill(
            mx, model, prompt_ids, mode, int(metadata["chunk_size"])
        )
        generated_ids, response = _greedy_response(
            mx,
            model,
            greedy_cache,
            greedy_logits,
            tokenizer,
            int(metadata["greedy_token_limit"]),
        )
        score = official_retrieval_score(
            response,
            answer,
            str(sample["random_string_to_prepend"]),
        )
        memory = _finish_mlx_measurement(mx, baseline)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "kind": "matrix",
            "mode": mode.name,
            "metadata": metadata,
            "metrics": {
                "dataset_prompt_sha256": sample["dataset_prompt_sha256"],
                "rendered_prompt_sha256": hash_text(rendered),
                "prompt_token_ids_sha256": hash_token_ids(prompt_ids),
                "answer_sha256": hash_text(answer),
                "answer_token_ids_sha256": hash_token_ids(answer_ids),
                "prompt_token_count": len(prompt_ids),
                "answer_token_count": len(answer_ids),
                "teacher_forced_replay_truncated": len(answer_ids) > len(replay_ids),
                "prefill": prefill,
                "teacher_forced": replay,
                "greedy_retrieval": {
                    "response": response,
                    "response_sha256": hash_text(response),
                    "token_count": len(generated_ids),
                    "token_ids_sha256": hash_token_ids(generated_ids),
                    "required_prefix": sample["random_string_to_prepend"],
                    "official_score": score,
                },
                "memory": memory,
            },
        }
    finally:
        primary_cache = None
        greedy_cache = None
        model = None
        tokenizer = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            logger.exception("failed to clear MLX cache after matrix child")


def _find_pool_model_id(pool: Any, model_path: Path) -> str:
    """Resolve the discovered EnginePool ID for an exact local model path."""
    expected = model_path.expanduser().resolve()
    matches = [
        model_id
        for model_id in pool.get_model_ids()
        if Path(pool.get_entry(model_id).model_path).expanduser().resolve() == expected
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"EnginePool discovered {len(matches)} entries for exact model path {expected}"
        )
    return matches[0]


def _capture_no_victim_organic_prefill(
    mx: Any,
    scheduler: Any,
    request: Any,
    prompt_ids: Sequence[int],
    replay_ids: Sequence[int],
) -> dict[str, Any]:
    """Run one sole-model prefill and require in-attempt conversion."""
    from omlx.scheduler import _PrefillEvictionNeeded

    captured: list[Any] = []
    original = scheduler._new_prefill_context

    def _capture(
        inner_request: Any,
        prompt_cache: list[Any],
        *,
        loop_label: str,
    ) -> Any:
        context = original(inner_request, prompt_cache, loop_label=loop_label)
        captured.append(context)
        return context

    scheduler._new_prefill_context = _capture
    baseline = _reset_mlx_measurement(mx)
    mx.synchronize()
    total_started = time.perf_counter()
    cache: list[Any] | None = None
    logits: Any | None = None
    try:
        try:
            (cache, last_token), external_seconds = _timed_synchronized(
                mx,
                partial(
                    scheduler._do_external_prefill,
                    request,
                    list(prompt_ids),
                    existing_cache=None,
                ),
            )
        except _PrefillEvictionNeeded as exc:
            raise ValidationError(
                "organic sole-model prefill requested an external-victim pause"
            ) from exc
        external_completed = time.perf_counter()
        with mx.stream(scheduler._stream):
            logits, final_seconds = _timed_synchronized(
                mx,
                partial(
                    _model_chunk,
                    mx,
                    scheduler.model,
                    cache,
                    tuple(last_token),
                ),
            )
        total_seconds = time.perf_counter() - total_started
        replay = _teacher_forced_replay(
            mx,
            scheduler.model,
            cache,
            logits,
            replay_ids,
            stream=scheduler._stream,
        )
        if len(captured) != 1:
            raise ValidationError(
                f"organic prefill captured {len(captured)} contexts instead of one"
            )
        context = captured[0]
        if not context.mid_triggered:
            raise ValidationError(
                "organic prefill completed without mid-prefill conversion"
            )
        if not context.conversion_attempted:
            raise ValidationError(
                "organic prefill triggered without recording its conversion attempt"
            )
        if int(request.prefill_eviction_retries) != 0:
            raise ValidationError(
                "organic sole-model prefill restarted for an external victim"
            )
        if request.turboquant_mid_prefill_attempted is not True:
            raise ValidationError(
                "organic request did not retain its one conversion attempt"
            )
        post_trigger_seconds = (
            external_completed - context.conversion_completed_at
            if context.conversion_completed_at is not None
            else 0.0
        )
        memory = _finish_mlx_measurement(mx, baseline)
        return {
            "actual_mid_prefill_trigger": True,
            "mid_prefill_conversion_attempt_count": 1,
            "prefill_attempt_count": 1,
            "eviction_pause_count": 0,
            "prefill_eviction_retries": 0,
            "eviction_callback_reclaimed": False,
            "eviction_pause_seconds": 0.0,
            "trigger_tokens": context.trigger_tokens,
            "conversion_seconds": float(context.conversion_seconds),
            "converted_layers": int(context.converted_layers),
            "conversion_slices": int(context.conversion_slices),
            "skipped_dense_layers": int(context.skipped_dense_layers),
            "external_prefill_attempt_seconds": [external_seconds],
            "external_prefill_seconds": external_seconds,
            "held_final_prompt_token_seconds": final_seconds,
            "total_prefill_seconds": total_seconds,
            "total_prefill_tokens_per_second": (
                len(prompt_ids) / total_seconds if total_seconds > 0 else 0.0
            ),
            "post_trigger_tokens": int(context.post_trigger_tokens),
            "post_trigger_seconds": post_trigger_seconds,
            "post_trigger_tokens_per_second": (
                context.post_trigger_tokens / post_trigger_seconds
                if post_trigger_seconds > 0
                else 0.0
            ),
            "teacher_forced": replay,
            "memory": memory,
        }
    finally:
        cache = None
        logits = None
        scheduler._new_prefill_context = original
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def _configure_organic_pressure(
    scheduler: Any,
    *,
    soft_limit_bytes: int,
    hard_limit_bytes: int,
    prefill_abort_margin: float,
    prefill_min_chunk_tokens: int,
) -> None:
    """Apply normal propagated scheduler pressure limits without a trigger hook."""
    if soft_limit_bytes <= 0 or hard_limit_bytes <= 0:
        raise ValidationError("organic scheduler limits must be positive")
    if soft_limit_bytes > hard_limit_bytes:
        raise ValidationError("organic soft limit exceeds hard limit")
    if not 0.0 < prefill_abort_margin <= 1.0:
        raise ValidationError("organic prefill abort margin must be in (0, 1]")
    if prefill_min_chunk_tokens <= 0:
        raise ValidationError("organic prefill minimum chunk must be positive")
    scheduler._memory_limit_bytes = soft_limit_bytes
    scheduler._memory_hard_limit_bytes = hard_limit_bytes
    scheduler._memory_hard_watermark_bytes = hard_limit_bytes
    scheduler._memory_abort_limit_bytes = hard_limit_bytes
    scheduler._memory_static_ceiling_bytes = hard_limit_bytes
    scheduler._memory_dynamic_ceiling_bytes = hard_limit_bytes
    scheduler._memory_metal_cap_bytes = hard_limit_bytes
    scheduler._memory_guard_tier = "custom"
    scheduler._prefill_abort_margin = prefill_abort_margin
    scheduler._prefill_min_chunk_tokens = prefill_min_chunk_tokens
    scheduler._prefill_memory_guard = True
    scheduler._memory_limits_propagated = True


async def _run_organic_child_async(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Load via EnginePool and run the production external-prefill pressure path."""
    import mlx.core as mx

    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import SchedulerConfig

    model_path = Path(str(spec["model_path"])).expanduser().resolve()
    chunk_size = int(spec["chunk_size"])
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        completion_batch_size=1,
        prefill_step_size=chunk_size,
        chunked_prefill=False,
        prefill_speed_priority=False,
        paged_ssd_cache_dir=None,
        hot_cache_max_size=0,
    )
    pool = EnginePool(scheduler_config=scheduler_config)
    engine: Any | None = None
    scheduler: Any | None = None
    request: Any | None = None
    try:
        pool.discover_models(str(model_path.parent))
        model_id = _find_pool_model_id(pool, model_path)
        settings = ModelSettings(
            enable_thinking=False,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            turboquant_kv_enabled=True,
            turboquant_mid_prefill=True,
            turboquant_kv_bits=8.0,
            turboquant_skip_last=True,
        )
        engine = await pool.get_engine(
            model_id,
            force_lm=True,
            runtime_settings=settings,
        )
        if pool.loaded_model_count != 1:
            raise ValidationError(
                "organic validation did not retain exclusive model count"
            )
        rendered = _apply_chat_template(
            engine.tokenizer,
            list(spec["sample"]["messages"]),
        )
        rendered_ids = _encode_prompt(engine.tokenizer, rendered)
        required = ORGANIC_PROMPT_TOKENS + FIXED_REPLAY_TOKENS
        if len(rendered_ids) < required:
            raise ValidationError(
                f"organic row rendered to {len(rendered_ids)} tokens, requires {required}"
            )
        prompt_ids = rendered_ids[:ORGANIC_PROMPT_TOKENS]
        replay_ids = rendered_ids[ORGANIC_PROMPT_TOKENS:required]
        core = engine._engine.engine
        await core.stop()
        scheduler = core.scheduler
        if scheduler._prefill_eviction_callback_configured is not True:
            raise ValidationError("organic engine has no production prefill callback")
        _configure_organic_pressure(
            scheduler,
            soft_limit_bytes=int(spec["scheduler_soft_limit_bytes"]),
            hard_limit_bytes=int(spec["scheduler_hard_limit_bytes"]),
            prefill_abort_margin=float(spec["prefill_abort_margin"]),
            prefill_min_chunk_tokens=int(spec["prefill_min_chunk_tokens"]),
        )
        request = Request(
            request_id="turboquant-organic-validation",
            prompt=list(prompt_ids),
            sampling_params=SamplingParams(
                max_tokens=FIXED_REPLAY_TOKENS,
                temperature=0.0,
            ),
            prompt_token_ids=list(prompt_ids),
            num_prompt_tokens=len(prompt_ids),
            skip_cache_store=True,
        )
        scheduler.requests[request.request_id] = request
        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            core._mlx_executor,
            _capture_no_victim_organic_prefill,
            mx,
            scheduler,
            request,
            prompt_ids,
            replay_ids,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "kind": "organic",
            "mode": "organic-mid-q8",
            "metadata": dict(spec["metadata"]),
            "metrics": {
                "dataset_prompt_sha256": spec["sample"]["dataset_prompt_sha256"],
                "full_rendered_sha256": hash_text(rendered),
                "full_rendered_token_count": len(rendered_ids),
                "prompt_token_ids_sha256": hash_token_ids(prompt_ids),
                "prompt_token_count": len(prompt_ids),
                "replay_token_ids_sha256": hash_token_ids(replay_ids),
                "replay_token_count": len(replay_ids),
                **metrics,
            },
        }
    finally:
        if scheduler is not None and request is not None:
            scheduler.requests.pop(request.request_id, None)
        await pool.shutdown()
        engine = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            logger.exception("failed to clear MLX cache after organic child")


def run_organic_child(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run the asynchronous organic child entry point."""
    return asyncio.run(_run_organic_child_async(spec))


def run_child(spec_path: Path, result_path: Path) -> int:
    """Execute a private child spec and always publish an atomic result."""
    try:
        decoded = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValidationError("child spec is not an object")
        kind = decoded.get("kind")
        if kind == "matrix":
            result = run_matrix_child(decoded)
        elif kind == "organic":
            result = run_organic_child(decoded)
        else:
            raise ValidationError(f"unknown child kind: {kind!r}")
        atomic_write_json(result_path, result)
        return 0
    except BaseException as exc:
        logger.exception("validation child failed")
        atomic_write_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


def _ensure_outside_model(model_path: Path, candidate: Path, label: str) -> None:
    """Reject every configured write location below the read-only model path."""
    model_root = model_path.expanduser().resolve()
    resolved = candidate.expanduser().resolve()
    if resolved == model_root or resolved.is_relative_to(model_root):
        raise ValidationError(f"{label} must not be under the model path")


def _validate_runtime_write_locations(
    model_path: Path,
    *,
    output_path: Path,
    dataset_cache_dir: Path | None,
) -> None:
    """Ensure every known cache/temp/result location is outside model sources."""
    _ensure_outside_model(model_path, output_path, "output")
    if dataset_cache_dir is not None:
        _ensure_outside_model(model_path, dataset_cache_dir, "dataset cache")
    _ensure_outside_model(
        model_path,
        Path(tempfile.gettempdir()),
        "temporary directory",
    )
    for variable in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
    ):
        value = os.environ.get(variable)
        if value:
            _ensure_outside_model(
                model_path,
                Path(value),
                f"{variable} environment location",
            )


def _run_supervised_spec(
    *,
    spec: Mapping[str, Any],
    workspace: Path,
    sequence: int,
    args: argparse.Namespace,
) -> SupervisedOutcome:
    """Persist one spec and supervise its fresh child."""
    spec_path = workspace / f"spec-{sequence:03d}.json"
    result_path = workspace / f"result-{sequence:03d}.json"
    atomic_write_json(spec_path, spec)
    command = build_child_command(Path(__file__), spec_path, result_path)
    return supervise_child(
        command=command,
        environment=child_environment(os.environ),
        result_path=result_path,
        poll_interval_seconds=float(args.poll_interval),
        child_limit_bytes=int(float(args.child_memory_gib) * GIB),
        host_minimum_bytes=int(float(args.host_headroom_gib) * GIB),
    )


def _matrix_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build JSON-compatible matrix configuration provenance."""
    return {
        "model_path": str(Path(args.model).expanduser().resolve()),
        "dataset_cache_dir": (
            str(Path(args.dataset_cache_dir).expanduser().resolve())
            if args.dataset_cache_dir
            else None
        ),
        "dataset": pinned_dataset_config(),
        "rows": list(args.rows),
        "modes": [asdict(mode) for mode in matrix_modes()],
        "chunk_size": args.chunk_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "greedy_token_limit": args.greedy_token_limit,
        "teacher_forced_replay_token_limit": FIXED_REPLAY_TOKENS,
        "thinking_enabled": False,
        "child_memory_limit_bytes": int(float(args.child_memory_gib) * GIB),
        "host_headroom_minimum_bytes": int(float(args.host_headroom_gib) * GIB),
        "poll_interval_seconds": args.poll_interval,
        "trust_remote_code": args.trust_remote_code,
    }


def _organic_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build JSON-compatible organic configuration provenance."""
    return {
        "model_path": str(Path(args.model).expanduser().resolve()),
        "dataset_cache_dir": (
            str(Path(args.dataset_cache_dir).expanduser().resolve())
            if args.dataset_cache_dir
            else None
        ),
        "dataset": {
            **pinned_dataset_config(),
            "row_index": ORGANIC_ROW_INDEX,
        },
        "mode": "organic-mid-q8",
        "prompt_tokens": ORGANIC_PROMPT_TOKENS,
        "teacher_forced_replay_tokens": FIXED_REPLAY_TOKENS,
        "chunk_size": args.chunk_size,
        "scheduler": {
            "max_num_seqs": 1,
            "max_num_batched_tokens": 8192,
            "completion_batch_size": 1,
            "prefill_step_size": args.chunk_size,
            "chunked_prefill": False,
            "prefill_speed_priority": False,
            "prefill_abort_margin": ORGANIC_PREFILL_ABORT_MARGIN,
            "prefill_min_chunk_tokens": ORGANIC_PREFILL_MIN_CHUNK_TOKENS,
        },
        "turboquant_bits": 8.0,
        "turboquant_skip_last": True,
        "exclusive_ownership": True,
        "no_cache": True,
        "forced_trigger": False,
        "thinking_enabled": False,
        "scheduler_soft_limit_bytes": int(float(args.scheduler_soft_limit_gib) * GIB),
        "scheduler_hard_limit_bytes": int(float(args.child_memory_gib) * GIB),
        "prefill_abort_margin": ORGANIC_PREFILL_ABORT_MARGIN,
        "prefill_min_chunk_tokens": ORGANIC_PREFILL_MIN_CHUNK_TOKENS,
        "child_memory_limit_bytes": int(float(args.child_memory_gib) * GIB),
        "host_headroom_minimum_bytes": int(float(args.host_headroom_gib) * GIB),
        "poll_interval_seconds": args.poll_interval,
        "trust_remote_code": args.trust_remote_code,
    }


def run_parent(args: argparse.Namespace) -> int:
    """Run all public work under fresh-child supervision and tensor verification."""
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_dir = (
        Path(args.dataset_cache_dir).expanduser().resolve()
        if args.dataset_cache_dir
        else None
    )
    _validate_runtime_write_locations(
        model_path,
        output_path=output_path,
        dataset_cache_dir=cache_dir,
    )
    config = _matrix_config(args) if args.command == "matrix" else _organic_config(args)
    provenance = collect_provenance(config)
    before = build_model_manifest(model_path)
    results: list[dict[str, Any]] = []
    error: str | None = None
    atomic_write_json(
        output_path,
        shape_validation_checkpoint(
            kind=str(args.command),
            provenance=provenance,
            config=config,
            before_manifest=before,
            results=results,
        ),
    )
    try:
        row_indices = (
            list(args.rows) if args.command == "matrix" else [ORGANIC_ROW_INDEX]
        )
        rows = fetch_mrcr_rows(row_indices, cache_dir)
        with tempfile.TemporaryDirectory(
            prefix="omlx-turboquant-validation-"
        ) as temporary:
            workspace = Path(temporary)
            sequence = 0
            if args.command == "matrix":
                for sample in rows:
                    metadata = matrix_shared_metadata(
                        row_index=int(sample["row_index"]),
                        chunk_size=int(args.chunk_size),
                        max_prompt_tokens=int(args.max_prompt_tokens),
                        greedy_token_limit=int(args.greedy_token_limit),
                    )
                    for mode in matrix_modes():
                        spec = {
                            "schema_version": SCHEMA_VERSION,
                            "kind": "matrix",
                            "model_path": str(model_path),
                            "trust_remote_code": bool(args.trust_remote_code),
                            "mode": asdict(mode),
                            "sample": sample,
                            "metadata": metadata,
                        }
                        outcome = _run_supervised_spec(
                            spec=spec,
                            workspace=workspace,
                            sequence=sequence,
                            args=args,
                        )
                        sequence += 1
                        child_result = (
                            dict(outcome.result)
                            if outcome.result is not None
                            else {
                                "schema_version": SCHEMA_VERSION,
                                "status": "failed",
                                "kind": "matrix",
                                "mode": mode.name,
                                "metadata": metadata,
                                "error": outcome.error,
                            }
                        )
                        child_result["supervision"] = outcome.telemetry
                        results.append(child_result)
                        atomic_write_json(
                            output_path,
                            shape_validation_checkpoint(
                                kind=str(args.command),
                                provenance=provenance,
                                config=config,
                                before_manifest=before,
                                results=results,
                            ),
                        )
                        if outcome.error is not None:
                            raise ValidationError(outcome.error)
                validate_matrix_results(results, expected_rows=args.rows)
            else:
                sample = rows[0]
                metadata = {
                    "dataset": {
                        "repo_id": DATASET_REPO,
                        "revision": DATASET_REVISION,
                        "filename": DATASET_FILE,
                        "row_index": ORGANIC_ROW_INDEX,
                    },
                    "thinking_enabled": False,
                    "prompt_tokens": ORGANIC_PROMPT_TOKENS,
                    "teacher_forced_replay_tokens": FIXED_REPLAY_TOKENS,
                    "chunk_size": int(args.chunk_size),
                    "no_cache": True,
                    "exclusive_ownership": True,
                    "forced_trigger": False,
                    "prefill_abort_margin": ORGANIC_PREFILL_ABORT_MARGIN,
                    "prefill_min_chunk_tokens": ORGANIC_PREFILL_MIN_CHUNK_TOKENS,
                }
                spec = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "organic",
                    "model_path": str(model_path),
                    "trust_remote_code": bool(args.trust_remote_code),
                    "sample": sample,
                    "metadata": metadata,
                    "chunk_size": int(args.chunk_size),
                    "scheduler_soft_limit_bytes": int(
                        float(args.scheduler_soft_limit_gib) * GIB
                    ),
                    "scheduler_hard_limit_bytes": int(
                        float(args.child_memory_gib) * GIB
                    ),
                    "prefill_abort_margin": ORGANIC_PREFILL_ABORT_MARGIN,
                    "prefill_min_chunk_tokens": ORGANIC_PREFILL_MIN_CHUNK_TOKENS,
                }
                outcome = _run_supervised_spec(
                    spec=spec,
                    workspace=workspace,
                    sequence=sequence,
                    args=args,
                )
                child_result = (
                    dict(outcome.result)
                    if outcome.result is not None
                    else {
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed",
                        "kind": "organic",
                        "mode": "organic-mid-q8",
                        "metadata": metadata,
                        "error": outcome.error,
                    }
                )
                child_result["supervision"] = outcome.telemetry
                results.append(child_result)
                atomic_write_json(
                    output_path,
                    shape_validation_checkpoint(
                        kind=str(args.command),
                        provenance=provenance,
                        config=config,
                        before_manifest=before,
                        results=results,
                    ),
                )
                if outcome.error is not None:
                    raise ValidationError(outcome.error)
    except BaseException as exc:
        logger.exception("validation parent failed")
        error = f"{type(exc).__name__}: {exc}"
    try:
        after = build_model_manifest(model_path)
    except BaseException as exc:
        logger.exception("post-run tensor manifest failed")
        after = []
        manifest_error = f"{type(exc).__name__}: {exc}"
        error = manifest_error if error is None else f"{error}; {manifest_error}"
    payload = shape_validation_result(
        kind=str(args.command),
        provenance=provenance,
        config=config,
        before_manifest=before,
        after_manifest=after,
        results=results,
        error=error,
    )
    atomic_write_json(output_path, payload)
    return 0 if payload["status"] == "ok" else 1


def _add_common_parent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared public supervisor arguments."""
    parser.add_argument(
        "--model", required=True, help="Local read-only model directory"
    )
    parser.add_argument("--output", required=True, help="Atomic provenance/result JSON")
    parser.add_argument(
        "--dataset-cache-dir",
        help="Optional Hugging Face dataset cache (must be outside the model)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Fixed prefill chunk size (default: 2048)",
    )
    parser.add_argument(
        "--child-memory-gib",
        type=float,
        default=37.0,
        help="Terminate at this child phys_footprint in GiB (default: 37)",
    )
    parser.add_argument(
        "--host-headroom-gib",
        type=float,
        default=6.0,
        help="Terminate below this kernel host headroom in GiB (default: 6)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Safety telemetry polling interval in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model remote code during loading",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the typed validation CLI parser."""
    parser = argparse.ArgumentParser(
        prog="validate_turboquant_mid_prefill.py",
        description=(
            "Validate TurboQuant mid-prefill quality, throughput, and memory "
            "with fail-closed parent supervision."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser(
        "matrix",
        help="Run pinned MRCR rows in dense, ordinary Q8/Q4, and mid Q8/Q4 modes",
    )
    _add_common_parent_arguments(matrix)
    matrix.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_MATRIX_ROWS),
        help="Pinned parquet row indices (default: 104 109 136 301 311 328)",
    )
    matrix.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=131072,
        help="Reject, never truncate, rendered prompts above this count",
    )
    matrix.add_argument(
        "--greedy-token-limit",
        type=int,
        default=1024,
        help="Maximum untimed greedy tokens for official MRCR scoring",
    )

    organic = subparsers.add_parser(
        "organic",
        help="Run the natural 131071-token production scheduler pressure path",
    )
    _add_common_parent_arguments(organic)
    organic.add_argument(
        "--scheduler-soft-limit-gib",
        type=float,
        default=32.0,
        help="Normal scheduler pressure target in GiB (default: 32)",
    )

    child = subparsers.add_parser("__child", help=argparse.SUPPRESS)
    child.add_argument("--spec", required=True)
    child.add_argument("--result", required=True)
    return parser


def _validate_cli_arguments(args: argparse.Namespace) -> None:
    """Reject nonsensical limits before hashing or launching children."""
    if args.command == "__child":
        return
    if args.chunk_size <= 0:
        raise ValidationError("chunk size must be positive")
    if args.child_memory_gib <= 0:
        raise ValidationError("child memory limit must be positive")
    if args.host_headroom_gib < 0:
        raise ValidationError("host headroom minimum must be non-negative")
    if args.poll_interval <= 0:
        raise ValidationError("poll interval must be positive")
    if args.command == "matrix":
        if args.max_prompt_tokens <= 1:
            raise ValidationError("max prompt tokens must exceed one")
        if args.greedy_token_limit <= 0:
            raise ValidationError("greedy token limit must be positive")
        rows = tuple(args.rows)
        if not rows:
            raise ValidationError("matrix rows must not be empty")
        if len(set(rows)) != len(rows):
            raise ValidationError("matrix rows must be unique")
        unsupported_rows = sorted(set(rows) - set(DEFAULT_MATRIX_ROWS))
        if unsupported_rows:
            raise ValidationError(
                "matrix rows must be selected from pinned rows "
                "104 109 136 301 311 328"
            )
    if args.command == "organic":
        if args.scheduler_soft_limit_gib <= 0:
            raise ValidationError("scheduler soft limit must be positive")
        if args.scheduler_soft_limit_gib > args.child_memory_gib:
            raise ValidationError("scheduler soft limit exceeds child hard limit")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_arguments(args)
        if args.command == "__child":
            return run_child(Path(args.spec), Path(args.result))
        return run_parent(args)
    except BaseException as exc:
        logger.error("validation failed: %s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
