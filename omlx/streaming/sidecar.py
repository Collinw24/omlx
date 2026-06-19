# SPDX-License-Identifier: Apache-2.0
"""StreamingExpertSidecar: SSD-backed expert weight streaming for MoE models.

Stores expert weights from safetensors in a single flat binary file with
16KB-aligned contiguous expert chunks (matching Apple Silicon's virtual memory
page size for zero-copy Metal buffer wrapping).  Uses Direct I/O
(``F_NOCACHE``) to bypass the macOS Unified Buffer Cache, preventing UBC
bloat and memory compression under sustained streaming of 100+ GB models.

Enables single ``os.pread()`` per expert during inference.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Map safetensors dtype strings to byte widths.
_DTYPE_BYTES: dict[str, int] = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1,  "U8": 1,  "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}

# Default alignment for expert chunks.
# Apple Silicon uses a 16 KiB virtual memory page size.  Any pointer wrapped
# into a Metal buffer (newBufferWithBytesNoCopy / mx.array) must be aligned
# to a 16 KiB boundary, otherwise the UMA subsystem rejects it or forces a
# blocking CPU-side copy.
_DEFAULT_ALIGNMENT = 16384  # 16 KiB

# macOS F_NOCACHE constant — bypasses the Unified Buffer Cache for direct
# NVMe → GPU buffer I/O.  Defined in <sys/fcntl.h>.
_F_NOCACHE: int = 48

# Projection order within each expert chunk.
_PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
_CATEGORY_ORDER = ("weight", "scales", "biases")

# ── Multi-architecture key detection registry ───────────────────────────
# Each entry: (regex_str, layer_group, proj_group, cat_group, expert_group_or_None, model_type_tag)
# - layer_group: 1-based regex group index for the layer number.
# - proj_group: 1-based group index for the projection name (gate/up/down).
# - cat_group: 1-based group index for the category (weight/scales/biases).
#   Use 0 if the pattern has no separate category (e.g. Mixtral w1/w2/w3,
#   where the matched token is the full weight — category is synthesised
#   as "weight").
# - expert_group_or_None: None if expert index is a tensor dimension (Qwen-style
#   3D tensors where shape[0] == num_experts); an int group index if the expert
#   index appears directly in the safetensors key (Mixtral/DeepSeek per-expert keys).
# - model_type_tag: human-readable architecture tag stored in the sidecar header.
_EXPERT_KEY_PARSERS: list[tuple[str, int, int, int, int | None, str]] = [
    # Qwen3.5-MoE / Qwen2-MoE: switch_mlp layout.
    # Keys: language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight
    # Expert index is shape[0] of the 3D tensor, NOT in the key.
    # Groups: 1=layer, 2=proj(gate/up/down), 3=cat(weight/scales/biases)
    (
        r".*\.layers\.(\d+)\.mlp\.switch_mlp\.(gate|up|down)_proj\.(weight|scales|biases)$",
        1, 2, 3, None, "qwen_moe",
    ),
    # Mixtral / Mistral-MoE: block_sparse_moe.experts.{idx}
    # Keys: model.layers.0.block_sparse_moe.experts.0.w1
    # Expert index IS in the key (group 2).  w1/w2/w3 are full weights with
    # no separate scales/biases — cat is synthesised as "weight" (cat_group=0).
    # Groups: 1=layer, 2=expert, 3=proj(w1/w2/w3)
    (
        r".*\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)$",
        1, 3, 0, 2, "mixtral",
    ),
    # DeepSeek-V2/V3/V4 MoE: mlp.experts.{idx}
    # Keys: model.layers.0.mlp.experts.0.gate_proj.weight
    # Expert index IS in the key (group 2).
    # Groups: 1=layer, 2=expert, 3=proj, 4=cat
    (
        r".*\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$",
        1, 3, 4, 2, "deepseek_moe",
    ),
]

# Map projection short names (w1/w2/w3 from Mixtral) to canonical names.
_MIXTRAL_PROJ_MAP: dict[str, str] = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _get_dtype_bytes(dtype: str) -> int:
    """Return the byte width for a safetensors dtype string."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported safetensors dtype: {dtype!r}")
    return _DTYPE_BYTES[dtype]


def _align_up(offset: int, alignment: int = _DEFAULT_ALIGNMENT) -> int:
    """Round *offset* up to the next multiple of *alignment* (default 16 KiB)."""
    return ((offset + alignment - 1) // alignment) * alignment


def _parse_safetensors_header(filepath: Path) -> tuple[int, dict[str, Any]]:
    """Parse the JSON header from a safetensors file.

    Returns:
        (data_section_offset, header_dict) where *data_section_offset* is the
        absolute byte offset where tensor data begins (``8 + header_len``) and
        *header_dict* is the decoded JSON header.
    """
    with open(filepath, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return 0, {}
        header_len = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_len)
    data_offset = 8 + header_len
    header = json.loads(header_json.decode("utf-8"))
    return data_offset, header


def _parse_expert_key(key: str) -> tuple[int, int | None, str, str, str] | None:
    """Try to parse a safetensors key as an expert projection tensor.

    Iterates the multi-architecture ``_EXPERT_KEY_PARSERS`` registry.
    Supports Qwen (switch_mlp), Mixtral (block_sparse_moe), and DeepSeek
    (mlp.experts) key layouts.

    Returns:
        ``(layer_idx, expert_idx_or_None, projection, category, model_type_tag)``
        or ``None`` if the key does not match any known MoE layout.

        ``expert_idx_or_None`` is ``None`` for Qwen-style (expert index is a
        tensor dimension) and an ``int`` for Mixtral/DeepSeek-style (expert
        index appears in the safetensors key).
    """
    for regex_str, layer_g, proj_g, cat_g, expert_g, tag in _EXPERT_KEY_PARSERS:
        m = re.match(regex_str, key)
        if m is None:
            continue
        layer_idx = int(m.group(layer_g))
        proj_raw = m.group(proj_g)
        expert_idx = int(m.group(expert_g)) if expert_g is not None else None

        # Resolve category: if cat_g is 0 (Mixtral), synthesize "weight"
        # since w1/w2/w3 are full weight tensors with no separate
        # scales/biases in the key.
        if cat_g == 0:
            cat_raw = "weight"
        else:
            cat_raw = m.group(cat_g)

        # Normalize projection names to canonical form.
        if proj_raw in _MIXTRAL_PROJ_MAP:
            proj_raw = _MIXTRAL_PROJ_MAP[proj_raw]
        elif not proj_raw.endswith("_proj"):
            proj_raw = proj_raw + "_proj"

        return layer_idx, expert_idx, proj_raw, cat_raw, tag

    return None


def _is_expert_key(key: str) -> bool:
    """Return True if *key* matches any known MoE expert tensor pattern."""
    return _parse_expert_key(key) is not None


def _layer_key(key: str, layer_idx: int) -> str:
    """Determine the layer key string for a safetensors key.

    Backbone layers (containing ``model.layers``) use the plain integer as the
    key string (e.g. ``"0"``). MTP layers (containing ``mtp.layers``) use
    ``"mtp_{idx}"`` (e.g. ``"mtp_0"``).
    """
    if "mtp" in key.lower():
        return f"mtp_{layer_idx}"
    return str(layer_idx)


# ---------------------------------------------------------------------------
# Tensor discovery
# ---------------------------------------------------------------------------

# Internal type for a discovered tensor record.
# (filepath, data_section_offset, start_offset, end_offset, shape, dtype)
_TensorRecord = tuple[Path, int, int, int, tuple[int, ...], str]

# Composite key: (layer_idx, proj, cat, expert_idx_or_None).
# expert_idx_or_None is None for Qwen-style (dimension-based experts) and an
# int for Mixtral/DeepSeek-style (key-based experts).
_ExpertKey = tuple[int, str, str, int | None]


def _discover_expert_tensors(
    model_path: Path,
) -> tuple[
    dict[str, dict[_ExpertKey, _TensorRecord]],
    str,  # detected model_type tag
]:
    """Scan all safetensors files and index every expert projection tensor.

    Detects the model architecture from the first matched parser in
    ``_EXPERT_KEY_PARSERS``.

    Returns:
        ``(per_file_index, detected_model_type)`` where *per_file_index* maps
        safetensors file path → dict of ``(layer_idx, proj, cat, expert_or_None)``
        → tensor record.
    """
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    result: dict[str, dict[_ExpertKey, _TensorRecord]] = {}
    detected_model_type: str | None = None

    for sf_path in st_files:
        data_offset, header = _parse_safetensors_header(sf_path)
        file_tensors: dict[_ExpertKey, _TensorRecord] = {}

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            parsed = _parse_expert_key(key)
            if parsed is None:
                continue

            layer_idx, expert_idx, projection, category, tag = parsed
            if detected_model_type is None:
                detected_model_type = tag

            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            start_off, end_off = meta["data_offsets"]

            file_tensors[(layer_idx, projection, category, expert_idx)] = (
                sf_path,
                data_offset,
                start_off,
                end_off,
                shape,
                dtype,
            )

        if file_tensors:
            result[str(sf_path)] = file_tensors
            logger.debug(
                "Found %d expert tensors in %s", len(file_tensors), sf_path.name
            )

    if not result:
        raise ValueError(
            f"No expert projection tensors found in {model_path}. "
            f"Expected MoE keys matching switch_mlp, block_sparse_moe.experts, "
            f"or mlp.experts patterns (Qwen/Mixtral/DeepSeek layouts)."
        )

    return result, detected_model_type or "unknown_moe"


# ---------------------------------------------------------------------------
# Sidecar writing helpers
# ---------------------------------------------------------------------------


def _build_layer_metadata(
    file_index: dict[_ExpertKey, _TensorRecord],
    layer_idx: int,
    is_dimension_based: bool,
) -> dict[str, Any]:
    """Build the per-layer metadata entry for the sidecar JSON header.

    Reads shapes and dtypes from the tensor records. Computes per-expert
    byte lengths for each projection/category.  Handles both Qwen-style
    (3D tensors, expert is a dimension) and Mixtral/DeepSeek-style (2D
    tensors, expert is in the key).

    Parameters
    ----------
    file_index:
        Combined expert tensor index keyed by (layer, proj, cat, expert_or_None).
    layer_idx:
        Numeric layer index to build metadata for.
    is_dimension_based:
        True for Qwen-style (shape[0] == num_experts), False for key-based
        architectures (tensor shape is already per-expert 2D).
    """
    import numpy as np

    projections: dict[str, dict[str, Any]] = {}
    total_expert_bytes = 0

    for proj in _PROJECTION_ORDER:
        proj_meta: dict[str, Any] = {}
        proj_total = 0

        for cat in _CATEGORY_ORDER:
            # For dimension-based (Qwen), look up with expert=None.
            # For key-based, any expert works; use 0 as reference.
            lookup_expert: int | None = None if is_dimension_based else 0
            key = (layer_idx, proj, cat, lookup_expert)

            # For key-based archs the expert may be numbered differently;
            # try the first available if lookup_expert=0 isn't found.
            if key not in file_index and not is_dimension_based:
                for (li, p, c, ei), _ in file_index.items():
                    if li == layer_idx and p == proj and c == cat:
                        key = (li, p, c, ei)
                        break

            if key not in file_index:
                # Missing category (e.g. no biases in unquantized models).
                proj_meta[f"{cat}_shape"] = []
                proj_meta[f"{cat}_dtype"] = ""
                proj_meta[f"{cat}_bytes"] = 0
                continue

            _, _, _, _, shape, dtype = file_index[key]
            bpe = _DTYPE_BYTES.get(dtype, 1)

            # Per-expert shape: for dimension-based, drop the first (expert)
            # dimension from the 3D tensor.  For key-based, the tensor is
            # already 2D per-expert.
            if is_dimension_based:
                per_expert_shape = tuple(int(s) for s in shape[1:])
            else:
                per_expert_shape = tuple(int(s) for s in shape)

            per_expert_bytes = int(np.prod(per_expert_shape) * bpe)

            proj_meta[f"{cat}_shape"] = list(per_expert_shape)
            proj_meta[f"{cat}_dtype"] = dtype
            proj_meta[f"{cat}_bytes"] = per_expert_bytes
            proj_total += per_expert_bytes

        projections[proj] = proj_meta
        total_expert_bytes += proj_total

    return {"projections": projections, "_expert_total_bytes": total_expert_bytes}


def _read_tensor_slice(
    fd: int,
    data_section_offset: int,
    start_offset: int,
    expert_idx: int,
    per_expert_bytes: int,
) -> bytes:
    """Read the raw bytes for one expert from a safetensors file descriptor."""
    offset = data_section_offset + start_offset + expert_idx * per_expert_bytes
    return os.pread(fd, per_expert_bytes, offset)


def _compute_per_expert_bytes(
    shape: tuple[int, ...],
    dtype: str,
    *,
    is_dimension_based: bool = True,
) -> int:
    """Compute the number of bytes for a single expert's slice of a tensor.

    For dimension-based layouts (Qwen), drops ``shape[0]`` (the expert dim).
    For key-based layouts (Mixtral/DeepSeek), the shape is already per-expert.
    """
    import numpy as np

    bpe = _DTYPE_BYTES.get(dtype, 1)
    if is_dimension_based:
        per_expert_shape = shape[1:]
    else:
        per_expert_shape = shape
    return int(np.prod(per_expert_shape) * bpe)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StreamingExpertSidecar:
    """Read-only access to a streaming expert sidecar file via Direct I/O.

    Provides single ``os.pread()`` access to 16KiB-aligned expert chunks
    containing all projection weights (gate, up, down) for a single expert
    in a single contiguous byte range.  The file descriptor is opened with
    ``F_NOCACHE`` to bypass the macOS Unified Buffer Cache, preventing UBC
    bloat during sustained streaming of 100+ GB models.

    Parameters
    ----------
    path:
        Path to an existing ``.sidecar`` file created by :meth:`create`.

    Attributes
    ----------
    header : dict
        The parsed JSON header containing layer/projection metadata.
    nocache_active : bool
        True if ``F_NOCACHE`` Direct I/O is active on the underlying file
        descriptor (read-only).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Sidecar file not found: {self._path}")

        self._fd = os.open(str(self._path), os.O_RDONLY)

        # Enable Direct I/O: bypass the Unified Buffer Cache so that data
        # flows directly from NVMe into the process's pre-allocated GPU
        # buffers, never polluting the UBC.
        try:
            fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
            self._nocache_active = True
        except OSError:
            # F_NOCACHE may not be supported on all volumes / macOS versions.
            # Fall back to buffered I/O — still correct, just not UBC-safe.
            self._nocache_active = False
            logger.debug("F_NOCACHE not available; using buffered I/O.")

        try:
            # Read header length (4 bytes, little-endian uint32).
            raw_len = os.pread(self._fd, 4, 0)
            if len(raw_len) < 4:
                raise ValueError("Sidecar file too short: cannot read header length.")
            header_json_len = struct.unpack("<I", raw_len)[0]

            # Read and parse the JSON header.
            header_bytes = os.pread(self._fd, header_json_len, 4)
            self.header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))

            # Build (layer, expert) → (offset, length) lookup tables.
            self._offsets: dict[tuple[str, int], int] = {}
            self._lengths: dict[tuple[str, int], int] = {}

            layers = self.header.get("layers", {})
            for layer_key, layer_data in layers.items():
                experts = layer_data.get("experts", {})
                for expert_key, expert_data in experts.items():
                    expert_idx = int(expert_key)
                    self._offsets[(layer_key, expert_idx)] = expert_data["offset"]
                    self._lengths[(layer_key, expert_idx)] = expert_data["length"]
        except Exception:
            os.close(self._fd)
            raise

        nocache_str = "active" if self._nocache_active else "fallback (buffered)"
        logger.debug(
            "Opened sidecar %s (F_NOCACHE %s): %d layers, %d experts tracked.",
            self._path.name,
            nocache_str,
            len(self.header.get("layers", {})),
            len(self._offsets),
        )

    @property
    def nocache_active(self) -> bool:
        """True if ``F_NOCACHE`` Direct I/O is active on this sidecar.

        When True, ``os.pread()`` calls bypass the macOS Unified Buffer Cache,
        streaming data directly from NVMe SSD into user-space buffers without
        polluting the UBC.  Essential for 100+ GB MoE models where UBC
        pressure would otherwise cause memory compression and swap thrashing.
        """
        return self._nocache_active

    def _resolve_offset_len(self, layer: int | str, expert: int) -> tuple[int, int]:
        """Resolve (layer, expert) to (byte_offset, byte_length) in sidecar."""
        layer_key = str(layer)
        key = (layer_key, expert)
        if key not in self._offsets:
            available = sorted(
                ei for (lk, ei) in self._offsets if lk == layer_key
            )
            if not available:
                raise KeyError(
                    f"Layer {layer_key!r} not found in sidecar. "
                    f"Available layers: {sorted(set(k[0] for k in self._offsets))}"
                )
            raise KeyError(
                f"Expert {expert} not found in layer {layer_key!r}. "
                f"Available experts: {available}"
            )
        return self._offsets[key], self._lengths[key]

    def read_expert(self, layer: int | str, expert: int) -> bytes:
        """Read the full expert chunk and return it as a new ``bytes`` object.

        This allocates a Python bytes object and is intended for tooling,
        validation, and debugging — not the hot inference path.  For the
        zero-copy hot path use :meth:`read_expert_into` with a pre-allocated
        Metal-compatible buffer.

        Data flows directly from NVMe SSD into the returned buffer (when
        ``F_NOCACHE`` is active), bypassing the Unified Buffer Cache.
        """
        offset, length = self._resolve_offset_len(layer, expert)
        return os.pread(self._fd, length, offset)

    def read_expert_into(
        self,
        layer: int | str,
        expert: int,
        buf: memoryview,
        buf_offset: int = 0,
    ) -> int:
        """Read expert bytes directly into a pre-allocated buffer.

        Zero-copy path for the hot inference loop.  Uses ``os.preadv()``
        (Python 3.12+) for scatter-gather I/O into the memoryview without
        a Python ``bytes`` intermediate.  Falls back to ``os.pread()`` +
        slice assignment on older Python.

        The buffer must be backed by a page-aligned allocation (16 KiB on
        Apple Silicon) if the result will be wrapped into a Metal buffer
        via ``newBufferWithBytesNoCopy``.

        Parameters
        ----------
        layer:
            Layer identifier. Integer for backbone layers, string like
            ``"mtp_0"`` for MTP layers.
        expert:
            Expert index within the layer.
        buf:
            Pre-allocated writable buffer supporting the buffer protocol
            (e.g. ``memoryview`` of a ``bytearray`` or an ``mx.array``
            memoryview obtained via ``ctypes``).
        buf_offset:
            Byte offset within *buf* to begin writing at.

        Returns
        -------
        int
            Number of bytes read (always equals the expert length from the
            sidecar header on success).
        """
        offset, length = self._resolve_offset_len(layer, expert)

        # Use the pread_into_array function which handles all buffer types
        # (mx.array, bytearray, memoryview, numpy.ndarray) via the Python
        # buffer protocol.  The C++ extension writes directly into the
        # backing buffer with zero intermediate allocations; the pure Python
        # fallback allocates a temporary bytes object per call.
        from . import pread_into_array as _pread_into

        nread = _pread_into(self._fd, offset, length, buf, buf_offset)

        if nread != length:
            raise RuntimeError(
                f"Short read on sidecar {self._path.name}: "
                f"layer={layer} expert={expert}: "
                f"expected {length} bytes, got {nread}"
            )
        return nread

    def verify(self, sample_layers: int = 3) -> bool:
        """Validate sidecar integrity by spot-checking experts from random layers.

        Reads expert 0 and the last expert from *sample_layers* randomly
        selected layers.  Confirms each read returns exactly the expected
        length from the header.  This is a quick integrity check — it does
        not verify byte-content correctness (that requires a model output
        comparison).

        Parameters
        ----------
        sample_layers:
            Maximum number of layers to spot-check (default 3).

        Returns
        -------
        bool
            ``True`` if all spot-checks pass, ``False`` if any read returned
            an unexpected length.
        """
        import random

        layers = list({lk for (lk, _) in self._offsets})
        if not layers:
            logger.warning("Sidecar has no layer entries to verify.")
            return True

        sample_count = min(sample_layers, len(layers))
        chosen = random.sample(layers, sample_count)

        checks_passed = 0
        for lk in chosen:
            # Find max expert index for this layer.
            experts_in_layer = sorted(ei for (ll, ei) in self._offsets if ll == lk)
            if not experts_in_layer:
                logger.warning("Layer %r has no expert entries; skipping.", lk)
                continue

            for test_expert in (experts_in_layer[0], experts_in_layer[-1]):
                expected_len = self._lengths.get((lk, test_expert))
                if expected_len is None:
                    logger.warning(
                        "Layer %r expert %d: missing length in lookup table.",
                        lk, test_expert,
                    )
                    return False

                try:
                    raw = self.read_expert(lk, test_expert)
                except OSError as exc:
                    logger.warning(
                        "Verification I/O error: layer %r expert %d: %s",
                        lk, test_expert, exc,
                    )
                    return False

                if len(raw) != expected_len:
                    logger.warning(
                        "Verification mismatch: layer %r expert %d: "
                        "expected %d bytes, got %d bytes.",
                        lk, test_expert, expected_len, len(raw),
                    )
                    return False
                checks_passed += 1

        logger.info(
            "Sidecar verification passed: %d spot-checks across %d layers OK.",
            checks_passed, sample_count,
        )
        return True

    def close(self) -> None:
        """Close the underlying file descriptor."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "StreamingExpertSidecar":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # create() – static factory that builds a sidecar from safetensors
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        model_path: str | Path,
        output_path: str | Path,
        alignment: int = _DEFAULT_ALIGNMENT,
        quant: int | None = None,
        quant_group_size: int = 64,
    ) -> "StreamingExpertSidecar":
        """Scan safetensors in *model_path*, extract expert weights, and write
        a contiguous sidecar file to *output_path*.

        Expert chunks are padded to *alignment* bytes (default 16 KiB to match
        Apple Silicon's virtual memory page size for zero-copy Metal buffer
        wrapping).

        Supports Qwen (switch_mlp), Mixtral (block_sparse_moe), and DeepSeek
        (mlp.experts) MoE key layouts.  The detected architecture is stored as
        ``model_type`` in the sidecar header.

        Parameters
        ----------
        model_path:
            Directory containing ``*.safetensors`` files with MoE expert weights.
        output_path:
            Destination path for the generated sidecar file.
        alignment:
            Byte alignment for expert chunks (default 16384 for Apple Silicon
            16 KiB page alignment).
        quant:
            Quantize expert weights to this many bits (e.g. 4 for 4-bit).
            When ``None`` (default), weights are stored as float32.
            When set, weights are packed in MLX's quantised layout so the
            streaming pipeline can dispatch ``mx.gather_qmm`` for fused
            dequant + gather + matmul.
        quant_group_size:
            Group size for quantisation (default 64, used when *quant* is set).

        Returns
        -------
        StreamingExpertSidecar
            A read-only sidecar instance opened on the newly created file.
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        logger.info(
            "Creating streaming expert sidecar from %s → %s", model_path, output_path
        )

        # ── Phase 1: Discover expert tensors ────────────────────────────
        per_file, detected_model_type = _discover_expert_tensors(model_path)
        logger.info("Detected MoE architecture: %s", detected_model_type)

        # Determine whether this is dimension-based (Qwen: expert in shape[0])
        # or key-based (Mixtral/DeepSeek: expert in safetensors key).
        is_dimension_based = detected_model_type in ("qwen_moe",)

        # Flatten into a combined index.
        combined: dict[_ExpertKey, _TensorRecord] = {}
        layer_key_map: dict[int, str] = {}

        for sf_path_str, file_tensors in per_file.items():
            for (layer_idx, proj, cat, expert_idx), rec in file_tensors.items():
                key = (layer_idx, proj, cat, expert_idx)
                if key in combined:
                    logger.warning(
                        "Duplicate tensor key layer=%d proj=%s cat=%s expert=%s; using last.",
                        layer_idx, proj, cat, expert_idx,
                    )
                combined[key] = rec

        # Determine layer keys by re-scanning headers to find original
        # safetensors keys for mtp vs backbone detection.
        for sf_path_str in per_file:
            sf_path = Path(sf_path_str)
            _, header = _parse_safetensors_header(sf_path)
            for key in header:
                if key == "__metadata__":
                    continue
                parsed = _parse_expert_key(key)
                if parsed is None:
                    continue
                layer_idx = parsed[0]
                if layer_idx not in layer_key_map:
                    layer_key_map[layer_idx] = _layer_key(key, layer_idx)

        sorted_layers = sorted(layer_key_map.keys())
        if not sorted_layers:
            raise ValueError("No expert layers discovered.")

        # Determine num_experts per layer.
        first_layer = sorted_layers[0]
        num_experts: dict[int, int] = {}
        if is_dimension_based:
            # Qwen-style: num_experts = shape[0] of any weight tensor.
            for li in sorted_layers:
                for (li2, proj, cat, ei), rec in combined.items():
                    if li2 == li and cat == "weight" and ei is None:
                        num_experts[li] = rec[4][0]
                        break
                if li not in num_experts:
                    num_experts[li] = 0
        else:
            # Key-based: count distinct expert indices.
            for li in sorted_layers:
                experts_in_layer: set[int] = set()
                for (li2, _, _, ei), _ in combined.items():
                    if li2 == li and ei is not None:
                        experts_in_layer.add(ei)
                num_experts[li] = len(experts_in_layer)

        # Determine hidden_size from the first layer's first available tensor.
        hidden_size = 0
        for li in sorted_layers:
            for (li2, proj, cat, ei), rec in combined.items():
                if li2 == li and cat == "weight":
                    if is_dimension_based:
                        hidden_size = int(min(rec[4][1], rec[4][2]))
                    else:
                        hidden_size = int(min(rec[4][0], rec[4][1]))
                    break
            if hidden_size > 0:
                break

        # ── Phase 2: Build per-layer metadata ───────────────────────────
        layers_header: dict[str, Any] = {}

        for layer_idx in sorted_layers:
            lk = layer_key_map[layer_idx]
            meta = _build_layer_metadata(combined, layer_idx, is_dimension_based)
            layers_header[lk] = {
                "projections": meta["projections"],
                "experts": {},  # filled in after offset computation
            }

        # ── Phase 3: Bounded single-pass header offset computation ─────
        # Build a placeholder header with MAX_LEN values to determine the
        # true upper bound for the JSON header size.  This avoids the
        # two-pass delta-shift retry that could theoretically loop.
        placeholder_experts = {"0": {"offset": 9999999999, "length": 9999999}}
        placeholder_layers = {
            lk: {
                "projections": layers_header[lk]["projections"],
                "experts": placeholder_experts,
            }
            for lk in (layer_key_map[li] for li in sorted_layers)
        }
        placeholder_header: dict[str, Any] = {
            "version": 1,
            "model_type": detected_model_type,
            "num_layers": len(sorted_layers),
            "num_experts": num_experts.get(first_layer, 0),
            "hidden_size": hidden_size,
            "alignment": alignment,
            "layers": placeholder_layers,
        }
        placeholder_json = json.dumps(placeholder_header, separators=(",", ":"))
        # A 4-byte-lengths header grows by at most a few bytes per expert
        # entry when switching from "9999999999" to real offsets.  64 bytes
        # of slack against a 16 KiB alignment is always sufficient.
        header_upper_bound = len(placeholder_json) + 64
        first_data_offset = _align_up(4 + header_upper_bound, alignment)

        # ── Phase 4: Extract expert bytes from safetensors ──────────────
        logger.info(
            "Extracting expert weights from %d safetensors files ...", len(per_file)
        )

        # Open all safetensors file descriptors for pread.
        st_fds: dict[str, int] = {}
        for sf_path_str in per_file:
            st_fds[sf_path_str] = os.open(sf_path_str, os.O_RDONLY)

        # (layer_key, expert_idx, offset, length, chunk_bytes)
        expert_layout: list[tuple[str, int, int, int, bytes]] = []

        try:
            current_offset = first_data_offset
            total_experts = 0

            for layer_idx in sorted_layers:
                lk = layer_key_map[layer_idx]
                n_exp = num_experts[layer_idx]
                lk_progress = 0

                if is_dimension_based:
                    # Qwen: iterate expert indices 0..n_exp-1, slice from 3D tensors.
                    for expert_idx in range(n_exp):
                        chunk, total_bytes = _build_expert_chunk_dimension_based(
                            st_fds, combined, layer_idx, expert_idx, alignment,
                        )
                        expert_layout.append(
                            (lk, expert_idx, current_offset, total_bytes, chunk)
                        )
                        padded_len = _align_up(total_bytes, alignment)
                        current_offset += padded_len
                        total_experts += 1
                        lk_progress += 1
                else:
                    # Mixtral/DeepSeek: enumerate distinct expert indices from keys.
                    layer_experts = sorted(
                        ei for (li, _, _, ei), _ in combined.items()
                        if li == layer_idx and ei is not None
                    )
                    # Deduplicate.
                    layer_experts = sorted(set(layer_experts))
                    for expert_idx in layer_experts:
                        chunk, total_bytes = _build_expert_chunk_key_based(
                            st_fds, combined, layer_idx, expert_idx, alignment,
                        )
                        expert_layout.append(
                            (lk, expert_idx, current_offset, total_bytes, chunk)
                        )
                        padded_len = _align_up(total_bytes, alignment)
                        current_offset += padded_len
                        total_experts += 1
                        lk_progress += 1

                logger.debug(
                    "Layer %s: %d experts extracted.",
                    lk, lk_progress,
                )

            # ── Phase 4b: Quantisation post-processing (optional) ────────
            if quant is not None and quant > 0:
                import mlx.core as _mx
                import numpy as _np

                logger.info(
                    "Quantising expert weights to %d-bit (group_size=%d) ...",
                    quant, quant_group_size,
                )

                # Determine segment layout from first layer's metadata.
                # For Qwen-style (dimension-based): each expert stores
                # gate(M,H) + up(M,H) + down(H,M) as contiguous float32.
                # For key-based: each projection is separate but still
                # packed contiguously in the chunk.
                #
                # We read the actual weight shapes from the combined dict
                # to get exact segment sizes.  Fallback: 3 equal parts.
                #
                # For each projection we:
                #   1. Convert raw float32 bytes → mx.array
                #   2. Call mx.quantize(weight, group_size, bits)
                #   3. Pack qw + scales + biases back as bytes
                #   4. Update the chunk and length

                for i, (lk, ei, _offset, _length, chunk) in enumerate(
                    expert_layout
                ):
                    # Build a list of (proj_name, shape) from the
                    # combined dict for this (layer, expert).
                    proj_shapes: list[tuple[str, int, int]] = []
                    for (
                        (_li, pn, _cat, _ei),
                        rec,
                    ) in combined.items():
                        # Match by layer key string
                        lk_from_idx = layer_key_map.get(_li, "")
                        if lk_from_idx == lk and _cat == "weight":
                            shape = rec[4]
                            if len(shape) >= 2:
                                if is_dimension_based:
                                    # shape = (e, M, H) — take dims 1,2
                                    proj_shapes.append(
                                        (pn, int(shape[1]), int(shape[2]))
                                    )
                                else:
                                    # shape = (M, H) or (H, M)
                                    proj_shapes.append(
                                        (pn, int(shape[0]), int(shape[1]))
                                    )

                    if not proj_shapes:
                        logger.debug(
                            "Layer %s expert %d: no projections found, "
                            "skipping quant",
                            lk, ei,
                        )
                        continue

                    # Sort: gate, up, down (standard order)
                    order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
                    proj_shapes.sort(key=lambda x: order.get(x[0], 99))

                    packed_chunks: list[bytes] = []
                    byte_offset = 0
                    for pn, rows, cols in proj_shapes:
                        seg_size = rows * cols * 4  # float32 bytes
                        raw = chunk[byte_offset : byte_offset + seg_size]
                        byte_offset += seg_size

                        # Reshape raw bytes → float32 mx.array
                        # Use numpy for byte reinterpretation since
                        # MLX lacks ``frombuffer``.
                        w_np = _np.frombuffer(raw[:seg_size], dtype=_np.float32).reshape(rows, cols)
                        w = _mx.array(w_np)

                        qw, scales, biases = _mx.quantize(
                            w, group_size=quant_group_size, bits=quant
                        )

                        # Convert to numpy and reinterpret as uint8 bytes.
                        # Note: ``.astype(mx.uint8)`` does element-wise
                        # conversion (wrong for uint32→uint8); we use
                        # numpy's ``.view(np.uint8)`` for true byte
                        # reinterpretation.
                        qw_np = _np.asarray(_mx.array(qw))
                        sc_np = _np.asarray(_mx.array(scales))
                        bi_np = _np.asarray(_mx.array(biases))
                        packed_chunks.append(
                            qw_np.view(_np.uint8).tobytes()
                            + sc_np.view(_np.uint8).tobytes()
                            + bi_np.view(_np.uint8).tobytes()
                        )

                    repacked = b"".join(packed_chunks)
                    expert_layout[i] = (lk, ei, _offset, len(repacked), repacked)

            # Recompute offsets and add alignment padding for quantized
            # chunks — the original offsets/padding were based on float32
            # sizes, and qw+sc+bi have different byte counts.
            if quant is not None and quant > 0:
                new_offset = expert_layout[0][2] if expert_layout else 0
                for i, (lk, ei, _old_off, new_len, chunk) in enumerate(
                    expert_layout
                ):
                    padded = _align_up(new_len, alignment)
                    padding_needed = padded - new_len
                    padded_chunk = chunk + (b"\x00" * padding_needed) if padding_needed > 0 else chunk
                    expert_layout[i] = (lk, ei, new_offset, new_len, padded_chunk)
                    new_offset += padded

            # ── Phase 5: Finalize header with real expert offsets ───────
            header_dict: dict[str, Any] = {
                "version": 1,
                "model_type": detected_model_type,
                "num_layers": len(sorted_layers),
                "num_experts": num_experts.get(first_layer, 0),
                "hidden_size": hidden_size,
                "alignment": alignment,
                "layers": {
                    lk: {
                        "projections": layers_header[lk]["projections"],
                        "experts": {},
                    }
                    for lk in (layer_key_map[li] for li in sorted_layers)
                },
            }
            # Add quantisation metadata to header if active.
            if quant is not None and quant > 0:
                _qm: dict[str, Any] = {
                    "bits": quant,
                    "group_size": quant_group_size,
                    "mode": "affine",
                }
                for lk in header_dict["layers"]:
                    header_dict["layers"][lk]["quant"] = dict(_qm)

            for lk, ei, offset, length, _chunk in expert_layout:
                header_dict["layers"][lk]["experts"][str(ei)] = {
                    "offset": offset,
                    "length": length,
                }

            # Serialize once — the upper-bound slack guarantees no overflow.
            header_json = json.dumps(header_dict, separators=(",", ":"))
            header_bytes = header_json.encode("utf-8")
            header_len_prefix = struct.pack("<I", len(header_bytes))
            actual_first = _align_up(4 + len(header_bytes), alignment)
            assert actual_first <= first_data_offset, (
                f"Header upper-bound slack insufficient: "
                f"actual={actual_first} > planned={first_data_offset}"
            )

            # ── Phase 6: Write the sidecar file ─────────────────────────
            logger.info(
                "Writing sidecar with %d experts (%d layers) to %s ...",
                total_experts, len(sorted_layers), output_path,
            )

            with open(output_path, "wb") as out:
                out.write(header_len_prefix)
                out.write(header_bytes)
                # Pad to planned first data offset.
                header_end = 4 + len(header_bytes)
                pad_needed = first_data_offset - header_end
                if pad_needed > 0:
                    out.write(b"\x00" * pad_needed)
                # Write expert chunks.
                for _lk, _ei, _offset, _length, chunk in expert_layout:
                    out.write(chunk)

            file_size = output_path.stat().st_size
            logger.info(
                "Sidecar written: %s (%.1f MB, %d experts)",
                output_path.name, file_size / (1024 * 1024), total_experts,
            )

        finally:
            for fd in st_fds.values():
                os.close(fd)

        # ── Phase 7: Validate and return ────────────────────────────────
        sidecar = StreamingExpertSidecar(output_path)
        if not sidecar.verify(sample_layers=3):
            sidecar.close()
            raise RuntimeError(
                f"Sidecar verification failed for {output_path}. "
                f"The file may be corrupt or truncated."
            )
        return sidecar


def _build_expert_chunk_dimension_based(
    st_fds: dict[str, int],
    combined: dict[_ExpertKey, _TensorRecord],
    layer_idx: int,
    expert_idx: int,
    alignment: int,
) -> tuple[bytes, int]:
    """Build a single expert chunk for Qwen-style (dimension-based) layouts.

    Expert index comes from the tensor dimension (shape[0]).
    """
    chunk_parts: list[bytes] = []
    total_bytes = 0

    for proj in _PROJECTION_ORDER:
        for cat in _CATEGORY_ORDER:
            key = (layer_idx, proj, cat, None)
            if key not in combined:
                continue
            sf_path, data_off, start_off, _end_off, shape, dtype = combined[key]
            per_expert_bytes = _compute_per_expert_bytes(shape, dtype, is_dimension_based=True)

            sf_fd = st_fds[str(sf_path)]
            raw = _read_tensor_slice(sf_fd, data_off, start_off, expert_idx, per_expert_bytes)
            chunk_parts.append(raw)
            total_bytes += len(raw)

    padded_len = _align_up(total_bytes, alignment)
    padding = b"\x00" * (padded_len - total_bytes)
    return b"".join(chunk_parts) + padding, total_bytes


def _build_expert_chunk_key_based(
    st_fds: dict[str, int],
    combined: dict[_ExpertKey, _TensorRecord],
    layer_idx: int,
    expert_idx: int,
    alignment: int,
) -> tuple[bytes, int]:
    """Build a single expert chunk for Mixtral/DeepSeek (key-based) layouts.

    Expert index comes from the safetensors key, not a tensor dimension.
    """
    chunk_parts: list[bytes] = []
    total_bytes = 0

    for proj in _PROJECTION_ORDER:
        for cat in _CATEGORY_ORDER:
            key = (layer_idx, proj, cat, expert_idx)
            if key not in combined:
                continue
            sf_path, data_off, start_off, _end_off, shape, dtype = combined[key]
            per_expert_bytes = _compute_per_expert_bytes(shape, dtype, is_dimension_based=False)

            sf_fd = st_fds[str(sf_path)]
            # For key-based, the entire tensor is a single expert — no slicing.
            raw = os.pread(sf_fd, per_expert_bytes, data_off + start_off)
            chunk_parts.append(raw)
            total_bytes += len(raw)

    padded_len = _align_up(total_bytes, alignment)
    padding = b"\x00" * (padded_len - total_bytes)
    return b"".join(chunk_parts) + padding, total_bytes


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def create_sidecar(
    model_path: str | Path,
    output_path: str | Path,
    alignment: int = _DEFAULT_ALIGNMENT,
    quant: int | None = None,
    quant_group_size: int = 64,
) -> StreamingExpertSidecar:
    """Convenience wrapper for :meth:`StreamingExpertSidecar.create`."""
    return StreamingExpertSidecar.create(
        model_path, output_path, alignment,
        quant=quant, quant_group_size=quant_group_size,
    )
