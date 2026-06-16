# SPDX-License-Identifier: Apache-2.0
"""StreamingExpertSidecar: SSD-backed expert weight streaming for MoE models.

Stores expert weights from safetensors in a single flat binary file with
4KB-aligned contiguous expert chunks. Enables single ``os.pread()`` per expert
during inference, allowing large MoE models to run on machines with limited
unified memory.
"""

from __future__ import annotations

import ctypes
import json
import logging
import mmap as _mmap
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

# Default alignment for expert chunks (4KB sector boundary).
_DEFAULT_ALIGNMENT = 4096

# madvise(2) constants for UBC control on macOS / BSD.
# From <sys/mman.h>: MADV_WILLNEED = 3, MADV_DONTNEED = 4.
_MADV_WILLNEED: int = 3
_MADV_DONTNEED: int = 4

# Cache the madvise libc function pointer (lazy-init).
_madvise_fn: Any | None = None


def _get_madvise():
    """Return the libc madvise(addr, length, advice) function."""
    global _madvise_fn
    if _madvise_fn is None:
        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        _madvise_fn = libc.madvise
        _madvise_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _madvise_fn.restype = ctypes.c_int
    return _madvise_fn

# Projection order within each expert chunk.
_PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
_CATEGORY_ORDER = ("weight", "scales", "biases")

# Regex components for expert tensor key detection.
# Examples:
#   language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight
#   language_model.mtp.layers.0.mlp.switch_mlp.up_proj.scales
#   mtp.layers.0.mlp.switch_mlp.down_proj.biases
_EXPERT_KEY_RE_STR = (
    r".*\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate|up|down)_proj\.(weight|scales|biases)$"
)

# Key prefixes for tensor discovery – any key containing one of these
# substrings AND ending in .weight/.scales/.biases is considered an expert
# tensor candidate.
_EXPERT_MARKERS = (".switch_mlp.",)


def _get_dtype_bytes(dtype: str) -> int:
    """Return the byte width for a safetensors dtype string."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported safetensors dtype: {dtype!r}")
    return _DTYPE_BYTES[dtype]


def _align_up(offset: int, alignment: int = _DEFAULT_ALIGNMENT) -> int:
    """Round *offset* up to the next multiple of *alignment*."""
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


def _parse_expert_key(key: str) -> tuple[int, str, str] | None:
    """Try to parse a safetensors key as an expert projection tensor.

    Returns:
        ``(layer_idx, projection, category)`` or ``None`` if the key does
        not match the expected expert tensor pattern.
    """
    m = re.match(_EXPERT_KEY_RE_STR, key)
    if m is None:
        return None
    return int(m.group(1)), m.group(2) + "_proj", m.group(3)


def _is_expert_key(key: str) -> bool:
    """Return True if *key* looks like an expert projection tensor."""
    if not any(marker in key for marker in _EXPERT_MARKERS):
        return False
    return key.endswith((".weight", ".scales", ".biases"))


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


def _discover_expert_tensors(
    model_path: Path,
) -> dict[str, dict[tuple[int, str, str], _TensorRecord]]:
    """Scan all safetensors files and index every expert projection tensor.

    Returns a dict keyed by the safetensors file path (as string), mapping to a
    dict of ``(layer_idx, projection, category) -> tensor_record`` for tensors
    contained in that file.
    """
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    result: dict[str, dict[tuple[int, str, str], _TensorRecord]] = {}

    for sf_path in st_files:
        data_offset, header = _parse_safetensors_header(sf_path)
        file_tensors: dict[tuple[int, str, str], _TensorRecord] = {}

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if not _is_expert_key(key):
                continue
            parsed = _parse_expert_key(key)
            if parsed is None:
                continue

            layer_idx, projection, category = parsed
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            start_off, end_off = meta["data_offsets"]

            file_tensors[(layer_idx, projection, category)] = (
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
            f"Expected keys containing 'switch_mlp'."
        )

    return result


# ---------------------------------------------------------------------------
# Sidecar writing
# ---------------------------------------------------------------------------


def _build_layer_metadata(
    file_index: dict[tuple[int, str, str], _TensorRecord],
    layer_idx: int,
) -> dict[str, Any]:
    """Build the per-layer metadata entry for the sidecar JSON header.

    Reads shapes and dtypes from the tensor records. Computes per-expert
    byte lengths for each projection/category.
    """
    import numpy as np

    projections: dict[str, dict[str, Any]] = {}
    total_expert_bytes = 0

    for proj in _PROJECTION_ORDER:
        proj_meta: dict[str, Any] = {}
        proj_total = 0

        for cat in _CATEGORY_ORDER:
            key = (layer_idx, proj, cat)
            if key not in file_index:
                # Missing category (e.g. no biases in unquantized models).
                proj_meta[f"{cat}_shape"] = []
                proj_meta[f"{cat}_dtype"] = ""
                proj_meta[f"{cat}_bytes"] = 0
                continue

            _, _, _, _, shape, dtype = file_index[key]
            bpe = _DTYPE_BYTES.get(dtype, 1)
            # Per-expert shape: drop the first (expert) dimension.
            per_expert_shape = tuple(int(s) for s in shape[1:])
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
    dtype: str,
) -> bytes:
    """Read the raw bytes for one expert from a safetensors file descriptor."""
    offset = data_section_offset + start_offset + expert_idx * per_expert_bytes
    return os.pread(fd, per_expert_bytes, offset)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StreamingExpertSidecar:
    """Read-only access to a streaming expert sidecar file.

    Provides single ``os.pread()`` access to 4KB-aligned expert chunks
    containing all projection weights (gate, up, down) for a single expert
    in a single contiguous byte range.

    Parameters
    ----------
    path:
        Path to an existing ``.sidecar`` file created by :meth:`create`.

    Attributes
    ----------
    header : dict
        The parsed JSON header containing layer/projection metadata.
    """

    def __init__(self, path: str | Path, *, use_mmap: bool = True) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Sidecar file not found: {self._path}")

        self._fd = os.open(str(self._path), os.O_RDWR if use_mmap else os.O_RDONLY)
        self._mm: _mmap.mmap | None = None
        self._mm_ptr: int = 0

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

            # mmap the entire file for zero-copy access + madvise control.
            if use_mmap:
                file_size = self._path.stat().st_size
                self._mm = _mmap.mmap(self._fd, file_size, prot=_mmap.PROT_READ | _mmap.PROT_WRITE)
                # Get the raw pointer to the mmap'd region for madvise.
                self._mm_ptr = ctypes.addressof(
                    (ctypes.c_ubyte * file_size).from_buffer(self._mm)
                )
        except Exception:
            self.close()
            raise

        logger.debug(
            "Opened sidecar %s (mmap=%s): %d layers, %d experts tracked.",
            self._path.name,
            self._mm is not None,
            len(self.header.get("layers", {})),
            len(self._offsets),
        )

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

    def get_ptr(self, layer: int | str, expert: int) -> int:
        """Return the absolute memory address of an expert's data in the mmap.

        Requires ``use_mmap=True`` at init. Raises ``RuntimeError`` if the
        sidecar was opened without mmap.
        """
        if self._mm is None:
            raise RuntimeError(
                "get_ptr requires mmap mode (use_mmap=True at init)."
            )
        offset, _length = self._resolve_offset_len(layer, expert)
        return self._mm_ptr + offset

    def madvise_willneed(self, layer: int | str, expert: int) -> int:
        """Tell the kernel to prefetch an expert's pages into the UBC.

        Calls ``madvise(MADV_WILLNEED)`` on the mmap'd region for the given
        expert.  The kernel begins reading pages from NVMe asynchronously;
        no Python I/O is performed.  Returns 0 on success, -1 on error.

        Requires ``use_mmap=True`` at init.
        """
        if self._mm is None:
            return -1
        offset, length = self._resolve_offset_len(layer, expert)
        fn = _get_madvise()
        return fn(self._mm_ptr + offset, length, _MADV_WILLNEED)

    def madvise_dontneed(self, layer: int | str, expert: int) -> int:
        """Immediately free an expert's physical pages from the UBC.

        Calls ``madvise(MADV_DONTNEED)`` on the mmap'd region.  Because the
        file mapping is read-only and backed by SSD, the kernel can drop the
        pages instantly without write-back.  Returns 0 on success, -1 on error.

        Requires ``use_mmap=True`` at init.
        """
        if self._mm is None:
            return -1
        offset, length = self._resolve_offset_len(layer, expert)
        fn = _get_madvise()
        return fn(self._mm_ptr + offset, length, _MADV_DONTNEED)

    def read_expert(self, layer: int | str, expert: int) -> bytes:
        """Read the full expert chunk via ``os.pread()``.

        Returns the raw bytes containing weight, scales, and biases for all
        projections (gate_proj, up_proj, down_proj) packed contiguously.
        Use ``self.header['layers'][str(layer)]['projections']`` to split
        the byte range into individual arrays.

        This uses OS pread for reliability. For zero-copy access from the
        mmap region, use :meth:`get_ptr`.
        """
        offset, length = self._resolve_offset_len(layer, expert)
        return os.pread(self._fd, length, offset)

    def close(self) -> None:
        """Close the underlying file descriptor and mmap."""
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
            self._mm_ptr = 0

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
    ) -> "StreamingExpertSidecar":
        """Scan safetensors in *model_path*, extract expert weights, and write
        a contiguous 4KB-aligned sidecar file to *output_path*.

        Parameters
        ----------
        model_path:
            Directory containing ``*.safetensors`` files with MoE expert weights.
        output_path:
            Destination path for the generated sidecar file.
        alignment:
            Byte alignment for expert chunks (default 4096 for NVMe sector
            alignment / ``F_NOCACHE`` compatibility).

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
        per_file = _discover_expert_tensors(model_path)

        # Flatten into a combined index keyed by (layer_idx, proj, cat).
        # Also record the original safetensors key so we can reconstruct
        # layer_key strings ("0" vs "mtp_0").
        combined: dict[tuple[int, str, str], _TensorRecord] = {}
        layer_key_map: dict[int, str] = {}

        for sf_path_str, file_tensors in per_file.items():
            for (layer_idx, proj, cat), rec in file_tensors.items():
                if (layer_idx, proj, cat) in combined:
                    logger.warning(
                        "Duplicate tensor key layer=%d proj=%s cat=%s; using last.",
                        layer_idx, proj, cat,
                    )
                combined[(layer_idx, proj, cat)] = rec

        # Determine layer keys and num_experts by re-scanning headers
        # to find the original safetensors keys.
        for sf_path_str in per_file:
            sf_path = Path(sf_path_str)
            _, header = _parse_safetensors_header(sf_path)
            for key in header:
                if key == "__metadata__":
                    continue
                if not _is_expert_key(key):
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

        # Determine num_experts and hidden_size from the first layer.
        first_layer = sorted_layers[0]
        num_experts: dict[int, int] = {}
        for li in sorted_layers:
            for (li2, proj, cat), rec in combined.items():
                if li2 == li and cat == "weight":
                    num_experts[li] = rec[4][0]
                    break
            if li not in num_experts:
                num_experts[li] = 0

        sample_key = (first_layer, "gate_proj", "weight")
        if sample_key not in combined:
            for proj in _PROJECTION_ORDER:
                sample_key = (first_layer, proj, "weight")
                if sample_key in combined:
                    break
        sample_shape = combined[sample_key][4]
        hidden_size = int(min(sample_shape[1], sample_shape[2]))

        # ── Phase 2: Build per-layer metadata ───────────────────────────
        layers_header: dict[str, Any] = {}
        # Expert entries: list of (layer_key, expert_idx, offset, length, chunk_bytes)
        expert_layout: list[tuple[str, int, int, int, bytes]] = []

        for layer_idx in sorted_layers:
            lk = layer_key_map[layer_idx]
            meta = _build_layer_metadata(combined, layer_idx)
            layers_header[lk] = {
                "projections": meta["projections"],
                "experts": {},  # filled in after offset computation
            }

        # ── Phase 3: Compute expert chunk layout ────────────────────────
        # First compute the JSON header size to know where data starts.
        # We'll build the header dict incrementally, serialize it, then
        # compute exact offsets.
        header_dict: dict[str, Any] = {
            "version": 1,
            "model_type": "qwen3_5_moe",
            "num_layers": len(sorted_layers),
            "num_experts": num_experts.get(first_layer, 0),
            "hidden_size": hidden_size,
            "alignment": alignment,
            "layers": {lk: {"projections": layers_header[lk]["projections"], "experts": {}}
                       for lk in (layer_key_map[li] for li in sorted_layers)},
        }

        header_json = json.dumps(header_dict, separators=(",", ":"))
        # Data starts at the first alignment boundary after 4 + len(header_json).
        first_data_offset = _align_up(4 + len(header_json), alignment)

        # Pre-compute all expert chunks (extract bytes from safetensors).
        logger.info("Extracting expert weights from %d safetensors files ...", len(per_file))

        # Open all safetensors file descriptors for pread.
        st_fds: dict[str, int] = {}
        for sf_path_str in per_file:
            st_fds[sf_path_str] = os.open(sf_path_str, os.O_RDONLY)

        try:
            current_offset = first_data_offset
            total_experts = 0

            for layer_idx in sorted_layers:
                lk = layer_key_map[layer_idx]
                n_exp = num_experts[layer_idx]
                lk_progress = 0

                for expert_idx in range(n_exp):
                    chunk_parts: list[bytes] = []
                    total_bytes = 0

                    for proj in _PROJECTION_ORDER:
                        for cat in _CATEGORY_ORDER:
                            key = (layer_idx, proj, cat)
                            if key not in combined:
                                continue
                            sf_path, data_off, start_off, end_off, shape, dtype = combined[key]
                            per_expert_bytes = _compute_per_expert_bytes(shape, dtype)

                            sf_fd = st_fds[str(sf_path)]
                            raw = _read_tensor_slice(
                                sf_fd, data_off, start_off, expert_idx,
                                per_expert_bytes, dtype,
                            )
                            chunk_parts.append(raw)
                            total_bytes += len(raw)

                    # Pad to alignment boundary.
                    padded_len = _align_up(total_bytes, alignment)
                    padding = b"\x00" * (padded_len - total_bytes)
                    chunk = b"".join(chunk_parts) + padding
                    expert_layout.append((lk, expert_idx, current_offset, total_bytes, chunk))
                    current_offset += padded_len
                    total_experts += 1
                    lk_progress += 1

                logger.debug(
                    "Layer %s (%s): %d experts extracted.",
                    lk, layer_key_map[layer_idx], lk_progress,
                )

            # ── Phase 4: Finalize header with expert offsets ────────────
            for lk, ei, offset, length, _chunk in expert_layout:
                header_dict["layers"][lk]["experts"][str(ei)] = {
                    "offset": offset,
                    "length": length,
                }

            # Re-serialize header with final offsets.
            header_json = json.dumps(header_dict, separators=(",", ":"))
            header_bytes = header_json.encode("utf-8")
            header_len_prefix = struct.pack("<I", len(header_bytes))

            # Verify first data offset is still valid.
            computed_first = _align_up(4 + len(header_bytes), alignment)
            if computed_first > first_data_offset:
                # JSON header grew due to expert entries – offsets shifted.
                # Recompute all offsets.
                delta = computed_first - first_data_offset
                for lk in header_dict["layers"]:
                    for ek in header_dict["layers"][lk]["experts"]:
                        header_dict["layers"][lk]["experts"][ek]["offset"] += delta
                header_json = json.dumps(header_dict, separators=(",", ":"))
                header_bytes = header_json.encode("utf-8")
                header_len_prefix = struct.pack("<I", len(header_bytes))
                first_data_offset = computed_first

            # ── Phase 5: Write the sidecar file ─────────────────────────
            logger.info(
                "Writing sidecar with %d experts (%d layers) to %s ...",
                total_experts, len(sorted_layers), output_path,
            )

            with open(output_path, "wb") as out:
                out.write(header_len_prefix)
                out.write(header_bytes)
                # Pad to first data offset.
                header_end = 4 + len(header_bytes)
                pad_needed = first_data_offset - header_end
                if pad_needed > 0:
                    out.write(b"\x00" * pad_needed)
                # Write expert chunks.
                for lk, ei, offset, length, chunk in expert_layout:
                    out.write(chunk)

            file_size = output_path.stat().st_size
            logger.info(
                "Sidecar written: %s (%.1f MB, %d experts)",
                output_path.name, file_size / (1024 * 1024), total_experts,
            )

        finally:
            for fd in st_fds.values():
                os.close(fd)

        # ── Phase 6: Validate and return ────────────────────────────────
        sidecar = StreamingExpertSidecar(output_path)

        # Spot-check: read a random expert from each layer.
        logger.info("Validating sidecar ...")
        for layer_idx in sorted_layers:
            lk = layer_key_map[layer_idx]
            n_exp = num_experts[layer_idx]
            if n_exp == 0:
                continue
            # Read expert 0 and the last expert.
            for test_expert in (0, n_exp - 1):
                try:
                    raw = sidecar.read_expert(lk, test_expert)
                    expected_len = header_dict["layers"][lk]["experts"][str(test_expert)]["length"]
                    if len(raw) != expected_len:
                        raise RuntimeError(
                            f"Validation failed: layer {lk} expert {test_expert}: "
                            f"expected {expected_len} bytes, got {len(raw)}"
                        )
                except Exception as exc:
                    sidecar.close()
                    raise RuntimeError(
                        f"Sidecar validation failed for layer {lk} expert {test_expert}: {exc}"
                    ) from exc

        logger.info("Sidecar validation passed.")
        return sidecar


def _compute_per_expert_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Compute the number of bytes for a single expert's slice of a tensor."""
    import numpy as np

    bpe = _DTYPE_BYTES.get(dtype, 1)
    per_expert_shape = shape[1:]  # drop the expert dimension
    return int(np.prod(per_expert_shape) * bpe)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def create_sidecar(
    model_path: str | Path,
    output_path: str | Path,
    alignment: int = _DEFAULT_ALIGNMENT,
) -> StreamingExpertSidecar:
    """Convenience wrapper for :meth:`StreamingExpertSidecar.create`."""
    return StreamingExpertSidecar.create(model_path, output_path, alignment)
