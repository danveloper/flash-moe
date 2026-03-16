"""
stream_infer.py — Streaming inference engine for Qwen3.5 MoE models.
Loads weights layer-by-layer from safetensors during inference.
Proves flash offloading works and measures overhead vs fully-loaded baseline.

Usage:
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode baseline
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode stream
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode layerwise
"""

import argparse
import time
import sys
import json
import re
import os
import atexit
import fcntl
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import subprocess
import psutil
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import mlx_lm
from safetensors import safe_open


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    import struct
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def read_tensors_direct(filepath, tensor_names, header_cache, file_handle_cache=None):
    """Read specific tensors from a safetensors file using direct I/O (not mmap).
    Returns dict of tensor_name -> mx.array (real Metal allocations, not mmap-backed).

    Handles dtype mapping:
    - U32 -> numpy uint32 -> mx uint32
    - F32 -> numpy float32 -> mx float32
    - F16 -> numpy float16 -> mx float16
    - I32 -> numpy int32 -> mx int32
    - BF16 -> read as uint16, shift to float32, cast to bfloat16

    If file_handle_cache is provided (dict), file handles are kept open across calls
    to avoid thousands of open/close cycles per token (many tensors x many layers).
    """
    if filepath not in header_cache:
        header_cache[filepath] = parse_safetensors_header(filepath)
    header, data_start = header_cache[filepath]

    NP_DTYPE = {
        'U32': np.uint32,
        'F32': np.float32,
        'F16': np.float16,
        'I32': np.int32,
        'I64': np.int64,
        'U8': np.uint8,
    }

    # Sort tensors by file offset for sequential I/O
    sorted_names = sorted(tensor_names, key=lambda n: header[n]['data_offsets'][0])

    # Use cached file handle if available, otherwise open (and optionally cache)
    owned_f = None
    if file_handle_cache is not None:
        if filepath not in file_handle_cache:
            file_handle_cache[filepath] = open(filepath, 'rb')
        f = file_handle_cache[filepath]
    else:
        owned_f = open(filepath, 'rb')
        f = owned_f

    result = {}
    try:
        for name in sorted_names:
            meta = header[name]
            off = meta['data_offsets']
            byte_len = off[1] - off[0]

            f.seek(data_start + off[0])
            raw = f.read(byte_len)

            dtype_str = meta['dtype']
            shape = meta['shape']

            if dtype_str in NP_DTYPE:
                np_arr = np.frombuffer(raw, dtype=NP_DTYPE[dtype_str]).reshape(shape)
                result[name] = mx.array(np_arr)
            elif dtype_str == 'BF16':
                # bfloat16 = top 16 bits of float32. Read as uint16,
                # shift left 16 bits to get float32, then convert to bfloat16.
                np_uint16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                np_f32 = (np_uint16.astype(np.uint32) << 16).view(np.float32)
                result[name] = mx.array(np_f32).astype(mx.bfloat16)
            else:
                raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")
    finally:
        if owned_f is not None:
            owned_f.close()

    return result


def read_expert_slices_direct(filepath, tensor_name, expert_indices, header_cache, file_handle_cache=None):
    """Read specific expert slices from a stacked [num_experts, ...] tensor using direct I/O.

    Args:
        filepath: path to safetensors file
        tensor_name: full tensor name in the safetensors file
        expert_indices: list/array of expert indices to read (e.g., [3, 17, 42, ...])
        header_cache: dict for caching parsed headers
        file_handle_cache: optional dict mapping filepath -> open file handle.
            If provided, file handles are kept open across calls to avoid
            repeated open/close overhead (~3456 cycles/token).

    Returns: mx.array of shape [len(expert_indices), ...rest_dims]
    """
    if filepath not in header_cache:
        header_cache[filepath] = parse_safetensors_header(filepath)
    header, data_start = header_cache[filepath]

    meta = header[tensor_name]
    shape = meta['shape']  # e.g., [256, 1024, 384]
    dtype_str = meta['dtype']
    tensor_offsets = meta['data_offsets']
    tensor_start = data_start + tensor_offsets[0]

    num_experts = shape[0]
    expert_shape = shape[1:]  # e.g., [1024, 384]

    # Calculate bytes per expert
    NP_DTYPE = {
        'U32': (np.uint32, 4),
        'F32': (np.float32, 4),
        'F16': (np.float16, 2),
        'BF16': (np.uint16, 2),  # Read as uint16
        'I32': (np.int32, 4),
    }
    np_dtype, elem_size = NP_DTYPE[dtype_str]
    expert_elems = 1
    for d in expert_shape:
        expert_elems *= d
    expert_bytes = expert_elems * elem_size

    # Use cached file handle if available, otherwise open (and optionally cache)
    owned_f = None
    if file_handle_cache is not None:
        if filepath not in file_handle_cache:
            file_handle_cache[filepath] = open(filepath, 'rb')
        f = file_handle_cache[filepath]
    else:
        owned_f = open(filepath, 'rb')
        f = owned_f

    # Read each selected expert's data
    expert_arrays = []
    try:
        for idx in expert_indices:
            offset = tensor_start + int(idx) * expert_bytes
            f.seek(offset)
            raw = f.read(expert_bytes)
            np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(expert_shape)
            expert_arrays.append(np_arr)
    finally:
        if owned_f is not None:
            owned_f.close()

    # Stack into [num_selected, ...] array
    stacked = np.stack(expert_arrays, axis=0)

    if dtype_str == 'BF16':
        # Convert uint16 bit pattern to bfloat16 via float32
        np_f32 = (stacked.astype(np.uint32) << 16).view(np.float32)
        return mx.array(np_f32).astype(mx.bfloat16)
    else:
        return mx.array(stacked)


def _read_single_expert_attrs(expert_idx, layer_idx, expert_file_map, header_cache):
    """Read all 9 tensor attributes for a single expert using coalesced I/O.

    Instead of 9 individual seek+read operations, groups reads by file, sorts by
    offset, and merges adjacent reads within a 64KB gap into larger sequential reads.
    This reduces ~9 seeks per expert to ~1-3 larger reads.

    Each thread opens its own file handles (no shared seek positions), reads all
    3 projections x 3 attributes, and returns a dict of results.

    Args:
        expert_idx: which expert to read (e.g. 42)
        layer_idx: which layer (for constructing tensor names)
        expert_file_map: dict of full tensor name -> filepath
        header_cache: dict for cached headers (read-only after init, thread-safe)

    Returns:
        (expert_idx, results_dict, io_stats_dict)
        results_dict: (proj_name, attr_name) -> mx.array of shape [1, ...expert_dims]
        io_stats_dict: {"bytes_read": int, "seek_count": int, "io_time_s": float, "array_time_s": float}
    """
    NP_DTYPE = {
        'U32': (np.uint32, 4),
        'F32': (np.float32, 4),
        'F16': (np.float16, 2),
        'BF16': (np.uint16, 2),
        'I32': (np.int32, 4),
    }

    COALESCE_GAP = 65536  # 64KB: merge reads with gaps smaller than this

    # Phase 1: Collect all (file, offset, size, metadata) tuples without doing I/O
    read_specs = []  # list of (filepath, abs_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype)
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}"
        for attr_name in ("weight", "scales", "biases"):
            full_key = f"{prefix}.{attr_name}"
            if full_key not in expert_file_map:
                continue
            filepath = expert_file_map[full_key]

            header, data_start = header_cache[filepath]
            meta = header[full_key]
            shape = meta['shape']
            dtype_str = meta['dtype']
            tensor_offsets = meta['data_offsets']
            tensor_start = data_start + tensor_offsets[0]
            expert_shape = shape[1:]

            np_dtype, elem_size = NP_DTYPE[dtype_str]
            expert_elems = 1
            for d in expert_shape:
                expert_elems *= d
            expert_bytes = expert_elems * elem_size

            abs_offset = tensor_start + int(expert_idx) * expert_bytes
            read_specs.append((filepath, abs_offset, expert_bytes, proj_name, attr_name, expert_shape, dtype_str, np_dtype))

    # Phase 2: Group by file, sort by offset, merge adjacent reads
    by_file = defaultdict(list)
    for spec in read_specs:
        by_file[spec[0]].append(spec)

    results = {}
    local_handles = {}
    io_bytes = 0
    io_seeks = 0
    io_time = 0.0
    array_time = 0.0

    try:
        for filepath, specs in by_file.items():
            # Sort by file offset
            specs.sort(key=lambda s: s[1])

            # Merge into coalesced read groups
            # Each group: (start_offset, total_len, [(local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype), ...])
            groups = []
            for spec in specs:
                _, abs_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype = spec
                if groups:
                    g_start, g_len, g_items = groups[-1]
                    g_end = g_start + g_len
                    gap = abs_offset - g_end
                    if gap <= COALESCE_GAP:
                        # Extend the current group to cover this read
                        new_end = abs_offset + byte_len
                        groups[-1] = (g_start, new_end - g_start, g_items)
                        local_offset = abs_offset - g_start
                        g_items.append((local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype))
                        continue
                # Start a new group
                groups.append((abs_offset, byte_len, [(0, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype)]))

            # Open file handle (reuse within this thread)
            if filepath not in local_handles:
                local_handles[filepath] = open(filepath, 'rb')
            f = local_handles[filepath]

            # Phase 3: Execute coalesced reads and split into individual tensors
            for g_start, g_len, g_items in groups:
                t_io = time.time()
                f.seek(g_start)
                raw_chunk = f.read(g_len)
                io_time += time.time() - t_io
                io_bytes += g_len
                io_seeks += 1

                t_arr = time.time()
                for local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype in g_items:
                    raw = raw_chunk[local_offset:local_offset + byte_len]
                    np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(expert_shape)

                    if dtype_str == 'BF16':
                        np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
                        results[(proj_name, attr_name)] = mx.array(np_f32).astype(mx.bfloat16)
                    else:
                        results[(proj_name, attr_name)] = mx.array(np_arr)
                array_time += time.time() - t_arr
    finally:
        for fh in local_handles.values():
            fh.close()

    io_stats = {
        "bytes_read": io_bytes,
        "seek_count": io_seeks,
        "io_time_s": io_time,
        "array_time_s": array_time,
    }
    return expert_idx, results, io_stats


# ---------------------------------------------------------------------------
# pread()-based expert loading (bypasses safetensors, uses pre-built index)
# ---------------------------------------------------------------------------

# Global state for pread file descriptors (cleaned up via atexit)
_pread_fds = {}  # filename -> fd


def _cleanup_pread_fds():
    """Close all pread file descriptors at exit."""
    for fd in _pread_fds.values():
        try:
            os.close(fd)
        except OSError:
            pass
    _pread_fds.clear()

atexit.register(_cleanup_pread_fds)


def load_expert_index(model_path):
    """Load expert_index.json and open all shard file descriptors.

    Args:
        model_path: Path to model directory (used to resolve shard filenames)

    Returns:
        (expert_index, fds) where:
        - expert_index: parsed expert_index.json dict
        - fds: dict mapping filename -> os file descriptor (O_RDONLY, F_NOCACHE)
    """
    index_path = Path("expert_index.json")
    if not index_path.exists():
        index_path = Path(model_path) / "expert_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"expert_index.json not found in working directory or {model_path}")

    with open(index_path) as f:
        expert_index = json.load(f)

    # Collect all unique shard filenames
    shard_files = set()
    for tensor_info in expert_index["tensors"].values():
        shard_files.add(tensor_info["file"])

    # Also collect from expert_reads (may reference different files)
    for layer_reads in expert_index.get("expert_reads", {}).values():
        for comp_info in layer_reads.values():
            shard_files.add(comp_info["file"])

    # Resolve full paths and open file descriptors
    resolved_model_path = Path(expert_index.get("model_path", str(model_path)))
    fds = {}
    for filename in sorted(shard_files):
        filepath = resolved_model_path / filename
        if not filepath.exists():
            # Try model_path argument as fallback
            filepath = Path(model_path) / filename
        fd = os.open(str(filepath), os.O_RDONLY)
        # NOTE: Do NOT use F_NOCACHE. The kernel page cache provides a valuable
        # second-level cache. With F_NOCACHE, every read hits SSD at ~1.8 GB/s.
        # With page cache, recently-read experts are served at ~40 GB/s (memcpy).
        # Our LRU cache handles the application-level caching; the page cache
        # handles the OS-level caching of recently-accessed pages.
        fds[filename] = fd

    # Store globally for atexit cleanup
    _pread_fds.update(fds)

    print(f"[pread] Loaded expert index: {len(expert_index.get('expert_reads', {}))} layers, "
          f"{len(fds)} shard files opened (page cache enabled)")

    return expert_index, fds


# Dtype mapping for pread path
_PREAD_NP_DTYPE = {
    'U32': (np.uint32, 4),
    'F32': (np.float32, 4),
    'BF16': (np.uint16, 2),  # read as uint16, convert to bfloat16
    'F16': (np.float16, 2),
    'I32': (np.int32, 4),
}


def load_packed_experts(model_path):
    """Load packed expert layout and open layer file descriptors.

    Returns (layout, packed_fds, packed_layers) or (None, None, set()) if not available.
    """
    packed_dir = Path(model_path) / "packed_experts"
    layout_path = packed_dir / "layout.json"
    if not layout_path.exists():
        return None, None, set()

    with open(layout_path) as f:
        layout = json.load(f)

    packed_fds = {}
    packed_layers = set()
    for layer_file in sorted(packed_dir.glob("layer_*.bin")):
        layer_idx = int(layer_file.stem.split("_")[1])
        fd = os.open(str(layer_file), os.O_RDONLY)
        packed_fds[layer_idx] = fd
        packed_layers.add(layer_idx)

    _pread_fds.update({f"packed_{k}": v for k, v in packed_fds.items()})
    print(f"[packed] Loaded {len(packed_layers)} packed layer files "
          f"(layers {min(packed_layers)}-{max(packed_layers)})")
    return layout, packed_fds, packed_layers


def read_expert_packed(packed_fd, layout, expert_idx):
    """Read a single expert from a packed layer file (1 pread for entire expert).

    Returns (expert_idx, results_dict, io_stats) matching _read_single_expert_attrs format.
    """
    expert_size = layout["expert_size"]
    offset = expert_idx * expert_size

    t_io = time.time()
    raw = os.pread(packed_fd, expert_size, offset)  # 1 read, 7.08MB contiguous
    io_time = time.time() - t_io

    t_arr = time.time()
    results = {}
    mv = memoryview(raw)  # zero-copy view for slicing
    for comp in layout["components"]:
        name = comp["name"]
        parts = name.split(".")
        proj_name, attr_name = parts[0], parts[1]
        comp_offset = comp["offset"]
        comp_size = comp["size"]
        dtype_str = comp["dtype"]
        shape = comp["shape"]

        chunk = mv[comp_offset:comp_offset + comp_size]  # zero-copy slice
        np_dtype, _ = _PREAD_NP_DTYPE[dtype_str]
        np_arr = np.frombuffer(chunk, dtype=np_dtype).reshape(shape)

        if dtype_str == 'BF16':
            np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
            results[(proj_name, attr_name)] = mx.array(np_f32).astype(mx.bfloat16)
        else:
            results[(proj_name, attr_name)] = mx.array(np_arr)
    array_time = time.time() - t_arr

    return expert_idx, results, {
        "bytes_read": expert_size,
        "seek_count": 1,
        "io_time_s": io_time,
        "array_time_s": array_time,
    }


def pread_expert(expert_index, fds, layer, expert_idx):
    """Read all 9 tensor components for a single expert using os.pread().

    Args:
        expert_index: parsed expert_index.json dict
        fds: dict mapping shard filename -> os file descriptor
        layer: layer index (int)
        expert_idx: expert index (int)

    Returns:
        (expert_idx, results_dict, io_stats_dict) matching _read_single_expert_attrs format.
        results_dict: {(proj_name, attr_name): mx.array} with per-expert shapes
        io_stats_dict: {"bytes_read": int, "seek_count": int, "io_time_s": float, "array_time_s": float}
    """
    layer_reads = expert_index["expert_reads"][str(layer)]
    results = {}
    io_bytes = 0
    io_seeks = 0
    io_time = 0.0
    array_time = 0.0

    for comp_name, comp_info in layer_reads.items():
        # comp_name is like "gate_proj.weight", "gate_proj.scales", etc.
        parts = comp_name.split(".")
        proj_name = parts[0]  # gate_proj, up_proj, down_proj
        attr_name = parts[1]  # weight, scales, biases

        fd = fds[comp_info["file"]]
        abs_offset = comp_info["abs_offset"]
        expert_stride = comp_info["expert_stride"]
        expert_size = comp_info["expert_size"]
        dtype_str = comp_info["dtype"]
        shape = comp_info["shape_per_expert"]

        offset = abs_offset + expert_idx * expert_stride

        t_io = time.time()
        raw = os.pread(fd, expert_size, offset)
        io_time += time.time() - t_io
        io_bytes += expert_size
        io_seeks += 1

        t_arr = time.time()
        np_dtype, _ = _PREAD_NP_DTYPE[dtype_str]
        np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

        if dtype_str == 'BF16':
            np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
            results[(proj_name, attr_name)] = mx.array(np_f32).astype(mx.bfloat16)
        else:
            results[(proj_name, attr_name)] = mx.array(np_arr)
        array_time += time.time() - t_arr

    io_stats = {
        "bytes_read": io_bytes,
        "seek_count": io_seeks,
        "io_time_s": io_time,
        "array_time_s": array_time,
    }
    return expert_idx, results, io_stats


def pread_expert_batch(expert_index, fds, layer, expert_indices):
    """Read all tensor components for multiple experts using sorted os.pread() calls.

    Collects all read operations across all requested experts, sorts them by
    (fd, offset) for sequential disk access, then executes all reads. This
    exploits the fact that adjacent expert indices are physically contiguous
    within each tensor shard.

    Args:
        expert_index: parsed expert_index.json dict
        fds: dict mapping shard filename -> os file descriptor
        layer: layer index (int)
        expert_indices: list of expert indices to read

    Returns:
        (batch_results, batch_io_stats) where:
        - batch_results: {expert_idx: {(proj_name, attr_name): mx.array}}
        - batch_io_stats: {"bytes_read": int, "seek_count": int,
                           "io_time_s": float, "array_time_s": float}
    """
    layer_reads = expert_index["expert_reads"][str(layer)]

    # Phase 1: Collect all read operations
    # Each entry: (fd_num, offset, size, expert_idx, proj_name, attr_name, dtype_str, shape)
    read_ops = []
    for expert_idx in expert_indices:
        for comp_name, comp_info in layer_reads.items():
            parts = comp_name.split(".")
            proj_name = parts[0]
            attr_name = parts[1]

            filename = comp_info["file"]
            fd = fds[filename]
            abs_offset = comp_info["abs_offset"]
            expert_stride = comp_info["expert_stride"]
            expert_size = comp_info["expert_size"]
            dtype_str = comp_info["dtype"]
            shape = comp_info["shape_per_expert"]

            offset = abs_offset + expert_idx * expert_stride
            read_ops.append((fd, offset, expert_size, expert_idx,
                             proj_name, attr_name, dtype_str, shape))

    # Phase 2: Sort by (fd, offset) for sequential access
    read_ops.sort(key=lambda x: (x[0], x[1]))

    # Phase 3: Execute reads with threading for parallel I/O
    batch_results = {idx: {} for idx in expert_indices}
    io_bytes = sum(op[2] for op in read_ops)
    io_seeks = len(read_ops)

    def _do_read(op):
        fd, offset, size, expert_idx, proj_name, attr_name, dtype_str, shape = op
        raw = os.pread(fd, size, offset)
        return (expert_idx, proj_name, attr_name, dtype_str, shape, raw)

    t_io = time.time()
    # Use ThreadPoolExecutor with 8 workers to saturate NVMe
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw_results = list(pool.map(_do_read, read_ops))
    io_time = time.time() - t_io

    t_arr = time.time()
    for expert_idx, proj_name, attr_name, dtype_str, shape, raw in raw_results:
        np_dtype, _ = _PREAD_NP_DTYPE[dtype_str]
        np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

        if dtype_str == 'BF16':
            np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
            batch_results[expert_idx][(proj_name, attr_name)] = mx.array(np_f32).astype(mx.bfloat16)
        else:
            batch_results[expert_idx][(proj_name, attr_name)] = mx.array(np_arr)
    array_time = time.time() - t_arr

    batch_io_stats = {
        "bytes_read": io_bytes,
        "seek_count": io_seeks,
        "io_time_s": io_time,
        "array_time_s": array_time,
    }
    return batch_results, batch_io_stats


class ExpertCache:
    """LRU cache for expert weight slices keyed by (layer, expert_id).

    Stores per-attribute arrays so that partially-populated entries can be
    built incrementally (one proj/attr at a time) then read back as a batch.

    Each entry is a dict mapping "proj_name.attr_name" -> mx.array, e.g.
    {"gate_proj.weight": mx.array(...), "gate_proj.scales": mx.array(...), ...}.

    Eviction policy: Least Recently Used via OrderedDict ordering.
    """

    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self.cache = OrderedDict()  # (layer_idx, expert_id) -> {attr_key: mx.array}
        self._protected = set()  # keys protected from eviction during current batch
        self.hits = 0
        self.misses = 0
        self._lock = __import__('threading').Lock()

    # -- internal helpers ---------------------------------------------------

    def protect(self, keys):
        """Mark keys as protected from eviction (call before batch loading)."""
        self._protected = set(keys)

    def unprotect(self):
        """Clear protection after batch loading is done."""
        self._protected = set()

    # -- public interface (unchanged signatures) ----------------------------

    def get_attr(self, layer_idx, expert_id, proj_name, attr_name):
        """Return a single cached array or None."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)
            entry = self.cache[key]
            attr_key = f"{proj_name}.{attr_name}"
            return entry.get(attr_key)
        return None

    def put_attr(self, layer_idx, expert_id, proj_name, attr_name, array):
        """Store a single attribute array for an expert."""
        key = (layer_idx, expert_id)
        attr_key = f"{proj_name}.{attr_name}"
        with self._lock:
            if key in self.cache:
                self.cache[key][attr_key] = array
            else:
                # Find the LRU entry that isn't protected
                while len(self.cache) >= self.max_entries:
                    # Iterate from LRU end to find first unprotected entry
                    for candidate_key in self.cache:
                        if candidate_key not in self._protected:
                            del self.cache[candidate_key]
                            break
                    else:
                        break  # all entries protected, can't evict
                self.cache[key] = {attr_key: array}

    def has_expert(self, layer_idx, expert_id):
        """Check whether all 9 attributes (3 projs x 3 attrs) are cached."""
        key = (layer_idx, expert_id)
        if key not in self.cache:
            return False
        return len(self.cache[key]) >= 9  # gate/up/down x weight/scales/biases

    def touch(self, layer_idx, expert_id):
        """Move entry to the most-recently-used end."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def preload_hot_experts(expert_cache, weight_index, model_path, num_layers, topk_per_layer, header_cache,
                        use_pread=False, pread_index=None, pread_fds=None):
    """Pre-load the hottest experts per layer into the ExpertCache at startup.

    Uses profiling data from expert_routing_profile.npz (if it exists) to identify
    the most frequently activated experts. Falls back to experts 0..topk-1 per layer
    if no profiling data is available.

    Args:
        expert_cache: ExpertCache instance to populate
        weight_index: dict from build_weight_index() mapping layer -> [(name, filepath)]
        model_path: Path to model directory (for locating profiling data)
        num_layers: number of MoE layers
        topk_per_layer: how many experts to preload per layer
        header_cache: dict for caching safetensors headers (populated in-place)
        use_pread: if True, use pread_expert_batch instead of _read_single_expert_attrs
        pread_index: expert_index dict (required if use_pread=True)
        pread_fds: fd dict (required if use_pread=True)
    """
    t_start = time.time()

    # --- Determine which experts to preload per layer ---
    profile_path = Path(model_path) / "expert_routing_profile.npz"
    if not profile_path.exists():
        # Also check working directory
        profile_path = Path("expert_routing_profile.npz")

    if profile_path.exists():
        data = np.load(str(profile_path))
        freq = data["counts"]  # [num_layers, num_experts]
        print(f"[preload] Using profiling data from {profile_path} "
              f"(shape {freq.shape})", flush=True)
        layer_topk = {}
        for layer_idx in range(min(num_layers, freq.shape[0])):
            topk_indices = np.argsort(freq[layer_idx])[::-1][:topk_per_layer]
            layer_topk[layer_idx] = topk_indices.tolist()
    else:
        print(f"[preload] No profiling data found, using experts 0..{topk_per_layer-1} "
              f"per layer (uniform fallback)", flush=True)
        layer_topk = {i: list(range(topk_per_layer)) for i in range(num_layers)}

    if not use_pread:
        # --- Build expert_file_map and pre-populate header_cache for all layers ---
        layer_expert_maps = {}
        all_expert_files = set()
        for layer_idx in range(num_layers):
            entries = weight_index.get(layer_idx, [])
            _, expert_entries = split_layer_entries(entries)
            expert_file_map = {}
            for name, filepath in expert_entries:
                expert_file_map[name] = filepath
                all_expert_files.add(filepath)
            layer_expert_maps[layer_idx] = expert_file_map

        # Parse all safetensors headers upfront (needed by _read_single_expert_attrs)
        for filepath in all_expert_files:
            if filepath not in header_cache:
                header_cache[filepath] = parse_safetensors_header(filepath)

    # --- Preload experts in batches of 10 layers using ThreadPoolExecutor ---
    total_loaded = 0
    total_bytes = 0
    batch_size = 10  # layers per progress report

    for batch_start in range(0, num_layers, batch_size):
        batch_end = min(batch_start + batch_size, num_layers)
        t_batch = time.time()
        batch_loaded = 0
        batch_bytes = 0

        if use_pread:
            # pread path: use pread_expert_batch per layer
            for layer_idx in range(batch_start, batch_end):
                experts_to_load = layer_topk.get(layer_idx, [])
                uncached = [eidx for eidx in experts_to_load
                            if not expert_cache.has_expert(layer_idx, eidx)]
                if not uncached:
                    continue
                batch_results, io_stats = pread_expert_batch(
                    pread_index, pread_fds, layer_idx, uncached)
                batch_bytes += io_stats["bytes_read"]
                for eidx, attrs in batch_results.items():
                    for (proj_name, attr_name), arr in attrs.items():
                        expert_cache.put_attr(layer_idx, eidx, proj_name, attr_name, arr)
                    batch_loaded += 1
        else:
            # Original safetensors path
            # Collect all (layer, expert) pairs for this batch
            work_items = []
            for layer_idx in range(batch_start, batch_end):
                expert_file_map = layer_expert_maps.get(layer_idx, {})
                if not expert_file_map:
                    continue
                experts_to_load = layer_topk.get(layer_idx, [])
                for expert_idx in experts_to_load:
                    # Skip if already in cache (shouldn't happen at startup, but be safe)
                    if not expert_cache.has_expert(layer_idx, expert_idx):
                        work_items.append((expert_idx, layer_idx, expert_file_map))

            if not work_items:
                continue

            # Parallel read with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_layer = {}
                for expert_idx, layer_idx, expert_file_map in work_items:
                    fut = executor.submit(
                        _read_single_expert_attrs,
                        expert_idx, layer_idx, expert_file_map, header_cache
                    )
                    future_to_layer[fut] = layer_idx
                for future in future_to_layer:
                    eidx, attrs, io_stats = future.result()
                    layer_idx = future_to_layer[future]
                    batch_bytes += io_stats["bytes_read"]
                    for (proj_name, attr_name), arr in attrs.items():
                        expert_cache.put_attr(layer_idx, eidx, proj_name, attr_name, arr)
                    batch_loaded += 1

        total_loaded += batch_loaded
        total_bytes += batch_bytes
        elapsed = time.time() - t_batch
        throughput = (batch_bytes / 1e9) / elapsed if elapsed > 0 else 0
        print(f"[preload] Layer {batch_start}-{batch_end-1}: loaded {batch_loaded} experts "
              f"({batch_bytes/1e9:.1f} GB), {throughput:.1f} GB/s", flush=True)

    total_time = time.time() - t_start
    total_throughput = (total_bytes / 1e9) / total_time if total_time > 0 else 0
    print(f"[preload] Done: {total_loaded} experts ({total_bytes/1e9:.1f} GB) "
          f"in {total_time:.1f}s ({total_throughput:.1f} GB/s, {get_mem_gb():.1f} GB RSS)",
          flush=True)


def get_mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)


def check_memory_pressure():
    """Check macOS system memory free percentage. Returns (level, free_pct)."""
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "free percentage" in line.lower():
                pct = int(line.split(":")[-1].strip().rstrip("%"))
                if pct < 10:
                    return "critical", pct
                elif pct < 25:
                    return "warn", pct
                return "normal", pct
        return "unknown", -1
    except Exception:
        return "unknown", -1


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def resolve_model_path(model_id):
    """Resolve a HF model ID to a local path."""
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def build_weight_index(model_path):
    """Build a mapping: layer_num -> [(tensor_name, file_path)].
    Also returns 'global' key for non-layer tensors (embed, norm, lm_head)."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)

    layer_weights = defaultdict(list)
    for name, filename in idx["weight_map"].items():
        filepath = model_path / filename
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_num = int(match.group(1))
            layer_weights[layer_num].append((name, str(filepath)))
        else:
            layer_weights["global"].append((name, str(filepath)))

    return layer_weights


def load_layer_from_safetensors(weight_index, layer_num, model, file_cache=None):
    """Load a single layer's weights from safetensors into the model.
    Uses mx.load() which handles bfloat16 natively via mmap.
    Returns the load time in seconds."""
    entries = weight_index.get(layer_num, [])
    if not entries:
        return 0.0

    t0 = time.time()

    # Group by file for efficiency
    by_file = defaultdict(list)
    for name, filepath in entries:
        by_file[filepath].append(name)

    weights = []
    for filepath, names in by_file.items():
        # mx.load() returns lazy mmap'd arrays — actual I/O at mx.eval()
        if file_cache is not None and filepath in file_cache:
            all_tensors = file_cache[filepath]
        else:
            all_tensors = mx.load(filepath)
            if file_cache is not None:
                file_cache[filepath] = all_tensors

        for name in names:
            if name not in all_tensors:
                continue
            # Sanitize: remove "language_model." prefix to match model's param paths
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            weights.append((san_name, all_tensors[name]))

    # Apply to model (language_model level)
    model.language_model.load_weights(weights, strict=False)
    # Force evaluation so weights are actually loaded from disk
    mx.eval(model.language_model.model.layers[layer_num].parameters())

    return time.time() - t0


def manual_forward(model, input_ids, cache):
    """Run the model's forward pass layer-by-layer, returning logits.
    This replicates the model's own forward but gives us per-layer control."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    # Embed
    h = text_model.embed_tokens(input_ids)

    # Create masks (same logic as the model)
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    # Process layers
    for i, (layer, c) in enumerate(zip(layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)

    # Norm
    h = text_model.norm(h)

    # LM head
    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)

    return logits


def manual_forward_layerwise(model, input_ids, cache, weight_index=None, file_cache=None):
    """Same as manual_forward but returns per-layer timing info.
    If weight_index is provided, reloads weights from safetensors per layer."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    h = text_model.embed_tokens(input_ids)
    mx.eval(h)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    layer_timings = []

    for i, (layer, c) in enumerate(zip(layers, cache)):
        load_time = 0.0
        if weight_index is not None:
            load_time = load_layer_from_safetensors(weight_index, i, model, file_cache)

        mask = ssm_mask if layer.is_linear else fa_mask

        t_compute = time.time()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        compute_time = time.time() - t_compute

        layer_timings.append({
            "layer": i,
            "is_linear": layer.is_linear,
            "load_ms": load_time * 1000,
            "compute_ms": compute_time * 1000,
        })

    h = text_model.norm(h)

    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    mx.eval(logits)

    return logits, layer_timings


def generate_baseline(model, tokenizer, prompt, max_tokens):
    """Generate using mlx_lm's built-in stream_generate. Reference baseline."""
    t_start = time.time()
    token_times = []
    generated_tokens = []
    peak_mem = get_mem_gb()

    t_gen_start = time.time()

    for i, response in enumerate(mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    )):
        t_now = time.time()

        if i == 0:
            ttft_ms = (t_now - t_gen_start) * 1000
            print(f"  [{fmt_time(t_now - t_start)}] Token 1/{max_tokens}... ttft={ttft_ms:.0f}ms")
        else:
            token_times.append(t_now - t_prev)

        t_prev = t_now
        generated_tokens.append(response.token)

        if (i + 1) % 5 == 0 or i == max_tokens - 1:
            cur_mem = get_mem_gb()
            peak_mem = max(peak_mem, cur_mem)
            elapsed = t_now - t_gen_start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{fmt_time(t_now - t_start)}] Token {i+1}/{max_tokens}... "
                  f"{tps:.1f} tok/s (mem: {cur_mem:.1f} GB)")

        if i + 1 >= max_tokens:
            break

    total_time = time.time() - t_gen_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": ttft_ms,
        "peak_mem_gb": peak_mem,
    }


def generate_manual(model, tokenizer, prompt, max_tokens, weight_index=None, mode="stream"):
    """Generate tokens using manual layer-by-layer forward pass.
    If weight_index is provided (mode=stream), reloads weights from safetensors each layer."""
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()
    # Cache file handles for mx.load() — avoids re-parsing safetensors headers
    file_cache = {} if mode == "stream" else None

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        if mode == "layerwise" or mode == "stream":
            logits, layer_timings = manual_forward_layerwise(
                model, input_ids, cache,
                weight_index=weight_index if mode == "stream" else None,
                file_cache=file_cache,
            )
            all_layer_timings.append(layer_timings)
        else:
            logits = manual_forward(model, input_ids, cache)
            layer_timings = None

        # Greedy sample
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        if token_idx == 0:
            ttft_ms = token_time * 1000
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (load={load_ms:.0f}ms compute={compute_ms:.0f}ms)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.1f} tok/s (load={load_ms:.0f}ms compute={compute_ms:.0f}ms "
                  f"mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]  # skip prompt token
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])

        # Per-layer breakdown
        num_layers = len(gen_timings[0])
        per_layer_load = [np.mean([tt[i]["load_ms"] for tt in gen_timings]) for i in range(num_layers)]
        per_layer_compute = [np.mean([tt[i]["compute_ms"] for tt in gen_timings]) for i in range(num_layers)]
    else:
        avg_load = 0
        avg_compute = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
    }


def clear_layer_weights(model, layer_num):
    """Replace all leaf parameters in a layer with tiny dummy arrays to free memory.
    This is the critical step that keeps DRAM usage bounded: after computing a layer,
    we throw away its ~1.3GB of weights so the next layer can be loaded."""
    layer = model.language_model.model.layers[layer_num]
    dummy_weights = []
    for name, param in mlx.utils.tree_flatten(layer.parameters()):
        dummy_weights.append((f"model.layers.{layer_num}.{name}", mx.zeros((1,), dtype=param.dtype)))
    model.language_model.load_weights(dummy_weights, strict=False)
    mx.clear_cache()


def clear_expert_weights(model, layer_num):
    """Clear only MoE expert weights (switch_mlp), preserving attention/norms/shared_expert.
    Expert weights are ~1.27GB per layer; non-expert weights are ~50MB and stay pinned."""
    layer = model.language_model.model.layers[layer_num]
    if not hasattr(layer.mlp, 'switch_mlp'):
        return  # Dense layer, nothing to clear
    switch = layer.mlp.switch_mlp
    dummy_weights = []
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj = getattr(switch, proj_name)
        for attr_name in ["weight", "scales", "biases"]:
            if hasattr(proj, attr_name):
                full_name = f"model.layers.{layer_num}.mlp.switch_mlp.{proj_name}.{attr_name}"
                dummy_weights.append((full_name, mx.zeros((1,), dtype=getattr(proj, attr_name).dtype)))
    if dummy_weights:
        model.language_model.load_weights(dummy_weights, strict=False)
    mx.clear_cache()  # Return freed Metal memory to OS


def compute_moe_direct(x, indices, expert_tensors, group_size=64, bits=4, mode="affine"):
    """Compute MoE SwitchGLU forward pass directly via mx.gather_qmm.

    Bypasses model weight mutation entirely -- no load_weights, no clear_expert_weights.
    Replicates the SwitchGLU.__call__ + QuantizedSwitchLinear.__call__ logic.

    Args:
        x: hidden states [batch, seq_len, hidden_dim]
        indices: remapped expert indices [batch, seq_len, top_k] with values in 0..num_unique-1
        expert_tensors: dict with keys like "gate_proj.weight", "gate_proj.scales", etc.
            Each value is [num_unique_experts, ...] stacked tensor.
        group_size: quantization group size (default 64)
        bits: quantization bits (default 4)
        mode: quantization mode (default "affine")

    Returns:
        output: [batch, seq_len, top_k, hidden_dim]
    """
    # Replicate SwitchGLU.__call__: expand dims for gather_qmm batch indexing
    x = mx.expand_dims(x, (-2, -3))

    # For single-token inference (indices.size < 64), no sorting needed
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        # Replicate _gather_sort
        *_, M = indices.shape
        flat_indices = indices.flatten()
        order = mx.argsort(flat_indices)
        inv_order = mx.argsort(order)
        x = x.flatten(0, -3)[order // M]
        idx = flat_indices[order]

    # up_proj: x -> intermediate
    x_up = mx.gather_qmm(
        x,
        expert_tensors["up_proj.weight"],
        expert_tensors["up_proj.scales"],
        expert_tensors.get("up_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # gate_proj: x -> intermediate
    x_gate = mx.gather_qmm(
        x,
        expert_tensors["gate_proj.weight"],
        expert_tensors["gate_proj.scales"],
        expert_tensors.get("gate_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # SwiGLU activation: silu(gate) * up
    x_act = nn.silu(x_gate) * x_up

    # down_proj: intermediate -> hidden
    out = mx.gather_qmm(
        x_act,
        expert_tensors["down_proj.weight"],
        expert_tensors["down_proj.scales"],
        expert_tensors.get("down_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # Unsort if we sorted
    if do_sort:
        # Replicate _scatter_unsort
        out = out[inv_order]
        out = mx.unflatten(out, 0, indices.shape)

    return out.squeeze(-2)


def generate_offload(model, tokenizer, prompt, max_tokens, weight_index, model_path, lazy_eval=False):
    """Generate tokens with explicit per-layer weight streaming for models larger than DRAM.

    For each token:
      1. Embed (weights already pinned in DRAM)
      2. For each layer:
         a. Load layer weights from safetensors (~1.3GB)
         b. Run layer forward pass
         c. Clear layer weights (replace with tiny dummies)
      3. Norm + lm_head (weights already pinned)
      4. Sample next token

    Peak DRAM usage: ~1GB global + ~1.3GB one layer = ~2.3GB active weights.
    The rest is KV cache and activations.

    If lazy_eval=True, skip mx.eval(layers[i].parameters()) after load_weights.
    This leaves weights as lazy mmap'd references. The subsequent layer forward
    pass + mx.eval(h) should only page in the expert weights actually accessed
    by the router (only active experts) instead of all experts per layer.
    """
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers
    num_layers = len(layers)

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()
    file_cache = {}  # Cache mx.load() dicts across tokens to avoid re-parsing safetensors headers

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        # --- Embed ---
        h = text_model.embed_tokens(input_ids)
        mx.eval(h)

        # --- Create masks ---
        fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

        layer_timings = []

        # --- Per-layer: load, compute, clear ---
        for i in range(num_layers):
            layer = layers[i]
            c = cache[i]

            # (a) Load this layer's weights from safetensors
            t_load = time.time()
            entries = weight_index.get(i, [])
            if entries:
                by_file = defaultdict(list)
                for name, filepath in entries:
                    by_file[filepath].append(name)

                layer_weights = []
                for filepath, names in by_file.items():
                    # mx.load() returns lazy mmap'd arrays — cache the dict so we
                    # only parse each safetensors file header once across all tokens
                    if filepath not in file_cache:
                        file_cache[filepath] = mx.load(filepath)
                    all_tensors = file_cache[filepath]
                    for name in names:
                        if name in all_tensors:
                            san_name = name
                            if san_name.startswith("language_model."):
                                san_name = san_name[len("language_model."):]
                            layer_weights.append((san_name, all_tensors[name]))

                lm.load_weights(layer_weights, strict=False)
                if not lazy_eval:
                    # Eager: pre-materialize all ~1.3GB of layer weights from SSD
                    mx.eval(layers[i].parameters())
                # else: lazy — skip mx.eval(params), let the forward pass
                # trigger lazy mmap eval so only accessed expert pages are read
                del layer_weights

            load_time = time.time() - t_load

            # (b) Compute this layer
            mask = ssm_mask if layer.is_linear else fa_mask

            t_compute = time.time()
            h = layer(h, mask=mask, cache=c)
            mx.eval(h)
            compute_time = time.time() - t_compute

            # (c) Clear this layer's weights — free ~1.3GB
            t_clear = time.time()
            clear_layer_weights(model, i)
            clear_time = time.time() - t_clear

            layer_timings.append({
                "layer": i,
                "is_linear": layer.is_linear,
                "load_ms": load_time * 1000,
                "compute_ms": compute_time * 1000,
                "clear_ms": clear_time * 1000,
            })

        all_layer_timings.append(layer_timings)

        # --- Norm + LM head (already pinned) ---
        h = text_model.norm(h)
        if lm.args.tie_word_embeddings:
            logits = text_model.embed_tokens.as_linear(h)
        else:
            logits = lm.lm_head(h)
        mx.eval(logits)

        # --- Sample ---
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        total_load = sum(lt["load_ms"] for lt in layer_timings)
        total_compute = sum(lt["compute_ms"] for lt in layer_timings)
        total_clear = sum(lt["clear_ms"] for lt in layer_timings)

        if token_idx == 0:
            ttft_ms = token_time * 1000
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (load={total_load:.0f}ms compute={total_compute:.0f}ms "
                  f"clear={total_clear:.0f}ms mem={cur_mem:.1f}GB)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.2f} tok/s (load={total_load:.0f}ms compute={total_compute:.0f}ms "
                  f"clear={total_clear:.0f}ms mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])
        avg_clear = np.mean([sum(lt["clear_ms"] for lt in tt) for tt in gen_timings])

        num_layers_t = len(gen_timings[0])
        per_layer_load = [np.mean([tt[i]["load_ms"] for tt in gen_timings]) for i in range(num_layers_t)]
        per_layer_compute = [np.mean([tt[i]["compute_ms"] for tt in gen_timings]) for i in range(num_layers_t)]
    else:
        avg_load = 0
        avg_compute = 0
        avg_clear = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "avg_clear_ms_per_token": avg_clear,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
    }


def split_layer_entries(entries):
    """Split a layer's weight index entries into non-expert and expert entries.

    Expert entries are the stacked expert weight tensors inside SwitchGLU:
        switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
    Everything else (layer norms, attention/SSM, router, shared expert) is non-expert.

    Returns (non_expert_entries, expert_entries).
    """
    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    non_expert = []
    expert = []
    for name, filepath in entries:
        if expert_pattern.search(name):
            expert.append((name, filepath))
        else:
            non_expert.append((name, filepath))
    return non_expert, expert


def generate_offload_selective(model, tokenizer, prompt, max_tokens, weight_index, model_path,
                               preload_topk=0, cache_gb=20.0, profile=False,
                               use_pread=False, pread_index=None, pread_fds=None,
                               use_cext=False, batch_experts=False,
                               top_k_override=0, no_expert_cache=False,
                               pin_experts=0, pin_topk=51,
                               async_pipeline=False,
                               skip_layers=0,
                               batch_layers=0,
                               numpy_cache_gb=0.0):
    """Generate tokens with selective expert loading for MoE models.

    At startup: pre-load all non-expert weights (~2.3GB) for all layers into DRAM.
    For each token, for each layer:
      Phase 2: Run attention + router (weights already resident, no loading needed)
      Phase 3: Load ONLY the selected expert slices, run MoE computation
      Phase 4: Clear only expert weights (keep attention/norms/shared_expert resident)

    This reduces per-token I/O to just expert slices (~40MB/layer), eliminating the
    ~38s of non-expert loading overhead from the previous implementation.

    When profile=True, collects detailed per-layer timing breakdown (routing, cache
    lookup, I/O, compute, mx.eval sync) and prints a summary table after generation.
    """
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers
    num_layers = len(layers)

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    # === Pre-load all non-expert weights using DIRECT I/O (not mmap) ===
    # mmap-backed arrays get evicted from page cache when expert weights are loaded.
    # Direct I/O creates real Metal allocations that stay pinned in DRAM.
    t_preload = time.time()
    header_cache = {}  # filepath -> (header_dict, data_start_offset)
    all_nonexpert_weights = []

    # Group all non-expert tensor names by filepath for sequential I/O
    file_to_names = defaultdict(list)
    file_to_san = {}  # (filepath, name) -> sanitized_name
    for layer_i in range(num_layers):
        entries = weight_index.get(layer_i, [])
        non_expert, _ = split_layer_entries(entries)
        for name, filepath in non_expert:
            file_to_names[filepath].append(name)
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            file_to_san[(filepath, name)] = san_name

    # Read from each file sequentially (sorted by offset within file)
    for filepath, names in sorted(file_to_names.items()):
        tensors = read_tensors_direct(filepath, names, header_cache, file_handle_cache=None)
        for name in names:
            if name in tensors:
                san_name = file_to_san[(filepath, name)]
                all_nonexpert_weights.append((san_name, tensors[name]))
        print(f"    read {len(names)} tensors from {Path(filepath).name} ({get_mem_gb():.1f}GB)", flush=True)

    print(f"    Total non-expert weights to load: {len(all_nonexpert_weights)}")
    lm.load_weights(all_nonexpert_weights, strict=False)
    # Skip mx.eval — direct I/O arrays are already real Metal allocations (not mmap-backed).
    # BF16 astype results will be lazily evaluated during first forward pass (cheap).
    del all_nonexpert_weights
    preload_time = time.time() - t_preload
    print(f"  Pre-loaded non-expert weights for {num_layers} layers in {preload_time:.1f}s ({get_mem_gb():.1f}GB)")

    # === File handle cache: keep safetensors files open for duration of inference ===
    # Avoids thousands of open/close cycles per token (many tensor reads per layer).
    file_handle_cache = {}

    # === LRU cache for expert weight slices ===
    # Size = num_layers * active_experts_per_token * N tokens of history.
    # Cap total cache memory at 20GB to leave headroom for OS on 48GB machine.
    active_experts = lm.args.num_experts_per_tok if top_k_override <= 0 else min(lm.args.num_experts_per_tok, top_k_override)
    if top_k_override > 0:
        print(f"  [top-k override] Using {active_experts} experts/token (model default: {lm.args.num_experts_per_tok})")

    expert_cache = None  # may remain None when --no-expert-cache

    # Try to load packed expert files (1 contiguous read per expert, 1.9x faster I/O)
    packed_layout, packed_fds, packed_layers = load_packed_experts(model_path)

    # mmap packed layer files for zero-copy access (60 GB/s when page-cached!)
    import mmap as _mmap_mod
    packed_mmaps = {}
    if packed_fds:
        for layer_idx, fd in packed_fds.items():
            packed_mmaps[layer_idx] = _mmap_mod.mmap(fd, 0, prot=_mmap_mod.PROT_READ)

    # Precompute packed component lookup for hot path
    _packed_comps = None
    if packed_layout is not None:
        _packed_comps = []
        for comp in packed_layout["components"]:
            parts = comp["name"].split(".")
            np_dtype, _ = _PREAD_NP_DTYPE[comp["dtype"]]
            is_bf16 = comp["dtype"] == "BF16"
            _packed_comps.append((parts[0], parts[1], comp["offset"], comp["size"],
                                  np_dtype, tuple(comp["shape"]), is_bf16))

    if no_expert_cache:
        print(f"  [no-expert-cache] Skipping ExpertCache — OS page cache handles all expert reads.")
        print(f"  [no-expert-cache] All {active_experts} active experts read from safetensors per layer per token.")

    # CPU-side numpy cache: stores raw pread bytes in Python heap (not Metal heap).
    # On hit: memcpy bytes -> mx.array (~0.11ms). On miss: pread from packed file (~0.7ms).
    numpy_cache = None
    numpy_cache_max_entries = 0
    numpy_cache_hits = 0
    numpy_cache_misses = 0
    if numpy_cache_gb > 0 and no_expert_cache and packed_layout is not None:
        expert_size = packed_layout["expert_size"]  # ~7.08MB per expert
        numpy_cache_max_entries = int(numpy_cache_gb * 1024**3 / expert_size)
        numpy_cache = OrderedDict()
        print(f"  [numpy-cache] CPU-side LRU: {numpy_cache_gb:.1f}GB = {numpy_cache_max_entries} entries "
              f"({expert_size/1e6:.2f}MB/entry). Stores raw bytes in Python heap, not Metal.")
    elif numpy_cache_gb > 0:
        if not no_expert_cache:
            print(f"  [numpy-cache] DISABLED — requires --no-expert-cache")
        elif packed_layout is None:
            print(f"  [numpy-cache] DISABLED — requires packed expert files")

    if not no_expert_cache:
        cache_entries = num_layers * active_experts * 8

        # Estimate per-entry bytes to enforce memory cap
        hidden = lm.args.hidden_size
        intermediate = lm.args.moe_intermediate_size
        # 3 projections (gate, up, down) each with weight(U32,4bit) + scales(BF16) + biases(BF16)
        # Per projection: intermediate*hidden/2 (weight) + intermediate*hidden/32 (scales+biases)
        # = intermediate*hidden*9/16 per projection, x3 projections
        per_expert_bytes = 3 * intermediate * hidden * 9 // 16
        max_cache_gb = cache_gb
        # Memory safety: cap total DRAM = non_expert (~5GB) + cache; leave ~5GB for OS on 48GB machine
        if max_cache_gb + 5 > 43:
            adjusted = 43.0 - 5.0
            print(f"[cache] WARNING: cache_gb={max_cache_gb:.1f} + non-expert (~5GB) exceeds 43GB safety limit. "
                  f"Reducing cache to {adjusted:.1f}GB")
            max_cache_gb = adjusted
        max_entries_by_mem = int(max_cache_gb * 1e9 / per_expert_bytes) if per_expert_bytes > 0 else cache_entries
        if cache_entries > max_entries_by_mem:
            print(f"[cache] Capping entries from {cache_entries} to {max_entries_by_mem} "
                  f"(~{max_cache_gb:.0f}GB limit, {per_expert_bytes/1e6:.1f}MB/entry)")
            cache_entries = max_entries_by_mem
        elif max_entries_by_mem > cache_entries:
            # When --cache-gb allows more than the heuristic, use full memory budget
            print(f"[cache] Expanding entries from {cache_entries} to {max_entries_by_mem} "
                  f"(~{max_cache_gb:.0f}GB budget, {per_expert_bytes/1e6:.1f}MB/entry)")
            cache_entries = max_entries_by_mem

        expert_cache = ExpertCache(max_entries=cache_entries)

        # === Pre-load hot experts if requested ===
        if preload_topk > 0:
            preload_hot_experts(expert_cache, weight_index, model_path, num_layers,
                                preload_topk, header_cache,
                                use_pread=use_pread, pread_index=pread_index,
                                pread_fds=pread_fds)

    # === Cache quantization parameters from model config (read once) ===
    # These are needed by compute_moe_direct for mx.gather_qmm calls.
    with open(model_path / "config.json") as f:
        _cfg = json.load(f)
    _qcfg = _cfg.get("quantization", _cfg.get("quantization_config", {}))
    qparams = {
        "group_size": _qcfg.get("group_size", 64),
        "bits": _qcfg.get("bits", 4),
        "mode": _qcfg.get("mode", "affine"),
    }
    del _cfg, _qcfg

    if batch_experts:
        print(f"  [batch-experts] Fused eval mode: 2 syncs/layer (was 4). "
              f"Eliminates {num_layers * 2} Metal syncs/token.")

    # Validate batch_layers prerequisites
    use_batch_layers = (batch_layers > 0 and no_expert_cache and batch_experts
                        and _packed_comps is not None and packed_fds)
    if batch_layers > 0 and not use_batch_layers:
        missing = []
        if not no_expert_cache:
            missing.append("--no-expert-cache")
        if not batch_experts:
            missing.append("--batch-experts")
        if _packed_comps is None or not packed_fds:
            missing.append("packed expert files")
        print(f"  [batch-layers] DISABLED — requires: {', '.join(missing)}")
    elif use_batch_layers:
        n_batches = (num_layers + batch_layers - 1) // batch_layers
        saved_evals = num_layers - n_batches
        print(f"  [batch-layers] N={batch_layers}: {n_batches} batched evals instead of {num_layers}. "
              f"Saves ~{saved_evals} Metal syncs/token. Routing approximate for layers 2-{batch_layers} in each batch.")

    # Validate async_pipeline prerequisites
    use_async_pipeline = (async_pipeline and no_expert_cache and batch_experts
                          and _packed_comps is not None and packed_fds)
    if async_pipeline and not use_async_pipeline:
        missing = []
        if not no_expert_cache:
            missing.append("--no-expert-cache")
        if not batch_experts:
            missing.append("--batch-experts")
        if _packed_comps is None or not packed_fds:
            missing.append("packed expert files")
        print(f"  [async-pipeline] DISABLED — requires: {', '.join(missing)}")
    elif use_async_pipeline:
        print(f"  [async-pipeline] Enabled: overlapping GPU routing eval with CPU I/O processing.")

    # === I/O instrumentation counters (per-token, reset each token) ===
    io_stats_history = []  # list of per-token dicts

    # === Profiling accumulators (only allocated when --profile is set) ===
    if profile:
        # Cumulative totals across all tokens (milliseconds)
        prof_totals = {
            "routing_ms": 0.0,
            "cache_lookup_ms": 0.0,
            "io_ms": 0.0,
            "compute_ms": 0.0,
            "eval_sync_ms": 0.0,
            "python_overhead_ms": 0.0,
        }
        # Per-layer detail: layer_idx -> {hits, misses, io_ms, bytes_read, eval_sync_ms}
        prof_per_layer = defaultdict(lambda: {
            "hits": 0, "misses": 0, "io_ms": 0.0, "bytes_read": 0,
            "eval_sync_ms": 0.0, "compute_ms": 0.0, "routing_ms": 0.0,
        })
        prof_token_count = 0  # tokens profiled (excludes prompt token 0)

    # === Online expert pinning state ===
    # When pin_experts > 0, we track expert activations during warmup, then pin the
    # top-K most frequently used experts per layer as persistent mx.arrays in GPU memory.
    # Pinned experts skip disk I/O entirely — they're already materialized in Metal.
    pinned_experts = {}  # {layer_idx: {expert_idx: {(proj, attr): mx.array}}}
    pin_counts = None  # [num_layers, num_experts] activation counter (during warmup)
    pin_phase = "disabled"  # "disabled" | "warmup" | "active"
    pin_total_hits = 0
    pin_total_lookups = 0
    if pin_experts > 0:
        num_experts_total = lm.args.num_experts
        pin_counts = np.zeros((num_layers, num_experts_total), dtype=np.int32)
        pin_phase = "warmup"
        print(f"  [pin] Online expert pinning enabled: warmup={pin_experts} tokens, "
              f"topk={pin_topk}/layer, num_experts={num_experts_total}")

    _io_pool = ThreadPoolExecutor(max_workers=8)  # reuse across tokens+layers
    _routing_stream = mx.new_stream(mx.default_device())  # separate Metal stream for routing

    for token_idx in range(max_tokens):
        t_token_start = time.perf_counter() if profile else time.time()

        # Per-token I/O counters
        token_io_bytes = 0
        token_io_seeks = 0
        token_io_time = 0.0
        token_array_time = 0.0
        token_moe_compute_time = 0.0
        # Reset per-token pin I/O miss counter
        if pin_phase == "active":
            generate_offload_selective._last_io_misses = 0

        # --- Embed ---
        h = text_model.embed_tokens(input_ids)
        mx.eval(h)

        # --- Create masks ---
        fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

        layer_timings = []

        # --- Async pipeline state (reset per token) ---
        # When use_async_pipeline is active, expert I/O for layer N is dispatched
        # during layer N's routing eval, and the I/O results are processed (converted
        # to mx.arrays, MoE computed) during layer N+1's routing eval. This overlaps
        # GPU routing computation with CPU I/O processing.
        _ap_prev_futures = None       # dict {eidx: Future} from previous layer's pread dispatch
        _ap_prev_unpinned = None      # list of unpinned expert indices from previous layer
        _ap_prev_np_cached = {}       # dict {eidx: raw_bytes} numpy-cache hits from previous layer
        _ap_prev_pinned_attrs = None  # dict {eidx: attrs} for pinned experts from previous layer
        _ap_prev_unique_list = None   # full unique_list from previous layer
        _ap_prev_layer_data = None    # (prev_i, prev_layer, remapped_inds, scores, h_mid, h_post)
        _ap_prev_io_bytes = 0         # bytes read by previous layer's I/O

        # --- Batch-layers state (reset per token) ---
        _bl_done = set()  # layers already processed by batch-layers path

        # --- Per-layer: selective load, compute (clearing deferred to end of token) ---
        for i in range(num_layers):
            # Layer skipping: skip every Nth layer in the middle to reduce compute+I/O
            if skip_layers > 0 and 10 <= i < num_layers - 10 and i % skip_layers != 0:
                continue

            # Skip layers already processed by batch-layers path
            if i in _bl_done:
                continue

            # ====== BATCH-LAYERS PATH ======
            # When batch_layers > 0, process N layers at once:
            #   Phase 1: Build attention+routing for N layers lazily (approximate h)
            #   Phase 2: Single mx.eval for all N routing decisions
            #   Phase 3: Batch-read experts for all N layers
            #   Phase 4: Compute MoE for all N layers with correct h chain
            #
            # Routing for layers 2-N in each batch uses h_mid (no expert contribution
            # from prior layers in the batch), making routing approximate. Quality impact
            # is ~7.5% wrong expert selections on layers 2-N of each batch.
            if use_batch_layers and i in packed_layers:
                batch_end = min(i + batch_layers, num_layers)
                # Collect layers in this batch (respecting skip_layers)
                batch_layer_indices = []
                for bi in range(i, batch_end):
                    if skip_layers > 0 and 10 <= bi < num_layers - 10 and bi % skip_layers != 0:
                        continue
                    if bi not in packed_layers:
                        break  # stop batch at first non-packed layer
                    batch_layer_indices.append(bi)

                if len(batch_layer_indices) >= 2:
                    # Worth batching (2+ layers)
                    t_batch_start = time.perf_counter() if profile else time.time()

                    # Phase 1: Build attention + routing for all batch layers lazily.
                    # h propagates as h_mid (attention output, no expert contribution).
                    # This makes routing approximate for layers after the first.
                    h_pre_batch = h  # save for Phase 4
                    all_batch_inds = []
                    all_batch_scores = []
                    all_batch_h_mids = []
                    all_batch_h_posts = []
                    all_batch_k = []

                    for bi in batch_layer_indices:
                        bl = layers[bi]
                        bc = cache[bi]
                        x_normed = bl.input_layernorm(h)
                        mask = ssm_mask if bl.is_linear else fa_mask
                        if bl.is_linear:
                            r = bl.linear_attn(x_normed, mask, bc)
                        else:
                            r = bl.self_attn(x_normed, mask, bc)
                        h_mid_b = h + r
                        h_post_b = bl.post_attention_layernorm(h_mid_b)
                        gates_b = bl.mlp.gate(h_post_b)
                        gates_b = mx.softmax(gates_b, axis=-1, precise=True)
                        k_b = bl.mlp.top_k if top_k_override <= 0 else min(bl.mlp.top_k, top_k_override)
                        inds_b = mx.argpartition(gates_b, kth=-k_b, axis=-1)[..., -k_b:]
                        scores_b = mx.take_along_axis(gates_b, inds_b, axis=-1)
                        scores_b = scores_b / scores_b.sum(axis=-1, keepdims=True)

                        all_batch_inds.append(inds_b)
                        all_batch_scores.append(scores_b)
                        all_batch_h_mids.append(h_mid_b)
                        all_batch_h_posts.append(h_post_b)
                        all_batch_k.append(k_b)

                        # Propagate approximate h (no expert contribution)
                        h = h_mid_b

                    # Phase 2: Single eval for all N routing decisions
                    mx.eval(*all_batch_inds)

                    t_batch_routing_done = time.perf_counter() if profile else time.time()

                    # Phase 3: Batch-read experts for all N layers
                    all_batch_expert_tensors = []
                    all_batch_remapped_inds = []
                    all_batch_unique_lists = []

                    for idx, bi in enumerate(batch_layer_indices):
                        bl = layers[bi]
                        inds_b = all_batch_inds[idx]

                        # Extract routing indices
                        inds_np = np.array(inds_b.tolist())
                        unique_experts = np.unique(inds_np)
                        num_unique = len(unique_experts)
                        unique_list = unique_experts.tolist()

                        # Build remap table
                        remap = np.zeros(bl.mlp.num_experts, dtype=np.int32)
                        remap[unique_experts] = np.arange(num_unique)
                        remapped_inds_b = mx.array(remap[inds_np])

                        # Track warmup routing
                        if pin_phase == "warmup":
                            for eidx in unique_list:
                                pin_counts[bi, eidx] += 1

                        # Split pinned vs unpinned
                        layer_pinned = pinned_experts.get(bi, {})
                        if pin_phase == "active" and layer_pinned:
                            pinned_list = [idx_e for idx_e in unique_list if idx_e in layer_pinned]
                            unpinned_list = [idx_e for idx_e in unique_list if idx_e not in layer_pinned]
                            pin_total_hits += len(pinned_list)
                            pin_total_lookups += num_unique
                        else:
                            pinned_list = []
                            unpinned_list = unique_list

                        # Read experts (packed path)
                        all_expert_attrs = {}
                        for idx_e in pinned_list:
                            all_expert_attrs[idx_e] = layer_pinned[idx_e]

                        if unpinned_list:
                            packed_fd = packed_fds[bi]
                            es = packed_layout["expert_size"]

                            # Split unpinned into numpy-cache hits vs disk reads
                            if numpy_cache is not None:
                                np_cached_list = []
                                np_uncached_list = []
                                for eidx in unpinned_list:
                                    nc_key = (bi, eidx)
                                    if nc_key in numpy_cache:
                                        numpy_cache.move_to_end(nc_key)
                                        np_cached_list.append((eidx, numpy_cache[nc_key]))
                                        numpy_cache_hits += 1
                                    else:
                                        np_uncached_list.append(eidx)
                                        numpy_cache_misses += 1
                            else:
                                np_cached_list = []
                                np_uncached_list = unpinned_list

                            # Numpy-cache hits: convert raw bytes -> mx.arrays
                            for eidx, raw in np_cached_list:
                                mv = memoryview(raw)
                                attrs = {}
                                for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                                    arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                                    if is_bf16:
                                        arr = arr.view(mx.bfloat16)
                                    attrs[(proj, attr)] = arr
                                all_expert_attrs[eidx] = attrs

                            # Numpy-cache misses: pread from packed file
                            if np_uncached_list:
                                read_futures = {eidx: _io_pool.submit(os.pread, packed_fd, es, eidx * es)
                                                for eidx in np_uncached_list}
                                for eidx in np_uncached_list:
                                    raw = read_futures[eidx].result()
                                    if numpy_cache is not None:
                                        nc_key = (bi, eidx)
                                        numpy_cache[nc_key] = raw
                                        while len(numpy_cache) > numpy_cache_max_entries:
                                            numpy_cache.popitem(last=False)
                                    mv = memoryview(raw)
                                    attrs = {}
                                    for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                                        arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                                        if is_bf16:
                                            arr = arr.view(mx.bfloat16)
                                        attrs[(proj, attr)] = arr
                                    all_expert_attrs[eidx] = attrs
                                    token_io_bytes += es
                                    token_io_seeks += 1

                        if pin_phase == "active":
                            generate_offload_selective._last_io_misses = getattr(
                                generate_offload_selective, '_last_io_misses', 0) + len(unpinned_list)

                        # Assemble expert tensors
                        expert_tensors_b = {}
                        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                            for attr_name in ["weight", "scales", "biases"]:
                                slices = []
                                for idx_e in unique_list:
                                    arr = all_expert_attrs[idx_e].get((proj_name, attr_name))
                                    if arr is not None:
                                        slices.append(arr)
                                    else:
                                        raise RuntimeError(
                                            f"Batch-layers read miss: layer={bi} expert={idx_e} "
                                            f"{proj_name}.{attr_name} not found"
                                        )
                                if slices:
                                    expert_tensors_b[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
                        del all_expert_attrs

                        all_batch_expert_tensors.append(expert_tensors_b)
                        all_batch_remapped_inds.append(remapped_inds_b)
                        all_batch_unique_lists.append(unique_list)

                    t_batch_io_done = time.perf_counter() if profile else time.time()

                    # Phase 4: Compute MoE for all N layers using pre-computed routing.
                    # We use the h_mid/h_post from Phase 1 directly. For the first layer
                    # in the batch these are exact; for subsequent layers they're approximate
                    # (based on h without expert contributions from prior batch layers).
                    # This is the core tradeoff: we accept approximate expert inputs to avoid
                    # re-running attention (which would double-write the KV cache). The I/O
                    # savings from batched routing far outweigh the quality cost.
                    for idx, bi in enumerate(batch_layer_indices):
                        bl = layers[bi]
                        h_mid_b = all_batch_h_mids[idx]
                        h_post_b = all_batch_h_posts[idx]

                        # Compute MoE with pre-loaded expert tensors
                        y = compute_moe_direct(
                            h_post_b, all_batch_remapped_inds[idx], all_batch_expert_tensors[idx],
                            group_size=qparams["group_size"],
                            bits=qparams["bits"],
                            mode=qparams["mode"],
                        )
                        y = (y * all_batch_scores[idx][..., None]).sum(axis=-2)

                        # Shared expert
                        shared_y = bl.mlp.shared_expert(h_post_b)
                        shared_y = mx.sigmoid(bl.mlp.shared_expert_gate(h_post_b)) * shared_y
                        y = y + shared_y

                        h = h_mid_b + y

                        # Release this layer's expert tensors (set to None, don't del from list)
                        all_batch_expert_tensors[idx] = None

                    # Eval final h after all batch layers
                    mx.eval(h)

                    t_batch_compute_done = time.perf_counter() if profile else time.time()

                    # Record timing for all layers in the batch
                    n_bl = len(batch_layer_indices)
                    batch_routing_ms = (t_batch_routing_done - t_batch_start) * 1000
                    batch_io_ms = (t_batch_io_done - t_batch_routing_done) * 1000
                    batch_compute_ms = (t_batch_compute_done - t_batch_io_done) * 1000
                    batch_total_ms = (t_batch_compute_done - t_batch_start) * 1000
                    for idx, bi in enumerate(batch_layer_indices):
                        # Distribute timing equally across layers in the batch (for profiling)
                        per_layer_ms = batch_total_ms / n_bl if n_bl > 0 else 0.0
                        layer_timings.append({
                            "layer": bi,
                            "is_linear": layers[bi].is_linear,
                            "attn_router_ms": batch_routing_ms / n_bl,
                            "expert_ms": (batch_io_ms + batch_compute_ms) / n_bl,
                            "clear_ms": 0.0,
                            "load_ms": batch_io_ms / n_bl,
                            "compute_ms": batch_compute_ms / n_bl,
                        })

                        if profile and token_idx > 0:
                            pld = prof_per_layer[bi]
                            pld["routing_ms"] += batch_routing_ms / n_bl
                            pld["io_ms"] += batch_io_ms / n_bl
                            pld["compute_ms"] += batch_compute_ms / n_bl
                            pld["eval_sync_ms"] += batch_compute_ms / n_bl
                            pld["hits"] += 0
                            pld["misses"] += len(all_batch_unique_lists[idx])

                    if profile and token_idx > 0:
                        prof_totals["routing_ms"] += batch_routing_ms
                        prof_totals["io_ms"] += batch_io_ms
                        prof_totals["compute_ms"] += batch_compute_ms
                        prof_totals["eval_sync_ms"] += batch_compute_ms

                    token_moe_compute_time += batch_compute_ms / 1000.0

                    # Mark all batch layers as done
                    _bl_done.update(batch_layer_indices)

                    # Clean up batch state
                    del all_batch_inds, all_batch_scores, all_batch_h_mids, all_batch_h_posts
                    del all_batch_remapped_inds, all_batch_unique_lists
                    del all_batch_expert_tensors

                    continue  # skip to next layer after the batch
                # else: single layer in batch, fall through to normal path

            if profile:
                t_layer_start = time.perf_counter()

            layer = layers[i]
            c = cache[i]

            entries = weight_index.get(i, [])
            _, expert_entries = split_layer_entries(entries)

            # ====== Phase 2: Run attention + router (weights already resident) ======
            if profile:
                t_attn = time.perf_counter()
            else:
                t_attn = time.time()

            # For async pipeline with pending prev-layer I/O: defer attention until
            # after we process the prev layer's MoE (which updates h). The attention
            # code will run inside the async pipeline block below.
            _ap_defer_attn = (use_async_pipeline and i in packed_layers
                              and _ap_prev_futures is not None)

            if not _ap_defer_attn:
                x_normed = layer.input_layernorm(h)
                mask = ssm_mask if layer.is_linear else fa_mask
                if layer.is_linear:
                    r = layer.linear_attn(x_normed, mask, c)
                else:
                    r = layer.self_attn(x_normed, mask, c)
                h_mid = h + r

                # Run router to discover which experts are needed
                h_post = layer.post_attention_layernorm(h_mid)
                gates = layer.mlp.gate(h_post)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = layer.mlp.top_k if top_k_override <= 0 else min(layer.mlp.top_k, top_k_override)
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)

            if use_async_pipeline and i in packed_layers:
                # ====== ASYNC PIPELINE PATH ======
                # Overlap disk I/O with GPU computation across layers.
                # Pattern:
                #   A. Process previous layer's I/O results -> compute MoE -> update h
                #      (I/O futures were dispatched at end of previous iteration and have
                #       been completing in background during the time between iterations)
                #   B. Build attention + routing graph for THIS layer (uses correct h)
                #      [already done above -- lines before this if-block]
                #   C. Dispatch routing to GPU (non-blocking)
                #   D. Sync routing (GPU computes while we set up the dispatch below)
                #   E. Extract indices, dispatch I/O for THIS layer (async)
                #
                # The speedup comes from disk I/O (pread futures from step E) overlapping
                # with the next layer's attention + routing GPU computation.

                # Step A: Process PREVIOUS layer's I/O (must complete before attention,
                # but attention graph was already built above using lazy h — the MoE
                # computation here updates h which will be correct for the NEXT layer)
                if _ap_prev_futures is not None:
                    prev_i, prev_layer_obj, prev_remapped_inds, prev_scores, prev_h_mid, prev_h_post = _ap_prev_layer_data

                    # Convert previous layer's raw pread data -> mx.arrays
                    prev_all_expert_attrs = dict(_ap_prev_pinned_attrs) if _ap_prev_pinned_attrs else {}
                    es = packed_layout["expert_size"]

                    # Process numpy-cache hits first (raw bytes, no pread needed)
                    if _ap_prev_np_cached:
                        for eidx, raw in _ap_prev_np_cached.items():
                            mv = memoryview(raw)
                            attrs = {}
                            for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                                arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                                if is_bf16:
                                    arr = arr.view(mx.bfloat16)
                                attrs[(proj, attr)] = arr
                            prev_all_expert_attrs[eidx] = attrs

                    # Process pread futures (disk reads) and store in numpy cache
                    for eidx in _ap_prev_unpinned:
                        raw = _ap_prev_futures[eidx].result()
                        if numpy_cache is not None:
                            nc_key = (prev_i, eidx)
                            numpy_cache[nc_key] = raw
                            while len(numpy_cache) > numpy_cache_max_entries:
                                numpy_cache.popitem(last=False)
                        mv = memoryview(raw)
                        attrs = {}
                        for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                            arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                            if is_bf16:
                                arr = arr.view(mx.bfloat16)
                            attrs[(proj, attr)] = arr
                        prev_all_expert_attrs[eidx] = attrs
                        token_io_bytes += es
                        token_io_seeks += 1

                    # Assemble expert_tensors for previous layer
                    prev_expert_tensors = {}
                    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                        for attr_name in ["weight", "scales", "biases"]:
                            slices = []
                            for idx in _ap_prev_unique_list:
                                arr = prev_all_expert_attrs[idx].get((proj_name, attr_name))
                                if arr is not None:
                                    slices.append(arr)
                                else:
                                    raise RuntimeError(
                                        f"Async pipeline: layer={prev_i} expert={idx} "
                                        f"{proj_name}.{attr_name} not found"
                                    )
                            if slices:
                                prev_expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
                    del prev_all_expert_attrs

                    # Compute MoE for previous layer (lazy graph build)
                    prev_y = compute_moe_direct(
                        prev_h_post, prev_remapped_inds, prev_expert_tensors,
                        group_size=qparams["group_size"],
                        bits=qparams["bits"],
                        mode=qparams["mode"],
                    )
                    prev_y = (prev_y * prev_scores[..., None]).sum(axis=-2)
                    prev_shared_y = prev_layer_obj.mlp.shared_expert(prev_h_post)
                    prev_shared_y = mx.sigmoid(prev_layer_obj.mlp.shared_expert_gate(prev_h_post)) * prev_shared_y
                    prev_y = prev_y + prev_shared_y

                    # Update h with prev layer MoE output, eval so attention below
                    # uses the correct hidden state (I/O futures overlapped with previous
                    # layer's GPU work, so the pread data should be ready by now).
                    h = prev_h_mid + prev_y
                    mx.eval(h)
                    del prev_expert_tensors

                # Build attention + routing for THIS layer (with correct h).
                # On first pipeline layer (no prev_futures), attention was already built
                # above. On subsequent layers, it was deferred until prev MoE completed.
                if _ap_defer_attn:
                    x_normed = layer.input_layernorm(h)
                    mask = ssm_mask if layer.is_linear else fa_mask
                    if layer.is_linear:
                        r = layer.linear_attn(x_normed, mask, c)
                    else:
                        r = layer.self_attn(x_normed, mask, c)
                    h_mid = h + r
                    h_post = layer.post_attention_layernorm(h_mid)
                    gates = layer.mlp.gate(h_post)
                    gates = mx.softmax(gates, axis=-1, precise=True)
                    k = layer.mlp.top_k if top_k_override <= 0 else min(layer.mlp.top_k, top_k_override)
                    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    scores = scores / scores.sum(axis=-1, keepdims=True)

                # Step C: Dispatch routing to GPU (non-blocking)
                mx.async_eval(inds)

                # Step D: Sync routing
                mx.eval(inds)
                attn_router_time = time.time() - t_attn
                if profile:
                    t_attn_eval_done = time.perf_counter()
                    layer_attn_eval_ms = (t_attn_eval_done - t_attn) * 1000
                    layer_route_eval_ms = 0.0
                    layer_routing_ms = layer_attn_eval_ms

                # Step E: Extract routing indices + dispatch I/O
                if profile:
                    t_expert = time.perf_counter()
                    layer_cache_lookup_ms = 0.0
                    t_io_start = time.perf_counter()
                    layer_io_bytes = 0
                else:
                    t_expert = time.time()

                inds_np = np.array(inds.tolist())
                unique_experts = np.unique(inds_np)
                num_unique = len(unique_experts)
                unique_list = unique_experts.tolist()

                remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
                remap[unique_experts] = np.arange(num_unique)
                remapped_inds = mx.array(remap[inds_np])

                if pin_phase == "warmup":
                    for eidx in unique_list:
                        pin_counts[i, eidx] += 1

                # Split pinned vs unpinned
                layer_pinned = pinned_experts.get(i, {})
                if pin_phase == "active" and layer_pinned:
                    pinned_list = [idx for idx in unique_list if idx in layer_pinned]
                    unpinned_list = [idx for idx in unique_list if idx not in layer_pinned]
                    layer_cache_hits = len(pinned_list)
                    layer_cache_misses = len(unpinned_list)
                    pin_total_hits += len(pinned_list)
                    pin_total_lookups += num_unique
                else:
                    pinned_list = []
                    unpinned_list = unique_list
                    layer_cache_hits = 0
                    layer_cache_misses = num_unique

                pinned_attrs = {}
                for idx in pinned_list:
                    pinned_attrs[idx] = layer_pinned[idx]

                if pin_phase == "active":
                    generate_offload_selective._last_io_misses = getattr(
                        generate_offload_selective, '_last_io_misses', 0) + len(unpinned_list)

                # Dispatch async I/O for THIS layer (pread futures start reading from SSD
                # immediately; they'll complete during the next layer's attention+routing)
                packed_fd = packed_fds[i]
                es = packed_layout["expert_size"]

                # Split unpinned into numpy-cache hits vs disk reads
                _ap_np_cached = {}
                if numpy_cache is not None:
                    ap_uncached_list = []
                    for eidx in unpinned_list:
                        nc_key = (i, eidx)
                        if nc_key in numpy_cache:
                            numpy_cache.move_to_end(nc_key)
                            _ap_np_cached[eidx] = numpy_cache[nc_key]
                            numpy_cache_hits += 1
                        else:
                            ap_uncached_list.append(eidx)
                            numpy_cache_misses += 1
                else:
                    ap_uncached_list = unpinned_list

                current_futures = {eidx: _io_pool.submit(os.pread, packed_fd, es, eidx * es)
                                   for eidx in ap_uncached_list}

                if profile:
                    t_io_done = time.perf_counter()
                    layer_io_ms = (t_io_done - t_io_start) * 1000

                # Save state for next iteration
                _ap_prev_futures = current_futures
                _ap_prev_unpinned = ap_uncached_list  # only disk-read experts
                _ap_prev_np_cached = _ap_np_cached    # numpy-cache hits (raw bytes)
                _ap_prev_pinned_attrs = pinned_attrs
                _ap_prev_unique_list = unique_list
                _ap_prev_layer_data = (i, layer, remapped_inds, scores, h_mid, h_post)
                _ap_prev_io_bytes = len(ap_uncached_list) * es

                # If last layer: process THIS layer's I/O immediately (no next iteration)
                if i == num_layers - 1:
                    all_expert_attrs = dict(pinned_attrs)

                    # Process numpy-cache hits (no pread needed)
                    for eidx, raw in _ap_np_cached.items():
                        mv = memoryview(raw)
                        attrs = {}
                        for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                            arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                            if is_bf16:
                                arr = arr.view(mx.bfloat16)
                            attrs[(proj, attr)] = arr
                        all_expert_attrs[eidx] = attrs

                    # Process pread futures (disk reads) and store in numpy cache
                    for eidx in ap_uncached_list:
                        raw = current_futures[eidx].result()
                        if numpy_cache is not None:
                            nc_key = (i, eidx)
                            numpy_cache[nc_key] = raw
                            while len(numpy_cache) > numpy_cache_max_entries:
                                numpy_cache.popitem(last=False)
                        mv = memoryview(raw)
                        attrs = {}
                        for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                            arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                            if is_bf16:
                                arr = arr.view(mx.bfloat16)
                            attrs[(proj, attr)] = arr
                        all_expert_attrs[eidx] = attrs
                        token_io_bytes += es
                        token_io_seeks += 1
                        if profile:
                            layer_io_bytes += es

                    expert_tensors = {}
                    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                        for attr_name in ["weight", "scales", "biases"]:
                            slices = []
                            for idx in unique_list:
                                arr = all_expert_attrs[idx].get((proj_name, attr_name))
                                if arr is not None:
                                    slices.append(arr)
                                else:
                                    raise RuntimeError(
                                        f"Async pipeline last-layer: layer={i} expert={idx} "
                                        f"{proj_name}.{attr_name} not found"
                                    )
                            if slices:
                                expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
                    del all_expert_attrs

                    t_moe = time.perf_counter() if profile else time.time()
                    y = compute_moe_direct(
                        h_post, remapped_inds, expert_tensors,
                        group_size=qparams["group_size"],
                        bits=qparams["bits"],
                        mode=qparams["mode"],
                    )
                    y = (y * scores[..., None]).sum(axis=-2)
                    shared_y = layer.mlp.shared_expert(h_post)
                    shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
                    y = y + shared_y
                    h = h_mid + y
                    mx.eval(h)
                    token_moe_compute_time += (time.perf_counter() if profile else time.time()) - t_moe
                    del expert_tensors

                    # Clear pipeline state
                    _ap_prev_futures = None

                # Set profiling variables expected by downstream code
                if profile:
                    layer_expert_eval_ms = 0.0
                    layer_compute_eval_ms = 0.0
                    layer_compute_ms = 0.0
                    # For non-last layers, MoE is deferred to next iteration (compute_ms = 0).
                    # For last layer, MoE was computed synchronously above.

            elif batch_experts:
                # Fused eval: evaluate attention + routing in a single Metal sync.
                # In the default path, h_mid and inds are eval'd separately (2 syncs).
                # Fusing them halves the sync overhead for this phase.
                if profile:
                    t_attn_eval = time.perf_counter()
                    mx.eval(inds)
                    t_attn_eval_done = time.perf_counter()
                    layer_attn_eval_ms = (t_attn_eval_done - t_attn_eval) * 1000
                    layer_route_eval_ms = 0.0
                    layer_routing_ms = (t_attn_eval - t_attn) * 1000
                else:
                    mx.eval(inds)
                    attn_router_time = time.time() - t_attn
            else:
                if profile:
                    # Separate the mx.eval sync time from the compute setup
                    t_attn_eval = time.perf_counter()
                    mx.eval(h_mid)
                    t_attn_eval_done = time.perf_counter()
                    t_route_eval = time.perf_counter()
                    mx.eval(inds)
                    t_route_eval_done = time.perf_counter()
                    layer_attn_eval_ms = (t_attn_eval_done - t_attn_eval) * 1000
                    layer_route_eval_ms = (t_route_eval_done - t_route_eval) * 1000
                    # Routing = total attn+route time MINUS the eval sync portions
                    layer_routing_ms = (t_route_eval_done - t_attn) * 1000 - layer_attn_eval_ms - layer_route_eval_ms
                else:
                    mx.eval(h_mid)
                    mx.eval(inds)
                    attn_router_time = time.time() - t_attn

            # ====== Phase 3: Selective expert loading + compute ======
            _ap_handled = use_async_pipeline and i in packed_layers  # set by async pipeline path above

            if not _ap_handled:

              if profile:
                  t_expert = time.perf_counter()
              else:
                  t_expert = time.time()

              # Extract UNIQUE expert IDs across all positions
              inds_np = np.array(inds.tolist())
              unique_experts = np.unique(inds_np)
              num_unique = len(unique_experts)
              unique_list = unique_experts.tolist()  # Python list for indexing

              # Build remap table: original expert ID -> index in sliced tensor
              remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
              remap[unique_experts] = np.arange(num_unique)
              remapped_inds = mx.array(remap[inds_np])

              # Build a map from expert tensor name -> filepath for this layer
              expert_file_map = {}
              for name, filepath in expert_entries:
                  expert_file_map[name] = filepath

              # --- Online expert pinning: track activations during warmup ---
              if pin_phase == "warmup":
                  for eidx in unique_list:
                      pin_counts[i, eidx] += 1

              # --- Expert loading: no-cache path vs LRU cache path ---
              if no_expert_cache:
                # ---- NO-CACHE PATH ----
                # Read ALL active experts directly from safetensors (mmap).
                # OS page cache handles caching at the VM level.
                # When pin_phase == "active", pinned experts skip I/O entirely.
                if profile:
                    layer_cache_lookup_ms = 0.0  # no cache to look up
                    t_io_start = time.perf_counter()
                    layer_io_bytes = 0

                # Split experts into pinned vs unpinned
                layer_pinned = pinned_experts.get(i, {})
                if pin_phase == "active" and layer_pinned:
                    pinned_list = [idx for idx in unique_list if idx in layer_pinned]
                    unpinned_list = [idx for idx in unique_list if idx not in layer_pinned]
                    layer_cache_hits = len(pinned_list)
                    layer_cache_misses = len(unpinned_list)
                    pin_total_hits += len(pinned_list)
                    pin_total_lookups += num_unique
                else:
                    pinned_list = []
                    unpinned_list = unique_list
                    layer_cache_hits = 0
                    layer_cache_misses = num_unique

                # Pinned experts: use pre-loaded weights (no I/O)
                all_expert_attrs = {}
                for idx in pinned_list:
                    all_expert_attrs[idx] = layer_pinned[idx]

                # Unpinned experts: read from disk
                if unpinned_list:
                    # Pre-populate header_cache for all expert files
                    for filepath in set(expert_file_map.values()):
                        if filepath not in header_cache:
                            header_cache[filepath] = parse_safetensors_header(filepath)

                    # Read only unpinned experts from disk
                    if _packed_comps is not None and i in packed_layers:
                        # PACKED PATH: pipelined I/O + conversion
                        # Submit all reads non-blocking, then process as
                        # each completes — conversion of expert N overlaps
                        # with pread of experts N+1..N+K (GIL released
                        # during pread syscall).
                        packed_fd = packed_fds[i]
                        es = packed_layout["expert_size"]

                        # Split unpinned into numpy-cache hits vs disk reads
                        if numpy_cache is not None:
                            np_cached_list = []
                            np_uncached_list = []
                            for eidx in unpinned_list:
                                nc_key = (i, eidx)
                                if nc_key in numpy_cache:
                                    numpy_cache.move_to_end(nc_key)  # LRU touch
                                    np_cached_list.append((eidx, numpy_cache[nc_key]))
                                    numpy_cache_hits += 1
                                else:
                                    np_uncached_list.append(eidx)
                                    numpy_cache_misses += 1
                        else:
                            np_cached_list = []
                            np_uncached_list = unpinned_list

                        # Numpy-cache hits: convert raw bytes -> mx.arrays (fast memcpy)
                        for eidx, raw in np_cached_list:
                            mv = memoryview(raw)
                            attrs = {}
                            for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                                arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                                if is_bf16:
                                    arr = arr.view(mx.bfloat16)
                                attrs[(proj, attr)] = arr
                            all_expert_attrs[eidx] = attrs

                        # Numpy-cache misses: pread from packed file
                        if np_uncached_list:
                            read_futures = {eidx: _io_pool.submit(os.pread, packed_fd, es, eidx * es)
                                            for eidx in np_uncached_list}
                            for eidx in np_uncached_list:
                                raw = read_futures[eidx].result()
                                # Store in numpy cache before converting
                                if numpy_cache is not None:
                                    nc_key = (i, eidx)
                                    numpy_cache[nc_key] = raw
                                    # LRU eviction
                                    while len(numpy_cache) > numpy_cache_max_entries:
                                        numpy_cache.popitem(last=False)
                                mv = memoryview(raw)
                                attrs = {}
                                for proj, attr, off, sz, npdtype, shape, is_bf16 in _packed_comps:
                                    arr = mx.array(np.frombuffer(mv[off:off+sz], dtype=npdtype).reshape(shape))
                                    if is_bf16:
                                        arr = arr.view(mx.bfloat16)
                                    attrs[(proj, attr)] = arr
                                all_expert_attrs[eidx] = attrs
                                token_io_bytes += es
                                token_io_seeks += 1
                                if profile:
                                    layer_io_bytes += es
                    else:
                        # SCATTERED PATH: 9 preads per expert from safetensors
                        num_unpinned = len(unpinned_list)
                        with ThreadPoolExecutor(max_workers=min(4, num_unpinned)) as executor:
                            futures = [
                                executor.submit(
                                    _read_single_expert_attrs,
                                    expert_idx, i, expert_file_map, header_cache
                                )
                                for expert_idx in unpinned_list
                            ]
                            for future in futures:
                                eidx, attrs, io_stats = future.result()
                                all_expert_attrs[eidx] = attrs
                                token_io_bytes += io_stats["bytes_read"]
                                token_io_seeks += io_stats["seek_count"]
                                token_io_time += io_stats["io_time_s"]
                                token_array_time += io_stats["array_time_s"]
                                if profile:
                                    layer_io_bytes += io_stats["bytes_read"]

                # Track I/O misses for per-token display
                if pin_phase == "active":
                    generate_offload_selective._last_io_misses = getattr(
                        generate_offload_selective, '_last_io_misses', 0) + len(unpinned_list)

                if profile:
                    t_io_done = time.perf_counter()
                    layer_io_ms = (t_io_done - t_io_start) * 1000

                # Assemble [num_unique, ...] weight tensors for compute_moe_direct
                expert_tensors = {}
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    for attr_name in ["weight", "scales", "biases"]:
                        slices = []
                        for idx in unique_list:
                            arr = all_expert_attrs[idx].get((proj_name, attr_name))
                            if arr is not None:
                                slices.append(arr)
                            else:
                                raise RuntimeError(
                                    f"No-cache read miss: layer={i} expert={idx} "
                                    f"{proj_name}.{attr_name} not found"
                                )
                        if slices:
                            expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
                del all_expert_attrs

              else:
                # ---- LRU CACHE PATH (original) ----
                # Determine which experts need disk reads.
                # When pin_phase == "active", pinned experts bypass cache entirely.
                if profile:
                    t_cache_lookup = time.perf_counter()

                uncached_list = []
                layer_cache_hits = 0
                layer_cache_misses = 0
                layer_pinned_c = pinned_experts.get(i, {})
                pin_active_here = (pin_phase == "active" and len(layer_pinned_c) > 0)

                # Split: pinned experts skip cache, remaining go through LRU cache
                if pin_active_here:
                    cache_check_list = [idx for idx in unique_list if idx not in layer_pinned_c]
                    pinned_hit_list = [idx for idx in unique_list if idx in layer_pinned_c]
                    layer_cache_hits += len(pinned_hit_list)
                    pin_total_hits += len(pinned_hit_list)
                    pin_total_lookups += num_unique
                else:
                    cache_check_list = unique_list

                # Protect all non-pinned experts needed for this layer from eviction
                expert_cache.protect([(i, idx) for idx in cache_check_list])
                for idx in cache_check_list:
                    if expert_cache.has_expert(i, idx):
                        expert_cache.record_hit()
                        expert_cache.touch(i, idx)
                        layer_cache_hits += 1
                    else:
                        expert_cache.record_miss()
                        uncached_list.append(idx)
                        layer_cache_misses += 1

                if profile:
                    t_cache_lookup_done = time.perf_counter()
                    layer_cache_lookup_ms = (t_cache_lookup_done - t_cache_lookup) * 1000

                # Read only uncached experts from disk (threaded: one expert per thread)
                if profile:
                    t_io_start = time.perf_counter()
                    layer_io_bytes = 0

                if uncached_list:
                    if use_cext:
                        # fast_expert_io C extension path (preadv + coalesced I/O)
                        import fast_expert_io
                        t_cext_io = time.time()
                        cext_results = fast_expert_io.batch_read(i, uncached_list)
                        cext_io_time = time.time() - t_cext_io

                        t_cext_arr = time.time()
                        cext_io_bytes = 0
                        for eidx, comp_dict in cext_results.items():
                            for comp_name, np_arr in comp_dict.items():
                                parts = comp_name.split(".")
                                proj_name = parts[0]
                                attr_name = parts[1]
                                cext_io_bytes += np_arr.nbytes

                                if np_arr.dtype == np.uint16:
                                    # BF16: stored as uint16, convert to bfloat16
                                    np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
                                    mx_arr = mx.array(np_f32).astype(mx.bfloat16)
                                else:
                                    mx_arr = mx.array(np_arr)
                                expert_cache.put_attr(i, eidx, proj_name, attr_name, mx_arr)
                        cext_arr_time = time.time() - t_cext_arr

                        token_io_bytes += cext_io_bytes
                        token_io_seeks += len(uncached_list) * 9  # 9 components per expert
                        token_io_time += cext_io_time
                        token_array_time += cext_arr_time
                        if profile:
                            layer_io_bytes += cext_io_bytes

                    elif use_pread and pread_index is not None:
                        # pread()-based expert loading (bypasses safetensors)
                        batch_results, io_stats = pread_expert_batch(
                            pread_index, pread_fds, i, uncached_list)
                        token_io_bytes += io_stats["bytes_read"]
                        token_io_seeks += io_stats["seek_count"]
                        token_io_time += io_stats["io_time_s"]
                        token_array_time += io_stats["array_time_s"]
                        if profile:
                            layer_io_bytes += io_stats["bytes_read"]
                        for eidx, attrs in batch_results.items():
                            for (proj_name, attr_name), arr in attrs.items():
                                expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
                    else:
                        # Original safetensors path
                        # Pre-populate header_cache for all expert files (thread-safe after this)
                        for filepath in set(expert_file_map.values()):
                            if filepath not in header_cache:
                                header_cache[filepath] = parse_safetensors_header(filepath)

                        # Parallel read: each thread reads all 9 attrs for one expert
                        with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                            futures = [
                                executor.submit(
                                    _read_single_expert_attrs,
                                    expert_idx, i, expert_file_map, header_cache
                                )
                                for expert_idx in uncached_list
                            ]
                            for future in futures:
                                eidx, attrs, io_stats = future.result()
                                # Accumulate I/O stats from this expert read
                                token_io_bytes += io_stats["bytes_read"]
                                token_io_seeks += io_stats["seek_count"]
                                token_io_time += io_stats["io_time_s"]
                                token_array_time += io_stats["array_time_s"]
                                if profile:
                                    layer_io_bytes += io_stats["bytes_read"]
                                for (proj_name, attr_name), arr in attrs.items():
                                    expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)

                if profile:
                    t_io_done = time.perf_counter()
                    layer_io_ms = (t_io_done - t_io_start) * 1000

                # Track I/O misses for per-token display (cache path)
                if pin_active_here:
                    generate_offload_selective._last_io_misses = getattr(
                        generate_offload_selective, '_last_io_misses', 0) + len(uncached_list)

                # Assemble [num_unique, ...] weight tensors into a dict for direct computation
                expert_tensors = {}
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    for attr_name in ["weight", "scales", "biases"]:
                        slices = []
                        for idx in unique_list:
                            # Check pinned first, then cache
                            if pin_active_here and idx in layer_pinned_c:
                                arr = layer_pinned_c[idx].get((proj_name, attr_name))
                            else:
                                arr = expert_cache.get_attr(i, idx, proj_name, attr_name)
                            if arr is not None:
                                slices.append(arr)
                            else:
                                raise RuntimeError(
                                    f"Expert cache miss: layer={i} expert={idx} "
                                    f"{proj_name}.{attr_name} not found after loading"
                                )
                        if slices:
                            expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)

              # ---- MoE computation (shared by both paths) ----
              if batch_experts:
                # Skip mx.eval(*expert_tensors.values()) — the stacked weight tensors
                # are already concrete (from ExpertCache or direct read). Letting gather_qmm
                # consume them directly avoids one Metal sync per layer. The mx.eval(h) below
                # evaluates the full MoE computation graph (stack + gather_qmm + activation)
                # in a single fused Metal submission.
                if profile:
                    layer_expert_eval_ms = 0.0
                    t_moe = time.perf_counter()
                else:
                    t_moe = time.time()

                y = compute_moe_direct(
                    h_post, remapped_inds, expert_tensors,
                    group_size=qparams["group_size"],
                    bits=qparams["bits"],
                    mode=qparams["mode"],
                )
                y = (y * scores[..., None]).sum(axis=-2)

                # Run shared expert (already loaded in phase 1)
                shared_y = layer.mlp.shared_expert(h_post)
                shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
                y = y + shared_y

                h = h_mid + y

                # Lazy eval: skip explicit sync, let computation accumulate.
                # The next layer's mx.eval(h_mid, inds) will evaluate this layer's
                # expert computation as part of its dependency chain. Eval every 4
                # layers to prevent unbounded graph growth.
                if i % 4 == 3 or i == num_layers - 1:
                    if profile:
                        t_compute_eval = time.perf_counter()
                        mx.eval(h)
                        t_compute_eval_done = time.perf_counter()
                        layer_compute_eval_ms = (t_compute_eval_done - t_compute_eval) * 1000
                        layer_compute_ms = (t_compute_eval - t_moe) * 1000
                        token_moe_compute_time += (t_compute_eval_done - t_moe)
                    else:
                        mx.eval(h)
                        token_moe_compute_time += time.time() - t_moe
                else:
                    if profile:
                        layer_compute_eval_ms = 0.0
                        layer_compute_ms = (time.perf_counter() - t_moe) * 1000
                        token_moe_compute_time += time.perf_counter() - t_moe
                    else:
                        token_moe_compute_time += time.time() - t_moe
              else:
                # Default path: eval expert weights separately, then compute
                # Force-eval the assembled expert weight tensors
                if profile:
                    t_expert_eval = time.perf_counter()
                    mx.eval(*expert_tensors.values())
                    t_expert_eval_done = time.perf_counter()
                    layer_expert_eval_ms = (t_expert_eval_done - t_expert_eval) * 1000
                else:
                    mx.eval(*expert_tensors.values())

                # Run expert MoE computation directly via mx.gather_qmm (no model weight mutation)
                if profile:
                    t_moe = time.perf_counter()
                else:
                    t_moe = time.time()

                y = compute_moe_direct(
                    h_post, remapped_inds, expert_tensors,
                    group_size=qparams["group_size"],
                    bits=qparams["bits"],
                    mode=qparams["mode"],
                )
                y = (y * scores[..., None]).sum(axis=-2)

                # Run shared expert (already loaded in phase 1)
                shared_y = layer.mlp.shared_expert(h_post)
                shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
                y = y + shared_y

                h = h_mid + y

                if profile:
                    t_compute_eval = time.perf_counter()
                    mx.eval(h)
                    t_compute_eval_done = time.perf_counter()
                    layer_compute_eval_ms = (t_compute_eval_done - t_compute_eval) * 1000
                    layer_compute_ms = (t_compute_eval - t_moe) * 1000  # pure compute setup (lazy graph build)
                    token_moe_compute_time += (t_compute_eval_done - t_moe)
                else:
                    mx.eval(h)
                    token_moe_compute_time += time.time() - t_moe

              # Clear protection / release expert tensors
              if not no_expert_cache:
                  expert_cache.unprotect()

              # Delete stacked expert tensors (no-cache: lets mx.arrays be GC'd;
              # cache path: they're copies from cache, not needed after compute).
              # The actual mx.clear_cache() is done once per token after all layers complete.
              del expert_tensors

            if profile:
                t_layer_end = time.perf_counter()
                layer_total_ms = (t_layer_end - t_layer_start) * 1000
                # Compute eval sync total for this layer
                layer_eval_sync_ms = layer_attn_eval_ms + layer_route_eval_ms + layer_expert_eval_ms + layer_compute_eval_ms
                # Python overhead = total - (routing + cache_lookup + io + compute + eval_sync)
                layer_python_overhead_ms = layer_total_ms - layer_routing_ms - layer_cache_lookup_ms - layer_io_ms - layer_compute_ms - layer_eval_sync_ms
                if layer_python_overhead_ms < 0:
                    layer_python_overhead_ms = 0.0

                # Skip prompt token (token_idx == 0) for profiling accumulation
                if token_idx > 0:
                    prof_totals["routing_ms"] += layer_routing_ms
                    prof_totals["cache_lookup_ms"] += layer_cache_lookup_ms
                    prof_totals["io_ms"] += layer_io_ms
                    prof_totals["compute_ms"] += layer_compute_ms
                    prof_totals["eval_sync_ms"] += layer_eval_sync_ms
                    prof_totals["python_overhead_ms"] += layer_python_overhead_ms

                    pld = prof_per_layer[i]
                    pld["hits"] += layer_cache_hits
                    pld["misses"] += layer_cache_misses
                    pld["io_ms"] += layer_io_ms
                    pld["bytes_read"] += layer_io_bytes
                    pld["eval_sync_ms"] += layer_eval_sync_ms
                    pld["compute_ms"] += layer_compute_ms
                    pld["routing_ms"] += layer_routing_ms

            if profile:
                # For compat with existing layer_timings: include eval sync in these totals
                attn_router_ms = layer_routing_ms + layer_attn_eval_ms + layer_route_eval_ms
                expert_ms = (t_layer_end - t_expert) * 1000
            else:
                attn_router_ms = attn_router_time * 1000
                expert_ms = (time.time() - t_expert) * 1000

            layer_timings.append({
                "layer": i,
                "is_linear": layer.is_linear,
                "attn_router_ms": attn_router_ms,
                "expert_ms": expert_ms,
                "clear_ms": 0.0,
                "load_ms": expert_ms,  # total I/O for compat (only experts now)
                "compute_ms": attn_router_ms,  # compute portion for compat
            })

        all_layer_timings.append(layer_timings)

        # Free Metal memory from stacked expert tensor copies accumulated across all layers.
        # Direct MoE doesn't mutate model weights (no dummy-weight clearing needed), but
        # the Metal allocator still holds dead stacked-tensor memory (~40MB/layer x num_layers).
        # Without clear_cache(), memory pressure triggers OS paging that slows file I/O.
        t_clear = time.time()
        mx.clear_cache()
        batch_clear_time = time.time() - t_clear

        # --- Norm + LM head (already pinned) ---
        h = text_model.norm(h)
        if lm.args.tie_word_embeddings:
            logits = text_model.embed_tokens.as_linear(h)
        else:
            logits = lm.lm_head(h)
        mx.eval(logits)

        # --- Sample ---
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.perf_counter() if profile else time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        if profile and token_idx > 0:
            prof_token_count += 1

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Store I/O stats for this token
        io_stats_history.append({
            "bytes_read": token_io_bytes,
            "seek_count": token_io_seeks,
            "io_time_s": token_io_time,
            "array_time_s": token_array_time,
            "moe_compute_time_s": token_moe_compute_time,
            "token_time_s": token_time,
        })

        # === Online expert pinning: transition from warmup to active after N tokens ===
        if pin_phase == "warmup" and (token_idx + 1) == pin_experts:
            t_pin_start = time.time()
            print(f"\n  [pin] Warmup complete ({pin_experts} tokens). Analyzing routing patterns...")

            # Compute coverage: what fraction of total activations are covered by top-K?
            total_activations = pin_counts.sum()
            coverage_per_layer = []
            for li in range(num_layers):
                layer_counts = pin_counts[li]
                layer_total = layer_counts.sum()
                if layer_total == 0:
                    coverage_per_layer.append(0.0)
                    continue
                top_indices = np.argsort(layer_counts)[::-1][:pin_topk]
                top_sum = layer_counts[top_indices].sum()
                coverage_per_layer.append(top_sum / layer_total)
            avg_coverage = np.mean(coverage_per_layer) if coverage_per_layer else 0.0
            print(f"  [pin] Expert coverage: top-{pin_topk}/layer covers "
                  f"{avg_coverage:.1%} of activations")

            # Estimate memory budget
            hidden = lm.args.hidden_size
            intermediate = lm.args.moe_intermediate_size
            per_expert_bytes = 3 * intermediate * hidden * 9 // 16  # 3 projs x (weight+scales+biases)
            total_pin_bytes = pin_topk * num_layers * per_expert_bytes
            total_pin_gb = total_pin_bytes / 1e9

            # Safety check: ensure pinned data fits in memory budget (~30GB max for pinned)
            max_pin_gb = 30.0
            actual_topk = pin_topk
            if total_pin_gb > max_pin_gb:
                actual_topk = int(max_pin_gb * 1e9 / (num_layers * per_expert_bytes))
                total_pin_gb = actual_topk * num_layers * per_expert_bytes / 1e9
                print(f"  [pin] WARNING: {pin_topk} experts/layer = {pin_topk * num_layers * per_expert_bytes / 1e9:.1f} GB "
                      f"exceeds {max_pin_gb:.0f} GB budget. Reducing to {actual_topk}/layer ({total_pin_gb:.1f} GB)")

            print(f"  [pin] Pinning {actual_topk} experts x {num_layers} layers = "
                  f"{actual_topk * num_layers} entries ({total_pin_gb:.1f} GB)...")

            # Pre-populate header_cache for expert files
            for li in range(num_layers):
                entries_li = weight_index.get(li, [])
                _, expert_entries_li = split_layer_entries(entries_li)
                for name, filepath in expert_entries_li:
                    if filepath not in header_cache:
                        header_cache[filepath] = parse_safetensors_header(filepath)

            # Pin top-K experts per layer
            for li in range(num_layers):
                layer_counts = pin_counts[li]
                top_indices = np.argsort(layer_counts)[::-1][:actual_topk]
                # Filter to experts that were actually activated
                top_indices = [int(idx) for idx in top_indices if layer_counts[idx] > 0]

                if not top_indices:
                    continue

                t_layer_pin = time.time()

                entries_li = weight_index.get(li, [])
                _, expert_entries_li = split_layer_entries(entries_li)
                efm = {name: filepath for name, filepath in expert_entries_li}

                pinned_experts[li] = {}

                if packed_layout is not None and li in packed_layers:
                    # PACKED PATH: read from packed expert files
                    packed_fd = packed_fds[li]
                    with ThreadPoolExecutor(max_workers=min(8, len(top_indices))) as executor:
                        futures = [
                            executor.submit(read_expert_packed, packed_fd, packed_layout, eidx)
                            for eidx in top_indices
                        ]
                        for future in futures:
                            eidx, attrs, _ = future.result()
                            pinned_experts[li][eidx] = attrs
                else:
                    # SCATTERED PATH: read from safetensors
                    with ThreadPoolExecutor(max_workers=min(8, len(top_indices))) as executor:
                        futures = [
                            executor.submit(
                                _read_single_expert_attrs,
                                eidx, li, efm, header_cache
                            )
                            for eidx in top_indices
                        ]
                        for future in futures:
                            eidx, attrs, _ = future.result()
                            pinned_experts[li][eidx] = attrs

                # Force-eval pinned arrays to ensure they're in Metal memory
                for eidx in pinned_experts[li]:
                    mx.eval(*pinned_experts[li][eidx].values())

                layer_pin_time = time.time() - t_layer_pin
                if li % 10 == 0 or li == num_layers - 1:
                    print(f"  [pin] Layer {li}: pinned {len(pinned_experts[li])} experts "
                          f"({layer_pin_time:.1f}s, mem={get_mem_gb():.1f}GB)")

            total_pin_time = time.time() - t_pin_start
            actual_pinned = sum(len(v) for v in pinned_experts.values())
            actual_pin_gb = actual_pinned * per_expert_bytes / 1e9
            print(f"  [pin] Done. {actual_pinned} experts pinned ({actual_pin_gb:.1f} GB) "
                  f"in {total_pin_time:.1f}s (mem={get_mem_gb():.1f}GB)")

            pin_phase = "active"
            # Free the counts array
            del pin_counts
            pin_counts = None

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        total_attn_router = sum(lt["attn_router_ms"] for lt in layer_timings)
        total_expert = sum(lt["expert_ms"] for lt in layer_timings)
        total_clear = batch_clear_time * 1000

        cache_hr = expert_cache.hit_rate if expert_cache is not None else 0.0

        # Compute pin hit rate for display
        pin_hr_str = ""
        if pin_phase == "active" and pin_total_lookups > 0:
            pin_hr = pin_total_hits / pin_total_lookups
            pin_hr_str = f" pin_hr={pin_hr:.0%}"
        elif pin_phase == "warmup":
            pin_hr_str = f" pin=warmup({token_idx+1}/{pin_experts})"

        if token_idx == 0:
            ttft_ms = token_time * 1000
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (attn+router={total_attn_router:.0f}ms "
                  f"expert={total_expert:.0f}ms clear={total_clear:.0f}ms "
                  f"cache_hr={cache_hr:.0%}{pin_hr_str} mem={cur_mem:.1f}GB)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            # Count I/O misses for pin mode
            io_miss_str = ""
            if pin_phase == "active" and pin_total_lookups > 0:
                # io_misses for this token = unique experts not pinned (tracked per-token below)
                io_miss_str = f" io_misses={getattr(generate_offload_selective, '_last_io_misses', 0)}"
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.2f} tok/s (attn+router={total_attn_router:.0f}ms "
                  f"expert={total_expert:.0f}ms clear={total_clear:.0f}ms "
                  f"cache_hr={cache_hr:.0%}{pin_hr_str}{io_miss_str} mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    # Close cached file handles
    for fh in file_handle_cache.values():
        try:
            fh.close()
        except Exception:
            pass

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # === I/O instrumentation summary (skip first token — prompt processing is different) ===
    if len(io_stats_history) > 1:
        gen_io = io_stats_history[1:]  # skip prompt token
        avg_bytes = np.mean([s["bytes_read"] for s in gen_io])
        avg_seeks = np.mean([s["seek_count"] for s in gen_io])
        avg_io_ms = np.mean([s["io_time_s"] for s in gen_io]) * 1000
        avg_arr_ms = np.mean([s["array_time_s"] for s in gen_io]) * 1000
        avg_moe_ms = np.mean([s["moe_compute_time_s"] for s in gen_io]) * 1000
        avg_tok_ms = np.mean([s["token_time_s"] for s in gen_io]) * 1000
        avg_other_ms = avg_tok_ms - avg_io_ms - avg_arr_ms - avg_moe_ms
        total_bytes = sum(s["bytes_read"] for s in gen_io)
        total_seeks = sum(s["seek_count"] for s in gen_io)

        print(f"\n  === I/O Instrumentation Summary (tokens 2-{total_tokens}) ===")
        print(f"  Avg bytes/token:  {avg_bytes/1024/1024:.1f} MB")
        print(f"  Avg seeks/token:  {avg_seeks:.0f}")
        print(f"  Avg file I/O:     {avg_io_ms:.1f} ms/token")
        print(f"  Avg arr build:    {avg_arr_ms:.1f} ms/token")
        print(f"  Avg MoE compute:  {avg_moe_ms:.1f} ms/token")
        print(f"  Avg other:        {avg_other_ms:.1f} ms/token (attn+router+clear+overhead)")
        print(f"  Avg total:        {avg_tok_ms:.1f} ms/token")
        print(f"  Total disk read:  {total_bytes/1024/1024:.1f} MB across {total_seeks} seeks")
        if avg_io_ms > 0:
            bw = (avg_bytes / 1024 / 1024) / (avg_io_ms / 1000)
            print(f"  Effective disk BW: {bw:.0f} MB/s")

        # Pin summary
        if pin_total_lookups > 0:
            pin_hr_final = pin_total_hits / pin_total_lookups
            print(f"\n  === Expert Pinning Summary ===")
            print(f"  Warmup tokens:    {pin_experts}")
            print(f"  Experts pinned:   {pin_topk}/layer x {num_layers} layers")
            print(f"  Pin hit rate:     {pin_hr_final:.1%} ({pin_total_hits}/{pin_total_lookups})")
            print(f"  Pin memory:       {sum(len(v) for v in pinned_experts.values()) * 3 * lm.args.moe_intermediate_size * lm.args.hidden_size * 9 // 16 / 1e9:.1f} GB")
    else:
        avg_bytes = 0
        avg_seeks = 0
        avg_io_ms = 0
        avg_arr_ms = 0
        avg_moe_ms = 0

    # === Detailed profiling summary (only when --profile is set) ===
    if profile and prof_token_count > 0:
        model_name = str(model_path).split("/")[-1]
        # Compute grand total for percentage calculation
        grand_total = sum(prof_totals.values())
        if grand_total <= 0:
            grand_total = 1.0  # avoid division by zero

        print(f"\n  === Per-Token Time Breakdown ({model_name}, {prof_token_count} tokens) ===")
        print(f"  {'Component':<20s} {'Total(ms)':>10s} {'Per-Token(ms)':>14s} {'Pct':>6s}")
        for comp_name, comp_key in [
            ("Routing",           "routing_ms"),
            ("Cache Lookup",      "cache_lookup_ms"),
            ("I/O (disk reads)",  "io_ms"),
            ("Compute (GPU)",     "compute_ms"),
            ("mx.eval() syncs",   "eval_sync_ms"),
            ("Python Overhead",   "python_overhead_ms"),
        ]:
            total_ms = prof_totals[comp_key]
            per_tok = total_ms / prof_token_count
            pct = total_ms / grand_total * 100
            print(f"  {comp_name:<20s} {total_ms:>10.1f} {per_tok:>14.2f} {pct:>5.1f}%")

        per_tok_total = grand_total / prof_token_count
        print(f"  {'TOTAL':<20s} {grand_total:>10.1f} {per_tok_total:>14.2f} {'100.0':>5s}%")

        # Per-layer I/O detail table
        print(f"\n  Per-Layer I/O Detail (averaged over {prof_token_count} tokens):")
        print(f"  {'Layer':>5s}  {'Hits':>5s}  {'Misses':>6s}  {'IO_ms':>8s}  {'Bytes_Read':>12s}  {'Throughput_GBs':>14s}")
        for layer_idx in sorted(prof_per_layer.keys()):
            pld = prof_per_layer[layer_idx]
            avg_hits = pld["hits"] / prof_token_count
            avg_misses = pld["misses"] / prof_token_count
            avg_io_layer_ms = pld["io_ms"] / prof_token_count
            avg_bytes = pld["bytes_read"] / prof_token_count
            if avg_io_layer_ms > 0:
                throughput_gbs = (avg_bytes / 1e9) / (avg_io_layer_ms / 1000)
            else:
                throughput_gbs = 0.0
            # Format bytes nicely
            if avg_bytes >= 1e6:
                bytes_str = f"{avg_bytes/1e6:.1f}MB"
            elif avg_bytes >= 1e3:
                bytes_str = f"{avg_bytes/1e3:.1f}KB"
            else:
                bytes_str = f"{avg_bytes:.0f}B"
            print(f"  {layer_idx:>5d}  {avg_hits:>5.1f}  {avg_misses:>6.1f}  {avg_io_layer_ms:>8.2f}  {bytes_str:>12s}  {throughput_gbs:>14.2f}")

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])
        avg_clear = np.mean([sum(lt["clear_ms"] for lt in tt) for tt in gen_timings])
        avg_expert = np.mean([sum(lt["expert_ms"] for lt in tt) for tt in gen_timings])
        avg_attn_router = np.mean([sum(lt["attn_router_ms"] for lt in tt) for tt in gen_timings])

        num_layers_t = len(gen_timings[0])
        per_layer_load = [np.mean([tt[j]["load_ms"] for tt in gen_timings]) for j in range(num_layers_t)]
        per_layer_compute = [np.mean([tt[j]["compute_ms"] for tt in gen_timings]) for j in range(num_layers_t)]
    else:
        avg_load = 0
        avg_compute = 0
        avg_clear = 0
        avg_expert = 0
        avg_attn_router = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "preload_time": preload_time,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "avg_clear_ms_per_token": avg_clear,
        "avg_expert_ms_per_token": avg_expert,
        "avg_attn_router_ms_per_token": avg_attn_router,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
        "expert_cache_hits": expert_cache.hits if expert_cache is not None else 0,
        "expert_cache_misses": expert_cache.misses if expert_cache is not None else 0,
        "expert_cache_hit_rate": expert_cache.hit_rate if expert_cache is not None else 0.0,
        "expert_cache_entries": len(expert_cache.cache) if expert_cache is not None else 0,
        "no_expert_cache": no_expert_cache,
        "batch_layers": batch_layers,
        "avg_io_bytes_per_token": avg_bytes,
        "avg_io_seeks_per_token": avg_seeks,
        "avg_io_ms_per_token": avg_io_ms,
        "avg_arr_ms_per_token": avg_arr_ms,
        "avg_moe_compute_ms_per_token": avg_moe_ms,
        "pin_experts": pin_experts,
        "pin_topk": pin_topk if pin_experts > 0 else 0,
        "pin_hit_rate": pin_total_hits / pin_total_lookups if pin_total_lookups > 0 else 0.0,
        "pin_total_hits": pin_total_hits,
        "pin_total_lookups": pin_total_lookups,
    }


def load_model_no_weights(model_path):
    """Create model architecture with quantization but load NO weights at all.
    For offload mode: weights are loaded per-layer during inference, not at init.
    This avoids mmap'ing 61GB of safetensors which causes OS thrashing on <DRAM machines."""
    model_path = Path(model_path)

    # 1. Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # 2. Create model architecture (empty — no weights)
    from mlx_lm.utils import _get_classes
    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # 3. Apply quantization (sets up quantized module structure, still no weights)
    qconfig = config.get("quantization", config.get("quantization_config", {}))
    if qconfig:
        nn.quantize(model, bits=qconfig["bits"], group_size=qconfig["group_size"])

    model.eval()

    # 4. Load ONLY global weights (embed_tokens, norm, lm_head) — ~1GB total
    weight_index = build_weight_index(model_path)
    global_entries = weight_index.get("global", [])
    if global_entries:
        by_file = defaultdict(list)
        for name, filepath in global_entries:
            by_file[filepath].append(name)

        global_weights = []
        for filepath, names in by_file.items():
            all_tensors = mx.load(filepath)
            for name in names:
                if name in all_tensors:
                    global_weights.append((name, all_tensors[name]))
            # Don't keep reference to all_tensors — let it be GC'd
            del all_tensors

        model.load_weights(global_weights, strict=False)
        # Force-eval global weights so they're resident in DRAM
        lm = model.language_model
        text_model = lm.model
        mx.eval(text_model.embed_tokens.parameters())
        mx.eval(text_model.norm.parameters())
        if hasattr(lm, 'lm_head'):
            mx.eval(lm.lm_head.parameters())
        del global_weights

    # 5. Load tokenizer
    from mlx_lm.utils import load_tokenizer
    eos_ids = config.get("eos_token_id", [])
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)

    return model, tokenizer


def load_model_custom(model_path):
    """Custom model loader that bypasses mlx_lm.load() overhead.
    Loads model architecture + lazy weights directly. Fast even for 100GB+ models."""
    import glob

    model_path = Path(model_path)

    # 1. Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # 2. Load all weights lazily (mmap, no actual I/O)
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # 3. Create model architecture
    from mlx_lm.utils import _get_classes
    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # 4. Sanitize weights (transforms HF format to MLX format)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # 5. Apply quantization
    qconfig = config.get("quantization", config.get("quantization_config", {}))
    if qconfig:
        nn.quantize(model, bits=qconfig["bits"], group_size=qconfig["group_size"])

    # 6. Load weights into model (still lazy — no actual I/O)
    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    # 7. Load tokenizer separately
    from mlx_lm.utils import load_tokenizer
    eos_ids = config.get("eos_token_id", [])
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)

    return model, tokenizer


def snapshot_cache(cache):
    """Deep-copy the verifier's KV/recurrent caches so we can rollback on rejection.

    The model uses two cache types:
      - KVCache (full_attention layers): has keys/values arrays + offset counter
      - ArraysCache (linear_attention layers): has cache list [conv_state, recurrent_state]

    We snapshot the raw tensor state and scalar offsets so restore is cheap.
    """
    from mlx_lm.models.cache import KVCache as _KVCache, ArraysCache as _ArraysCache

    snapshots = []
    for c in cache:
        if isinstance(c, _KVCache):
            # Snapshot: (keys_copy, values_copy, offset)
            # Only copy the live region to save memory
            if c.keys is not None:
                k = c.keys[..., :c.offset, :]
                v = c.values[..., :c.offset, :]
                mx.eval(k, v)
                snapshots.append(("kv", k, v, c.offset))
            else:
                snapshots.append(("kv", None, None, 0))
        elif isinstance(c, _ArraysCache):
            # Snapshot each element of the cache list
            parts = []
            for item in c.cache:
                if item is not None:
                    cp = mx.array(item)
                    mx.eval(cp)
                    parts.append(cp)
                else:
                    parts.append(None)
            snapshots.append(("arrays", parts))
        else:
            # Unknown cache type -- store None (will skip restore)
            snapshots.append(("unknown",))
    return snapshots


def restore_cache(cache, snapshots):
    """Restore verifier caches from a snapshot taken before speculative verification."""
    from mlx_lm.models.cache import KVCache as _KVCache, ArraysCache as _ArraysCache

    for c, snap in zip(cache, snapshots):
        if snap[0] == "kv":
            _, k, v, offset = snap
            if k is not None:
                # Overwrite the cache arrays and offset
                c.keys = None
                c.values = None
                c.offset = 0
                # Re-insert via update_and_fetch to reallocate properly
                # The keys/values are [B, n_heads, seq, dim] -- feed all at once
                c.update_and_fetch(k, v)
                c.offset = offset
            else:
                c.keys = None
                c.values = None
                c.offset = 0
        elif snap[0] == "arrays":
            _, parts = snap
            for idx, item in enumerate(parts):
                c.cache[idx] = item


def trim_verifier_cache(cache, num_to_trim):
    """Trim the verifier cache by num_to_trim positions.

    For KVCache layers: use the built-in trim method (adjusts offset).
    For ArraysCache layers (linear attention): these use recurrent state that
    cannot be partially trimmed. We leave them as-is since the recurrent state
    from rejected tokens is a minor approximation error that does not affect
    correctness significantly (the state is a compressed summary, not per-token).

    This is faster than full snapshot/restore when we only reject the tail.
    """
    from mlx_lm.models.cache import KVCache as _KVCache

    for c in cache:
        if isinstance(c, _KVCache) and c.offset >= num_to_trim:
            c.trim(num_to_trim)


def generate_speculative(
    draft_model,
    verifier_model,
    tokenizer,
    prompt,
    max_tokens,
    weight_index,
    model_path,
    draft_k=8,
    preload_topk=0,
    cache_gb=20.0,
    use_pread=False,
    pread_index=None,
    pread_fds=None,
):
    """Speculative decoding: draft K tokens with small model, verify with large model.

    Uses Leviathan et al. 2023 approach with greedy (argmax) acceptance:
      1. Draft K tokens autoregressively with the 35B model (fast, in DRAM)
      2. Feed all K draft tokens to the 397B verifier in a single batched forward pass
      3. Accept tokens greedily: for each position, if verifier agrees with draft, accept
      4. On first disagreement, use verifier's token, discard remaining drafts
      5. Rollback caches to the rejection point

    Args:
        draft_model: Small model loaded fully in DRAM (e.g. Qwen3.5-35B-A3B)
        verifier_model: Large model with only global+non-expert weights loaded
        tokenizer: Shared tokenizer (identical for both models)
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        weight_index: Weight index for verifier model (from build_weight_index)
        model_path: Path to verifier model safetensors
        draft_k: Number of draft tokens per speculation round
    """
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]  # [1, seq_len]

    # === Initialize caches for both models ===
    draft_cache = draft_model.make_cache()
    verifier_cache = verifier_model.make_cache()

    lm_v = verifier_model.language_model
    text_model_v = lm_v.model
    layers_v = text_model_v.layers
    num_layers_v = len(layers_v)

    generated_tokens = []
    peak_mem = get_mem_gb()

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    # === Pre-load verifier non-expert weights (same as offload_selective) ===
    t_preload = time.time()
    header_cache = {}
    all_nonexpert_weights = []

    file_to_names = defaultdict(list)
    file_to_san = {}
    for layer_i in range(num_layers_v):
        entries = weight_index.get(layer_i, [])
        non_expert, _ = split_layer_entries(entries)
        for name, filepath in non_expert:
            file_to_names[filepath].append(name)
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            file_to_san[(filepath, name)] = san_name

    for filepath, names in sorted(file_to_names.items()):
        tensors = read_tensors_direct(filepath, names, header_cache, file_handle_cache=None)
        for name in names:
            if name in tensors:
                san_name = file_to_san[(filepath, name)]
                all_nonexpert_weights.append((san_name, tensors[name]))
        print(f"    read {len(names)} tensors from {Path(filepath).name} "
              f"({get_mem_gb():.1f}GB)", flush=True)

    print(f"    Total non-expert weights to load: {len(all_nonexpert_weights)}")
    lm_v.load_weights(all_nonexpert_weights, strict=False)
    del all_nonexpert_weights
    preload_time = time.time() - t_preload
    print(f"  Pre-loaded verifier non-expert weights in {preload_time:.1f}s "
          f"({get_mem_gb():.1f}GB)")

    # === File handle cache for expert I/O ===
    file_handle_cache = {}

    # === Expert LRU cache ===
    active_experts = lm_v.args.num_experts_per_tok
    cache_entries = num_layers_v * active_experts * 8
    hidden = lm_v.args.hidden_size
    intermediate = lm_v.args.moe_intermediate_size
    per_expert_bytes = 3 * intermediate * hidden * 9 // 16
    max_cache_gb = cache_gb
    # Memory safety: cap total DRAM = non_expert (~5GB) + draft (~7GB) + cache
    if max_cache_gb + 12 > 43:
        adjusted = 43.0 - 12.0
        print(f"[cache] WARNING: cache_gb={max_cache_gb:.1f} + models (~12GB) exceeds 43GB safety limit. "
              f"Reducing cache to {adjusted:.1f}GB")
        max_cache_gb = adjusted
    max_entries_by_mem = int(max_cache_gb * 1e9 / per_expert_bytes) if per_expert_bytes > 0 else cache_entries
    if cache_entries > max_entries_by_mem:
        print(f"[cache] Capping entries from {cache_entries} to {max_entries_by_mem} "
              f"(~{max_cache_gb:.0f}GB limit, {per_expert_bytes/1e6:.1f}MB/entry)")
        cache_entries = max_entries_by_mem
    elif max_entries_by_mem > cache_entries:
        print(f"[cache] Expanding entries from {cache_entries} to {max_entries_by_mem} "
              f"(~{max_cache_gb:.0f}GB budget, {per_expert_bytes/1e6:.1f}MB/entry)")
        cache_entries = max_entries_by_mem
    expert_cache = ExpertCache(max_entries=cache_entries)

    # === Pre-load hot experts if requested ===
    if preload_topk > 0:
        preload_hot_experts(expert_cache, weight_index, model_path, num_layers_v,
                            preload_topk, header_cache,
                            use_pread=use_pread, pread_index=pread_index,
                            pread_fds=pread_fds)

    # === Quantization params ===
    with open(model_path / "config.json") as f:
        _cfg = json.load(f)
    _qcfg = _cfg.get("quantization", _cfg.get("quantization_config", {}))
    qparams = {
        "group_size": _qcfg.get("group_size", 64),
        "bits": _qcfg.get("bits", 4),
        "mode": _qcfg.get("mode", "affine"),
    }
    del _cfg, _qcfg

    # === Statistics tracking ===
    total_drafted = 0
    total_accepted = 0
    total_rounds = 0
    round_times = []  # wall time per speculation round
    draft_times = []
    verify_times = []

    # === Prompt prefill for BOTH models ===
    # Draft model: run prompt through to populate its KV cache
    print(f"  Prefilling draft model...", flush=True)
    t_prefill_draft = time.time()
    draft_logits = manual_forward(draft_model, input_ids, draft_cache)
    mx.eval(draft_logits)
    prefill_draft_time = time.time() - t_prefill_draft

    # Get first token from draft model's prompt logits
    first_token = mx.argmax(draft_logits[:, -1, :], axis=-1)
    mx.eval(first_token)

    # Verifier model: run prompt through to populate its KV/recurrent cache
    # We need to run the FULL verifier forward including MoE expert loading
    print(f"  Prefilling verifier model...", flush=True)
    t_prefill_verify = time.time()

    # Run verifier prefill token-by-token for prompt (uses the offload_selective path)
    # For simplicity, feed the whole prompt as a sequence
    h_v = text_model_v.embed_tokens(input_ids)
    mx.eval(h_v)

    fa_mask = create_attention_mask(h_v, verifier_cache[text_model_v.fa_idx])
    ssm_mask = create_ssm_mask(h_v, verifier_cache[text_model_v.ssm_idx])

    for i in range(num_layers_v):
        layer = layers_v[i]
        c = verifier_cache[i]
        entries = weight_index.get(i, [])
        _, expert_entries = split_layer_entries(entries)

        # Phase 1: Attention + router
        x_normed = layer.input_layernorm(h_v)
        mask = ssm_mask if layer.is_linear else fa_mask
        if layer.is_linear:
            r = layer.linear_attn(x_normed, mask, c)
        else:
            r = layer.self_attn(x_normed, mask, c)
        h_mid = h_v + r
        mx.eval(h_mid)

        # Phase 2: MoE with expert loading
        h_post = layer.post_attention_layernorm(h_mid)
        gates = layer.mlp.gate(h_post)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k_experts = layer.mlp.top_k
        inds = mx.argpartition(gates, kth=-k_experts, axis=-1)[..., -k_experts:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(inds)

        inds_np = np.array(inds.tolist())
        unique_experts = np.unique(inds_np)
        num_unique = len(unique_experts)
        unique_list = unique_experts.tolist()

        remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
        remap[unique_experts] = np.arange(num_unique)
        remapped_inds = mx.array(remap[inds_np])

        expert_file_map = {}
        for name, filepath in expert_entries:
            expert_file_map[name] = filepath

        expert_cache.protect([(i, idx) for idx in unique_list])
        uncached_list = []
        for idx in unique_list:
            if expert_cache.has_expert(i, idx):
                expert_cache.record_hit()
                expert_cache.touch(i, idx)
            else:
                expert_cache.record_miss()
                uncached_list.append(idx)

        if uncached_list:
            if use_pread and pread_index is not None:
                batch_results, io_stats = pread_expert_batch(
                    pread_index, pread_fds, i, uncached_list)
                for eidx, attrs in batch_results.items():
                    for (proj_name, attr_name), arr in attrs.items():
                        expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
            else:
                for filepath in set(expert_file_map.values()):
                    if filepath not in header_cache:
                        header_cache[filepath] = parse_safetensors_header(filepath)

                with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                    futures = [
                        executor.submit(
                            _read_single_expert_attrs,
                            expert_idx, i, expert_file_map, header_cache
                        )
                        for expert_idx in uncached_list
                    ]
                    for future in futures:
                        eidx, attrs, io_stats = future.result()
                        for (proj_name, attr_name), arr in attrs.items():
                            expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)

        expert_tensors = {}
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            for attr_name in ["weight", "scales", "biases"]:
                slices = []
                for idx in unique_list:
                    arr = expert_cache.get_attr(i, idx, proj_name, attr_name)
                    if arr is not None:
                        slices.append(arr)
                    else:
                        raise RuntimeError(
                            f"Expert cache miss during prefill: layer={i} expert={idx} "
                            f"{proj_name}.{attr_name}")
                if slices:
                    expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)

        mx.eval(*expert_tensors.values())

        y = compute_moe_direct(
            h_post, remapped_inds, expert_tensors,
            group_size=qparams["group_size"],
            bits=qparams["bits"],
            mode=qparams["mode"],
        )
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = layer.mlp.shared_expert(h_post)
        shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
        y = y + shared_y

        h_v = h_mid + y
        mx.eval(h_v)
        expert_cache.unprotect()
        del expert_tensors

    mx.clear_cache()

    h_v = text_model_v.norm(h_v)
    if lm_v.args.tie_word_embeddings:
        verifier_logits = text_model_v.embed_tokens.as_linear(h_v)
    else:
        verifier_logits = lm_v.lm_head(h_v)
    mx.eval(verifier_logits)

    # Verifier's first token prediction (what should come after the prompt)
    verifier_first = mx.argmax(verifier_logits[:, -1, :], axis=-1)
    mx.eval(verifier_first)

    prefill_verify_time = time.time() - t_prefill_verify
    print(f"  Prefill: draft={prefill_draft_time:.1f}s, verifier={prefill_verify_time:.1f}s")

    # Use verifier's token as the authoritative first token
    first_token_id = verifier_first.item()
    generated_tokens.append(first_token_id)

    # Track the verifier's prediction for the NEXT position.
    # After prefill, verifier_logits[:, -1, :] predicted first_token (committed above).
    # Now we need the verifier's prediction for position 2 (what follows first_token).
    # We'll get this by running first_token through the verifier for one step.
    #
    # For the speculative loop, verifier_pending_pred stores the verifier's argmax
    # prediction for the next token (the one the draft needs to match as d_0).
    # We get this by running first_token through the verifier.
    _vp_input = verifier_first.reshape(1, 1)
    h_vp = text_model_v.embed_tokens(_vp_input)
    mx.eval(h_vp)
    fa_mask = create_attention_mask(h_vp, verifier_cache[text_model_v.fa_idx])
    ssm_mask = create_ssm_mask(h_vp, verifier_cache[text_model_v.ssm_idx])
    for i in range(num_layers_v):
        layer = layers_v[i]
        c = verifier_cache[i]
        entries = weight_index.get(i, [])
        _, expert_entries = split_layer_entries(entries)

        x_normed = layer.input_layernorm(h_vp)
        mask = ssm_mask if layer.is_linear else fa_mask
        if layer.is_linear:
            r = layer.linear_attn(x_normed, mask, c)
        else:
            r = layer.self_attn(x_normed, mask, c)
        h_mid_vp = h_vp + r
        mx.eval(h_mid_vp)

        h_post_vp = layer.post_attention_layernorm(h_mid_vp)
        gates = layer.mlp.gate(h_post_vp)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k_experts = layer.mlp.top_k
        inds = mx.argpartition(gates, kth=-k_experts, axis=-1)[..., -k_experts:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(inds)

        inds_np = np.array(inds.tolist())
        unique_experts = np.unique(inds_np)
        unique_list = unique_experts.tolist()
        remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
        remap[unique_experts] = np.arange(len(unique_experts))
        remapped_inds = mx.array(remap[inds_np])

        expert_file_map = {}
        for name, filepath in expert_entries:
            expert_file_map[name] = filepath

        expert_cache.protect([(i, idx) for idx in unique_list])
        uncached_list = []
        for idx in unique_list:
            if expert_cache.has_expert(i, idx):
                expert_cache.record_hit()
                expert_cache.touch(i, idx)
            else:
                expert_cache.record_miss()
                uncached_list.append(idx)
        if uncached_list:
            if use_pread and pread_index is not None:
                batch_results, io_stats = pread_expert_batch(
                    pread_index, pread_fds, i, uncached_list)
                for eidx, attrs in batch_results.items():
                    for (pn, an), arr in attrs.items():
                        expert_cache.put_attr(i, eidx, pn, an, arr)
            else:
                for filepath in set(expert_file_map.values()):
                    if filepath not in header_cache:
                        header_cache[filepath] = parse_safetensors_header(filepath)
                with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                    futures = [
                        executor.submit(_read_single_expert_attrs, eidx, i, expert_file_map, header_cache)
                        for eidx in uncached_list
                    ]
                    for future in futures:
                        eidx, attrs, _ = future.result()
                        for (pn, an), arr in attrs.items():
                            expert_cache.put_attr(i, eidx, pn, an, arr)

        expert_tensors = {}
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            for attr_name in ["weight", "scales", "biases"]:
                slices = [expert_cache.get_attr(i, idx, proj_name, attr_name) for idx in unique_list]
                if all(s is not None for s in slices):
                    expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
        mx.eval(*expert_tensors.values())

        y = compute_moe_direct(
            h_post_vp, remapped_inds, expert_tensors,
            group_size=qparams["group_size"], bits=qparams["bits"], mode=qparams["mode"],
        )
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = layer.mlp.shared_expert(h_post_vp)
        shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post_vp)) * shared_y
        y = y + shared_y
        h_vp = h_mid_vp + y
        mx.eval(h_vp)
        expert_cache.unprotect()
        del expert_tensors
    mx.clear_cache()

    h_vp = text_model_v.norm(h_vp)
    if lm_v.args.tie_word_embeddings:
        _vp_logits = text_model_v.embed_tokens.as_linear(h_vp)
    else:
        _vp_logits = lm_v.lm_head(h_vp)
    mx.eval(_vp_logits)

    # This is the verifier's prediction for position 2 (what follows first_token)
    verifier_pending_pred = mx.argmax(_vp_logits[:, -1, :], axis=-1).item()
    del _vp_logits, h_vp

    # Set up the next input for draft model
    draft_input = verifier_first.reshape(1, 1)

    print(f"  Starting speculative decode loop (K={draft_k})...", flush=True)

    # === Main speculative decode loop ===
    while len(generated_tokens) < max_tokens:
        t_round_start = time.time()
        remaining = max_tokens - len(generated_tokens)
        K = min(draft_k, remaining)

        # ============================================================
        # PHASE 1: Draft K tokens with the small model (fast, in DRAM)
        # ============================================================
        t_draft_start = time.time()
        draft_token_ids = []
        cur_draft_input = draft_input

        for d in range(K):
            d_logits = manual_forward(draft_model, cur_draft_input, draft_cache)
            mx.eval(d_logits)
            d_token = mx.argmax(d_logits[:, -1, :], axis=-1)
            mx.eval(d_token)
            draft_token_ids.append(d_token.item())
            cur_draft_input = d_token.reshape(1, 1)

        draft_time = time.time() - t_draft_start
        draft_times.append(draft_time)

        # ============================================================
        # PHASE 2: Verify all K tokens with the 397B model (batched)
        # ============================================================
        t_verify_start = time.time()

        # Snapshot verifier cache BEFORE verification so we can rollback
        # Only snapshot on first few rounds or when K > 1 to avoid overhead
        cache_snapshot = snapshot_cache(verifier_cache)

        # Build the verification input: all K draft tokens as a sequence
        # The verifier's cache already contains everything up to the last accepted token.
        # We feed [draft_0, draft_1, ..., draft_{K-1}] as a single sequence.
        verify_input = mx.array([[t for t in draft_token_ids]])  # [1, K]

        # Run verifier forward pass through all layers (batched K tokens)
        h_v = text_model_v.embed_tokens(verify_input)
        mx.eval(h_v)

        fa_mask = create_attention_mask(h_v, verifier_cache[text_model_v.fa_idx])
        ssm_mask = create_ssm_mask(h_v, verifier_cache[text_model_v.ssm_idx])

        token_io_bytes = 0
        token_io_time = 0.0

        for i in range(num_layers_v):
            layer = layers_v[i]
            c = verifier_cache[i]
            entries = weight_index.get(i, [])
            _, expert_entries = split_layer_entries(entries)

            # Attention (handles K tokens naturally via the cache)
            x_normed = layer.input_layernorm(h_v)
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer.is_linear:
                r = layer.linear_attn(x_normed, mask, c)
            else:
                r = layer.self_attn(x_normed, mask, c)
            h_mid = h_v + r
            mx.eval(h_mid)

            # Router: discovers experts for ALL K positions at once
            h_post = layer.post_attention_layernorm(h_mid)
            gates = layer.mlp.gate(h_post)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k_experts = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k_experts, axis=-1)[..., -k_experts:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds)

            # Key optimization: collect unique experts across ALL K positions
            # Each position routes to k_experts, but many overlap across positions
            inds_np = np.array(inds.tolist())
            unique_experts = np.unique(inds_np)
            num_unique = len(unique_experts)
            unique_list = unique_experts.tolist()

            remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
            remap[unique_experts] = np.arange(num_unique)
            remapped_inds = mx.array(remap[inds_np])

            expert_file_map = {}
            for name, filepath in expert_entries:
                expert_file_map[name] = filepath

            # LRU cache check
            expert_cache.protect([(i, idx) for idx in unique_list])
            uncached_list = []
            for idx in unique_list:
                if expert_cache.has_expert(i, idx):
                    expert_cache.record_hit()
                    expert_cache.touch(i, idx)
                else:
                    expert_cache.record_miss()
                    uncached_list.append(idx)

            # Load uncached experts from disk
            if uncached_list:
                if use_pread and pread_index is not None:
                    batch_results, io_stats = pread_expert_batch(
                        pread_index, pread_fds, i, uncached_list)
                    token_io_bytes += io_stats["bytes_read"]
                    token_io_time += io_stats["io_time_s"]
                    for eidx, attrs in batch_results.items():
                        for (proj_name, attr_name), arr in attrs.items():
                            expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
                else:
                    for filepath in set(expert_file_map.values()):
                        if filepath not in header_cache:
                            header_cache[filepath] = parse_safetensors_header(filepath)

                    with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                        futures = [
                            executor.submit(
                                _read_single_expert_attrs,
                                expert_idx, i, expert_file_map, header_cache
                            )
                            for expert_idx in uncached_list
                        ]
                        for future in futures:
                            eidx, attrs, io_stats = future.result()
                            token_io_bytes += io_stats["bytes_read"]
                            token_io_time += io_stats["io_time_s"]
                            for (proj_name, attr_name), arr in attrs.items():
                                expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)

            # Assemble expert tensors
            expert_tensors = {}
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                for attr_name in ["weight", "scales", "biases"]:
                    slices = []
                    for idx in unique_list:
                        arr = expert_cache.get_attr(i, idx, proj_name, attr_name)
                        if arr is not None:
                            slices.append(arr)
                        else:
                            raise RuntimeError(
                                f"Expert cache miss: layer={i} expert={idx} "
                                f"{proj_name}.{attr_name}")
                    if slices:
                        expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)

            mx.eval(*expert_tensors.values())

            # Batched MoE computation: compute_moe_direct handles [1, K, hidden]
            y = compute_moe_direct(
                h_post, remapped_inds, expert_tensors,
                group_size=qparams["group_size"],
                bits=qparams["bits"],
                mode=qparams["mode"],
            )
            y = (y * scores[..., None]).sum(axis=-2)

            shared_y = layer.mlp.shared_expert(h_post)
            shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
            y = y + shared_y

            h_v = h_mid + y
            mx.eval(h_v)
            expert_cache.unprotect()
            del expert_tensors

        mx.clear_cache()

        # Norm + LM head
        h_v = text_model_v.norm(h_v)
        if lm_v.args.tie_word_embeddings:
            v_logits = text_model_v.embed_tokens.as_linear(h_v)
        else:
            v_logits = lm_v.lm_head(h_v)
        mx.eval(v_logits)

        verify_time = time.time() - t_verify_start
        verify_times.append(verify_time)

        # ============================================================
        # PHASE 3: Greedy acceptance (Leviathan et al. 2023, argmax)
        # ============================================================
        # verifier_pending_pred: the verifier's argmax prediction for the NEXT token
        #   (carried from previous round or prefill). This validates d_0.
        #
        # v_logits has shape [1, K, vocab_size] from the batched verification.
        # After processing [d_0, d_1, ..., d_{K-1}]:
        #   v_logits[:, j, :] = P(next | context ++ [d_0, ..., d_j])
        # So:
        #   v_logits[:, 0, :] predicts what follows d_0 -> validates d_1
        #   v_logits[:, 1, :] predicts what follows d_0,d_1 -> validates d_2
        #   ...
        #   v_logits[:, K-1, :] predicts what follows all K -> bonus token
        #
        # Acceptance:
        #   1. Check d_0 against verifier_pending_pred (from previous round)
        #   2. For j=0..K-2: check d_{j+1} against argmax(v_logits[:, j, :])
        #   3. If all pass, bonus = argmax(v_logits[:, K-1, :])

        v_argmax = mx.argmax(v_logits, axis=-1)  # [1, K]
        mx.eval(v_argmax)
        v_preds = v_argmax[0].tolist()  # list of K ints

        accepted_tokens = []

        # Step 1: validate d_0 against verifier's pending prediction
        if draft_token_ids[0] == verifier_pending_pred:
            # d_0 matches verifier's expectation
            accepted_tokens.append(draft_token_ids[0])

            # Step 2: validate d_1 through d_{K-1}
            for j in range(K - 1):
                # v_preds[j] = verifier's argmax after seeing d_0..d_j
                # This should match draft_token_ids[j+1]
                if v_preds[j] == draft_token_ids[j + 1]:
                    accepted_tokens.append(draft_token_ids[j + 1])
                else:
                    # Reject d_{j+1}: use verifier's prediction as correction
                    accepted_tokens.append(v_preds[j])
                    break
            else:
                # All K draft tokens accepted -- bonus token from verifier
                accepted_tokens.append(v_preds[K - 1])
        else:
            # d_0 itself was wrong. Use verifier's pending prediction as correction.
            # No draft tokens accepted.
            accepted_tokens.append(verifier_pending_pred)

        generated_tokens.extend(accepted_tokens)
        total_drafted += K
        total_accepted += len(accepted_tokens)
        total_rounds += 1

        # ============================================================
        # PHASE 4: Cache management and verifier_pending_pred update
        # ============================================================
        # The verifier processed all K draft tokens in its forward pass.
        # accepted_tokens has len N_acc where the last token is either:
        #   - A bonus token (all K accepted, N_acc = K+1)
        #   - A correction token (draft rejected at some point, N_acc <= K)
        #   - The verifier's pending pred (d_0 itself was wrong, N_acc = 1)
        #
        # We need to:
        #   1. Determine how many of the K draft tokens in the cache are valid
        #   2. Rollback/trim the verifier cache if needed
        #   3. Set verifier_pending_pred for the next round
        #   4. Trim draft model cache similarly
        #
        # Case analysis for accepted_tokens:
        #   If d_0 was wrong: accepted = [verifier_pending_pred]. 0 draft tokens in cache are valid.
        #   If d_0 OK, rejected at d_{j+1}: accepted = [d_0, ..., d_j, correction].
        #     (j+1) draft tokens in cache are valid.
        #   If all K accepted + bonus: accepted = [d_0, ..., d_{K-1}, bonus].
        #     All K draft tokens in cache are valid.

        d0_was_wrong = (draft_token_ids[0] != verifier_pending_pred)
        all_accepted_with_bonus = (len(accepted_tokens) == K + 1) and not d0_was_wrong

        if d0_was_wrong:
            # d_0 was wrong. The verifier cache has K positions from the forward pass,
            # but NONE are valid (since d_0 itself was wrong). Restore from snapshot.
            n_valid_in_cache = 0
        elif all_accepted_with_bonus:
            # All K accepted. Cache has K valid positions.
            n_valid_in_cache = K
        else:
            # Rejected at some point. accepted_tokens = [d_0, ..., d_j, correction].
            # d_0 through d_j were correct = len(accepted_tokens) - 1 valid positions.
            n_valid_in_cache = len(accepted_tokens) - 1

        # Rollback verifier cache if needed
        if n_valid_in_cache < K:
            # Restore from snapshot and re-run only the valid portion
            restore_cache(verifier_cache, cache_snapshot)

            if n_valid_in_cache > 0:
                rerun_input = mx.array([draft_token_ids[:n_valid_in_cache]])  # [1, n_valid]

                h_v = text_model_v.embed_tokens(rerun_input)
                mx.eval(h_v)

                fa_mask = create_attention_mask(h_v, verifier_cache[text_model_v.fa_idx])
                ssm_mask = create_ssm_mask(h_v, verifier_cache[text_model_v.ssm_idx])

                for i in range(num_layers_v):
                    layer = layers_v[i]
                    c_v = verifier_cache[i]
                    entries = weight_index.get(i, [])
                    _, expert_entries = split_layer_entries(entries)

                    x_normed = layer.input_layernorm(h_v)
                    mask = ssm_mask if layer.is_linear else fa_mask
                    if layer.is_linear:
                        r = layer.linear_attn(x_normed, mask, c_v)
                    else:
                        r = layer.self_attn(x_normed, mask, c_v)
                    h_mid = h_v + r
                    mx.eval(h_mid)

                    h_post = layer.post_attention_layernorm(h_mid)
                    gates = layer.mlp.gate(h_post)
                    gates = mx.softmax(gates, axis=-1, precise=True)
                    k_exp = layer.mlp.top_k
                    inds = mx.argpartition(gates, kth=-k_exp, axis=-1)[..., -k_exp:]
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                    mx.eval(inds)

                    inds_np = np.array(inds.tolist())
                    unique_experts = np.unique(inds_np)
                    unique_list = unique_experts.tolist()

                    remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
                    remap[unique_experts] = np.arange(len(unique_experts))
                    remapped_inds = mx.array(remap[inds_np])

                    expert_file_map = {}
                    for name, filepath in expert_entries:
                        expert_file_map[name] = filepath

                    expert_cache.protect([(i, idx) for idx in unique_list])
                    uncached_list = []
                    for idx in unique_list:
                        if expert_cache.has_expert(i, idx):
                            expert_cache.record_hit()
                            expert_cache.touch(i, idx)
                        else:
                            expert_cache.record_miss()
                            uncached_list.append(idx)

                    if uncached_list:
                        if use_pread and pread_index is not None:
                            batch_results, io_stats = pread_expert_batch(
                                pread_index, pread_fds, i, uncached_list)
                            for eidx, attrs in batch_results.items():
                                for (proj_name, attr_name), arr in attrs.items():
                                    expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)
                        else:
                            for filepath in set(expert_file_map.values()):
                                if filepath not in header_cache:
                                    header_cache[filepath] = parse_safetensors_header(filepath)
                            with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                                futures = [
                                    executor.submit(
                                        _read_single_expert_attrs,
                                        expert_idx, i, expert_file_map, header_cache
                                    )
                                    for expert_idx in uncached_list
                                ]
                                for future in futures:
                                    eidx, attrs, io_stats = future.result()
                                    for (proj_name, attr_name), arr in attrs.items():
                                        expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)

                    expert_tensors = {}
                    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                        for attr_name in ["weight", "scales", "biases"]:
                            slices = []
                            for idx in unique_list:
                                arr = expert_cache.get_attr(i, idx, proj_name, attr_name)
                                if arr is not None:
                                    slices.append(arr)
                                else:
                                    raise RuntimeError(
                                        f"Expert cache miss in re-run: layer={i} expert={idx} "
                                        f"{proj_name}.{attr_name}")
                            if slices:
                                expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)

                    mx.eval(*expert_tensors.values())

                    y = compute_moe_direct(
                        h_post, remapped_inds, expert_tensors,
                        group_size=qparams["group_size"],
                        bits=qparams["bits"],
                        mode=qparams["mode"],
                    )
                    y = (y * scores[..., None]).sum(axis=-2)

                    shared_y = layer.mlp.shared_expert(h_post)
                    shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
                    y = y + shared_y

                    h_v = h_mid + y
                    mx.eval(h_v)
                    expert_cache.unprotect()
                    del expert_tensors

                mx.clear_cache()

        del cache_snapshot

        # Update verifier_pending_pred for the next round.
        # This is the verifier's argmax prediction for the token AFTER the last
        # accepted token. We already have this from v_logits if the rejection
        # point gave us the right position.
        #
        # Cases:
        #   d0 wrong: accepted = [verifier_pending_pred]. The verifier's cache is
        #     unchanged (restored). v_logits are from the bad forward pass, unusable.
        #     We don't have a valid pending pred from v_logits. But the correction
        #     token (verifier_pending_pred from last round) was committed. We need
        #     the verifier's prediction for what follows THAT token. This requires
        #     running the correction token through the verifier. We'll do that as
        #     the last accepted token feed-through below.
        #
        #   Rejected d_{j+1}: accepted = [d_0, ..., d_j, correction].
        #     correction = v_preds[j]. The verifier's cache has been rolled back
        #     to d_0..d_j (n_valid_in_cache = j+1). We committed d_0..d_j plus the
        #     correction. We need to feed the correction token to the verifier and
        #     get its next prediction. OR we can use the v_logits from the original
        #     forward pass since v_preds[j] came from v_logits[:, j, :]. The token
        #     after the correction is NOT available from v_logits (the correction
        #     token was never fed to the verifier).
        #
        #   All K accepted + bonus: accepted = [d_0, ..., d_{K-1}, bonus].
        #     bonus = v_preds[K-1]. Cache has all K positions. We committed all K
        #     plus the bonus. We need the verifier's prediction for what follows
        #     the bonus token. This requires feeding the bonus token to the verifier.
        #
        # In all cases where we committed a token that the verifier hasn't "seen"
        # (correction or bonus), we need to run that token through the verifier
        # to get the next pending prediction. The last accepted token is always
        # either a correction, bonus, or verifier_pending_pred -- none of which
        # are in the verifier cache yet.

        last_accepted = accepted_tokens[-1]
        last_accepted_input = mx.array([[last_accepted]])

        # Feed the last accepted token through the verifier to:
        # 1. Update the verifier cache to include this token
        # 2. Get verifier_pending_pred for the next round
        h_v = text_model_v.embed_tokens(last_accepted_input)
        mx.eval(h_v)
        fa_mask = create_attention_mask(h_v, verifier_cache[text_model_v.fa_idx])
        ssm_mask = create_ssm_mask(h_v, verifier_cache[text_model_v.ssm_idx])
        for i in range(num_layers_v):
            layer = layers_v[i]
            c_v = verifier_cache[i]
            entries = weight_index.get(i, [])
            _, expert_entries = split_layer_entries(entries)

            x_normed = layer.input_layernorm(h_v)
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer.is_linear:
                r = layer.linear_attn(x_normed, mask, c_v)
            else:
                r = layer.self_attn(x_normed, mask, c_v)
            h_mid = h_v + r
            mx.eval(h_mid)

            h_post = layer.post_attention_layernorm(h_mid)
            gates = layer.mlp.gate(h_post)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k_exp = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k_exp, axis=-1)[..., -k_exp:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds)

            inds_np = np.array(inds.tolist())
            unique_experts = np.unique(inds_np)
            unique_list = unique_experts.tolist()
            remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
            remap[unique_experts] = np.arange(len(unique_experts))
            remapped_inds = mx.array(remap[inds_np])

            expert_file_map = {}
            for name, filepath in expert_entries:
                expert_file_map[name] = filepath

            expert_cache.protect([(i, idx) for idx in unique_list])
            uncached_list = []
            for idx in unique_list:
                if expert_cache.has_expert(i, idx):
                    expert_cache.record_hit()
                    expert_cache.touch(i, idx)
                else:
                    expert_cache.record_miss()
                    uncached_list.append(idx)
            if uncached_list:
                if use_pread and pread_index is not None:
                    batch_results, io_stats = pread_expert_batch(
                        pread_index, pread_fds, i, uncached_list)
                    for eidx, attrs in batch_results.items():
                        for (pn, an), arr in attrs.items():
                            expert_cache.put_attr(i, eidx, pn, an, arr)
                else:
                    for filepath in set(expert_file_map.values()):
                        if filepath not in header_cache:
                            header_cache[filepath] = parse_safetensors_header(filepath)
                    with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                        futures = [
                            executor.submit(_read_single_expert_attrs, eidx, i, expert_file_map, header_cache)
                            for eidx in uncached_list
                        ]
                        for future in futures:
                            eidx, attrs, _ = future.result()
                            for (pn, an), arr in attrs.items():
                                expert_cache.put_attr(i, eidx, pn, an, arr)

            expert_tensors = {}
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                for attr_name in ["weight", "scales", "biases"]:
                    slices = [expert_cache.get_attr(i, idx, proj_name, attr_name) for idx in unique_list]
                    if all(s is not None for s in slices):
                        expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)
            mx.eval(*expert_tensors.values())

            y = compute_moe_direct(
                h_post, remapped_inds, expert_tensors,
                group_size=qparams["group_size"], bits=qparams["bits"], mode=qparams["mode"],
            )
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = layer.mlp.shared_expert(h_post)
            shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
            y = y + shared_y
            h_v = h_mid + y
            mx.eval(h_v)
            expert_cache.unprotect()
            del expert_tensors
        mx.clear_cache()

        h_v = text_model_v.norm(h_v)
        if lm_v.args.tie_word_embeddings:
            _pp_logits = text_model_v.embed_tokens.as_linear(h_v)
        else:
            _pp_logits = lm_v.lm_head(h_v)
        mx.eval(_pp_logits)
        verifier_pending_pred = mx.argmax(_pp_logits[:, -1, :], axis=-1).item()
        del _pp_logits

        # Draft model cache rollback.
        # The draft model ran K tokens autoregressively, building up its cache.
        # We accepted n_valid_in_cache draft tokens (0 if d_0 was wrong).
        # Trim the excess.
        trim_amount = K - n_valid_in_cache
        if trim_amount > 0:
            from mlx_lm.models.cache import trim_prompt_cache
            trim_prompt_cache(draft_cache, trim_amount)

        # Set up next draft input = last accepted token
        draft_input = mx.array([[last_accepted]])

        # ============================================================
        # Progress reporting
        # ============================================================
        t_round_end = time.time()
        round_time = t_round_end - t_round_start
        round_times.append(round_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        n_generated = len(generated_tokens)
        elapsed = t_round_end - t_start
        effective_tps = n_generated / elapsed if elapsed > 0 else 0
        accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
        cache_hr = expert_cache.hit_rate

        n_acc_this_round = len(accepted_tokens)
        if all_accepted_with_bonus:
            round_label = f"+1 bonus"
        elif d0_was_wrong:
            round_label = "d0 rejected"
        else:
            round_label = "correction"
        print(f"  [token {n_generated}] accepted {n_acc_this_round}/{K} "
              f"({round_label}), "
              f"effective {effective_tps:.1f} tok/s, "
              f"accept_rate {accept_rate:.0%}, cache_hit {cache_hr:.0%}, "
              f"draft {draft_time*1000:.0f}ms verify {verify_time*1000:.0f}ms "
              f"mem {cur_mem:.1f}GB",
              flush=True)

        # Safety: check memory pressure every few rounds
        if total_rounds % 3 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical).")
                break

        # Truncate if we generated more than requested
        if len(generated_tokens) > max_tokens:
            generated_tokens = generated_tokens[:max_tokens]
            break

    # === Cleanup ===
    for fh in file_handle_cache.values():
        try:
            fh.close()
        except Exception:
            pass

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Summary stats
    avg_draft_ms = np.mean(draft_times) * 1000 if draft_times else 0
    avg_verify_ms = np.mean(verify_times) * 1000 if verify_times else 0
    avg_round_ms = np.mean(round_times) * 1000 if round_times else 0
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tokens_per_round = total_tokens / total_rounds if total_rounds > 0 else 0

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": (prefill_draft_time + prefill_verify_time) * 1000,
        "peak_mem_gb": peak_mem,
        "preload_time": preload_time,
        "prefill_draft_time": prefill_draft_time,
        "prefill_verify_time": prefill_verify_time,
        "total_rounds": total_rounds,
        "draft_k": draft_k,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate": accept_rate,
        "tokens_per_round": tokens_per_round,
        "avg_draft_ms": avg_draft_ms,
        "avg_verify_ms": avg_verify_ms,
        "avg_round_ms": avg_round_ms,
        "expert_cache_hits": expert_cache.hits,
        "expert_cache_misses": expert_cache.misses,
        "expert_cache_hit_rate": expert_cache.hit_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming inference engine")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--mode", choices=["baseline", "layerwise", "stream", "lazy", "offload", "offload_lazy",
                                          "offload_selective", "speculative"],
                        default="stream",
                        help="baseline=mlx_lm native, layerwise=manual forward with timing, "
                             "stream=reload weights from safetensors per layer, "
                             "lazy=load with lazy=True (mmap, OS handles paging), "
                             "offload=per-layer load/compute/clear for models larger than DRAM, "
                             "offload_lazy=like offload but skips mx.eval(params) so only "
                             "accessed expert pages are read via lazy mmap (~40MB vs ~1.3GB/layer), "
                             "offload_selective=run router first, then load only selected "
                             "expert slices per layer (large I/O reduction vs full layer load), "
                             "speculative=draft tokens with small model, verify with large model")
    parser.add_argument("--max-mem-gb", type=float, default=40.0,
                        help="Abort if RSS exceeds this (GB)")
    parser.add_argument("--draft-model", default="mlx-community/Qwen3.5-35B-A3B-4bit",
                        help="Draft model for speculative decoding (must fit in DRAM)")
    parser.add_argument("--draft-k", type=int, default=8,
                        help="Number of draft tokens per speculation round")
    parser.add_argument("--preload-topk", type=int, default=0,
                        help="Pre-load N hottest experts per layer into cache at startup. "
                             "0=disabled (default). 50 covers ~93.6%% of activations (~20 GiB), "
                             "75 covers ~98.4%% (~30 GiB). Requires offload_selective or speculative mode.")
    parser.add_argument("--cache-gb", type=float, default=20.0,
                        help="Expert cache size limit in GB (default: 20). "
                             "Increase when using --preload-topk (e.g. 30 for topk=75). Max: 35.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable detailed per-layer profiling instrumentation in offload_selective mode. "
                             "Prints routing/cache/IO/compute/sync breakdown per layer after generation.")
    parser.add_argument("--use-pread", action="store_true", default=False,
                        help="Use pread()-based expert loading instead of safetensors mmap. "
                             "Requires expert_index.json in working directory. "
                             "Bypasses kernel buffer cache (F_NOCACHE) for direct SSD reads.")
    parser.add_argument("--use-cext", action="store_true", default=False,
                        help="Use fast_expert_io C extension for expert loading. "
                             "Requires expert_index.json and compiled fast_expert_io.so. "
                             "Uses preadv + coalesced I/O with dedicated thread pool.")
    parser.add_argument("--batch-experts", action="store_true", default=False,
                        help="Reduce Metal kernel launch overhead by fusing mx.eval() syncs. "
                             "Merges attention+routing into one sync and skips the separate "
                             "expert weight eval, cutting per-layer syncs from 4 to 2. "
                             "Only affects offload_selective mode. Output is identical.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Override number of active experts per token (0 = use model default). "
                             "Lower values reduce I/O at the cost of quality. Model default is 10.")
    parser.add_argument("--no-expert-cache", action="store_true", default=False,
                        help="Skip the explicit ExpertCache and rely on OS page cache for expert "
                             "weight access. Reads ALL active experts from mmap'd safetensors per "
                             "token (memcpy from page cache after warmup). Frees ~26GB Metal heap "
                             "for OS page cache, trading per-token read volume for page-cache speed.")
    parser.add_argument("--pin-experts", type=int, default=0,
                        help="Online expert pinning: warmup for N tokens tracking routing, then "
                             "pin the most frequently activated experts in GPU memory. Pinned "
                             "experts skip disk I/O entirely. 0=disabled (default). Typical: 20.")
    parser.add_argument("--pin-topk", type=int, default=51,
                        help="Number of experts to pin per layer after warmup (default: 51, "
                             "~top 10%% of 512 experts). Higher = more memory, higher hit rate.")
    parser.add_argument("--async-pipeline", action="store_true", default=False,
                        help="Overlap disk I/O with GPU computation by dispatching pread futures "
                             "for layer N while GPU processes layer N+1's attention+routing. "
                             "Requires --batch-experts --no-expert-cache and packed expert files. "
                             "Standalone benchmark showed 3.86x speedup (364ms -> 94ms).")
    parser.add_argument("--skip-layers", type=int, default=0,
                        help="In layers 10..N-10, keep only every Nth layer (0=disabled). "
                             "Reduces compute+I/O at the cost of quality. --skip-layers 2 = 40/60 layers.")
    parser.add_argument("--batch-layers", type=int, default=0,
                        help="Batch N layers of attention+routing into a single mx.eval() call, "
                             "then batch-read all needed experts for all N layers at once. "
                             "Reduces Metal sync overhead from 60 evals to 60/N evals. "
                             "Routing for layers 2-N in each batch uses approximate h (no expert "
                             "contribution from prior layers in the batch). 0=disabled. Typical: 4. "
                             "Requires --batch-experts --no-expert-cache and packed expert files.")
    parser.add_argument("--numpy-cache-gb", type=float, default=0.0,
                        help="CPU-side numpy LRU cache for expert raw bytes (GB). When > 0, "
                             "caches pread results as Python bytes objects in CPU memory (not Metal "
                             "heap). On hit: memcpy to mx.array (~0.11ms). On miss: pread from "
                             "packed file (~0.7ms). Requires --no-expert-cache and packed expert "
                             "files. 0=disabled (default). Typical: 5.")
    args = parser.parse_args()

    # Validate --cache-gb range
    if args.cache_gb > 35:
        print(f"WARNING: --cache-gb {args.cache_gb:.1f} exceeds safety max of 35 GB. Capping to 35.")
        args.cache_gb = 35.0
    if args.cache_gb < 1:
        print(f"WARNING: --cache-gb {args.cache_gb:.1f} too small. Using 1 GB.")
        args.cache_gb = 1.0

    # Validate --preload-topk is only used with compatible modes
    if args.preload_topk > 0 and args.mode not in ("offload_selective", "speculative"):
        print(f"WARNING: --preload-topk only works with offload_selective or speculative mode. Ignoring.")
        args.preload_topk = 0

    # Validate --pin-experts is only used with offload_selective
    if args.pin_experts > 0 and args.mode != "offload_selective":
        print(f"WARNING: --pin-experts only works with offload_selective mode. Ignoring.")
        args.pin_experts = 0

    # Validate --async-pipeline is only used with offload_selective
    if args.async_pipeline and args.mode != "offload_selective":
        print(f"WARNING: --async-pipeline only works with offload_selective mode. Ignoring.")
        args.async_pipeline = False

    # Validate --batch-layers is only used with offload_selective
    if args.batch_layers > 0 and args.mode != "offload_selective":
        print(f"WARNING: --batch-layers only works with offload_selective mode. Ignoring.")
        args.batch_layers = 0

    if args.pin_experts > 0 and args.pin_experts >= args.tokens:
        print(f"WARNING: --pin-experts ({args.pin_experts}) >= --tokens ({args.tokens}). "
              f"Need at least 1 token after warmup. Reducing to {args.tokens - 1}.")
        args.pin_experts = max(1, args.tokens - 1)

    t_start = time.time()
    mem_before = get_mem_gb()

    print(f"[{fmt_time(0)}] Mode: {args.mode}")
    print(f"[{fmt_time(0)}] Loading model: {args.model}")
    print(f"[{fmt_time(0)}] Memory before load: {mem_before:.1f} GB")
    if args.preload_topk > 0:
        print(f"[{fmt_time(0)}] Expert preload: top-{args.preload_topk}/layer, cache={args.cache_gb:.0f} GB")

    # Resolve model path (for safetensors access)
    model_path = resolve_model_path(args.model)

    if args.mode == "speculative":
        # Speculative decoding: load BOTH draft (small) and verifier (large) models.
        # Draft model: fully in DRAM via mlx_lm.load()
        # Verifier model: empty shell + global weights (experts loaded from SSD)
        draft_model_path = resolve_model_path(args.draft_model)

        print(f"[{fmt_time(time.time() - t_start)}] Loading draft model: {args.draft_model}")
        draft_model, tokenizer = mlx_lm.load(str(draft_model_path))
        mx.eval(draft_model.parameters())
        mem_after_draft = get_mem_gb()
        print(f"[{fmt_time(time.time() - t_start)}] Draft model loaded. "
              f"Memory: {mem_after_draft:.1f} GB (+{mem_after_draft - mem_before:.1f} GB)")

        print(f"[{fmt_time(time.time() - t_start)}] Loading verifier model: {args.model}")
        model, _ = load_model_no_weights(model_path)
        # Cap wired memory: draft (~7GB) + verifier non-expert (~5GB) + expert cache
        # With preloaded experts, cache may be larger than default 20GB
        wired_gb = min(args.cache_gb + 15, args.max_mem_gb * 0.85, 43)
        mx.set_wired_limit(int(wired_gb * 1024**3))
        print(f"[{fmt_time(time.time() - t_start)}] Verifier loaded (global weights only). "
              f"wired limit={wired_gb:.0f}GB")

    elif args.mode in ("offload", "offload_lazy", "offload_selective"):
        # Offload mode: create empty model, load ONLY global weights (~1GB).
        # Layer weights are loaded/cleared per-layer during inference.
        # This is the only mode that works for model_size > DRAM.
        # offload_lazy variant skips mx.eval(params) to test lazy mmap expert paging.
        # offload_selective variant runs router first, then loads only selected expert slices.
        print(f"[{fmt_time(time.time() - t_start)}] Using offload loader (no layer weights)...")
        model, tokenizer = load_model_no_weights(model_path)
        # Cap wired memory — needs non-expert weights (~3-5GB) + expert cache (if used)
        # With --no-expert-cache, no Metal heap for experts; just non-expert + 5GB headroom.
        # With --pin-experts, need extra headroom for pinned expert arrays (~20-30GB).
        if args.pin_experts > 0 and args.mode == "offload_selective":
            # Pinned experts need significant GPU memory: non-expert (~5GB) + pinned (~20-30GB)
            wired_gb = min(args.cache_gb + 8 + 25, args.max_mem_gb * 0.9, 43)
        elif getattr(args, 'no_expert_cache', False) and args.mode == "offload_selective":
            wired_gb = min(5 + 5, args.max_mem_gb * 0.8, 43)  # non-expert (~5GB) + 5GB headroom
        else:
            wired_gb = min(args.cache_gb + 8, args.max_mem_gb * 0.8, 43)
        mx.set_wired_limit(int(wired_gb * 1024**3))
        mode_note = ""
        if args.mode == "offload_lazy":
            mode_note = " [lazy_eval=True]"
        elif args.mode == "offload_selective":
            mode_note = " [selective expert loading]"
            if getattr(args, 'no_expert_cache', False):
                mode_note += " [no-expert-cache]"
            if args.pin_experts > 0:
                mode_note += f" [pin-experts={args.pin_experts}, pin-topk={args.pin_topk}]"
        print(f"[{fmt_time(time.time() - t_start)}] Loaded (global weights only). "
              f"wired limit={wired_gb:.0f}GB{mode_note}")
    elif args.mode == "lazy":
        # Lazy mode: load all weights via mmap, let OS handle paging.
        # WARNING: thrashes on models larger than DRAM. Use offload instead.
        print(f"[{fmt_time(time.time() - t_start)}] Using custom loader (lazy mmap)...")
        model, tokenizer = load_model_custom(model_path)
        # Pin only embed_tokens, norm, and lm_head in DRAM
        lm = model.language_model
        text_model = lm.model
        mx.eval(text_model.embed_tokens.parameters())
        mx.eval(text_model.norm.parameters())
        if hasattr(lm, 'lm_head'):
            mx.eval(lm.lm_head.parameters())
        # Cap wired memory to leave room for the rest of the system
        wired_gb = min(args.max_mem_gb * 0.6, 28)  # ~60% of limit or 28GB max
        mx.set_wired_limit(int(wired_gb * 1024**3))
        print(f"[{fmt_time(time.time() - t_start)}] Loaded. Pinned essentials, "
              f"wired limit={wired_gb:.0f}GB")
    else:
        # Full load: everything in DRAM
        model, tokenizer = mlx_lm.load(str(model_path))
        mx.eval(model.parameters())

    mem_after_load = get_mem_gb()
    t_loaded = time.time() - t_start
    print(f"[{fmt_time(t_loaded)}] Model loaded. Memory: {mem_after_load:.1f} GB "
          f"(+{mem_after_load - mem_before:.1f} GB)")

    if args.mode not in ("lazy", "offload", "offload_lazy", "offload_selective", "speculative") and mem_after_load > args.max_mem_gb:
        print(f"ABORT: Memory {mem_after_load:.1f} GB exceeds limit {args.max_mem_gb} GB")
        sys.exit(1)

    # Count parameters
    total_params = sum(p.size for n, p in mlx.utils.tree_flatten(model.parameters()))
    params_b = total_params / 1e9

    # Build weight index for streaming/offload modes
    weight_index = build_weight_index(model_path) if args.mode in ("stream", "offload", "offload_lazy", "offload_selective", "speculative") else None

    # Load pread expert index if requested
    pread_index = None
    pread_fds = None
    if args.use_pread and args.mode in ("offload_selective", "speculative"):
        print(f"[{fmt_time(time.time() - t_start)}] Loading expert_index.json for pread() path...")
        pread_index, pread_fds = load_expert_index(model_path)

    # Initialize fast_expert_io C extension if requested
    if args.use_cext and args.mode in ("offload_selective", "speculative"):
        print(f"[{fmt_time(time.time() - t_start)}] Initializing fast_expert_io C extension...")
        import fast_expert_io

        # Check for packed_experts/ directory (1 read per expert, much faster)
        packed_dir = Path(model_path) / "packed_experts"
        packed_layout_path = packed_dir / "layout.json"

        if packed_layout_path.exists():
            # Packed mode: one file per layer, contiguous expert blocks
            with open(packed_layout_path) as f:
                packed_layout = json.load(f)

            fast_expert_io.init(num_workers=8)
            fast_expert_io.register_packed_files(str(packed_dir), packed_layout)
            atexit.register(fast_expert_io.shutdown)

            num_packed_layers = packed_layout.get("num_layers", 0)
            num_packed_comps = len(packed_layout.get("components", []))
            expert_size_mb = packed_layout.get("expert_size", 0) / 1e6
            print(f"[cext] Packed mode: {num_packed_layers} layers, "
                  f"{num_packed_comps} components/expert, "
                  f"{expert_size_mb:.1f} MB/expert, 8 workers")
            del packed_layout
        else:
            # Scattered mode: read from safetensors shards (9 reads per expert)
            index_path = Path("expert_index.json")
            if not index_path.exists():
                index_path = Path(model_path) / "expert_index.json"
            if not index_path.exists():
                print("ERROR: expert_index.json not found. Required for --use-cext.")
                sys.exit(1)
            with open(index_path) as f:
                cext_expert_index = json.load(f)

            # Build file_dict: {filename: full_path}
            shard_files = set()
            for tensor_info in cext_expert_index["tensors"].values():
                shard_files.add(tensor_info["file"])
            for layer_reads in cext_expert_index.get("expert_reads", {}).values():
                for comp_info in layer_reads.values():
                    shard_files.add(comp_info["file"])

            resolved_model_path = Path(cext_expert_index.get("model_path", str(model_path)))
            cext_file_dict = {}
            for filename in sorted(shard_files):
                filepath = resolved_model_path / filename
                if not filepath.exists():
                    filepath = Path(model_path) / filename
                cext_file_dict[filename] = str(filepath)

            fast_expert_io.init(num_workers=8)
            fast_expert_io.register_files(cext_file_dict, cext_expert_index)
            atexit.register(fast_expert_io.shutdown)

            num_cext_layers = len(cext_expert_index.get("expert_reads", {}))
            print(f"[cext] Scattered mode: {num_cext_layers} layers, "
                  f"{len(cext_file_dict)} shard files, 8 workers")
            del cext_expert_index, cext_file_dict

    # Generate
    print(f"[{fmt_time(time.time() - t_start)}] Generating {args.tokens} tokens ({args.mode})...")
    print(f"[{fmt_time(time.time() - t_start)}] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")

    if args.mode == "baseline":
        result = generate_baseline(model, tokenizer, args.prompt, args.tokens)
    elif args.mode == "speculative":
        result = generate_speculative(
            draft_model, model, tokenizer, args.prompt, args.tokens,
            weight_index, model_path, draft_k=args.draft_k,
            preload_topk=args.preload_topk, cache_gb=args.cache_gb,
            use_pread=args.use_pread, pread_index=pread_index, pread_fds=pread_fds,
        )
    elif args.mode == "offload_selective":
        # Selective offload: run attention+router first, then load only selected expert slices.
        # Large I/O reduction vs full offload (only active experts loaded per layer).
        result = generate_offload_selective(model, tokenizer, args.prompt, args.tokens,
                                            weight_index, model_path,
                                            preload_topk=args.preload_topk,
                                            cache_gb=args.cache_gb,
                                            profile=args.profile,
                                            use_pread=args.use_pread,
                                            pread_index=pread_index,
                                            pread_fds=pread_fds,
                                            use_cext=args.use_cext,
                                            batch_experts=args.batch_experts,
                                            top_k_override=args.top_k,
                                            no_expert_cache=args.no_expert_cache,
                                            pin_experts=args.pin_experts,
                                            pin_topk=args.pin_topk,
                                            async_pipeline=args.async_pipeline,
                                            skip_layers=args.skip_layers,
                                            batch_layers=args.batch_layers,
                                            numpy_cache_gb=args.numpy_cache_gb)
    elif args.mode in ("offload", "offload_lazy"):
        # Offload mode: explicit per-layer load -> compute -> clear cycle.
        # Only way to run models larger than DRAM without OS thrashing.
        # offload_lazy: skip mx.eval(params) so lazy mmap only reads accessed experts.
        result = generate_offload(model, tokenizer, args.prompt, args.tokens,
                                  weight_index, model_path,
                                  lazy_eval=(args.mode == "offload_lazy"))
    elif args.mode == "lazy":
        # Lazy mode: all weights mmap'd, OS handles paging
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=None, mode="layerwise")
    else:
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=weight_index, mode=args.mode)

    # Summary
    peak_mem = max(result["peak_mem_gb"], get_mem_gb())
    print(f"\n[{fmt_time(time.time() - t_start)}] Done. {result['tokens']} tokens in "
          f"{result['total_time']:.1f}s")
    print(f"Generated: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
    print()

    # Mode-specific stats
    if args.mode == "speculative":
        print(f"Speculative decoding summary:")
        print(f"  Draft model:      {args.draft_model}")
        print(f"  Draft K:          {result.get('draft_k', 0)}")
        print(f"  Total rounds:     {result.get('total_rounds', 0)}")
        print(f"  Total drafted:    {result.get('total_drafted', 0)}")
        print(f"  Total accepted:   {result.get('total_accepted', 0)}")
        print(f"  Accept rate:      {result.get('accept_rate', 0):.1%}")
        print(f"  Tokens/round:     {result.get('tokens_per_round', 0):.1f}")
        print(f"  Avg draft time:   {result.get('avg_draft_ms', 0):.0f}ms")
        print(f"  Avg verify time:  {result.get('avg_verify_ms', 0):.0f}ms")
        print(f"  Avg round time:   {result.get('avg_round_ms', 0):.0f}ms")
        print(f"  Prefill draft:    {result.get('prefill_draft_time', 0):.1f}s")
        print(f"  Prefill verifier: {result.get('prefill_verify_time', 0):.1f}s")
        print(f"  Expert cache HR:  {result.get('expert_cache_hit_rate', 0):.1%}")
        print()

    elif args.mode in ("layerwise", "stream", "lazy", "offload", "offload_lazy", "offload_selective"):
        print(f"Per-token breakdown (generation phase, excluding prompt):")
        print(f"  Avg weight load: {result.get('avg_load_ms_per_token', 0):.1f}ms")
        print(f"  Avg compute:     {result.get('avg_compute_ms_per_token', 0):.1f}ms")
        if result.get('avg_clear_ms_per_token', 0) > 0:
            print(f"  Avg clear:       {result.get('avg_clear_ms_per_token', 0):.1f}ms")
        if args.mode == "offload_selective":
            print(f"  Avg non-expert load: {result.get('avg_load_nonexpert_ms_per_token', 0):.1f}ms")
            print(f"  Avg attn+router:     {result.get('avg_attn_router_ms_per_token', 0):.1f}ms")
            print(f"  Avg expert load+run: {result.get('avg_expert_ms_per_token', 0):.1f}ms")

        if result.get("per_layer_compute_ms"):
            linear_times = []
            fa_times = []
            layers = model.language_model.model.layers
            for i, t in enumerate(result["per_layer_compute_ms"]):
                if layers[i].is_linear:
                    linear_times.append(t)
                else:
                    fa_times.append(t)

            if linear_times:
                print(f"  Avg linear_attn layer: {np.mean(linear_times):.2f}ms "
                      f"({len(linear_times)} layers)")
            if fa_times:
                print(f"  Avg full_attn layer:   {np.mean(fa_times):.2f}ms "
                      f"({len(fa_times)} layers)")

        if result.get("per_layer_load_ms"):
            avg_load_per_layer = np.mean(result["per_layer_load_ms"])
            total_load = sum(result["per_layer_load_ms"])
            print(f"  Avg safetensors load/layer: {avg_load_per_layer:.2f}ms")
            print(f"  Total load per token: {total_load:.0f}ms")
        print()

    # Compute active params (MoE heuristic)
    active_b = params_b
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    match = re.search(r'A(\d+)B', model_name, re.IGNORECASE)
    if match:
        active_b = float(match.group(1))

    # Machine-parseable result line
    print(f"RESULT model={model_name} params_B={params_b:.1f} active_B={active_b:.1f} "
          f"tok_sec={result['tok_sec']:.2f} ttft_ms={result['ttft_ms']:.0f} "
          f"mem_gb={peak_mem:.1f} tokens={result['tokens']} mode={args.mode}")


if __name__ == "__main__":
    main()
