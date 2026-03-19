#!/usr/bin/env python3
"""
generate_expert_index.py — Generate expert_index.json from Qwen3.5-397B-A17B safetensors.

Scans safetensors file headers to build a mapping of (layer, component) -> (file, offset, stride).
This index is required by repack_experts.py.

Usage:
    python generate_expert_index.py [--model PATH] [--output expert_index.json]
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


# Expected component sizes per expert (bytes)
COMPONENT_SIZES = {
    "gate_proj.weight": 2097152,   # [1024, 512] uint32
    "gate_proj.scales": 131072,    # [1024, 64] uint16
    "gate_proj.biases": 131072,    # [1024, 64] uint16
    "up_proj.weight":   2097152,   # [1024, 512] uint32
    "up_proj.scales":   131072,    # [1024, 64] uint16
    "up_proj.biases":   131072,    # [1024, 64] uint16
    "down_proj.weight": 2097152,   # [4096, 128] uint32
    "down_proj.scales": 131072,    # [4096, 16] uint16
    "down_proj.biases": 131072,    # [4096, 16] uint16
}

NUM_EXPERTS = 512
NUM_LAYERS = 60

# Pattern: language_model.model.layers.{L}.mlp.switch_mlp.{component}
EXPERT_PATTERN = re.compile(
    r'^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.((?:gate|up|down)_proj\.(?:weight|scales|biases))$'
)


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description='Generate expert_index.json from safetensors')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory (containing safetensors files)')
    parser.add_argument('--output', type=str, default='expert_index.json',
                        help='Output path for expert_index.json')
    args = parser.parse_args()

    model_path = Path(args.model)

    # Load weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Find all expert tensors and group by (layer, component) -> filename
    expert_tensors = {}  # (layer_idx, component) -> (tensor_name, filename)
    for name, filename in weight_map.items():
        m = EXPERT_PATTERN.match(name)
        if m:
            layer_idx = int(m.group(1))
            component = m.group(2)
            expert_tensors[(layer_idx, component)] = (name, filename)

    print(f"Model: {model_path}")
    print(f"Found {len(expert_tensors)} expert tensors")
    print(f"Expected: {NUM_LAYERS * len(COMPONENT_SIZES)} = {NUM_LAYERS} layers x {len(COMPONENT_SIZES)} components")

    if len(expert_tensors) != NUM_LAYERS * len(COMPONENT_SIZES):
        print("WARNING: tensor count mismatch", file=sys.stderr)

    # Parse safetensors headers for all needed files
    needed_files = set(fn for _, fn in expert_tensors.values())
    print(f"\nParsing {len(needed_files)} safetensors file headers...")

    header_cache = {}
    for filename in sorted(needed_files):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))
        print(f"  {filename}: header parsed")

    # Build expert_reads index
    expert_reads = defaultdict(dict)

    for (layer_idx, component), (tensor_name, filename) in sorted(expert_tensors.items()):
        header, data_start = header_cache[filename]

        if tensor_name not in header:
            # Skip __metadata__ key
            if tensor_name == '__metadata__':
                continue
            print(f"WARNING: {tensor_name} not in {filename} header", file=sys.stderr)
            continue

        meta = header[tensor_name]
        tensor_offset = meta['data_offsets'][0]
        tensor_size = meta['data_offsets'][1] - meta['data_offsets'][0]

        # The tensor contains all 512 experts contiguously
        # expert_size = total_tensor_size / num_experts
        expert_size = tensor_size // NUM_EXPERTS

        expected_size = COMPONENT_SIZES.get(component)
        if expected_size and expert_size != expected_size:
            print(f"WARNING: {tensor_name}: computed expert_size={expert_size}, "
                  f"expected={expected_size}", file=sys.stderr)

        # abs_offset = data section start + tensor's offset within data section
        abs_offset = data_start + tensor_offset

        # expert_stride = expert_size (experts are packed contiguously)
        expert_stride = expert_size

        expert_reads[str(layer_idx)][component] = {
            "file": filename,
            "abs_offset": abs_offset,
            "expert_stride": expert_stride,
            "expert_size": expert_size,
        }

    # Verify completeness
    complete = True
    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"ERROR: layer {layer_idx} missing entirely", file=sys.stderr)
            complete = False
            continue
        for comp in COMPONENT_SIZES:
            if comp not in expert_reads[layer_key]:
                print(f"ERROR: layer {layer_idx} missing {comp}", file=sys.stderr)
                complete = False

    if not complete:
        print("\nERROR: Index is incomplete", file=sys.stderr)
        sys.exit(1)

    # Write output
    output = {
        "model_path": str(model_path),
        "expert_reads": dict(expert_reads),
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"  {len(expert_reads)} layers, {len(COMPONENT_SIZES)} components each")
    print(f"  Total: {len(expert_reads) * len(COMPONENT_SIZES)} entries")


if __name__ == '__main__':
    main()
