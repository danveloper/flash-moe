#!/usr/bin/env python3
"""Build expert_index.json from model safetensors for repack_experts.py.

Scans the safetensors headers to find expert weight locations and writes
an index file compatible with repack_experts.py.

Usage:
    python build_expert_index.py --model /path/to/model-safetensors --output expert_index.json
"""

import argparse
import json
import struct
import os
import re
from pathlib import Path


def parse_safetensors_header(filepath):
    """Parse safetensors header, return (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model directory with safetensors')
    parser.add_argument('--output', default='expert_index.json', help='Output index file')
    args = parser.parse_args()

    model_path = Path(args.model)
    index_path = model_path / 'model.safetensors.index.json'

    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    # Find all expert weight tensor names
    # Matches both "language_model.model.layers.{L}.mlp.switch_mlp.{proj}.{comp}"
    # and "model.layers.{L}.mlp.switch_mlp.{proj}.{comp}" (no language_model prefix)
    expert_pattern = re.compile(
        r'(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$'
    )

    # Group by (layer, proj.component) -> filename
    expert_tensors = {}  # (layer, "gate_proj.weight") -> (tensor_name, filename)
    for name, filename in weight_map.items():
        m = expert_pattern.match(name)
        if m:
            layer = int(m.group(1))
            proj = m.group(2)
            comp = m.group(3)
            key = (layer, f"{proj}.{comp}")
            expert_tensors[key] = (name, filename)

    # Collect unique layers
    layers = sorted(set(l for l, _ in expert_tensors.keys()))
    print(f"Found {len(layers)} layers with expert weights")
    print(f"Found {len(expert_tensors)} expert tensor entries")

    # Parse headers for all needed files
    needed_files = set(fn for _, fn in expert_tensors.values())
    print(f"Parsing {len(needed_files)} safetensors headers...")

    header_cache = {}
    for fn in sorted(needed_files):
        fp = model_path / fn
        header_cache[fn] = parse_safetensors_header(str(fp))

    # Build expert_reads index
    # For each layer and component, compute:
    #   - file: safetensors filename
    #   - abs_offset: absolute byte offset to expert 0's data
    #   - expert_stride: byte stride between consecutive experts
    #   - expert_size: bytes per expert for this component
    #   - total_size: total bytes for all 512 experts
    #   - shape: [num_experts, out_dim, packed_cols_or_groups]

    expert_reads = {}
    for layer in layers:
        layer_reads = {}
        for comp_key in ['gate_proj.weight', 'gate_proj.scales', 'gate_proj.biases',
                         'up_proj.weight', 'up_proj.scales', 'up_proj.biases',
                         'down_proj.weight', 'down_proj.scales', 'down_proj.biases']:
            key = (layer, comp_key)
            if key not in expert_tensors:
                print(f"  WARNING: missing {comp_key} for layer {layer}")
                continue

            tensor_name, filename = expert_tensors[key]
            header, data_start = header_cache[filename]

            # Find tensor in header (skip __metadata__)
            tensor_info = None
            for k, v in header.items():
                if k == '__metadata__':
                    continue
                if k == tensor_name:
                    tensor_info = v
                    break

            if tensor_info is None:
                print(f"  WARNING: tensor {tensor_name} not found in {filename}")
                continue

            offsets = tensor_info['data_offsets']
            shape = tensor_info['shape']
            dtype = tensor_info['dtype']

            abs_offset = data_start + offsets[0]
            total_size = offsets[1] - offsets[0]

            # shape is [num_experts, out_dim, packed_dim]
            num_experts = shape[0]
            expert_size = total_size // num_experts
            expert_stride = expert_size

            layer_reads[comp_key] = {
                'file': filename,
                'abs_offset': abs_offset,
                'expert_stride': expert_stride,
                'expert_size': expert_size,
                'total_size': total_size,
                'shape': shape,
            }

        expert_reads[str(layer)] = layer_reads

    # Write index
    index = {
        'model_path': str(model_path),
        'expert_reads': expert_reads,
    }

    with open(args.output, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"  {len(layers)} layers, 9 components each")

    # Verify sizes are consistent across layers
    first_layer = expert_reads[str(layers[0])]
    expert_size_total = sum(first_layer[c]['expert_size'] for c in first_layer)
    num_experts = first_layer[list(first_layer.keys())[0]]['shape'][0]
    print(f"  Experts per layer: {num_experts}")
    print(f"  Expert size: {expert_size_total} bytes ({expert_size_total/1024/1024:.2f} MB)")
    for comp, info in first_layer.items():
        print(f"    {comp:25s} {info['expert_size']:>8d} bytes  shape={info['shape']}")

    ok = True
    for layer in layers[1:]:
        for comp in first_layer:
            if comp not in expert_reads[str(layer)]:
                print(f"  MISSING: layer {layer} {comp}")
                ok = False
                continue
            if expert_reads[str(layer)][comp]['expert_size'] != first_layer[comp]['expert_size']:
                print(f"  MISMATCH: layer {layer} {comp}")
                ok = False
    if ok:
        print("  Cross-layer consistency: OK")


if __name__ == '__main__':
    main()
