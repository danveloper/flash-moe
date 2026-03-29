#!/usr/bin/env python3
"""
extract_weights.py — Extract all non-expert weights from any Qwen3.5 MoE model
into a single binary file that the C inference engine can mmap.

Outputs:
  - model_weights.bin: binary blob containing all non-expert weight tensors
  - model_weights.json: manifest describing each tensor's location, shape, dtype

The binary format is simple:
  - Tensors are packed contiguously, 64-byte aligned
  - Each tensor is stored in its native format (U32 packed, BF16 as uint16, F32)
  - The JSON manifest maps tensor names to {offset, size, shape, dtype}

Usage:
    python extract_weights.py [--model PATH] [--output DIR]
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory (safetensors)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for model_weights.bin and .json')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Filter: keep only language_model weights, skip vision_tower
    # Also skip expert weights (switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases})
    # unless --include-experts is set
    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}  # name -> filename
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not args.include_experts and expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    # Group by shard file for sequential I/O
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    # Parse headers and plan layout
    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    # Sanitize tensor names: remove "language_model." prefix for the C engine
    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    # Plan the output layout
    # Sort tensors for deterministic output
    all_tensors = []  # (sanitized_name, original_name, filename)
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    # Read model config from config.json (auto-detect text_config nesting for VL models)
    config_path = model_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            raw_config = json.load(f)
        # VL models nest the text config under "text_config"
        cfg = raw_config.get('text_config', raw_config)
        print(f"Config: {config_path} (text_config={'text_config' in raw_config})")
    else:
        print(f"WARNING: {config_path} not found, using defaults for Qwen3.5-397B")
        cfg = {}

    num_layers = cfg.get('num_hidden_layers', 60)
    full_attn_interval = cfg.get('full_attention_interval', 4)

    # Write binary file
    bin_path = output_dir / 'model_weights.bin'
    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        # Model config for the C engine — auto-detected from config.json
        "config": {
            "hidden_size": cfg.get('hidden_size', 4096),
            "num_hidden_layers": num_layers,
            "num_attention_heads": cfg.get('num_attention_heads', 32),
            "num_key_value_heads": cfg.get('num_key_value_heads', 2),
            "head_dim": cfg.get('head_dim', 256),
            "vocab_size": cfg.get('vocab_size', 248320),
            "rms_norm_eps": cfg.get('rms_norm_eps', 1e-6),
            "num_experts": cfg.get('num_experts', 512),
            "num_experts_per_tok": cfg.get('num_experts_per_tok', 4),
            "moe_intermediate_size": cfg.get('moe_intermediate_size', 1024),
            "shared_expert_intermediate_size": cfg.get('shared_expert_intermediate_size', 1024),
            "full_attention_interval": full_attn_interval,
            "linear_num_value_heads": cfg.get('linear_num_value_heads', 64),
            "linear_num_key_heads": cfg.get('linear_num_key_heads', 16),
            "linear_key_head_dim": cfg.get('linear_key_head_dim', 128),
            "linear_value_head_dim": cfg.get('linear_value_head_dim', 128),
            "linear_conv_kernel_dim": cfg.get('linear_conv_kernel_dim', 4),
            "partial_rotary_factor": cfg.get('partial_rotary_factor', 0.25),
            "rope_theta": cfg.get('rope_theta', 10000000.0),
        }
    }

    # Layer type map — auto-generated from config
    layer_types = cfg.get('layer_types', None)
    if layer_types is None:
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % full_attn_interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0

    ALIGN = 64  # 64-byte alignment for Metal buffers

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            filepath = model_path / filename
            header, data_start = header_cache[filename]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                continue

            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            shape = meta['shape']
            dtype = meta['dtype']

            # Align offset
            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            # Read tensor data from safetensors
            with open(filepath, 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            out_f.write(data)

            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    # Write manifest
    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Print summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
