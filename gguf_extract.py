#!/usr/bin/env python3
"""Extract weights from a GGUF MoE model for Flash-MoE inference.

Reads a .gguf file and produces:
  1. model_weights.bin  — non-expert weights (contiguous binary, 64-byte aligned)
  2. model_weights.json — manifest with tensor offsets, shapes, dtypes, and model config
  3. packed_experts/layer_XX.bin — per-layer expert binaries
  4. vocab.bin          — vocabulary for token decoding

Usage:
    python gguf_extract.py --gguf model.gguf --output ./model_dir
    python gguf_extract.py --gguf model.gguf --output ./model_dir --dry-run
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

# ============================================================================
# GGUF constants
# ============================================================================
GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

# Quant type → (block_elements, block_bytes)
QUANT_SIZES = {
    0:  (1, 4),      # F32
    1:  (1, 2),      # F16
    2:  (32, 18),    # Q4_0
    3:  (32, 20),    # Q4_1
    6:  (32, 22),    # Q5_0
    7:  (32, 24),    # Q5_1
    8:  (32, 34),    # Q8_0
    9:  (32, 40),    # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    30: (1, 2),      # BF16
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


def tensor_nbytes(n_elements, quant_type):
    block_elems, block_bytes = QUANT_SIZES[quant_type]
    return (n_elements // block_elems) * block_bytes


# ============================================================================
# GGUF → engine tensor name mapping
# ============================================================================
# full_attention_interval=4: layers 3, 7, 11, ... are full attention (0-indexed)

def map_tensor_name(gguf_name, full_attn_interval=4):
    """Map a GGUF tensor name to the engine's expected name.

    Returns the mapped name, or the original name if no mapping applies.
    """
    # Global tensors
    if gguf_name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    if gguf_name == 'output.weight':
        return 'lm_head.weight'
    if gguf_name == 'output_norm.weight':
        return 'model.norm.weight'

    # Per-layer tensors: blk.{L}.xxx
    if not gguf_name.startswith('blk.'):
        return gguf_name

    parts = gguf_name.split('.')
    layer = int(parts[1])
    rest = '.'.join(parts[2:])  # e.g. "attn_norm.weight"
    is_full = ((layer + 1) % full_attn_interval == 0)

    # Common mappings (both layer types)
    mapping = {
        # Norms
        'attn_norm.weight': f'model.layers.{layer}.input_layernorm.weight',
        'post_attention_norm.weight': f'model.layers.{layer}.post_attention_layernorm.weight',
        # MoE routing + shared expert
        'ffn_gate_inp.weight': f'model.layers.{layer}.mlp.gate.weight',
        'ffn_gate_shexp.weight': f'model.layers.{layer}.mlp.shared_expert.gate_proj.weight',
        'ffn_up_shexp.weight': f'model.layers.{layer}.mlp.shared_expert.up_proj.weight',
        'ffn_down_shexp.weight': f'model.layers.{layer}.mlp.shared_expert.down_proj.weight',
        'ffn_gate_inp_shexp.weight': f'model.layers.{layer}.mlp.shared_expert_gate.weight',
    }

    if rest in mapping:
        return mapping[rest]

    if is_full:
        # Full attention layers have SEPARATE Q/K/V (not fused)
        full_mapping = {
            'attn_q.weight': f'model.layers.{layer}.self_attn.q_proj.weight',
            'attn_k.weight': f'model.layers.{layer}.self_attn.k_proj.weight',
            'attn_v.weight': f'model.layers.{layer}.self_attn.v_proj.weight',
            'attn_output.weight': f'model.layers.{layer}.self_attn.o_proj.weight',
            'attn_q_norm.weight': f'model.layers.{layer}.self_attn.q_norm.weight',
            'attn_k_norm.weight': f'model.layers.{layer}.self_attn.k_norm.weight',
        }
        if rest in full_mapping:
            return full_mapping[rest]
    else:
        # Linear attention layers have fused QKV + SSM parameters
        linear_mapping = {
            'attn_qkv.weight': f'model.layers.{layer}.linear_attn.in_proj_qkv.weight',
            'attn_output.weight': f'model.layers.{layer}.self_attn.o_proj.weight',
            # attn_gate in linear layers is the Z/output gate (maps to in_proj_z)
            'attn_gate.weight': f'model.layers.{layer}.linear_attn.in_proj_z.weight',
            'attn_q_norm.weight': f'model.layers.{layer}.self_attn.q_norm.weight',
            'attn_k_norm.weight': f'model.layers.{layer}.self_attn.k_norm.weight',
            # GatedDeltaNet / SSM
            'ssm_conv1d.weight': f'model.layers.{layer}.linear_attn.conv1d.weight',
            'ssm_a': f'model.layers.{layer}.linear_attn.A_log',
            'ssm_dt.bias': f'model.layers.{layer}.linear_attn.dt_bias',
            'ssm_norm.weight': f'model.layers.{layer}.linear_attn.norm.weight',
            'ssm_out.weight': f'model.layers.{layer}.linear_attn.out_proj.weight',
            'ssm_alpha.weight': f'model.layers.{layer}.linear_attn.in_proj_a.weight',
            'ssm_beta.weight': f'model.layers.{layer}.linear_attn.in_proj_b.weight',
        }
        if rest in linear_mapping:
            return linear_mapping[rest]

    return gguf_name


# ============================================================================
# GGUF parser
# ============================================================================
class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.fd = open(path, 'rb')
        self.metadata = {}
        self.tensors = []  # list of {name, dims, type, offset, nbytes}
        self.tensor_data_start = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self._parse()

    def _read_str(self):
        n = struct.unpack('<Q', self.fd.read(8))[0]
        return self.fd.read(n).decode('utf-8')

    def _read_value(self, vtype):
        simple = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                  6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if vtype in simple:
            fmt = simple[vtype]
            return struct.unpack(f'<{fmt}', self.fd.read(struct.calcsize(fmt)))[0]
        elif vtype == 8:  # STRING
            return self._read_str()
        elif vtype == 9:  # ARRAY
            etype = struct.unpack('<I', self.fd.read(4))[0]
            cnt = struct.unpack('<Q', self.fd.read(8))[0]
            return [self._read_value(etype) for _ in range(cnt)]
        else:
            raise ValueError(f"Unknown metadata type {vtype}")

    def _parse(self):
        magic, version = struct.unpack('<II', self.fd.read(8))
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic={hex(magic)})"
        assert version >= 2, f"Unsupported GGUF version {version}"
        tensor_count, kv_count = struct.unpack('<QQ', self.fd.read(16))

        # Read metadata
        for _ in range(kv_count):
            key = self._read_str()
            vtype = struct.unpack('<I', self.fd.read(4))[0]
            value = self._read_value(vtype)
            self.metadata[key] = value

        self.alignment = self.metadata.get('general.alignment', GGUF_DEFAULT_ALIGNMENT)

        # Read tensor infos
        for _ in range(tensor_count):
            name = self._read_str()
            ndim = struct.unpack('<I', self.fd.read(4))[0]
            dims = list(struct.unpack(f'<{ndim}Q', self.fd.read(8 * ndim)))
            ttype = struct.unpack('<I', self.fd.read(4))[0]
            offset = struct.unpack('<Q', self.fd.read(8))[0]
            n_elements = 1
            for d in dims:
                n_elements *= d
            nbytes = tensor_nbytes(n_elements, ttype)
            self.tensors.append({
                'name': name, 'dims': dims, 'type': ttype,
                'offset': offset, 'nbytes': nbytes,
                'n_elements': n_elements,
            })

        # Compute tensor data start (aligned)
        pos = self.fd.tell()
        pad = (self.alignment - (pos % self.alignment)) % self.alignment
        self.tensor_data_start = pos + pad

    def read_tensor_data(self, tensor_info):
        abs_offset = self.tensor_data_start + tensor_info['offset']
        self.fd.seek(abs_offset)
        return self.fd.read(tensor_info['nbytes'])

    def read_tensor_slice(self, tensor_info, offset, size):
        abs_offset = self.tensor_data_start + tensor_info['offset'] + offset
        self.fd.seek(abs_offset)
        return self.fd.read(size)

    def close(self):
        self.fd.close()


# ============================================================================
# Config extraction
# ============================================================================
def extract_config(meta, arch):
    """Build model config from GGUF metadata."""
    prefix = arch + '.'

    def get(key, default=None):
        return meta.get(prefix + key, meta.get(key, default))

    hidden_size = get('embedding_length', 4096)
    num_layers = get('block_count', 60)
    num_heads = get('attention.head_count', 32)
    num_kv_heads = get('attention.head_count_kv', 2)
    vocab_size = len(meta.get('tokenizer.ggml.tokens', []))
    if vocab_size == 0:
        vocab_size = get('vocab_size', 248320)

    full_attn_interval = get('full_attention_interval', 4)

    # Layer types
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": get('attention.key_length', 256),
        "vocab_size": vocab_size,
        "rms_norm_eps": get('attention.layer_norm_rms_epsilon', 1e-6),
        "num_experts": get('expert_count', 256),
        "num_experts_per_tok": get('expert_used_count', 8),
        "moe_intermediate_size": get('expert_feed_forward_length', 512),
        "shared_expert_intermediate_size": get('expert_shared_feed_forward_length', 512),
        "full_attention_interval": full_attn_interval,
        "linear_num_value_heads": get('ssm.time_step_rank', 32),
        "linear_num_key_heads": get('ssm.group_count', 16),
        "linear_key_head_dim": get('ssm.state_size', 128),
        "linear_value_head_dim": get('ssm.state_size', 128),
        "linear_conv_kernel_dim": get('ssm.conv_kernel', 4),
        "partial_rotary_factor": get('rope.partial_rotary_factor', 0.25),
        "rope_theta": get('rope.freq_base', 10000000.0),
        "layer_types": layer_types,
        "quant_format": "gguf",
    }
    return config


# ============================================================================
# Vocab extraction
# ============================================================================
def extract_vocab(meta, output_dir):
    """Extract vocab.bin from GGUF metadata.

    Format must match the CUDA engine's expected layout:
      u32 num_entries
      u32 max_id (= num_entries - 1 for GGUF since tokens are dense 0..N-1)
      per entry: u16 len + bytes
    """
    tokens = meta.get('tokenizer.ggml.tokens', [])
    if not tokens:
        print("WARNING: No tokenizer tokens in GGUF metadata")
        return

    vocab_path = output_dir / 'vocab.bin'
    with open(vocab_path, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        f.write(struct.pack('<I', len(tokens) - 1))  # max_id
        for tok in tokens:
            if isinstance(tok, str):
                tok = tok.encode('utf-8')
            # Engine expects u16 length, not u32
            f.write(struct.pack('<H', len(tok)))
            f.write(tok)

    print(f"Wrote {vocab_path} ({len(tokens)} tokens)")


# ============================================================================
# Main extraction
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Extract GGUF MoE model for Flash-MoE')
    parser.add_argument('--gguf', required=True, help='Path to .gguf file')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Parse only, no extraction')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.gguf}...")
    reader = GGUFReader(args.gguf)

    arch = reader.metadata.get('general.architecture', 'unknown')
    model_name = reader.metadata.get('general.name', 'unknown')
    print(f"Architecture: {arch}")
    print(f"Model: {model_name}")
    print(f"Tensors: {len(reader.tensors)}")
    print(f"Tensor data starts at: {reader.tensor_data_start}")

    # Extract config
    config = extract_config(reader.metadata, arch)
    num_layers = config['num_hidden_layers']
    num_experts = config['num_experts']
    print(f"Config: {num_layers} layers, {num_experts} experts, "
          f"K={config['num_experts_per_tok']}, hidden={config['hidden_size']}")

    # Classify tensors
    expert_tensors = {}   # (layer, "gate"/"up"/"down") → tensor_info
    nonexpert_tensors = []

    for t in reader.tensors:
        name = t['name']
        # Expert tensors: blk.{L}.ffn_{gate|up|down}_exps.weight
        if '_exps.' in name:
            parts = name.split('.')
            layer = int(parts[1])
            if 'gate_exps' in name:
                proj = 'gate'
            elif 'up_exps' in name:
                proj = 'up'
            elif 'down_exps' in name:
                proj = 'down'
            else:
                print(f"  Unknown expert tensor: {name}")
                nonexpert_tensors.append(t)
                continue
            expert_tensors[(layer, proj)] = t
        else:
            nonexpert_tensors.append(t)

    expert_layers = sorted(set(l for l, _ in expert_tensors.keys()))
    print(f"Expert layers: {len(expert_layers)} ({expert_layers[0]}-{expert_layers[-1]})")
    print(f"Non-expert tensors: {len(nonexpert_tensors)}")

    # Compute expert sizes
    if expert_layers:
        l0 = expert_layers[0]
        expert_size = 0
        for proj in ['gate', 'up', 'down']:
            t = expert_tensors.get((l0, proj))
            if t:
                per_expert = t['nbytes'] // num_experts
                expert_size += per_expert
                tname = QUANT_NAMES.get(t['type'], f"type{t['type']}")
                print(f"  {proj}_exps: {t['dims']} {tname}, "
                      f"{per_expert} bytes/expert")
        print(f"  Total expert size: {expert_size} bytes ({expert_size/1024:.1f} KB)")

    if args.dry_run:
        print("\nDry run complete.")
        reader.close()
        return

    # =========================================================================
    # Step 1: Extract non-expert weights
    # =========================================================================
    print(f"\n=== Extracting non-expert weights ===")
    bin_path = output_dir / 'model_weights.bin'
    ALIGN = 64

    manifest = {
        "model": str(args.gguf),
        "num_tensors": len(nonexpert_tensors),
        "tensors": {},
        "config": config,
    }

    full_attn_interval = config.get('full_attention_interval', 4)
    t0 = time.time()
    offset = 0
    with open(bin_path, 'wb') as out_f:
        for i, t in enumerate(nonexpert_tensors):
            # Map GGUF name → engine name (e.g. blk.0.attn_norm.weight → model.layers.0.input_layernorm.weight)
            san_name = map_tensor_name(t['name'], full_attn_interval)

            data = reader.read_tensor_data(t)
            assert len(data) == t['nbytes'], f"Short read for {t['name']}"

            # Align
            pad = (ALIGN - (offset % ALIGN)) % ALIGN
            if pad:
                out_f.write(b'\x00' * pad)
                offset += pad

            out_f.write(data)
            manifest['tensors'][san_name] = {
                'offset': offset,
                'size': t['nbytes'],
                'shape': t['dims'],
                'dtype': QUANT_NAMES.get(t['type'], f"type{t['type']}"),
                'gguf_type': t['type'],
            }
            offset += t['nbytes']

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(nonexpert_tensors)}] {offset/1024/1024:.1f} MB")

    elapsed = time.time() - t0
    print(f"  Wrote {bin_path} ({offset/1024/1024:.1f} MB in {elapsed:.1f}s)")

    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {json_path}")

    # =========================================================================
    # Step 2: Extract vocab
    # =========================================================================
    print(f"\n=== Extracting vocabulary ===")
    extract_vocab(reader.metadata, output_dir)

    # =========================================================================
    # Step 3: Repack experts into per-layer binary files
    # =========================================================================
    print(f"\n=== Repacking experts ===")
    expert_dir = output_dir / 'packed_experts'
    expert_dir.mkdir(exist_ok=True)

    # Compute per-component max sizes across all layers (i1 quant mixes Q4_K/Q6_K)
    max_comp_size = {}
    for proj in ['gate', 'up', 'down']:
        sizes = set()
        types = set()
        for layer_idx in expert_layers:
            t = expert_tensors[(layer_idx, proj)]
            per_expert = t['nbytes'] // num_experts
            sizes.add(per_expert)
            types.add(t['type'])
        max_comp_size[proj] = max(sizes)
        if len(types) > 1:
            print(f"  NOTE: {proj}_exps has mixed quant types across layers: "
                  f"{', '.join(QUANT_NAMES.get(t, str(t)) for t in sorted(types))}")

    # Use max sizes for uniform expert layout (smaller data is zero-padded)
    components = []
    comp_offset = 0
    for proj in ['gate', 'up', 'down']:
        components.append({
            'name': f'{proj}_exps',
            'offset': comp_offset,
            'slot_size': max_comp_size[proj],  # allocated slot (max across layers)
        })
        comp_offset += max_comp_size[proj]
    expert_size = comp_offset

    # Per-layer component types and actual sizes
    layer_info = []
    for layer_idx in expert_layers:
        li = {}
        for proj in ['gate', 'up', 'down']:
            t = expert_tensors[(layer_idx, proj)]
            per_expert = t['nbytes'] // num_experts
            li[f'{proj}_type'] = t['type']
            li[f'{proj}_size'] = per_expert
        layer_info.append(li)

    layout = {
        'expert_size': expert_size,
        'num_layers': len(expert_layers),
        'num_experts': num_experts,
        'components': components,
        'layer_info': layer_info,
        'format': 'gguf',
    }
    layout_path = expert_dir / 'layout.json'
    with open(layout_path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"  Layout: {expert_size} bytes/expert (uniform), {num_experts} experts/layer")

    layer_size = num_experts * expert_size
    t0 = time.time()
    total_written = 0

    for i, layer_idx in enumerate(expert_layers):
        out_path = expert_dir / f'layer_{layer_idx:02d}.bin'
        fd_out = os.open(str(out_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, layer_size)  # zero-filled

        for proj_idx, proj in enumerate(['gate', 'up', 'down']):
            t = expert_tensors[(layer_idx, proj)]
            per_expert = t['nbytes'] // num_experts
            slot_offset = components[proj_idx]['offset']

            data = reader.read_tensor_data(t)

            for e in range(num_experts):
                src_start = e * per_expert
                # Write actual data at the component's slot offset (zero-padded if smaller)
                os.pwrite(fd_out, data[src_start:src_start + per_expert],
                          e * expert_size + slot_offset)

        os.close(fd_out)
        total_written += layer_size

        elapsed = time.time() - t0
        throughput = total_written / elapsed / (1024**3) if elapsed > 0 else 0
        eta = (len(expert_layers) - i - 1) * (elapsed / (i + 1))
        print(f"  Layer {layer_idx:2d}: {layer_size/1024/1024:.1f} MB | "
              f"Total: {total_written/1024**3:.1f}/{len(expert_layers)*layer_size/1024**3:.1f} GB "
              f"({throughput:.1f} GB/s) | ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Expert data: {total_written/1024**3:.1f} GB in {elapsed:.1f}s")
    print(f"Output: {output_dir}")

    reader.close()


if __name__ == '__main__':
    main()
