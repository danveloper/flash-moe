#!/usr/bin/env python3
"""Generate build command for any MoE model from its model_weights.json config.

Usage:
    python3 configure.py [--manifest model_weights.json] [--output Makefile.model]

Reads the config section from model_weights.json and outputs either:
  1. A Makefile with the correct -D flags (default)
  2. The nvcc command to stdout (--print-cmd)
"""
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Configure build for a specific MoE model')
    parser.add_argument('--manifest', default='model_weights.json',
                        help='Path to model_weights.json')
    parser.add_argument('--output', default=None,
                        help='Output Makefile (default: print command to stdout)')
    parser.add_argument('--print-cmd', action='store_true',
                        help='Print nvcc command instead of writing Makefile')
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    cfg = manifest.get('config', {})

    # Map config keys to C #define names
    defines = {
        'HIDDEN_DIM': cfg.get('hidden_size', 4096),
        'NUM_LAYERS': cfg.get('num_hidden_layers', 60),
        'NUM_ATTN_HEADS': cfg.get('num_attention_heads', 32),
        'NUM_KV_HEADS': cfg.get('num_key_value_heads', 2),
        'HEAD_DIM': cfg.get('head_dim', 256),
        'VOCAB_SIZE': cfg.get('vocab_size', 248320),
        'NUM_EXPERTS': cfg.get('num_experts', 512),
        'MOE_INTERMEDIATE': cfg.get('moe_intermediate_size', 1024),
        'SHARED_INTERMEDIATE': cfg.get('shared_expert_intermediate_size', 1024),
        'FULL_ATTN_INTERVAL': cfg.get('full_attention_interval', 4),
        'LINEAR_NUM_V_HEADS': cfg.get('linear_num_value_heads', 64),
        'LINEAR_NUM_K_HEADS': cfg.get('linear_num_key_heads', 16),
        'LINEAR_KEY_DIM': cfg.get('linear_key_head_dim', 128),
        'LINEAR_VALUE_DIM': cfg.get('linear_value_head_dim', 128),
        'CONV_KERNEL_SIZE': cfg.get('linear_conv_kernel_dim', 4),
    }

    # Float defines
    float_defines = {
        'ROPE_THETA': cfg.get('rope_theta', 10000000.0),
        'PARTIAL_ROTARY': cfg.get('partial_rotary_factor', 0.25),
    }

    # Build -D flags
    dflags = ' '.join(f'-D{k}={v}' for k, v in defines.items())
    dflags += ' ' + ' '.join(f'-D{k}={v}f' for k, v in float_defines.items())

    # Model name for the binary
    hidden = defines['HIDDEN_DIM']
    layers = defines['NUM_LAYERS']
    experts = defines['NUM_EXPERTS']
    model_name = f"qwen_{hidden}x{layers}x{experts}"

    nvcc_cmd = (
        f'nvcc -O2 -Wno-deprecated-gpu-targets -diag-suppress 1650 '
        f'{dflags} '
        f'-o infer_{model_name} infer.cu tokenizer_impl.o -lpthread'
    )

    if args.print_cmd:
        print(nvcc_cmd)
        return

    print(f'Model: {model_name}')
    print(f'  hidden={hidden}, layers={layers}, experts={experts}')
    print(f'  K={cfg.get("num_experts_per_tok", "?")}, '
          f'intermediate={defines["MOE_INTERMEDIATE"]}')

    if args.output:
        with open(args.output, 'w') as f:
            f.write(f'# Auto-generated for {model_name}\n')
            f.write(f'# From: {args.manifest}\n\n')
            f.write(f'NVCC ?= nvcc\n')
            f.write(f'MODEL_DFLAGS = {dflags}\n\n')
            f.write(f'infer_{model_name}: infer.cu kernels.cuh tokenizer_impl.o\n')
            f.write(f'\t$(NVCC) -O2 -Wno-deprecated-gpu-targets -diag-suppress 1650 '
                    f'$(MODEL_DFLAGS) -o $@ infer.cu tokenizer_impl.o -lpthread\n\n')
            f.write(f'tokenizer_impl.o: tokenizer_impl.c ../metal_infer/tokenizer.h\n')
            f.write(f'\tgcc -O2 -c tokenizer_impl.c -o tokenizer_impl.o\n')
        print(f'Wrote {args.output}')
        print(f'Build with: make -f {args.output}')
    else:
        print(f'\nBuild command:')
        print(f'  {nvcc_cmd}')

if __name__ == '__main__':
    main()
