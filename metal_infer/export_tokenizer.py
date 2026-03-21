#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json to a compact binary format for C.

Usage: python export_tokenizer.py [tokenizer.json] [output.bin]

Binary format:
  Header:
    magic: "BPET" (4 bytes)
    version: uint32
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section (sorted by token_id):
    For each entry: uint32 token_id, uint16 str_len, char[str_len] (UTF-8 bytes of the BPE string)
  Merges section (ordered by priority, index 0 = highest priority):
    For each entry: uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens section:
    For each entry: uint32 token_id, uint16 str_len, char[str_len]
"""
import json
import os
import struct
import sys

def main():
    import glob
    default_tok = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit'
        '/snapshots/*/tokenizer.json'))
    tok_path = sys.argv[1] if len(sys.argv) > 1 else (
        default_tok[0] if default_tok else 'tokenizer.json'
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'tokenizer.bin'

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    model = t['model']
    vocab = model['vocab']       # str -> int
    merges = model['merges']     # list of [str, str]
    added = t['added_tokens']    # list of {id, content, special, ...}

    # Sort vocab by token_id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_path, 'wb') as f:
        # Header
        f.write(b'BPET')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(sorted_vocab)))
        f.write(struct.pack('<I', len(merges)))
        f.write(struct.pack('<I', len(added)))

        # Vocab
        for token_str, token_id in sorted_vocab:
            b = token_str.encode('utf-8')
            f.write(struct.pack('<I', token_id))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

        # Merges
        for pair in merges:
            a, b = pair[0], pair[1]
            ab = a.encode('utf-8')
            bb = b.encode('utf-8')
            f.write(struct.pack('<H', len(ab)))
            f.write(ab)
            f.write(struct.pack('<H', len(bb)))
            f.write(bb)

        # Added tokens
        for tok in added:
            b = tok['content'].encode('utf-8')
            f.write(struct.pack('<I', tok['id']))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

    print(f"Exported to {out_path}:")
    print(f"  Vocab: {len(sorted_vocab)} entries")
    print(f"  Merges: {len(merges)} rules")
    print(f"  Added tokens: {len(added)} entries")

    import os
    sz = os.path.getsize(out_path)
    print(f"  File size: {sz / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
