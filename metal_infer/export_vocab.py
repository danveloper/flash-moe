#!/usr/bin/env python3
"""Export vocab.bin in the simple format expected by infer.m's load_vocab().

Format:
  [num_entries: uint32] [max_id: uint32]
  For each entry (0..max_id): [byte_len: uint16] [utf8_bytes: byte_len]

Usage:
    python export_vocab.py <tokenizer.json> [output.bin]
"""
import json
import struct
import sys

def main():
    tok_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']       # str -> int
    added = t.get('added_tokens', [])  # list of {id, content, ...}

    # Merge added tokens into vocab
    for tok in added:
        vocab[tok['content']] = tok['id']

    max_id = max(vocab.values())
    num_entries = max_id + 1

    # BPE byte-level encoding uses Unicode chars for bytes:
    # Ġ (U+0120) = space, Ċ (U+010A) = newline, etc.
    # Build the reverse mapping to decode these back to real bytes.
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    bpe_decode_map = {chr(c): bytes([b]) for b, c in zip(bs, cs)}

    def decode_bpe_token(s):
        """Convert BPE token string to actual bytes."""
        try:
            return b''.join(bpe_decode_map.get(ch, ch.encode('utf-8')) for ch in s)
        except Exception:
            return s.encode('utf-8')

    # Build id -> string mapping with BPE decoding
    id_to_str = {}
    for s, tid in vocab.items():
        id_to_str[tid] = decode_bpe_token(s)

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', num_entries))
        f.write(struct.pack('<I', max_id))

        for i in range(num_entries):
            b = id_to_str.get(i, b'')
            f.write(struct.pack('<H', len(b)))
            if b:
                f.write(b)

    import os
    sz = os.path.getsize(out_path)
    print(f"Exported vocab.bin: {num_entries} entries (max_id={max_id}), {sz / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    main()
