#!/usr/bin/env python3
import json, struct, os, sys

tok_path = sys.argv[1] if len(sys.argv) > 1 else 'tokenizer.json'
out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

with open(tok_path) as f:
    t = json.load(f)

vocab = t['model']['vocab']
added = {tok['content']: tok['id'] for tok in t['added_tokens']}

all_tokens = dict(vocab)
all_tokens.update(added)

max_id = max(all_tokens.values())
num_entries = max_id + 1

id_to_str = [''] * num_entries
for s, i in all_tokens.items():
    id_to_str[i] = s

with open(out_path, 'wb') as f:
    f.write(struct.pack('<I', num_entries))
    f.write(struct.pack('<I', max_id))
    for i in range(num_entries):
        b = id_to_str[i].encode('utf-8')
        f.write(struct.pack('<H', len(b)))
        if b:
            f.write(b)

sz = os.path.getsize(out_path)
print(f"Wrote {out_path}: {sz / 1024:.0f} KB, {num_entries} entries, max_id={max_id}")
