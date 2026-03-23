#!/usr/bin/env python3
"""
Co-activation expert clustering for flash-moe.

Analyzes routing logs to find which experts are frequently co-activated,
then rewrites packed expert files so co-activated experts are physically
adjacent on disk. This improves cold SSD read throughput by ~38% for
cache misses (measured: scattered=3.2GB/s, adjacent=4.4GB/s on M1 Pro).

Usage:
    # Step 1: Generate routing log during inference
    ./infer --prompt "..." --tokens 200 --k 4 --collect-routing routing.bin

    # Step 2: Analyze and cluster
    python3 cluster_experts.py --routing routing.bin --packed-dir metal_infer/packed_experts

    # Step 3: Verify
    python3 cluster_experts.py --routing routing.bin --verify
"""

import argparse
import os
import struct
import sys
import time
import numpy as np
from collections import defaultdict

EXPERT_SIZE = 7077888
NUM_EXPERTS = 512
NUM_LAYERS = 60
HIDDEN_DIM = 4096


def load_routing_log(path):
    """Load binary routing log. Format per sample: int32 layer, int32 K, float32[4096] hidden, int32[K] experts."""
    routing = defaultdict(list)  # layer -> list of expert_index tuples

    with open(path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            layer_idx, K = struct.unpack('<ii', header)
            hidden = f.read(HIDDEN_DIM * 4)  # skip hidden state
            if len(hidden) < HIDDEN_DIM * 4:
                break
            experts_data = f.read(K * 4)
            if len(experts_data) < K * 4:
                break
            experts = struct.unpack(f'<{K}i', experts_data)
            routing[layer_idx].append(experts)

    total = sum(len(v) for v in routing.values())
    print(f"Loaded {total} routing decisions across {len(routing)} layers")
    return routing


def build_coactivation_matrix(routing, layer_idx):
    """Build co-activation count matrix for a layer."""
    coact = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.int32)
    freq = np.zeros(NUM_EXPERTS, dtype=np.int32)

    for experts in routing.get(layer_idx, []):
        for e in experts:
            freq[e] += 1
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                coact[experts[i], experts[j]] += 1
                coact[experts[j], experts[i]] += 1

    return coact, freq


def greedy_cluster_order(coact, freq):
    """Greedy clustering: start with most frequent expert, greedily add the
    most co-activated neighbor. This produces an ordering where co-activated
    experts are physically adjacent."""
    N = len(freq)
    visited = [False] * N
    order = []

    # Start with the most frequently activated expert
    current = int(np.argmax(freq))
    visited[current] = True
    order.append(current)

    for _ in range(N - 1):
        # Find the unvisited expert most co-activated with current
        best = -1
        best_score = -1
        for e in range(N):
            if not visited[e] and coact[current, e] > best_score:
                best_score = coact[current, e]
                best = e

        if best < 0:
            # No co-activation data — pick most frequent unvisited
            for e in range(N):
                if not visited[e]:
                    best = e
                    break

        visited[best] = True
        order.append(best)
        current = best

    return order


def repack_layer(layer_idx, order, packed_dir):
    """Rewrite a packed expert file using the new ordering."""
    src_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")
    tmp_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.tmp")

    if not os.path.exists(src_path):
        print(f"  Layer {layer_idx}: MISSING")
        return False

    # Check if real data (not sparse)
    actual = os.stat(src_path).st_blocks * 512
    if actual < 3e9:
        print(f"  Layer {layer_idx}: sparse/synthetic, skipping")
        return False

    # Read all experts in new order, write contiguously
    print(f"  Layer {layer_idx}: repacking ({actual/1e9:.1f}GB)...", end="", flush=True)
    t0 = time.time()

    fd_in = os.open(src_path, os.O_RDONLY)
    fd_out = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    for new_pos, old_expert_idx in enumerate(order):
        data = os.pread(fd_in, EXPERT_SIZE, old_expert_idx * EXPERT_SIZE)
        os.pwrite(fd_out, data, new_pos * EXPERT_SIZE)

    os.close(fd_in)
    os.close(fd_out)

    # Atomic swap
    os.rename(tmp_path, src_path)
    print(f" {time.time()-t0:.1f}s")
    return True


def save_mapping(layer_idx, order, packed_dir):
    """Save the old->new mapping so inference can translate expert indices."""
    map_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.map")
    # order[new_pos] = old_expert_idx
    # We need: for a given old expert idx, what's its new position in the file?
    inverse = [0] * NUM_EXPERTS
    for new_pos, old_idx in enumerate(order):
        inverse[old_idx] = new_pos

    with open(map_path, 'wb') as f:
        # Format: uint16[512] mapping old_expert_idx -> new_file_position
        for old_idx in range(NUM_EXPERTS):
            f.write(struct.pack('<H', inverse[old_idx]))

    return inverse


def main():
    parser = argparse.ArgumentParser(description="Co-activation expert clustering")
    parser.add_argument('--routing', required=True, help='Binary routing log from --collect-routing')
    parser.add_argument('--packed-dir', default='metal_infer/packed_experts',
                        help='Directory containing packed expert files')
    parser.add_argument('--verify', action='store_true', help='Verify clustering quality')
    parser.add_argument('--layers', default='all', help='Layer spec: "all", "0-27", "0,5,10"')
    args = parser.parse_args()

    routing = load_routing_log(args.routing)

    # Parse layers
    if args.layers == 'all':
        layers = list(range(NUM_LAYERS))
    elif '-' in args.layers:
        a, b = args.layers.split('-')
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    if args.verify:
        # Check clustering quality: what fraction of co-activated experts are adjacent?
        print("=== Clustering Quality Verification ===")
        for layer in layers:
            if layer not in routing:
                continue
            coact, freq = build_coactivation_matrix(routing, layer)
            map_path = os.path.join(args.packed_dir, f"layer_{layer:02d}.map")
            if not os.path.exists(map_path):
                print(f"  Layer {layer}: no mapping file")
                continue
            with open(map_path, 'rb') as f:
                mapping = struct.unpack(f'<{NUM_EXPERTS}H', f.read(NUM_EXPERTS * 2))

            total_pairs = 0
            adjacent_pairs = 0
            for experts in routing[layer]:
                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        new_i = mapping[experts[i]]
                        new_j = mapping[experts[j]]
                        dist = abs(new_i - new_j)
                        total_pairs += 1
                        if dist <= 4:  # within 4 positions
                            adjacent_pairs += 1

            pct = 100.0 * adjacent_pairs / total_pairs if total_pairs > 0 else 0
            print(f"  Layer {layer}: {adjacent_pairs}/{total_pairs} pairs within distance 4 ({pct:.1f}%)")
        return

    print(f"=== Co-activation Expert Clustering ===")
    print(f"Layers: {layers[0]}-{layers[-1]}")

    for layer in layers:
        coact, freq = build_coactivation_matrix(routing, layer)
        active = int(np.sum(freq > 0))
        print(f"\n  Layer {layer}: {active} active experts")

        order = greedy_cluster_order(coact, freq)
        mapping = save_mapping(layer, order, args.packed_dir)

        # Show top co-activated pairs and their new positions
        top_pairs = []
        for i in range(NUM_EXPERTS):
            for j in range(i + 1, NUM_EXPERTS):
                if coact[i, j] > 0:
                    top_pairs.append((coact[i, j], i, j))
        top_pairs.sort(reverse=True)

        if top_pairs:
            print(f"    Top co-activated pair: E{top_pairs[0][1]}+E{top_pairs[0][2]} ({top_pairs[0][0]} times)")
            old_dist = abs(top_pairs[0][1] - top_pairs[0][2])
            new_dist = abs(mapping[top_pairs[0][1]] - mapping[top_pairs[0][2]])
            print(f"    Distance: {old_dist} -> {new_dist}")

        repacked = repack_layer(layer, order, args.packed_dir)
        if not repacked:
            print(f"    Skipped (no repack)")

    print("\n=== Done ===")
    print("Expert mapping files (.map) saved alongside packed files.")
    print("Inference engine needs to load .map files and translate expert indices.")


if __name__ == "__main__":
    main()
