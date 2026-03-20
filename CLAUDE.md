# Flash-MoE: Running Massive MoE Models on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine for **Qwen3.5 Mixture-of-Experts** models on Apple Silicon. Runs models from 35B to 397B parameters on machines with as little as 24GB RAM, streaming expert weights from SSD through a custom Metal compute pipeline.

No Python runtime. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders. Model architecture is auto-detected from HuggingFace `config.json` — switch models with a single `--model` flag.

## Compatible Models

Any **Qwen3.5 MoE** model with MLX quantization (`model_type: qwen3_5_moe`) is supported. Use the model manager to discover and download compatible models:

| Model | Params | Active | Quant | Disk | Min RAM |
|-------|--------|--------|-------|------|---------|
| Qwen3.5-35B-A3B | 35B | 3B | 4-bit | ~18GB | 24GB |
| Qwen3.5-35B-A3B | 35B | 3B | 8-bit | ~35GB | 48GB |
| Qwen3.5-122B-A10B | 122B | 10B | 4-bit | ~65GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 4-bit | ~209GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 6-bit | ~280GB | 64GB |
| Qwen3.5-397B-A17B | 397B | 17B | 8-bit | ~397GB | 96GB |

The engine auto-detects architecture, dimensions, expert counts, quantization, and layer types from `config.json`. No recompilation needed.

## Results

![Progress](progress.png)

Results below are for Qwen3.5-397B-A17B on MacBook Pro M3 Max (48GB):

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Current best. Full tool calling. 209GB on disk. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.

## Hardware

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

## Architecture

Qwen3.5 MoE models use a hybrid attention architecture with GatedDeltaNet (linear attention) and standard full attention layers, each containing a Mixture-of-Experts MLP. Model dimensions, expert counts, and layer types vary per model and are read from `config.json` at startup. For example, the 397B model has 60 layers (45 linear + 15 full), 512 experts (K=4 active), hidden dim 4096; the 35B model has 40 layers (30 linear + 10 full), 256 experts (K=8 active), hidden dim 2048.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

### Unified Memory Constraint

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## Model Manager

The model manager helps you find, download, and validate compatible models:

```bash
# List local models and search HuggingFace for compatible ones
python model_manager.py

# Search HuggingFace only
python model_manager.py --search

# List local models only
python model_manager.py --local

# Download a specific model
python model_manager.py --download mlx-community/Qwen3.5-35B-A3B-4bit

# Check if a local model is compatible
python model_manager.py --check /path/to/model
```

After downloading, prepare the model for inference:

```bash
# 1. Pack expert weights into per-expert files
python repack_experts.py --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 2. Extract non-expert weights into a single binary
python metal_infer/extract_weights.py --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 3. Run inference
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit --prompt "Hello" --tokens 20
```

## Quick Start

```bash
cd metal_infer
make

# Run with a specific model (auto-detects architecture from config.json)
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Explain quantum computing" --tokens 100

# Or set FLASH_MOE_MODEL to avoid passing --model every time
export FLASH_MOE_MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
./infer --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (faster but breaks tool calling)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# Interactive chat with tool calling
./chat

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## Project Structure

```
model_manager.py       # Model discovery, download, and compatibility checking
repack_experts.py      # 4-bit expert packing from safetensors
progress.py            # Results visualization (Q2/Q4 tracks)
results.tsv            # Experiment log (58 experiments)

metal_infer/
  infer.m              # Complete inference engine (~7500 lines)
                       #   - ModelConfig struct + config.json parser
                       #   - Runtime model auto-detection
                       #   - Metal compute pipeline
  shaders.metal        # Metal compute kernels (~1200 lines)
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization
  train_predictor.py   # Expert routing prediction analysis
  model_weights.bin    # Non-expert weights (model-specific, mmap'd)
  model_weights.json   # Tensor manifest
  vocab.bin            # Vocabulary for token decoding
  tokenizer.bin        # Pre-exported BPE tokenizer data
```

## What We Tried (and What Worked)

### Kept
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |

### Discarded (58 experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: model-dependent (e.g., 5.5GB for 397B, ~1.5GB for 35B, mmap'd read-only)
- Metal scratch buffers: ~200MB
- Expert data streams from SSD on demand — no full model load required
- No custom caches. Trust the OS page cache for expert LRU.
- Minimum RAM: 24GB (35B-A3B 4-bit), 48GB (397B-A17B 4-bit)
