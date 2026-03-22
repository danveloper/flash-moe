# Flash-MoE CUDA: Running Qwen3.5-397B on a Single NVIDIA GPU

CUDA/C port of [Flash-MoE](../CLAUDE.md) for x86 PCs with NVIDIA GPUs. Runs **Qwen3.5-397B-A17B** (397 billion parameter MoE model) on a single RTX 4090 with 24GB VRAM, streaming 209GB of expert weights from NVMe SSD.

**2.45 tokens/second** with production-quality output. No Python. No frameworks. One CUDA file + one kernel header.

## How It Works

The full model is 209GB at 4-bit quantization. Only 5.2GB of non-expert weights fit in GPU VRAM. The remaining 203GB of expert weights (512 experts per layer, K=4 activated per token) stream from NVMe SSD on demand:

```
SSD (203GB experts) ──GDS──> GPU VRAM (24GB)
                              ↕ CUDA kernels
CPU RAM (64GB page cache) ←──> GPU compute
```

Each token requires loading 4 experts × 60 layers = 240 expert reads (~6.75MB each) from SSD. NVIDIA GPUDirect Storage (GDS) enables direct NVMe-to-GPU DMA transfers, bypassing the CPU.

## Results

| Configuration | tok/s | Hardware | Notes |
|--------------|-------|----------|-------|
| **Flash-MoE CUDA (GDS)** | **2.45** | 1x RTX 4090, 64GB RAM, 2TB NVMe | This project. Direct SSD→GPU. |
| Flash-MoE Metal (Apple) | 4.36 | M3 Max 48GB, 1TB NVMe | Original project. Unified memory. |

### Comparison with Other Solutions

| System | Qwen3.5-397B | Hardware Required | Approach |
|--------|-------------|-------------------|----------|
| **Flash-MoE CUDA** | **2.45 tok/s** | **1x RTX 4090 + 64GB RAM + NVMe** | SSD expert streaming, GDS |
| KTransformers | ~14 tok/s* | 1x RTX 4090 + **384GB RAM** | CPU expert compute (AMX), GPU attention |
| llama.cpp (offload) | ~1-2 tok/s | 1x RTX 4090 + **256GB RAM** | CPU/GPU layer split, GGUF |
| KTransformers (full) | ~150 tok/s | **4x RTX 4090 + 800GB RAM** | Multi-GPU tensor parallelism |

*KTransformers single-GPU numbers are for Qwen3-235B (smaller model); 397B numbers not published for single GPU.

**Key advantage**: Flash-MoE CUDA requires only **64GB RAM** (vs 256-384GB for alternatives) by streaming experts from SSD instead of storing them in system memory.

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on RTX 4090)
- **RAM**: 64GB+ system memory (for OS page cache)
- **SSD**: NVMe SSD with 250GB+ free space, PCIe 4.0+ recommended
- **CUDA**: 12.8+ with GDS support (optional but recommended)
- **OS**: Linux (tested on Ubuntu 24.04)

## Quick Start

### 1. Build

```bash
cd cuda_infer

# Requires CUDA toolkit 12.8+ and GDS library
make
```

### 2. Download and prepare model weights

```bash
# Install Python dependencies
python3 -m venv flash-moe-env
source flash-moe-env/bin/activate
pip install huggingface_hub safetensors numpy

# Download MLX 4-bit quantized model (~209GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-397B-A17B-4bit', local_dir='model-safetensors')
"

# Build expert index and repack into per-layer binary files (~203GB)
python3 build_expert_index.py --model model-safetensors --output expert_index.json
python3 ../repack_experts.py --index expert_index.json

# Extract non-expert weights (~5.2GB)
python3 ../metal_infer/extract_weights.py --model model-safetensors --output .

# Export tokenizer and vocabulary
python3 ../metal_infer/export_tokenizer.py model-safetensors/tokenizer.json tokenizer.bin
python3 export_vocab.py model-safetensors/tokenizer.json vocab.bin
```

### 3. Run

```bash
./infer --prompt "Explain quantum computing" --tokens 50

# With timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## Architecture

### Files

```
cuda_infer/
  infer.cu         # Complete inference engine (~1200 lines)
  kernels.cuh      # 15 CUDA compute kernels
  Makefile
  build_expert_index.py   # Generate expert_index.json from safetensors
  export_vocab.py         # Generate vocab.bin from tokenizer.json

bench_transfer.cu  # Transfer path benchmarks (GDS, pread, cudaMemcpy)
```

### CUDA Kernels (ported from Metal)

| Kernel | Purpose |
|--------|---------|
| `dequant_matvec_4bit_fma` | FMA-optimized 4-bit dequant matrix-vector multiply |
| `swiglu_fused` | SiLU(gate) × up activation |
| `rms_norm` / `rms_norm_bf16` | RMS normalization with f32/bf16 weights |
| `gated_delta_net_step` | GatedDeltaNet linear attention recurrence |
| `conv1d_step` | Depthwise conv1d (kernel=4) with SiLU |
| `attn_scores` / `attn_softmax` / `attn_values` | Full attention (Q@K^T, softmax, scores@V) |
| `moe_combine_residual` | Weighted expert sum + shared expert + residual |

### Transfer Path Benchmarks

Measured on RTX 4090 + Samsung 990 EVO Plus (PCIe 4.0 x4):

| Path | Time (K=4/layer) | Throughput |
|------|-----------------|-----------|
| pread → cudaMemcpy (cold) | 8.3 ms | 3.4 GB/s |
| **GDS cuFileRead (cold)** | **5.3 ms** | **5.3 GB/s** |
| Warm cache (page cache hit) | 2.7 ms | 10.4 GB/s |
| GPU dequant K=4 experts | 0.08 ms | negligible |

GDS provides a **37% speedup** over the traditional pread+cudaMemcpy path by enabling direct NVMe-to-GPU DMA transfers.

### Key Differences from Apple Silicon Version

| Aspect | Apple Silicon (Metal) | NVIDIA (CUDA) |
|--------|----------------------|---------------|
| Memory | Unified (GPU=CPU=SSD) | Discrete (PCIe bus) |
| SSD bandwidth | 17.5 GB/s | 5-7 GB/s (PCIe 4.0 x4) |
| GPU memory BW | ~400 GB/s | 1008 GB/s |
| SSD→GPU path | Direct (shared memory) | GDS or pread+cudaMemcpy |
| I/O+compute overlap | Cannot overlap (shared bus) | **Can overlap** (separate buses) |
| Pipeline | 3 Metal command buffers | CUDA streams |

## Technical Details

### Per-Token Pipeline (60 layers)

For each layer:
1. **RMS norm** (input layernorm) — GPU
2. **Attention projections** (4-bit dequant matvec) — GPU
3. **Attention compute**:
   - Linear (45 layers): conv1d → RMS norm Q/K → decay/beta → GatedDeltaNet recurrence → gated RMS norm
   - Full (15 layers): Q/K RMS norm → RoPE → KV cache update → Q@K^T → softmax → scores@V → sigmoid gate
4. **Output projection** (dequant matvec) — GPU
5. **Residual + post-attention RMS norm** — GPU
6. **MoE routing**: dequant matvec → softmax → topK — GPU + CPU
7. **Shared expert forward**: gate+up → SwiGLU → down — GPU (overlapped with expert I/O)
8. **Expert loading**: K=4 parallel reads from SSD — GDS or pread
9. **Expert forward**: gate+up → SwiGLU → down × K — GPU
10. **MoE combine + residual** — GPU

### Memory Usage

| Component | Size | Location |
|-----------|------|----------|
| Non-expert weights | 5.2 GB | GPU VRAM |
| Scratch buffers | ~200 MB | GPU VRAM |
| KV cache (15 full-attn layers) | ~200 MB | GPU VRAM |
| Delta-net state (45 linear layers) | ~180 MB | GPU VRAM |
| **Total GPU VRAM** | **~5.8 GB** | |
| Expert staging (pinned) | ~28 MB | CPU RAM |
| OS page cache | ~58 GB | CPU RAM (dynamic) |
| Expert data on disk | 203 GB | NVMe SSD |
