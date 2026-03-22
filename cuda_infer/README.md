# Flash-MoE CUDA: Running Qwen3.5-397B on a Single NVIDIA GPU

CUDA/C port of [Flash-MoE](../CLAUDE.md) for x86 PCs with NVIDIA GPUs. Runs **Qwen3.5-397B-A17B** (397 billion parameter MoE model) on a single RTX 4090 with 24GB VRAM, streaming 209GB of expert weights from NVMe SSD.

**3.55 tokens/second** (warm cache) with tool calling, OpenAI-compatible API, and SSE streaming. No Python. No frameworks. One CUDA file + one kernel header.

## How It Works

The full model is 209GB at 4-bit quantization. Only 5.2GB of non-expert weights fit in GPU VRAM. The remaining 203GB of expert weights (512 experts per layer, K=4 activated per token) stream from NVMe SSD on demand:

```
SSD (203GB experts) ──pread──> CPU RAM (page cache) ──cudaMemcpy──> GPU VRAM
                                                                     ↕
                                               VRAM expert cache (17GB, ~2500 experts)
                                                                     ↕
                                                               CUDA kernels
```

Each token requires loading 4 experts × 60 layers = 240 expert reads (~6.75MB each). A three-tier caching hierarchy minimizes SSD access:

1. **VRAM expert cache** (~17GB, ~2500 experts): LRU cache in GPU memory. Hot experts are served instantly without any I/O. After a few requests, ~95% of expert accesses hit the VRAM cache.
2. **OS page cache** (~50GB with 64GB RAM): Experts not in VRAM are read via `pread()`, which populates the OS page cache. Repeat accesses hit RAM at ~10 GB/s.
3. **NVMe SSD**: Cold misses go to SSD at ~5-7 GB/s.

The VRAM cache warms progressively: 2.49 tok/s cold → 3.22 after one request → **3.55 tok/s** after a few requests. GDS (direct NVMe-to-GPU DMA) is available for low-RAM systems via `ENABLE_GDS=1` but bypasses the page cache, so it's slower for sustained generation.

## Results

| Configuration | tok/s | Hardware | Notes |
|--------------|-------|----------|-------|
| **Flash-MoE CUDA** | **3.55** | 1x RTX 4090, 64GB RAM, 2TB NVMe | This project. VRAM cache + page cache + SSD. |
| Flash-MoE Metal (Apple) | 4.36 | M3 Max 48GB, 1TB NVMe | Original project. Unified memory. |

### Comparison with Other Solutions

| System | Qwen3.5-397B | Hardware Required | Approach |
|--------|-------------|-------------------|----------|
| **Flash-MoE CUDA** | **3.55 tok/s** | **1x RTX 4090 + 16GB+ RAM + NVMe** | VRAM cache + page cache + SSD |
| KTransformers | ~14 tok/s* | 1x RTX 4090 + **384GB RAM** | CPU expert compute (AMX), GPU attention |
| llama.cpp (offload) | ~1-2 tok/s | 1x RTX 4090 + **256GB RAM** | CPU/GPU layer split, GGUF |
| KTransformers (full) | ~150 tok/s | **4x RTX 4090 + 800GB RAM** | Multi-GPU tensor parallelism |

*KTransformers single-GPU numbers are for Qwen3-235B (smaller model); 397B numbers not published for single GPU.

**Key advantage**: Flash-MoE CUDA requires only **16GB RAM** (process uses 5.5GB; more RAM = better page cache but not required) vs 256-384GB for alternatives.

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on RTX 4090)
- **RAM**: 16GB minimum, 64GB+ recommended (process uses 5.5GB; extra RAM serves as page cache for experts — 64GB caches ~50% of expert data, significantly improving throughput)
- **SSD**: NVMe SSD with 250GB+ free space, PCIe 4.0+ recommended
- **CUDA**: 12.8+ with GDS support (optional but recommended)
- **OS**: Linux (tested on Ubuntu 24.04)

## Quick Start

### 1. Build

```bash
cd cuda_infer
make      # requires CUDA toolkit 12.8+ and libcufile
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
# Direct generation
./infer --prompt "Explain quantum computing" --tokens 50

# HTTP server (OpenAI-compatible API)
./infer --serve 8080

# With timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## HTTP Server (OpenAI-Compatible API)

Start the server with `--serve PORT`:

```bash
./infer --serve 8080
```

On startup, the server prefills and caches the system prompt (~4s). All subsequent requests restore from this snapshot instantly — no repeated prefill cost. Custom system prompt can be placed at `~/.flash-moe/system.md`.

### Endpoints

- `POST /v1/chat/completions` — OpenAI Chat Completions API (SSE streaming)
- `POST /v1/messages` — Anthropic Messages API (SSE streaming)
- `GET /v1/models` — List available models
- `GET /health` — Health check

Both chat endpoints support tool calling and produce correct streaming events for their respective formats.

### Basic chat

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

### Tool calling (function calling)

The server supports OpenAI-compatible tool calling. Pass `tools` in the request and the model will generate `tool_calls` in the response:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 200,
    "stream": true
  }'
```

Response includes tool calls in OpenAI format:

```json
{"choices": [{"delta": {"tool_calls": [{"id": "call_1", "type": "function",
  "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]}}]}
```

The model correctly generates `<tool_call>` tags which are parsed and converted to OpenAI `tool_calls` format. Generation stops after the tool call so the client can execute the function and send results back.

### Sending tool results back

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_1",
       "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"}
    ],
    "max_tokens": 200,
    "stream": true
  }'
```

### Using with Claude Code

The server natively supports the Anthropic Messages API (`POST /v1/messages`) — no proxy needed:

```bash
# Start the Flash-MoE server
./infer --serve 8080

# Point Claude Code at it
export ANTHROPIC_BASE_URL=http://localhost:8080
claude --model qwen3.5-397b-a17b
```

This gives you a 397B parameter model with tool calling running locally through Claude Code's agent framework — reading files, running commands, editing code — all on a single GPU.

### Using with other OpenAI-compatible clients

The server works directly with any OpenAI-compatible client:

```python
# Python (openai SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3.5-397b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

```bash
# aider
aider --model openai/qwen3.5-397b --openai-api-base http://localhost:8080/v1

# continue.dev (VS Code) — add to config.json:
# {"models": [{"provider": "openai", "model": "qwen3.5-397b",
#   "apiBase": "http://localhost:8080/v1"}]}
```

### Custom system prompt

Place a file at `~/.flash-moe/system.md` to override the default system prompt.

## Architecture

### Files

```
cuda_infer/
  infer.cu         # Complete inference engine + HTTP server (~1700 lines)
  kernels.cuh      # 15 CUDA compute kernels (~570 lines)
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

For cold reads, GDS is 37% faster than pread+cudaMemcpy. However, **pread with page cache** is the default because hot experts cached in RAM (2.7ms) beat GDS cold reads (5.3ms). With 64GB RAM, the page cache grows to ~50GB during sustained generation, caching roughly half the expert data. Set `ENABLE_GDS=1` to force GDS on low-RAM systems.

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
| **VRAM expert cache** | **~17 GB** | **GPU VRAM (LRU, ~2500 experts)** |
| Scratch buffers | ~200 MB | GPU VRAM |
| KV cache (15 full-attn layers) | ~200 MB | GPU VRAM |
| Delta-net state (45 linear layers) | ~180 MB | GPU VRAM |
| **Total GPU VRAM** | **~23 GB** | |
| Process RSS | ~5.5 GB | CPU RAM |
| OS page cache | ~50 GB | CPU RAM (dynamic, caches SSD reads) |
| Expert data on disk | 203 GB | NVMe SSD |
