# Runtime Model Config from HuggingFace config.json

**Date:** 2026-03-20
**Status:** Approved

## Problem

Switching Flash-MoE between models (e.g., Qwen3.5-397B-A17B vs Qwen3.5-35B-A3B) requires manually editing ~40 `#define` constants and recompiling. This is error-prone (the NaN bug was caused by stale expert offsets) and prevents runtime model selection.

## Solution

Replace all model-specific `#define` constants with a global `ModelConfig` struct populated at startup by parsing the HuggingFace `config.json` and `tokenizer_config.json` using NSJSONSerialization. Expert byte offsets are computed from dimensions and quantization parameters. The `--model` CLI flag selects which model to load.

## Design

### ModelConfig struct

A single global `ModelConfig cfg` replaces all model `#define`s:

```c
typedef struct {
    // Core architecture (from config.json -> text_config)
    int hidden_dim;           // text_config.hidden_size
    int num_layers;           // text_config.num_hidden_layers
    int num_attn_heads;       // text_config.num_attention_heads
    int num_kv_heads;         // text_config.num_key_value_heads
    int head_dim;             // text_config.head_dim
    int vocab_size;           // text_config.vocab_size
    float rms_norm_eps;       // text_config.rms_norm_eps

    // MoE (from config.json -> text_config)
    int num_experts;          // text_config.num_experts
    int num_experts_per_tok;  // text_config.num_experts_per_tok
    int moe_intermediate;     // text_config.moe_intermediate_size
    int shared_intermediate;  // text_config.shared_expert_intermediate_size
    int group_size;           // quantization.group_size (or quantization_config)
    int bits;                 // quantization.bits (or quantization_config)

    // Linear attention / GatedDeltaNet (from config.json -> text_config)
    int linear_num_v_heads;   // text_config.linear_num_value_heads
    int linear_num_k_heads;   // text_config.linear_num_key_heads
    int linear_key_dim;       // text_config.linear_key_head_dim
    int linear_value_dim;     // text_config.linear_value_head_dim
    int conv_kernel_size;     // text_config.linear_conv_kernel_dim

    // Full attention (from config.json -> text_config)
    float rope_theta;         // text_config.rope_parameters.rope_theta
    float partial_rotary;     // text_config.rope_parameters.partial_rotary_factor

    // Layer type map (from config.json -> text_config.layer_types)
    int num_full_attn_layers;
    int num_linear_layers;
    bool *is_full_attn;       // [num_layers] — true if full attention
    int *full_attn_index;     // [num_layers] — index into full-attn arrays, or -1
    int *linear_index;        // [num_layers] — index into linear-attn arrays, or -1

    // Derived: expert byte offsets (computed from dims + quantization)
    size_t expert_size_4bit;
    size_t gate_w_off_4, gate_s_off_4, gate_b_off_4;
    size_t up_w_off_4, up_s_off_4, up_b_off_4;
    size_t down_w_off_4, down_s_off_4, down_b_off_4;
    size_t expert_size_2bit;
    size_t gate_w_off_2, gate_s_off_2, gate_b_off_2;
    size_t up_w_off_2, up_s_off_2, up_b_off_2;
    size_t down_w_off_2, down_s_off_2, down_b_off_2;

    // Derived dimensions (computed from above)
    int linear_total_key;     // linear_num_k_heads * linear_key_dim
    int linear_total_value;   // linear_num_v_heads * linear_value_dim
    int linear_conv_dim;      // linear_total_key * 2 + linear_total_value
    int rotary_dim;           // (int)(head_dim * partial_rotary)

    // Special tokens
    int eos_token_ids[8];     // from config.json eos_token_id (can be array)
    int num_eos_tokens;
    int think_start_token;    // from tokenizer_config.json added_tokens_decoder
    int think_end_token;      // from tokenizer_config.json added_tokens_decoder

    // Context limits (kept as defaults, could be overridden by CLI)
    int max_seq_len;          // default: text_config.max_position_embeddings
    int gpu_kv_seq;           // default: 8192 (pre-allocation, not model-specific)

    // Model path (resolved HF snapshot dir)
    char model_path[1024];
} ModelConfig;

static ModelConfig cfg;
```

### Config loading function

```c
static void load_model_config(const char *model_dir);
```

Steps:
1. Resolve HF snapshot directory (walk `snapshots/` if needed, same logic as existing code)
2. Read and parse `config.json` via NSJSONSerialization
3. Extract `text_config` sub-dictionary for architecture params
4. Read `layer_types` array to build `is_full_attn[]`, `full_attn_index[]`, `linear_index[]`. Fallback: if `layer_types` is absent but `full_attn_interval` exists, compute `is_full[i] = ((i+1) % interval == 0)`. If neither exists, fatal error.
5. Read `quantization` or `quantization_config` for group_size and bits
6. Read `eos_token_id` (handles both single int and array)
7. Read `rope_parameters` sub-dict for rope_theta and partial_rotary_factor
8. Read and parse `tokenizer_config.json` for think tokens:
   - Walk `added_tokens_decoder` object, match entries where `content` == `"<think>"` or `"</think>"`
   - Extract their integer key as the token ID
9. Compute expert byte offsets via `compute_expert_offsets()`
10. Compute derived dimensions (linear_total_key, etc.)
11. Print summary to stderr for verification

### Expert offset computation

Offsets are deterministic from `moe_intermediate`, `hidden_dim`, `group_size`, and `bits`:

```c
static void compute_expert_offsets(ModelConfig *c) {
    int mid = c->moe_intermediate;   // e.g. 512
    int hid = c->hidden_dim;         // e.g. 2048
    int gs = c->group_size;          // e.g. 64
    int bits = c->bits;              // 4 or 2

    // For a [out_dim, in_dim] weight at N bits, group_size gs:
    //   weight_bytes = out_dim * ceil(in_dim / (32/bits)) * 4
    //   scales_bytes = out_dim * ceil(in_dim / gs) * 2  (bf16)
    //   biases_bytes = scales_bytes

    // gate_proj: [mid, hid], up_proj: [mid, hid], down_proj: [hid, mid]
    // Compute for 4-bit and 2-bit layouts
    for (int b = 4; b >= 2; b -= 2) {
        int vals_per_u32 = 32 / b;
        // gate_proj [mid, hid]
        size_t gw = (size_t)mid * ((hid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t gs_bytes = (size_t)mid * ((hid + gs - 1) / gs) * 2;
        size_t gb = gs_bytes;
        // up_proj [mid, hid] — same shape as gate
        size_t uw = gw, us = gs_bytes, ub = gb;
        // down_proj [hid, mid]
        size_t dw = (size_t)hid * ((mid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t ds = (size_t)hid * ((mid + gs - 1) / gs) * 2;
        size_t db = ds;

        size_t off = 0;
        if (b == 4) {
            c->gate_w_off_4 = off; off += gw;
            c->gate_s_off_4 = off; off += gs_bytes;
            c->gate_b_off_4 = off; off += gb;
            c->up_w_off_4 = off;   off += uw;
            c->up_s_off_4 = off;   off += us;
            c->up_b_off_4 = off;   off += ub;
            c->down_w_off_4 = off; off += dw;
            c->down_s_off_4 = off; off += ds;
            c->down_b_off_4 = off; off += db;
            c->expert_size_4bit = off;
        } else {
            c->gate_w_off_2 = off; off += gw;
            c->gate_s_off_2 = off; off += gs_bytes;
            c->gate_b_off_2 = off; off += gb;
            c->up_w_off_2 = off;   off += uw;
            c->up_s_off_2 = off;   off += us;
            c->up_b_off_2 = off;   off += ub;
            c->down_w_off_2 = off; off += dw;
            c->down_s_off_2 = off; off += ds;
            c->down_b_off_2 = off; off += db;
            c->expert_size_2bit = off;
        }
    }
}
```

### Static arrays to dynamic allocation

These static arrays use compile-time `NUM_LAYERS`/`NUM_EXPERTS` and must become dynamically allocated after config loading:

| Current | New |
|---------|-----|
| `static int g_expert_freq[NUM_LAYERS][NUM_EXPERTS]` | `int *g_expert_freq` (malloc `num_layers * num_experts * sizeof(int)`) |
| `static uint8_t g_expert_seen[NUM_LAYERS][NUM_EXPERTS/8]` | `uint8_t *g_expert_seen` (malloc `num_layers * ceil(num_experts/8)`) |
| `static uint8_t g_cache_seen[NUM_LAYERS][NUM_EXPERTS]` | `uint8_t *g_cache_seen` (malloc) |
| `static uint64_t g_cache_last_touch_token[NUM_LAYERS][NUM_EXPERTS]` | `uint64_t *g_cache_last_touch_token` (malloc) |
| `static uint64_t g_cache_last_evict_token[NUM_LAYERS][NUM_EXPERTS]` | `uint64_t *g_cache_last_evict_token` (malloc) |
| `static LayerWeightCache layer_cache[NUM_LAYERS]` | `LayerWeightCache *layer_cache` (malloc `num_layers * sizeof(...)`) |
| `LZ4IndexEntry *g_lz4_index[NUM_LAYERS]` | `LZ4IndexEntry **g_lz4_index` (malloc `num_layers` pointers) |
| `g_pred_experts[60][MAX_K]` | `int *g_pred_experts` (malloc `num_layers * MAX_K`, note: currently hardcoded to 60, a latent bug) |
| `g_pred_count[60]` | `int *g_pred_count` (malloc `num_layers`) |
| `id<MTLBuffer> buf_kv_k[NUM_FULL_ATTN_LAYERS]` | Dynamically allocated array in MetalCtx |
| `id<MTLBuffer> buf_kv_v[NUM_FULL_ATTN_LAYERS]` | Same |
| `id<MTLBuffer> buf_delta_state[NUM_LINEAR_LAYERS]` | Same |
| `id<MTLBuffer> buf_conv_state[NUM_LINEAR_LAYERS]` | Same |
| `id<MTLBuffer> buf_multi_expert_data[MAX_K]` | Stays `MAX_K` (hardware limit, not model-specific) |
| Stack VLAs `gpu_delta_snapshots[NUM_LINEAR_LAYERS]` (serve loop) | `malloc`'d at serve entry, freed at exit |
| Stack VLAs `gpu_conv_snapshots[NUM_LINEAR_LAYERS]` (serve loop) | Same |

Access pattern changes from `g_expert_freq[layer][expert]` to `g_expert_freq[layer * cfg.num_experts + expert]` (flattened 2D indexing). Helper macros can simplify this:

```c
#define FREQ(l, e)       g_expert_freq[(l) * cfg.num_experts + (e)]
#define CACHE_SEEN(l, e) g_cache_seen[(l) * cfg.num_experts + (e)]
```

### MetalCtx changes

The `MetalCtx` struct's fixed-size arrays become pointers, allocated in `metal_setup()` after config is loaded:

```c
typedef struct {
    // ...existing fields...
    id<MTLBuffer> *buf_kv_k;         // [num_full_attn_layers]
    id<MTLBuffer> *buf_kv_v;         // [num_full_attn_layers]
    id<MTLBuffer> *buf_delta_state;  // [num_linear_layers]
    id<MTLBuffer> *buf_conv_state;   // [num_linear_layers]
    // buf_multi_expert_data[MAX_K] stays fixed (MAX_K is hardware limit)
} MetalCtx;
```

### What remains as #define

Only non-model-specific constants:

- `MAX_K 8` — maximum supported experts per token (array sizing). Note: `MAX_BATCH_SLOTS` (currently 8) is coupled to `MAX_K` — keep them in sync.
- `GPU_KV_SEQ 8192` — GPU KV pre-allocation (tuning parameter)
- `TENSOR_HT_SIZE 8192` — hash table size (implementation detail)
- `NUM_IO_THREADS 8` — I/O thread pool size (hardware tuning)
- Metal threadgroup sizes (256, 64, 128) — GPU hardware tuning

### Layer type detection

Currently uses `(i + 1) % FULL_ATTN_INTERVAL == 0`. Replaced with config-driven lookup:

```c
// Old
int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);

// New
int is_full = cfg.is_full_attn[i];
```

The `full_attn_index[]` and `linear_index[]` arrays map global layer index to per-type index (used for KV cache / delta-net state buffer indexing). All arithmetic formulas like `(layer_idx + 1) / FULL_ATTN_INTERVAL - 1` must be replaced with `cfg.full_attn_index[i]` / `cfg.linear_index[i]` lookups. This includes `build_layer_cache()`, `fused_layer_forward()`, GPU snapshot save/restore, and any other site using the interval formula.

### Startup sequence

```
main()
  -> parse CLI args (--model path)
  -> load_model_config(model_path)    // NEW: populates cfg
  -> alloc_tracking_arrays()          // NEW: malloc g_expert_freq etc.
  -> metal_setup()                    // uses cfg.* for buffer sizes
  -> load_weights()                   // uses cfg.model_path
  -> inference loop                   // uses cfg.* throughout
```

### Thread safety invariant

`cfg` is immutable after `load_model_config()` returns. It is populated once at startup before any threads are spawned, and is read-only for the entire lifetime of the process. No locking required.

### CLI change

The existing `--model` flag already accepts a path. No new flags needed. The only change is that `load_model_config()` is called with this path before any other initialization. If `--model` is omitted, `load_model_config()` searches for any Qwen model in `~/.cache/huggingface/hub/` or prints a clear error asking the user to provide `--model`.

### Files changed

| File | Change |
|------|--------|
| `metal_infer/infer.m` | Remove ~40 `#define`s, add `ModelConfig` struct + `load_model_config()` (~150 lines), convert ~13 static/stack arrays to malloc, update ~200 references from `DEFINE` to `cfg.field`, replace all `FULL_ATTN_INTERVAL` formula sites with array lookups |
| `metal_infer/chat.m` | No changes needed (pure HTTP/SSE client, references no model constants) |
| `metal_infer/shaders.metal` | No changes (already parameterized via kernel arguments) |

### Validation

`load_model_config()` prints a summary to stderr on startup:

```
[config] Qwen3.5-35B-A3B: 40 layers (30 linear + 10 full), 2048 hidden, 256 experts (K=8)
[config] 4-bit expert size: 1769472 bytes, group_size=64
[config] EOS tokens: [248046, 248044], think: 248068/248069
```

This makes it immediately visible which model is loaded and whether the config was parsed correctly.

### Error handling

- Missing `config.json` → fatal error with clear message
- Missing `tokenizer_config.json` → warning, think tokens default to -1 (disabled)
- Missing `text_config` key → fatal error
- Missing optional keys (e.g., `linear_conv_kernel_dim`) → use sensible defaults with warning
- Computed expert offsets are validated against `expert_index.json` if present

### Note on extract_weights.py

`extract_weights.py` currently hardcodes model parameters (layer types, dimensions) when generating the weight manifest. This is acceptable for now — the manifest is generated once per model. A future improvement could make it config-driven too, but it's out of scope for this spec since the runtime engine is the priority.
