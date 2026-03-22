/*
 * infer.cu — CUDA inference engine for Qwen3.5-397B-A17B (MoE)
 *
 * Port of metal_infer/infer.m for NVIDIA GPUs.
 * Single-file engine: model loading, tokenization, forward pass, sampling.
 *
 * Architecture: Qwen3.5-397B-A17B
 *   - 60 layers: 45 GatedDeltaNet (linear attention) + 15 full attention
 *   - hidden_size=4096, head_dim=256, num_attn_heads=32, num_kv_heads=2
 *   - 512 experts/layer, K=4 active + 1 shared expert
 *
 * Build:
 *   nvcc -O2 -o infer infer.cu -lpthread
 *
 * Run:
 *   ./infer --prompt "Hello" --tokens 20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include "kernels.cuh"

// tokenizer.h: declarations only (impl in tokenizer_impl.c, linked separately)
extern "C" {
#include "../metal_infer/tokenizer.h"
}

// ============================================================================
// Model constants
// ============================================================================

#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define NUM_ATTN_HEADS      32
#define NUM_KV_HEADS        2
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         512
#define MOE_INTERMEDIATE    1024
#define SHARED_INTERMEDIATE 1024
#define FULL_ATTN_INTERVAL  4
#define GROUP_SIZE_C        64

// Linear attention (GatedDeltaNet)
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE    4

// Full attention
#define ROPE_THETA          10000000.0f
#define PARTIAL_ROTARY      0.25f
#define ROTARY_DIM          ((int)(HEAD_DIM * PARTIAL_ROTARY))  // 64
#define MAX_SEQ_LEN         4096

// Expert layout
#define EXPERT_SIZE         7077888
#define MAX_K               8

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline float bf16_to_f32_host(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &tmp, sizeof(f));
    return f;
}

// ============================================================================
// BPE byte-unicode decoding (Ġ→space, Ċ→newline, etc.)
// ============================================================================
// GPT-2 BPE maps bytes 0-255 to Unicode codepoints to avoid control chars.
// We need to reverse this mapping when displaying tokens.

static int g_bpe_byte_table[256];  // unicode codepoint → original byte
static int g_bpe_byte_table_built = 0;

static void build_bpe_decode_table(void) {
    if (g_bpe_byte_table_built) return;
    // Build forward map: byte → unicode (same as GPT-2 bytes_to_unicode)
    int unicode_map[256];
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            unicode_map[b] = b;
        } else {
            unicode_map[b] = 256 + n;
            n++;
        }
    }
    // Build reverse: unicode → byte
    memset(g_bpe_byte_table, -1, sizeof(g_bpe_byte_table));
    for (int b = 0; b < 256; b++) {
        if (unicode_map[b] < 256)
            g_bpe_byte_table[unicode_map[b]] = b;
    }
    // Handle the 256+ range (Ġ=288→0x20=space, Ċ=266→0x0A=newline, etc.)
    g_bpe_byte_table_built = 1;
}

// Decode a BPE token string to raw bytes, return length
static int bpe_decode_token(const char *token, char *out, int max_out) {
    build_bpe_decode_table();
    int j = 0;
    const unsigned char *p = (const unsigned char *)token;
    while (*p && j < max_out - 1) {
        // Decode UTF-8 codepoint
        uint32_t cp;
        int bytes;
        if (*p < 0x80) { cp = *p; bytes = 1; }
        else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F; bytes = 2; }
        else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F; bytes = 3; }
        else { cp = *p & 0x07; bytes = 4; }
        for (int i = 1; i < bytes && p[i]; i++)
            cp = (cp << 6) | (p[i] & 0x3F);
        p += bytes;

        // Map unicode codepoint back to byte
        if (cp < 256 && g_bpe_byte_table[cp] >= 0) {
            out[j++] = (char)g_bpe_byte_table[cp];
        } else if (cp >= 256 && cp < 512) {
            // Extended range: codepoints 256-511 map to bytes that were remapped
            // Build on-the-fly: find the byte whose unicode_map == cp
            int found = 0;
            int n = 0;
            for (int b = 0; b < 256 && !found; b++) {
                if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
                    continue;
                if (256 + n == (int)cp) { out[j++] = (char)b; found = 1; }
                n++;
            }
            if (!found) out[j++] = '?';
        } else {
            // Pass through other Unicode as-is (UTF-8 encode back)
            if (cp < 0x80) out[j++] = cp;
            else if (cp < 0x800 && j + 1 < max_out) {
                out[j++] = 0xC0 | (cp >> 6);
                out[j++] = 0x80 | (cp & 0x3F);
            } else if (cp < 0x10000 && j + 2 < max_out) {
                out[j++] = 0xE0 | (cp >> 12);
                out[j++] = 0x80 | ((cp >> 6) & 0x3F);
                out[j++] = 0x80 | (cp & 0x3F);
            }
        }
    }
    out[j] = '\0';
    return j;
}

static void print_token(const char *token) {
    char decoded[1024];
    bpe_decode_token(token, decoded, sizeof(decoded));
    printf("%s", decoded);
}

// ============================================================================
// Minimal JSON parser for model_weights.json manifest
// ============================================================================

typedef struct {
    char name[256];
    size_t offset;
    size_t size;
    char dtype[8];    // "U32", "BF16", "F32"
    int shape[4];
    int ndim;
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
} TensorManifest;

// Simple JSON string extraction (no dependencies)
static const char *json_find_key(const char *json, const char *key) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    return strstr(json, pattern);
}

static TensorManifest *load_manifest(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open manifest %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = (char *)malloc(sz + 1);
    fread(json, 1, sz, f);
    json[sz] = '\0';
    fclose(f);

    // Find "tensors" section
    const char *tensors_start = json_find_key(json, "tensors");
    if (!tensors_start) { fprintf(stderr, "No tensors in manifest\n"); free(json); return NULL; }

    // Count tensors (count "offset" occurrences)
    int count = 0;
    const char *p = tensors_start;
    while ((p = strstr(p + 1, "\"offset\"")) != NULL) count++;

    TensorManifest *m = (TensorManifest *)calloc(1, sizeof(TensorManifest));
    m->tensors = (TensorInfo *)calloc(count, sizeof(TensorInfo));
    m->num_tensors = 0;

    // Parse each tensor entry
    p = tensors_start;
    while (m->num_tensors < count) {
        // Find next tensor name (key before the {)
        p = strchr(p + 1, '"');
        if (!p) break;
        p++; // skip opening quote
        const char *name_end = strchr(p, '"');
        if (!name_end) break;

        // Skip non-tensor keys
        const char *brace = strchr(name_end, '{');
        if (!brace) break;

        // Check if this is a tensor entry (has "offset")
        const char *next_brace = strchr(brace + 1, '}');
        if (!next_brace) break;
        char *offset_key = strstr((char *)brace, "\"offset\"");
        if (!offset_key || offset_key > next_brace) {
            p = next_brace;
            continue;
        }

        TensorInfo *t = &m->tensors[m->num_tensors];
        size_t nlen = name_end - p;
        if (nlen >= sizeof(t->name)) nlen = sizeof(t->name) - 1;
        memcpy(t->name, p, nlen);
        t->name[nlen] = '\0';

        // Parse offset
        char *colon = strchr(offset_key, ':');
        if (colon) t->offset = strtoul(colon + 1, NULL, 10);

        // Parse size
        char *size_key = strstr((char *)brace, "\"size\"");
        if (size_key && size_key < next_brace) {
            colon = strchr(size_key, ':');
            if (colon) t->size = strtoul(colon + 1, NULL, 10);
        }

        // Parse dtype
        char *dtype_key = strstr((char *)brace, "\"dtype\"");
        if (dtype_key && dtype_key < next_brace) {
            char *dq = strchr(dtype_key + 7, '"');
            if (dq) {
                dq++;
                char *dq_end = strchr(dq, '"');
                if (dq_end) {
                    size_t dl = dq_end - dq;
                    if (dl < sizeof(t->dtype)) { memcpy(t->dtype, dq, dl); t->dtype[dl] = '\0'; }
                }
            }
        }

        // Parse shape
        char *shape_key = strstr((char *)brace, "\"shape\"");
        if (shape_key && shape_key < next_brace) {
            char *sb = strchr(shape_key, '[');
            if (sb) {
                t->ndim = 0;
                char *sp = sb + 1;
                while (t->ndim < 4) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp == ']') break;
                    t->shape[t->ndim++] = atoi(sp);
                    while (*sp && *sp != ',' && *sp != ']') sp++;
                }
            }
        }

        m->num_tensors++;
        p = next_brace;
    }

    free(json);
    return m;
}

// Raw JSON for fallback tensor lookup
static char *g_manifest_json = NULL;
static size_t g_manifest_json_len = 0;

// ============================================================================
// Weight file (mmap'd model_weights.bin)
// ============================================================================

typedef struct {
    void *data;
    size_t size;
    TensorManifest *manifest;
} WeightFile;

static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    // Load raw JSON for fallback tensor lookup
    {
        FILE *jf = fopen(json_path, "r");
        if (jf) {
            fseek(jf, 0, SEEK_END);
            g_manifest_json_len = ftell(jf);
            fseek(jf, 0, SEEK_SET);
            g_manifest_json = (char *)malloc(g_manifest_json_len + 1);
            fread(g_manifest_json, 1, g_manifest_json_len, jf);
            g_manifest_json[g_manifest_json_len] = '\0';
            fclose(jf);
        }
    }

    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) { perror(bin_path); return NULL; }
    struct stat st;
    fstat(fd, &st);
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("mmap"); return NULL; }
    madvise(data, st.st_size, MADV_SEQUENTIAL);

    WeightFile *wf = (WeightFile *)calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = st.st_size;
    wf->manifest = load_manifest(json_path);
    if (!wf->manifest) { munmap(data, st.st_size); free(wf); return NULL; }

    printf("[weights] Loaded %.2f GB, %d tensors\n",
           wf->size / (1024.0*1024*1024), wf->manifest->num_tensors);
    return wf;
}

static TensorInfo *find_tensor(WeightFile *wf, const char *name) {
    for (int i = 0; i < wf->manifest->num_tensors; i++) {
        if (strcmp(wf->manifest->tensors[i].name, name) == 0)
            return &wf->manifest->tensors[i];
    }
    // Fallback: if not found in parsed manifest, search raw JSON
    // (the custom parser may have missed some entries)
    return NULL;
}

// Forward declarations for raw JSON fallback
// (defined below, used in upload_tensor and open_weights)

static int find_tensor_in_json(const char *name, size_t *out_offset, size_t *out_size) {
    if (!g_manifest_json) return 0;
    // Search for "name": { ... "offset": N, "size": N ... }
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "\"%s\"", name);
    const char *pos = strstr(g_manifest_json, pattern);
    if (!pos) return 0;
    const char *brace = strchr(pos + strlen(pattern), '{');
    if (!brace) return 0;
    const char *end_brace = strchr(brace, '}');
    if (!end_brace) return 0;

    // Extract offset
    const char *off_key = strstr(brace, "\"offset\"");
    if (off_key && off_key < end_brace) {
        const char *colon = strchr(off_key, ':');
        if (colon) *out_offset = strtoul(colon + 1, NULL, 10);
    } else return 0;

    // Extract size
    const char *sz_key = strstr(brace, "\"size\"");
    if (sz_key && sz_key < end_brace) {
        const char *colon = strchr(sz_key, ':');
        if (colon) *out_size = strtoul(colon + 1, NULL, 10);
    } else return 0;

    return 1;
}

static void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf, name);
    if (!t) return NULL;
    return (char *)wf->data + t->offset;
}

// ============================================================================
// GPU weight storage — all non-expert weights uploaded to VRAM
// ============================================================================

typedef struct {
    // Per-layer weight pointers (on GPU)
    struct {
        // Input/post-attention norms (bf16)
        uint16_t *input_norm_w;
        uint16_t *post_attn_norm_w;

        // Full attention (15 layers)
        uint32_t *q_w; uint16_t *q_s, *q_b;
        uint32_t *k_w; uint16_t *k_s, *k_b;
        uint32_t *v_w; uint16_t *v_s, *v_b;
        uint32_t *o_w; uint16_t *o_s, *o_b;
        uint16_t *q_norm_w, *k_norm_w;

        // Linear attention (45 layers)
        uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
        uint32_t *z_w;   uint16_t *z_s, *z_b;
        uint32_t *b_w;   uint16_t *b_s, *b_b;   // beta projection
        uint32_t *a_w;   uint16_t *a_s, *a_b;   // alpha projection
        uint16_t *conv1d_w;
        float *A_log;
        uint16_t *dt_bias;
        uint16_t *gated_norm_w;
        uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;

        // MoE routing + shared expert
        uint32_t *gate_w; uint16_t *gate_s, *gate_b;
        uint32_t *sg_w;   uint16_t *sg_s, *sg_b;
        uint32_t *su_w;   uint16_t *su_s, *su_b;
        uint32_t *sd_w;   uint16_t *sd_s, *sd_b;
        uint32_t *seg_w;  uint16_t *seg_s, *seg_b;  // shared_expert_gate

        int is_full;
    } layers[NUM_LAYERS];

    // Global weights
    uint32_t *embed_w; uint16_t *embed_s, *embed_b;
    uint32_t *lm_head_w; uint16_t *lm_head_s, *lm_head_b;
    uint16_t *final_norm_w;

    // Scratch buffers (GPU)
    float *buf_hidden;       // [HIDDEN_DIM]
    float *buf_normed;       // [HIDDEN_DIM]
    float *buf_residual;     // [HIDDEN_DIM]
    float *buf_attn_out;     // [max(NUM_ATTN_HEADS*HEAD_DIM, LINEAR_TOTAL_VALUE)]

    // Attention projection outputs
    float *buf_q_proj;       // [NUM_ATTN_HEADS * HEAD_DIM * 2] or [LINEAR_CONV_DIM]
    float *buf_k_proj;       // [NUM_KV_HEADS * HEAD_DIM]
    float *buf_v_proj;       // [NUM_KV_HEADS * HEAD_DIM]
    float *buf_z_proj;       // [LINEAR_TOTAL_VALUE]
    float *buf_beta_proj;    // [LINEAR_NUM_V_HEADS]
    float *buf_alpha_proj;   // [LINEAR_NUM_V_HEADS]

    // Post-attention
    float *buf_h_mid;        // [HIDDEN_DIM] after o_proj + residual
    float *buf_gate_scores;  // [NUM_EXPERTS]
    float *buf_shared_gate;  // [SHARED_INTERMEDIATE]
    float *buf_shared_up;    // [SHARED_INTERMEDIATE]
    float *buf_shared_out;   // [HIDDEN_DIM]

    // Expert buffers
    float *buf_expert_outs;  // [MAX_K * HIDDEN_DIM]
    void  *buf_expert_data;  // [MAX_K * EXPERT_SIZE] raw expert data on GPU

    // Linear attention state (persistent across tokens)
    float *delta_state[NUM_LAYERS];  // [64 * 128 * 128] per linear layer
    float *conv_state[NUM_LAYERS];   // [3 * 12288] per linear layer
    float *buf_conv_output;          // [LINEAR_CONV_DIM]
    float *buf_g_decay;              // [LINEAR_NUM_V_HEADS]
    float *buf_beta_gate;            // [LINEAR_NUM_V_HEADS]
    float *buf_delta_output;         // [LINEAR_TOTAL_VALUE]

    // Full attention KV cache (persistent)
    float *kv_k[NUM_LAYERS];  // [MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM]
    float *kv_v[NUM_LAYERS];  // same
    int kv_len[NUM_LAYERS];   // current seq length per full-attn layer
    float *buf_attn_scores;   // [NUM_ATTN_HEADS * MAX_SEQ_LEN]
    float *buf_q;             // [NUM_ATTN_HEADS * HEAD_DIM] deinterleaved Q
    float *buf_q_gate;        // [NUM_ATTN_HEADS * HEAD_DIM] Q gate

    // Logits
    float *buf_logits;  // [VOCAB_SIZE] on GPU
    float *h_logits;    // [VOCAB_SIZE] pinned host memory

    // Expert I/O staging (pinned host)
    void *h_expert_buf[MAX_K];

    // Pre-allocated expert weights buffer (avoids per-layer cudaMalloc)
    float *buf_expert_weights;  // [MAX_K] on GPU

    // GDS handles (NULL if GDS not available)
    int gds_available;
    CUfileHandle_t gds_handles[NUM_LAYERS];

    // CUDA streams for I/O overlap
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;

    // Expert file descriptors
    int expert_fds[NUM_LAYERS];

    // GPU memory for bulk weight upload
    void *d_weights;  // single allocation for all non-expert weights
    size_t d_weights_size;

} Model;

// ============================================================================
// Upload a tensor from mmap to GPU, return device pointer
// ============================================================================

static void *upload_tensor(WeightFile *wf, const char *name, void *d_base, size_t *d_offset) {
    TensorInfo *t = find_tensor(wf, name);
    size_t t_offset = 0, t_size = 0;
    if (t) {
        t_offset = t->offset;
        t_size = t->size;
    } else if (find_tensor_in_json(name, &t_offset, &t_size)) {
        // Fallback: found in raw JSON
    } else {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    void *src = (char *)wf->data + t_offset;
    void *dst = (char *)d_base + *d_offset;
    CHECK_CUDA(cudaMemcpy(dst, src, t_size, cudaMemcpyHostToDevice));
    *d_offset += (t_size + 63) & ~63ULL;
    return dst;
}

// Helper macro for uploading weight triplets (weight, scales, biases)
#define UPLOAD_WEIGHT_TRIPLET(prefix, w_field, s_field, b_field) do { \
    char _n[256]; \
    snprintf(_n, sizeof(_n), "%s.weight", prefix); \
    model->w_field = (uint32_t *)upload_tensor(wf, _n, model->d_weights, &off); \
    snprintf(_n, sizeof(_n), "%s.scales", prefix); \
    model->s_field = (uint16_t *)upload_tensor(wf, _n, model->d_weights, &off); \
    snprintf(_n, sizeof(_n), "%s.biases", prefix); \
    model->b_field = (uint16_t *)upload_tensor(wf, _n, model->d_weights, &off); \
} while(0)

#define UPLOAD_LAYER_TRIPLET(layer_idx, prefix, w_field, s_field, b_field) do { \
    char _n[256]; \
    snprintf(_n, sizeof(_n), "model.layers.%d." prefix ".weight", layer_idx); \
    model->layers[layer_idx].w_field = (uint32_t *)upload_tensor(wf, _n, model->d_weights, &off); \
    snprintf(_n, sizeof(_n), "model.layers.%d." prefix ".scales", layer_idx); \
    model->layers[layer_idx].s_field = (uint16_t *)upload_tensor(wf, _n, model->d_weights, &off); \
    snprintf(_n, sizeof(_n), "model.layers.%d." prefix ".biases", layer_idx); \
    model->layers[layer_idx].b_field = (uint16_t *)upload_tensor(wf, _n, model->d_weights, &off); \
} while(0)

// ============================================================================
// Model initialization
// ============================================================================

static Model *model_init(WeightFile *wf, const char *expert_dir, int K) {
    Model *model = (Model *)calloc(1, sizeof(Model));

    printf("[init] Uploading %.2f GB of non-expert weights to GPU...\n",
           wf->size / (1024.0*1024*1024));
    double t0 = now_ms();

    // Allocate single GPU buffer for all weights (slightly over-allocate for alignment)
    model->d_weights_size = wf->size + NUM_LAYERS * 64 * 100;  // extra for alignment padding
    CHECK_CUDA(cudaMalloc(&model->d_weights, model->d_weights_size));

    size_t off = 0;

    // Global weights
    UPLOAD_WEIGHT_TRIPLET("model.embed_tokens", embed_w, embed_s, embed_b);
    UPLOAD_WEIGHT_TRIPLET("lm_head", lm_head_w, lm_head_s, lm_head_b);

    {
        char n[] = "model.norm.weight";
        model->final_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
    }

    // Per-layer weights
    for (int i = 0; i < NUM_LAYERS; i++) {
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        model->layers[i].is_full = is_full;

        // Norms
        {
            char n[256];
            snprintf(n, sizeof(n), "model.layers.%d.input_layernorm.weight", i);
            model->layers[i].input_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
            snprintf(n, sizeof(n), "model.layers.%d.post_attention_layernorm.weight", i);
            model->layers[i].post_attn_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
        }

        if (is_full) {
            UPLOAD_LAYER_TRIPLET(i, "self_attn.q_proj", q_w, q_s, q_b);
            UPLOAD_LAYER_TRIPLET(i, "self_attn.k_proj", k_w, k_s, k_b);
            UPLOAD_LAYER_TRIPLET(i, "self_attn.v_proj", v_w, v_s, v_b);
            UPLOAD_LAYER_TRIPLET(i, "self_attn.o_proj", o_w, o_s, o_b);
            {
                char n[256];
                snprintf(n, sizeof(n), "model.layers.%d.self_attn.q_norm.weight", i);
                model->layers[i].q_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
                snprintf(n, sizeof(n), "model.layers.%d.self_attn.k_norm.weight", i);
                model->layers[i].k_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
            }
        } else {
            UPLOAD_LAYER_TRIPLET(i, "linear_attn.in_proj_qkv", qkv_w, qkv_s, qkv_b);
            UPLOAD_LAYER_TRIPLET(i, "linear_attn.in_proj_z", z_w, z_s, z_b);
            UPLOAD_LAYER_TRIPLET(i, "linear_attn.in_proj_b", b_w, b_s, b_b);
            UPLOAD_LAYER_TRIPLET(i, "linear_attn.in_proj_a", a_w, a_s, a_b);
            UPLOAD_LAYER_TRIPLET(i, "linear_attn.out_proj", out_proj_w, out_proj_s, out_proj_b);
            {
                char n[256];
                snprintf(n, sizeof(n), "model.layers.%d.linear_attn.conv1d.weight", i);
                model->layers[i].conv1d_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
                snprintf(n, sizeof(n), "model.layers.%d.linear_attn.A_log", i);
                model->layers[i].A_log = (float *)upload_tensor(wf, n, model->d_weights, &off);
                snprintf(n, sizeof(n), "model.layers.%d.linear_attn.dt_bias", i);
                model->layers[i].dt_bias = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
                snprintf(n, sizeof(n), "model.layers.%d.linear_attn.norm.weight", i);
                model->layers[i].gated_norm_w = (uint16_t *)upload_tensor(wf, n, model->d_weights, &off);
            }
        }

        // MoE routing + shared expert (all layers)
        UPLOAD_LAYER_TRIPLET(i, "mlp.gate", gate_w, gate_s, gate_b);
        UPLOAD_LAYER_TRIPLET(i, "mlp.shared_expert.gate_proj", sg_w, sg_s, sg_b);
        UPLOAD_LAYER_TRIPLET(i, "mlp.shared_expert.up_proj", su_w, su_s, su_b);
        UPLOAD_LAYER_TRIPLET(i, "mlp.shared_expert.down_proj", sd_w, sd_s, sd_b);
        UPLOAD_LAYER_TRIPLET(i, "mlp.shared_expert_gate", seg_w, seg_s, seg_b);
    }

    printf("[init] Uploaded %.2f GB in %.1f ms (offset=%zu)\n",
           off / (1024.0*1024*1024), now_ms() - t0, off);

    // Allocate scratch buffers
    CHECK_CUDA(cudaMalloc(&model->buf_hidden, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_normed, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_residual, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_attn_out, LINEAR_TOTAL_VALUE * sizeof(float)));

    int max_proj = NUM_ATTN_HEADS * HEAD_DIM * 2;  // full attn q_proj
    if (LINEAR_CONV_DIM > max_proj) max_proj = LINEAR_CONV_DIM;
    CHECK_CUDA(cudaMalloc(&model->buf_q_proj, max_proj * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_k_proj, NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_v_proj, NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_z_proj, LINEAR_TOTAL_VALUE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_beta_proj, LINEAR_NUM_V_HEADS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_alpha_proj, LINEAR_NUM_V_HEADS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&model->buf_h_mid, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_gate_scores, NUM_EXPERTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_shared_gate, SHARED_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_shared_up, SHARED_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_shared_out, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_expert_outs, MAX_K * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_expert_data, MAX_K * EXPERT_SIZE));

    // Linear attention persistent state
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (!model->layers[i].is_full) {
            CHECK_CUDA(cudaMalloc(&model->delta_state[i],
                LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float)));
            CHECK_CUDA(cudaMemset(model->delta_state[i], 0,
                LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&model->conv_state[i],
                (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float)));
            CHECK_CUDA(cudaMemset(model->conv_state[i], 0,
                (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float)));
        }
    }
    CHECK_CUDA(cudaMalloc(&model->buf_conv_output, LINEAR_CONV_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_g_decay, LINEAR_NUM_V_HEADS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_beta_gate, LINEAR_NUM_V_HEADS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_delta_output, LINEAR_TOTAL_VALUE * sizeof(float)));

    // Full attention KV caches
    int kv_size = MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (model->layers[i].is_full) {
            CHECK_CUDA(cudaMalloc(&model->kv_k[i], kv_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&model->kv_v[i], kv_size * sizeof(float)));
            CHECK_CUDA(cudaMemset(model->kv_k[i], 0, kv_size * sizeof(float)));
            CHECK_CUDA(cudaMemset(model->kv_v[i], 0, kv_size * sizeof(float)));
            model->kv_len[i] = 0;
        }
    }
    CHECK_CUDA(cudaMalloc(&model->buf_attn_scores, NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_q, NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&model->buf_q_gate, NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)));

    // Logits
    CHECK_CUDA(cudaMalloc(&model->buf_logits, VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&model->h_logits, VOCAB_SIZE * sizeof(float)));

    // Pre-allocated expert weights buffer
    CHECK_CUDA(cudaMalloc(&model->buf_expert_weights, MAX_K * sizeof(float)));

    // CUDA streams for I/O overlap
    CHECK_CUDA(cudaStreamCreate(&model->stream_compute));
    CHECK_CUDA(cudaStreamCreate(&model->stream_transfer));

    // Expert staging (pinned host)
    for (int i = 0; i < K; i++)
        CHECK_CUDA(cudaMallocHost(&model->h_expert_buf[i], EXPERT_SIZE));

    // Open expert files
    for (int i = 0; i < NUM_LAYERS; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", expert_dir, i);
        model->expert_fds[i] = open(path, O_RDONLY);
        if (model->expert_fds[i] < 0) {
            fprintf(stderr, "WARNING: Cannot open %s: %s\n", path, strerror(errno));
        }
    }

    // GDS vs page cache: pread with page cache is faster for sustained generation
    // because hot experts stay in RAM (~3ms vs 5.3ms). GDS bypasses page cache.
    // Use --gds flag or ENABLE_GDS=1 env var to force GDS (useful if RAM < 32GB).
    model->gds_available = 0;
    int want_gds = (getenv("ENABLE_GDS") != NULL);
    CUfileError_t gds_status;
    if (!want_gds) { gds_status.err = (CUfileOpError)999; }
    else { gds_status = cuFileDriverOpen(); }
    if (gds_status.err == CU_FILE_SUCCESS) {
        int gds_ok = 1;
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (model->expert_fds[i] < 0) continue;
            // Re-open with O_DIRECT for GDS
            char path[512];
            snprintf(path, sizeof(path), "%s/layer_%02d.bin", expert_dir, i);
            int dfd = open(path, O_RDONLY | O_DIRECT);
            if (dfd < 0) { gds_ok = 0; break; }

            CUfileDescr_t desc = {};
            desc.handle.fd = dfd;
            desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            CUfileError_t s = cuFileHandleRegister(&model->gds_handles[i], &desc);
            if (s.err != CU_FILE_SUCCESS) { close(dfd); gds_ok = 0; break; }
        }
        if (gds_ok) {
            // Register expert data buffer for GDS
            cuFileBufRegister(model->buf_expert_data, MAX_K * EXPERT_SIZE, 0);
            model->gds_available = 1;
            printf("[init] GDS: enabled (direct SSD→GPU, set ENABLE_GDS=1)\n");
        } else {
            printf("[init] Using pread + page cache (best for 32GB+ RAM)\n");
            cuFileDriverClose();
        }
    } else {
        printf("[init] Using pread + page cache (set ENABLE_GDS=1 to force GDS)\n");
    }

    // Print GPU memory usage
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    printf("[init] GPU memory: %.2f GB used, %.2f GB free / %.2f GB total\n",
           (total_mem - free_mem) / (1024.0*1024*1024),
           free_mem / (1024.0*1024*1024),
           total_mem / (1024.0*1024*1024));

    return model;
}

// ============================================================================
// Embedding lookup (GPU dequant one row)
// ============================================================================

static void embed_token(Model *model, int token_id) {
    // Compute row offset into packed embedding table
    uint32_t packed_cols = HIDDEN_DIM / 8;  // 512
    uint32_t num_groups = HIDDEN_DIM / GROUP_SIZE_C;  // 64

    uint32_t *W = model->embed_w + token_id * packed_cols;
    uint16_t *S = model->embed_s + token_id * num_groups;
    uint16_t *B = model->embed_b + token_id * num_groups;

    // Use dequant matvec with a "unit" input to extract the row
    // Actually, embedding is just dequantizing a single row, not a matvec.
    // Let's do it with a simple kernel or CPU-side.
    // For now, CPU dequant (embedding is a one-time cost per token):
    uint32_t h_W[512];
    uint16_t h_S[64], h_B[64];
    CHECK_CUDA(cudaMemcpy(h_W, W, packed_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_S, S, num_groups * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, B, num_groups * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    float h_out[HIDDEN_DIM];
    for (uint32_t g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32_host(h_S[g]);
        float bias  = bf16_to_f32_host(h_B[g]);
        uint32_t base = g * (GROUP_SIZE_C / 8);
        for (uint32_t p = 0; p < GROUP_SIZE_C / 8; p++) {
            uint32_t packed = h_W[base + p];
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                h_out[g * GROUP_SIZE_C + p * 8 + n] = (float)nibble * scale + bias;
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(model->buf_hidden, h_out, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
}

// ============================================================================
// Expert I/O: parallel pread + cudaMemcpy
// ============================================================================

typedef struct {
    int fd;
    void *buf;
    size_t size;
    off_t offset;
} PreadArg;

static void *pread_worker(void *arg) {
    PreadArg *a = (PreadArg *)arg;
    (void)pread(a->fd, a->buf, a->size, a->offset);
    return NULL;
}

// GDS expert loading: direct SSD→GPU
typedef struct {
    CUfileHandle_t handle;
    void *d_buf;
    size_t size;
    off_t offset;
} GDSArg;

static void *gds_worker(void *arg) {
    GDSArg *a = (GDSArg *)arg;
    cuFileRead(a->handle, a->d_buf, a->size, a->offset, 0);
    return NULL;
}

static void load_experts(Model *model, int layer_idx, const int *expert_ids, int K) {
    if (model->gds_available) {
        // GDS path: parallel cuFileRead directly to GPU memory
        pthread_t threads[MAX_K];
        GDSArg args[MAX_K];
        for (int i = 0; i < K; i++) {
            args[i].handle = model->gds_handles[layer_idx];
            args[i].d_buf = (char *)model->buf_expert_data + i * EXPERT_SIZE;
            args[i].size = EXPERT_SIZE;
            args[i].offset = (off_t)expert_ids[i] * EXPERT_SIZE;
            pthread_create(&threads[i], NULL, gds_worker, &args[i]);
        }
        for (int i = 0; i < K; i++)
            pthread_join(threads[i], NULL);
    } else {
        // Fallback: parallel pread → pinned host → cudaMemcpyAsync
        pthread_t threads[MAX_K];
        PreadArg args[MAX_K];
        int fd = model->expert_fds[layer_idx];
        for (int i = 0; i < K; i++) {
            args[i].fd = fd;
            args[i].buf = model->h_expert_buf[i];
            args[i].size = EXPERT_SIZE;
            args[i].offset = (off_t)expert_ids[i] * EXPERT_SIZE;
            pthread_create(&threads[i], NULL, pread_worker, &args[i]);
        }
        for (int i = 0; i < K; i++)
            pthread_join(threads[i], NULL);
        // Async copy to GPU
        for (int i = 0; i < K; i++) {
            CHECK_CUDA(cudaMemcpyAsync(
                (char *)model->buf_expert_data + i * EXPERT_SIZE,
                model->h_expert_buf[i], EXPERT_SIZE,
                cudaMemcpyHostToDevice, model->stream_transfer));
        }
        CHECK_CUDA(cudaStreamSynchronize(model->stream_transfer));
    }
}

// ============================================================================
// Expert forward pass (one expert on GPU)
// ============================================================================

// Expert component offsets within EXPERT_SIZE bytes
#define EXP_GATE_W   0
#define EXP_GATE_S   2097152
#define EXP_GATE_B   2228224
#define EXP_UP_W     2359296
#define EXP_UP_S     4456448
#define EXP_UP_B     4587520
#define EXP_DOWN_W   4718592
#define EXP_DOWN_S   6815744
#define EXP_DOWN_B   6946816

static void expert_forward(Model *model, int expert_slot, const float *input, float *output) {
    void *base = (char *)model->buf_expert_data + expert_slot * EXPERT_SIZE;

    uint32_t *gate_w = (uint32_t *)((char *)base + EXP_GATE_W);
    uint16_t *gate_s = (uint16_t *)((char *)base + EXP_GATE_S);
    uint16_t *gate_b = (uint16_t *)((char *)base + EXP_GATE_B);
    uint32_t *up_w   = (uint32_t *)((char *)base + EXP_UP_W);
    uint16_t *up_s   = (uint16_t *)((char *)base + EXP_UP_S);
    uint16_t *up_b   = (uint16_t *)((char *)base + EXP_UP_B);
    uint32_t *down_w = (uint32_t *)((char *)base + EXP_DOWN_W);
    uint16_t *down_s = (uint16_t *)((char *)base + EXP_DOWN_S);
    uint16_t *down_b = (uint16_t *)((char *)base + EXP_DOWN_B);

    // gate_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] → buf_shared_gate
    launch_dequant_matvec(gate_w, gate_s, gate_b, input, model->buf_shared_gate,
                          MOE_INTERMEDIATE, HIDDEN_DIM);
    // up_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] → buf_shared_up
    launch_dequant_matvec(up_w, up_s, up_b, input, model->buf_shared_up,
                          MOE_INTERMEDIATE, HIDDEN_DIM);
    // SwiGLU
    launch_swiglu(model->buf_shared_gate, model->buf_shared_up, model->buf_shared_gate,
                  MOE_INTERMEDIATE);
    // down_proj: [HIDDEN_DIM, MOE_INTERMEDIATE] → output
    launch_dequant_matvec(down_w, down_s, down_b, model->buf_shared_gate, output,
                          HIDDEN_DIM, MOE_INTERMEDIATE);
}

// ============================================================================
// CPU-side routing: softmax + topK
// ============================================================================

static void cpu_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void topk(const float *scores, int n, int k, int *indices, float *weights) {
    // Simple selection sort for small k
    uint8_t *used = (uint8_t *)calloc(n, 1);
    for (int j = 0; j < k; j++) {
        int best = -1;
        float best_val = -1e30f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && scores[i] > best_val) {
                best_val = scores[i];
                best = i;
            }
        }
        indices[j] = best;
        weights[j] = best_val;
        if (best >= 0) used[best] = 1;
    }
    free(used);

    // Renormalize weights
    float sum = 0.0f;
    for (int j = 0; j < k; j++) sum += weights[j];
    if (sum > 0) for (int j = 0; j < k; j++) weights[j] /= sum;
}

// ============================================================================
// CPU-side RoPE for full attention
// ============================================================================

static void apply_rope(float *q, float *k, int pos) {
    int half = ROTARY_DIM / 2;
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        float *qh = q + h * HEAD_DIM;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / ROTARY_DIM);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * c - q1 * s;
            qh[i + half]  = q0 * s + q1 * c;
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        float *kh = k + h * HEAD_DIM;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / ROTARY_DIM);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float k0 = kh[i], k1 = kh[i + half];
            kh[i]        = k0 * c - k1 * s;
            kh[i + half]  = k0 * s + k1 * c;
        }
    }
}

// ============================================================================
// Per-layer forward pass
// ============================================================================

// Timing accumulator for per-phase breakdown
static int g_timing_enabled = 0;
static struct {
    double input_norm, attn_proj, attn_compute, oproj_residual;
    double routing, shared_expert, expert_io, expert_compute, combine;
    double total;
    int count;
} g_layer_timing;

static void layer_forward(Model *model, int layer_idx, int pos, int K) {
    auto &L = model->layers[layer_idx];
    double t0, t1;

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t0 = now_ms(); }

    // 1. Input RMS norm
    launch_rms_norm_bf16(model->buf_hidden, L.input_norm_w, model->buf_normed,
                         HIDDEN_DIM, RMS_NORM_EPS);

    // Save residual
    CHECK_CUDA(cudaMemcpy(model->buf_residual, model->buf_hidden,
                          HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.input_norm += t1-t0; t0=t1; }

    // 2. Attention projections + attention compute
    if (L.is_full) {
        // Full attention path
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;  // interleaved Q + gate
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;

        launch_dequant_matvec(L.q_w, L.q_s, L.q_b, model->buf_normed,
                              model->buf_q_proj, q_proj_dim, HIDDEN_DIM);
        launch_dequant_matvec(L.k_w, L.k_s, L.k_b, model->buf_normed,
                              model->buf_k_proj, kv_dim, HIDDEN_DIM);
        launch_dequant_matvec(L.v_w, L.v_s, L.v_b, model->buf_normed,
                              model->buf_v_proj, kv_dim, HIDDEN_DIM);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Deinterleave Q and Q_gate, apply Q/K norms, RoPE, attention — on CPU
        // (CPU is fine since it's only 15 layers and attention is memory-bound at low seq_len)
        float h_q_proj[NUM_ATTN_HEADS * HEAD_DIM * 2];
        float h_k[NUM_KV_HEADS * HEAD_DIM];
        float h_v[NUM_KV_HEADS * HEAD_DIM];
        CHECK_CUDA(cudaMemcpy(h_q_proj, model->buf_q_proj, q_proj_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_k, model->buf_k_proj, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_v, model->buf_v_proj, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));

        // Deinterleave: q_proj is [num_heads, 2*head_dim] → split into q[num_heads, head_dim] + gate
        float h_q[NUM_ATTN_HEADS * HEAD_DIM];
        float h_qg[NUM_ATTN_HEADS * HEAD_DIM];
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            memcpy(h_q + h * HEAD_DIM, h_q_proj + h * 2 * HEAD_DIM, HEAD_DIM * sizeof(float));
            memcpy(h_qg + h * HEAD_DIM, h_q_proj + h * 2 * HEAD_DIM + HEAD_DIM, HEAD_DIM * sizeof(float));
        }

        // Q/K RMS norm with learned weights
        uint16_t h_qnorm[HEAD_DIM], h_knorm[HEAD_DIM];
        CHECK_CUDA(cudaMemcpy(h_qnorm, L.q_norm_w, HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_knorm, L.k_norm_w, HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            float *qh = h_q + h * HEAD_DIM;
            float sum_sq = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum_sq += qh[d] * qh[d];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int d = 0; d < HEAD_DIM; d++) qh[d] *= inv_rms * bf16_to_f32_host(h_qnorm[d]);
        }
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            float *kh = h_k + h * HEAD_DIM;
            float sum_sq = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum_sq += kh[d] * kh[d];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int d = 0; d < HEAD_DIM; d++) kh[d] *= inv_rms * bf16_to_f32_host(h_knorm[d]);
        }

        // RoPE
        apply_rope(h_q, h_k, pos);

        // KV cache update
        int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
        int cache_pos = model->kv_len[layer_idx];
        CHECK_CUDA(cudaMemcpy(
            model->kv_k[layer_idx] + cache_pos * kv_dim,
            h_k, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(
            model->kv_v[layer_idx] + cache_pos * kv_dim,
            h_v, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
        model->kv_len[layer_idx]++;

        // Attention: Q@K^T → softmax → @V on GPU
        int seq_len = model->kv_len[layer_idx];
        int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);

        CHECK_CUDA(cudaMemcpy(model->buf_q, h_q, NUM_ATTN_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));

        attn_scores<<<seq_len * NUM_ATTN_HEADS, 256>>>(
            model->buf_q, model->kv_k[layer_idx], model->buf_attn_scores,
            HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, scale, heads_per_kv, seq_len);

        attn_softmax<<<NUM_ATTN_HEADS, 256>>>(
            model->buf_attn_scores, seq_len, MAX_SEQ_LEN);

        int attn_threads = NUM_ATTN_HEADS * HEAD_DIM;
        attn_values<<<(attn_threads + 255) / 256, 256>>>(
            model->buf_attn_scores, model->kv_v[layer_idx], model->buf_attn_out,
            HEAD_DIM, kv_dim, seq_len, MAX_SEQ_LEN, heads_per_kv);

        // Sigmoid gate: attn_out *= sigmoid(q_gate)
        CHECK_CUDA(cudaMemcpy(model->buf_q_gate, h_qg,
            NUM_ATTN_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
        sigmoid_gate<<<(attn_threads + 255) / 256, 256>>>(
            model->buf_attn_out, model->buf_q_gate, attn_threads);

        // O projection
        int oproj_in = NUM_ATTN_HEADS * HEAD_DIM;
        launch_dequant_matvec(L.o_w, L.o_s, L.o_b, model->buf_attn_out,
                              model->buf_h_mid, HIDDEN_DIM, oproj_in);

    } else {
        // Linear attention (GatedDeltaNet) path — all on GPU
        launch_dequant_matvec(L.qkv_w, L.qkv_s, L.qkv_b, model->buf_normed,
                              model->buf_q_proj, LINEAR_CONV_DIM, HIDDEN_DIM);
        launch_dequant_matvec(L.z_w, L.z_s, L.z_b, model->buf_normed,
                              model->buf_z_proj, LINEAR_TOTAL_VALUE, HIDDEN_DIM);
        launch_dequant_matvec(L.b_w, L.b_s, L.b_b, model->buf_normed,
                              model->buf_beta_proj, LINEAR_NUM_V_HEADS, HIDDEN_DIM);
        launch_dequant_matvec(L.a_w, L.a_s, L.a_b, model->buf_normed,
                              model->buf_alpha_proj, LINEAR_NUM_V_HEADS, HIDDEN_DIM);

        // Conv1d step
        conv1d_step<<<(LINEAR_CONV_DIM + 255) / 256, 256>>>(
            model->conv_state[layer_idx], model->buf_q_proj,
            L.conv1d_w, model->buf_conv_output, LINEAR_CONV_DIM);

        // RMS norm Q and K
        float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
        rms_norm_qk<<<LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM>>>(
            model->buf_conv_output,
            model->buf_conv_output + LINEAR_TOTAL_KEY,  // k starts after q
            LINEAR_KEY_DIM, inv_scale);

        // Compute decay and beta gate
        compute_decay_beta<<<1, LINEAR_NUM_V_HEADS>>>(
            model->buf_alpha_proj, model->buf_beta_proj,
            L.A_log, L.dt_bias,
            model->buf_g_decay, model->buf_beta_gate);

        // GatedDeltaNet recurrence
        uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
        gated_delta_net_step<<<LINEAR_NUM_V_HEADS, 128>>>(
            model->delta_state[layer_idx],
            model->buf_conv_output,                           // q [2048]
            model->buf_conv_output + LINEAR_TOTAL_KEY,        // k [2048]
            model->buf_conv_output + 2 * LINEAR_TOTAL_KEY,    // v [8192]
            model->buf_g_decay, model->buf_beta_gate,
            model->buf_delta_output, khpv);

        // Gated RMS norm
        gated_rms_norm<<<LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM>>>(
            model->buf_delta_output, model->buf_z_proj,
            L.gated_norm_w, model->buf_attn_out,
            LINEAR_VALUE_DIM, RMS_NORM_EPS);

        // Output projection
        launch_dequant_matvec(L.out_proj_w, L.out_proj_s, L.out_proj_b,
                              model->buf_attn_out, model->buf_h_mid,
                              HIDDEN_DIM, LINEAR_TOTAL_VALUE);
    }

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.attn_compute += t1-t0; t0=t1; }

    // 3. Residual + post-attention norm
    launch_residual_add(model->buf_residual, model->buf_h_mid, model->buf_h_mid, HIDDEN_DIM);
    launch_rms_norm_bf16(model->buf_h_mid, L.post_attn_norm_w, model->buf_normed,
                         HIDDEN_DIM, RMS_NORM_EPS);

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.oproj_residual += t1-t0; t0=t1; }

    // 4. MoE routing
    launch_dequant_matvec(L.gate_w, L.gate_s, L.gate_b, model->buf_normed,
                          model->buf_gate_scores, NUM_EXPERTS, HIDDEN_DIM);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_scores[NUM_EXPERTS];
    CHECK_CUDA(cudaMemcpy(h_scores, model->buf_gate_scores, NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_softmax(h_scores, NUM_EXPERTS);

    int expert_ids[MAX_K];
    float expert_weights[MAX_K];
    topk(h_scores, NUM_EXPERTS, K, expert_ids, expert_weights);

    if (g_timing_enabled) { t1 = now_ms(); g_layer_timing.routing += t1-t0; t0=t1; }

    // 5. Shared expert forward + expert I/O OVERLAP
    // Launch shared expert on GPU compute stream while loading experts from SSD
    launch_dequant_matvec(L.sg_w, L.sg_s, L.sg_b, model->buf_normed,
                          model->buf_shared_gate, SHARED_INTERMEDIATE, HIDDEN_DIM);
    launch_dequant_matvec(L.su_w, L.su_s, L.su_b, model->buf_normed,
                          model->buf_shared_up, SHARED_INTERMEDIATE, HIDDEN_DIM);
    launch_swiglu(model->buf_shared_gate, model->buf_shared_up, model->buf_shared_gate,
                  SHARED_INTERMEDIATE);
    launch_dequant_matvec(L.sd_w, L.sd_s, L.sd_b, model->buf_shared_gate,
                          model->buf_shared_out, HIDDEN_DIM, SHARED_INTERMEDIATE);

    // Shared expert gate score (can overlap with I/O)
    launch_dequant_matvec(L.seg_w, L.seg_s, L.seg_b, model->buf_normed,
                          model->buf_gate_scores, 1, HIDDEN_DIM);

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.shared_expert += t1-t0; t0=t1; }

    // 6. Load K experts from SSD (overlaps with shared expert GPU work above)
    load_experts(model, layer_idx, expert_ids, K);

    if (g_timing_enabled) { t1 = now_ms(); g_layer_timing.expert_io += t1-t0; t0=t1; }

    // 7. Expert forward (K experts on GPU)
    for (int k = 0; k < K; k++) {
        expert_forward(model, k, model->buf_normed,
                       model->buf_expert_outs + k * HIDDEN_DIM);
    }

    // 8. MoE combine + residual (no per-layer malloc)
    float h_seg_score;
    CHECK_CUDA(cudaMemcpy(&h_seg_score, model->buf_gate_scores, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemcpy(model->buf_expert_weights, expert_weights,
                          K * sizeof(float), cudaMemcpyHostToDevice));

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.expert_compute += t1-t0; t0=t1; }

    moe_combine_residual<<<(HIDDEN_DIM + 255) / 256, 256>>>(
        model->buf_h_mid, model->buf_shared_out, model->buf_hidden,
        model->buf_expert_outs, model->buf_expert_weights, h_seg_score,
        HIDDEN_DIM, K);

    if (g_timing_enabled) { CHECK_CUDA(cudaDeviceSynchronize()); t1 = now_ms(); g_layer_timing.combine += t1-t0;
        g_layer_timing.count++;
    }
}

// ============================================================================
// Full forward pass: embedding → 60 layers → norm → lm_head → argmax
// ============================================================================

static int forward(Model *model, int token_id, int pos, int K) {
    // Embedding
    embed_token(model, token_id);

    // Reset timing
    if (g_timing_enabled) memset(&g_layer_timing, 0, sizeof(g_layer_timing));

    // 60 layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        layer_forward(model, i, pos, K);
    }

    // Print timing summary
    if (g_timing_enabled && g_layer_timing.count > 0) {
        int n = g_layer_timing.count;
        fprintf(stderr, "[timing] Per-layer avg (%.0f layers): "
                "norm=%.2f attn=%.2f oproj=%.2f route=%.2f "
                "shared=%.2f io=%.2f expert=%.2f combine=%.2f ms\n",
                (double)n,
                g_layer_timing.input_norm/n, g_layer_timing.attn_compute/n,
                g_layer_timing.oproj_residual/n, g_layer_timing.routing/n,
                g_layer_timing.shared_expert/n, g_layer_timing.expert_io/n,
                g_layer_timing.expert_compute/n, g_layer_timing.combine/n);
    }

    // Final RMS norm
    launch_rms_norm_bf16(model->buf_hidden, model->final_norm_w, model->buf_normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
    // LM head: [VOCAB_SIZE, HIDDEN_DIM] → logits
    launch_dequant_matvec(model->lm_head_w, model->lm_head_s, model->lm_head_b,
                          model->buf_normed, model->buf_logits,
                          VOCAB_SIZE, HIDDEN_DIM);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy logits to host and argmax
    CHECK_CUDA(cudaMemcpy(model->h_logits, model->buf_logits,
                          VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    int best = 0;
    float best_val = model->h_logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (model->h_logits[i] > best_val) {
            best_val = model->h_logits[i];
            best = i;
        }
    }
    return best;
}

// ============================================================================
// HTTP Server — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================

#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>

static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 && buf[total-4]=='\r' && buf[total-3]=='\n' &&
            buf[total-2]=='\r' && buf[total-1]=='\n') break;
    }
    buf[total] = '\0';
    // Read body if Content-Length present
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            int got = 0;
            while (got < content_len) {
                ssize_t r = read(fd, buf + total + got, content_len - got);
                if (r <= 0) break;
                got += r;
            }
            total += got;
            buf[total] = '\0';
        }
    }
    return total;
}

static void http_write_str(int fd, const char *s) {
    int len = strlen(s), sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, s + sent, len - sent);
        if (w <= 0) break;
        sent += w;
    }
}

static char *extract_last_content(char *buf) {
    char *last = NULL, *p = buf;
    for (;;) {
        p = strstr(p, "\"content\"");
        if (!p) break;
        p += 9;
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') { p++; last = p; while (*p && !(*p == '"' && *(p-1) != '\\')) p++; }
    }
    if (last) {
        char *end = last;
        while (*end && !(*end == '"' && (end == last || *(end-1) != '\\'))) end++;
        *end = '\0';
        // Unescape
        char *r = last, *w = last;
        while (*r) {
            if (*r == '\\' && *(r+1)) {
                r++;
                switch (*r) {
                    case 'n': *w++ = '\n'; r++; break;
                    case 't': *w++ = '\t'; r++; break;
                    case '"': *w++ = '"'; r++; break;
                    case '\\': *w++ = '\\'; r++; break;
                    default: *w++ = '\\'; *w++ = *r++; break;
                }
            } else *w++ = *r++;
        }
        *w = '\0';
    }
    return last;
}

// Extract a string value for a given key from JSON body. Returns 0 if not found.
static int extract_string_field(const char *buf, const char *key, char *out, int out_sz) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(buf, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < out_sz - 1) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

static int extract_max_tokens(const char *buf, int def) {
    const char *p = strstr(buf, "\"max_completion_tokens\"");
    if (!p) p = strstr(buf, "\"max_tokens\"");
    if (!p) return def;
    p = strchr(p, ':');
    return p ? atoi(p + 1) : def;
}

static int sse_send_delta(int fd, const char *req_id, const char *token_text) {
    char chunk[4096], escaped[2048];
    char *w = escaped;
    for (const char *r = token_text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"': *w++ = '\\'; *w++ = '"'; break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n'; break;
            case '\r': *w++ = '\\'; *w++ = 'r'; break;
            case '\t': *w++ = '\\'; *w++ = 't'; break;
            default: *w++ = *r; break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        req_id, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done(int fd, const char *req_id) {
    char chunk[1024];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n", req_id);
    http_write_str(fd, chunk);
}

#define EOS_TOKEN_1 248044  // <|endoftext|>
#define EOS_TOKEN_2 248046  // <|im_end|>
#define IM_START    248045  // <|im_start|>
#define THINK_START 248068  // <think>
#define THINK_END   248069  // </think>

// ============================================================================
// Tool calling support — extract tools from request, format for Qwen, parse output
// ============================================================================

// Extract the "tools" JSON array from the request body (returns malloc'd string or NULL)
static char *extract_tools_json(const char *body) {
    const char *p = strstr(body, "\"tools\"");
    if (!p) return NULL;
    p = strchr(p + 7, '[');
    if (!p) return NULL;
    // Find matching ]
    int depth = 1;
    const char *start = p;
    p++;
    while (*p && depth > 0) {
        if (*p == '[') depth++;
        else if (*p == ']') depth--;
        p++;
    }
    if (depth != 0) return NULL;
    size_t len = p - start;
    char *result = (char *)malloc(len + 1);
    memcpy(result, start, len);
    result[len] = '\0';
    return result;
}

// Build a full ChatML prompt from the OpenAI messages array + tools.
// Returns malloc'd string ready for tokenization.
// Build a per-request prompt from OpenAI messages array + tools.
// System prompt is already in KV cache — this only generates the user turn(s).
static char *build_chat_prompt(const char *body, const char *tools_json) {
    size_t bufsize = strlen(body) * 2 + (tools_json ? strlen(tools_json) * 2 : 0) + 65536;
    char *prompt = (char *)calloc(1, bufsize);
    char *w = prompt;

    // If tools provided, inject as a system addendum before user messages
    if (tools_json) {
        w += sprintf(w, "<|im_start|>system\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n%s\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            "{\"name\": \"<function-name>\", \"arguments\": {<args-json-object>}}\n</tool_call>"
            "<|im_end|>\n", tools_json);
    }

    // Parse messages array — find each role/content pair
    const char *msgs = strstr(body, "\"messages\"");
    if (!msgs) { w += sprintf(w, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"); return prompt; }

    const char *arr = strchr(msgs, '[');
    if (!arr) { w += sprintf(w, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"); return prompt; }

    // Simple message parser: find each {"role":"...", "content":"..."} pair
    const char *p = arr + 1;
    while (*p) {
        // Find next "role"
        const char *role_key = strstr(p, "\"role\"");
        if (!role_key) break;
        const char *role_val = strchr(role_key + 6, '"');
        if (!role_val) break;
        role_val++; // skip quote
        const char *role_end = strchr(role_val, '"');
        if (!role_end) break;

        char role[32] = {};
        size_t rlen = role_end - role_val;
        if (rlen >= sizeof(role)) rlen = sizeof(role) - 1;
        memcpy(role, role_val, rlen);

        // Find content for this message
        const char *content_key = strstr(role_end, "\"content\"");
        if (!content_key) { p = role_end + 1; continue; }

        // Skip to value
        const char *cv = content_key + 9;
        while (*cv == ' ' || *cv == ':' || *cv == '\t') cv++;

        if (*cv == '"') {
            cv++; // skip opening quote
            // Find end quote (handle escapes)
            const char *ce = cv;
            while (*ce && !(*ce == '"' && *(ce-1) != '\\')) ce++;

            // Write ChatML turn
            if (strcmp(role, "system") == 0) {
                // System message already handled above, skip
            } else if (strcmp(role, "tool") == 0) {
                // Tool result — find tool_call_id and name
                w += sprintf(w, "<|im_start|>user\n[Tool result]: ");
                size_t clen = ce - cv;
                memcpy(w, cv, clen); w += clen;
                w += sprintf(w, "<|im_end|>\n");
            } else {
                w += sprintf(w, "<|im_start|>%s\n", role);
                // Copy content, unescaping JSON escapes
                const char *r = cv;
                while (r < ce) {
                    if (*r == '\\' && r + 1 < ce) {
                        r++;
                        switch (*r) {
                            case 'n': *w++ = '\n'; break;
                            case 't': *w++ = '\t'; break;
                            case '"': *w++ = '"'; break;
                            case '\\': *w++ = '\\'; break;
                            default: *w++ = '\\'; *w++ = *r; break;
                        }
                        r++;
                    } else {
                        *w++ = *r++;
                    }
                }
                w += sprintf(w, "<|im_end|>\n");
            }
            p = ce + 1;
        } else if (*cv == 'n') {
            // null content (e.g., assistant tool call message)
            if (strcmp(role, "assistant") == 0) {
                // Check for tool_calls in this message
                const char *tc = strstr(role_end, "\"tool_calls\"");
                if (tc) {
                    w += sprintf(w, "<|im_start|>assistant\n<tool_call>\n");
                    // Extract function name and arguments
                    const char *fname = strstr(tc, "\"name\"");
                    if (fname) {
                        const char *fv = strchr(fname + 6, '"');
                        if (fv) {
                            fv++;
                            const char *fe = strchr(fv, '"');
                            if (fe) {
                                w += sprintf(w, "{\"name\": \"");
                                memcpy(w, fv, fe - fv); w += (fe - fv);
                                w += sprintf(w, "\", \"arguments\": ");
                            }
                        }
                    }
                    const char *fargs = strstr(tc, "\"arguments\"");
                    if (fargs) {
                        const char *av = strchr(fargs + 11, '"');
                        if (av) {
                            av++;
                            const char *ae = av;
                            while (*ae && !(*ae == '"' && *(ae-1) != '\\')) ae++;
                            // Unescape and write
                            const char *r = av;
                            while (r < ae) {
                                if (*r == '\\' && r + 1 < ae) {
                                    r++;
                                    switch (*r) {
                                        case 'n': *w++ = '\n'; break;
                                        case '"': *w++ = '"'; break;
                                        case '\\': *w++ = '\\'; break;
                                        default: *w++ = *r; break;
                                    }
                                    r++;
                                } else *w++ = *r++;
                            }
                        }
                    }
                    w += sprintf(w, "}\n</tool_call><|im_end|>\n");
                }
            }
            p = cv + 4;
        } else {
            p = cv + 1;
        }
    }

    // End with assistant prompt
    w += sprintf(w, "<|im_start|>assistant\n");
    return prompt;
}

// Send a tool_call SSE chunk (OpenAI format)
static int sse_send_tool_call(int fd, const char *req_id, const char *call_id,
                              const char *func_name, const char *arguments) {
    char chunk[8192], esc_args[4096];
    // Escape arguments for JSON
    char *w = esc_args;
    for (const char *r = arguments; *r && w < esc_args + sizeof(esc_args) - 8; r++) {
        switch (*r) {
            case '"': *w++ = '\\'; *w++ = '"'; break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n'; break;
            default: *w++ = *r; break;
        }
    }
    *w = '\0';

    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"%s\","
        "\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},"
        "\"finish_reason\":null}]}\n\n",
        req_id, call_id, func_name, esc_args);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_tool_done(int fd, const char *req_id) {
    char chunk[1024];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
        "data: [DONE]\n\n", req_id);
    http_write_str(fd, chunk);
}

// Parse a tool call from generated text. Looks for <tool_call>...</tool_call> pattern.
// Returns 1 if found, fills name and arguments buffers.
static int parse_tool_call(const char *text, char *name, int name_sz, char *args, int args_sz) {
    const char *start = strstr(text, "<tool_call>");
    if (!start) return 0;
    start += 11; // skip tag
    const char *end = strstr(start, "</tool_call>");

    // Find "name" in the JSON
    const char *np = strstr(start, "\"name\"");
    if (!np || (end && np > end)) return 0;
    np = strchr(np + 6, '"');
    if (!np) return 0;
    np++;
    const char *ne = strchr(np, '"');
    if (!ne) return 0;
    int nlen = ne - np;
    if (nlen >= name_sz) nlen = name_sz - 1;
    memcpy(name, np, nlen);
    name[nlen] = '\0';

    // Find "arguments" in the JSON
    const char *ap = strstr(start, "\"arguments\"");
    if (!ap || (end && ap > end)) { args[0] = '{'; args[1] = '}'; args[2] = '\0'; return 1; }
    ap = strchr(ap + 11, '{');
    if (!ap) { args[0] = '{'; args[1] = '}'; args[2] = '\0'; return 1; }
    // Find matching }
    int depth = 1;
    const char *aps = ap + 1;
    while (*aps && depth > 0) {
        if (*aps == '{') depth++;
        else if (*aps == '}') depth--;
        aps++;
    }
    int alen = aps - ap;
    if (alen >= args_sz) alen = args_sz - 1;
    memcpy(args, ap, alen);
    args[alen] = '\0';
    return 1;
}

// ============================================================================
// Anthropic Messages API — SSE helpers
// ============================================================================

static void anth_send_message_start(int fd, const char *msg_id, const char *model_name) {
    char buf[2048];
    int n = snprintf(buf, sizeof(buf),
        "event: message_start\n"
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"content\":[],\"model\":\"%s\","
        "\"stop_reason\":null,\"stop_sequence\":null,"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
        msg_id, model_name);
    http_write_str(fd, buf);
}

static void anth_send_content_block_start(int fd, int index, const char *type) {
    char buf[512];
    if (strcmp(type, "text") == 0) {
        snprintf(buf, sizeof(buf),
            "event: content_block_start\n"
            "data: {\"type\":\"content_block_start\",\"index\":%d,"
            "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n", index);
    }
    http_write_str(fd, buf);
}

static void anth_send_content_block_start_tool(int fd, int index, const char *tool_id,
                                                const char *func_name) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":%d,"
        "\"content_block\":{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":\"%s\",\"input\":{}}}\n\n",
        index, tool_id, func_name);
    http_write_str(fd, buf);
}

static int anth_send_text_delta(int fd, int index, const char *text) {
    char chunk[4096], escaped[2048];
    char *w = escaped;
    for (const char *r = text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"': *w++ = '\\'; *w++ = '"'; break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n'; break;
            case '\r': *w++ = '\\'; *w++ = 'r'; break;
            case '\t': *w++ = '\\'; *w++ = 't'; break;
            default: *w++ = *r; break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":%d,"
        "\"delta\":{\"type\":\"text_delta\",\"text\":\"%s\"}}\n\n",
        index, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static int anth_send_tool_input_delta(int fd, int index, const char *json_delta) {
    char chunk[8192], escaped[4096];
    char *w = escaped;
    for (const char *r = json_delta; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"': *w++ = '\\'; *w++ = '"'; break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n'; break;
            default: *w++ = *r; break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":%d,"
        "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"%s\"}}\n\n",
        index, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void anth_send_content_block_stop(int fd, int index) {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":%d}\n\n", index);
    http_write_str(fd, buf);
}

static void anth_send_message_delta(int fd, const char *stop_reason, int output_tokens) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "event: message_delta\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
        "\"usage\":{\"output_tokens\":%d}}\n\n",
        stop_reason, output_tokens);
    http_write_str(fd, buf);
}

static void anth_send_message_stop(int fd) {
    http_write_str(fd,
        "event: message_stop\n"
        "data: {\"type\":\"message_stop\"}\n\n");
}

// Build per-request ChatML prompt from Anthropic Messages API request.
// System prompt is already in KV cache — this only generates user turn(s) + tools.
static char *build_anthropic_prompt(const char *body, const char *system_prompt) {
    size_t bufsize = strlen(body) * 2 + 65536;
    char *prompt = (char *)calloc(1, bufsize);
    char *w = prompt;

    // If tools provided, inject as system addendum
    char *tools_json = extract_tools_json(body);
    if (tools_json) {
        w += sprintf(w, "<|im_start|>system\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            "%s\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            "{\"name\": \"<function-name>\", \"arguments\": {<args-json-object>}}\n</tool_call>"
            "<|im_end|>\n", tools_json);
        free(tools_json);
    }

    // Parse messages array
    const char *msgs = strstr(body, "\"messages\"");
    if (!msgs) { w += sprintf(w, "<|im_start|>assistant\n"); return prompt; }
    const char *arr = strchr(msgs, '[');
    if (!arr) { w += sprintf(w, "<|im_start|>assistant\n"); return prompt; }

    // Iterate messages — Anthropic format:
    // {"role": "user"/"assistant", "content": "string" or [{"type":"text","text":"..."},{"type":"tool_result",...}]}
    const char *p = arr + 1;
    while (*p) {
        const char *role_key = strstr(p, "\"role\"");
        if (!role_key) break;
        const char *role_val = strchr(role_key + 6, '"');
        if (!role_val) break;
        role_val++;
        const char *role_end = strchr(role_val, '"');
        if (!role_end) break;

        char role[32] = {};
        size_t rlen = role_end - role_val;
        if (rlen >= sizeof(role)) rlen = sizeof(role) - 1;
        memcpy(role, role_val, rlen);

        // Find content
        const char *content_key = strstr(role_end, "\"content\"");
        if (!content_key) { p = role_end + 1; continue; }
        const char *cv = content_key + 9;
        while (*cv == ' ' || *cv == ':' || *cv == '\t') cv++;

        if (*cv == '"') {
            // Simple string content
            cv++;
            const char *ce = cv;
            while (*ce && !(*ce == '"' && *(ce-1) != '\\')) ce++;

            w += sprintf(w, "<|im_start|>%s\n", role);
            const char *r = cv;
            while (r < ce) {
                if (*r == '\\' && r + 1 < ce) {
                    r++;
                    switch (*r) { case 'n': *w++ = '\n'; break; case 't': *w++ = '\t'; break;
                                  case '"': *w++ = '"'; break; case '\\': *w++ = '\\'; break;
                                  default: *w++ = '\\'; *w++ = *r; break; }
                    r++;
                } else *w++ = *r++;
            }
            w += sprintf(w, "<|im_end|>\n");
            p = ce + 1;
        } else if (*cv == '[') {
            // Array content — may contain text blocks and tool_result blocks
            w += sprintf(w, "<|im_start|>%s\n", role);
            // Find matching ]
            int depth = 1;
            const char *as = cv + 1;
            while (*as && depth > 0) {
                if (*as == '[') depth++;
                else if (*as == ']') depth--;
                if (depth > 0) as++;
            }
            // Scan for text and tool_result blocks within the array
            const char *scan = cv + 1;
            while (scan < as) {
                const char *type_key = strstr(scan, "\"type\"");
                if (!type_key || type_key >= as) break;
                const char *tv = strchr(type_key + 6, '"');
                if (!tv || tv >= as) break;
                tv++;
                if (strncmp(tv, "text\"", 5) == 0) {
                    // Find "text" field
                    const char *tk = strstr(tv, "\"text\"");
                    if (tk && tk < as) {
                        const char *tval = strchr(tk + 6, '"');
                        if (tval) { tval++;
                            const char *te = tval;
                            while (*te && !(*te == '"' && *(te-1) != '\\')) te++;
                            const char *r = tval;
                            while (r < te) {
                                if (*r == '\\' && r + 1 < te) { r++;
                                    switch (*r) { case 'n': *w++ = '\n'; break; case '"': *w++ = '"'; break;
                                                  case '\\': *w++ = '\\'; break; default: *w++ = *r; break; }
                                    r++;
                                } else *w++ = *r++;
                            }
                            scan = te + 1;
                            continue;
                        }
                    }
                } else if (strncmp(tv, "tool_use\"", 9) == 0) {
                    // Tool use block from assistant — add as <tool_call>
                    w += sprintf(w, "<tool_call>\n");
                    const char *nk = strstr(tv, "\"name\"");
                    if (nk && nk < as) {
                        const char *nv = strchr(nk + 6, '"'); if (nv) { nv++;
                            const char *ne = strchr(nv, '"');
                            if (ne) { w += sprintf(w, "{\"name\": \""); memcpy(w, nv, ne-nv); w += (ne-nv); w += sprintf(w, "\", \"arguments\": "); }
                        }
                    }
                    const char *ik = strstr(tv, "\"input\"");
                    if (ik && ik < as) {
                        const char *iv = strchr(ik + 7, '{');
                        if (iv) { int d = 1; const char *ie = iv + 1;
                            while (*ie && d > 0) { if (*ie == '{') d++; else if (*ie == '}') d--; ie++; }
                            memcpy(w, iv, ie - iv); w += (ie - iv);
                        }
                    }
                    w += sprintf(w, "}\n</tool_call>");
                    scan = tv + 9;
                    continue;
                } else if (strncmp(tv, "tool_result\"", 12) == 0) {
                    // Tool result — extract content
                    w += sprintf(w, "[Tool result]: ");
                    const char *ck = strstr(tv, "\"content\"");
                    if (ck && ck < as) {
                        const char *cval = strchr(ck + 9, '"');
                        if (cval) { cval++;
                            const char *ce = cval;
                            while (*ce && !(*ce == '"' && *(ce-1) != '\\')) ce++;
                            memcpy(w, cval, ce - cval); w += (ce - cval);
                            scan = ce + 1;
                            continue;
                        }
                    }
                }
                scan = tv + 5;
            }
            w += sprintf(w, "<|im_end|>\n");
            p = as + 1;
        } else {
            p = cv + 1;
        }
    }

    w += sprintf(w, "<|im_start|>assistant\n");
    return prompt;
}

static void serve_loop(Model *model, char **vocab_strings, bpe_tokenizer *tokenizer,
                       int port, int K) {
    signal(SIGPIPE, SIG_IGN);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); close(server_fd); return;
    }

    printf("[serve] Listening on http://0.0.0.0:%d\n", port);
    printf("[serve] Endpoints:\n");
    printf("[serve]   POST /v1/chat/completions  (OpenAI format)\n");
    printf("[serve]   POST /v1/messages          (Anthropic format)\n");
    printf("[serve]   GET  /v1/models\n");
    printf("[serve]   GET  /health\n");
    fflush(stdout);

    size_t delta_sz = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    size_t conv_sz = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // ---- System prompt pre-cache ----
    // Tokenize and prefill the system prompt once at startup.
    // Snapshot the resulting state so each request restores from here
    // instead of starting from scratch (~8s saved per request).
    const char *sys_text = "You are a helpful assistant.";
    {
        const char *home = getenv("HOME");
        if (home) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/.flash-moe/system.md", home);
            FILE *f = fopen(path, "r");
            if (f) {
                fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                char *buf = (char *)malloc(sz + 1);
                buf[fread(buf, 1, sz, f)] = 0;
                fclose(f);
                sys_text = buf;
                fprintf(stderr, "[serve] Custom system prompt from %s (%ld bytes)\n", path, sz);
            }
        }
    }

    // Build system prompt in ChatML format
    char *sys_chatml = (char *)malloc(strlen(sys_text) + 256);
    sprintf(sys_chatml, "<|im_start|>system\n%s<|im_end|>\n", sys_text);

    uint32_t sys_ids[4096];
    int sys_ntokens = bpe_encode(tokenizer, sys_chatml, sys_ids, 4096);
    free(sys_chatml);

    fprintf(stderr, "[serve] System prompt: %d tokens, prefilling...\n", sys_ntokens);
    double t_prefill = now_ms();
    for (int i = 0; i < sys_ntokens; i++)
        forward(model, sys_ids[i], i, K);
    int sys_pos = sys_ntokens;
    fprintf(stderr, "[serve] System prompt cached in %.0f ms\n", now_ms() - t_prefill);

    // Snapshot KV caches + delta-net + conv states after system prompt
    void *snap_kv_k[NUM_LAYERS] = {}, *snap_kv_v[NUM_LAYERS] = {};
    int snap_kv_len[NUM_LAYERS] = {};
    void *snap_delta[NUM_LAYERS] = {}, *snap_conv[NUM_LAYERS] = {};

    for (int i = 0; i < NUM_LAYERS; i++) {
        if (model->layers[i].is_full) {
            size_t sz = sys_pos * kv_dim * sizeof(float);
            snap_kv_k[i] = malloc(sz); snap_kv_v[i] = malloc(sz);
            CHECK_CUDA(cudaMemcpy(snap_kv_k[i], model->kv_k[i], sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(snap_kv_v[i], model->kv_v[i], sz, cudaMemcpyDeviceToHost));
            snap_kv_len[i] = model->kv_len[i];
        } else {
            snap_delta[i] = malloc(delta_sz); snap_conv[i] = malloc(conv_sz);
            CHECK_CUDA(cudaMemcpy(snap_delta[i], model->delta_state[i], delta_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(snap_conv[i], model->conv_state[i], conv_sz, cudaMemcpyDeviceToHost));
        }
    }
    fprintf(stderr, "[serve] State snapshot saved (%d layers)\n", NUM_LAYERS);

    uint64_t req_counter = 0;

    // Session tracking — keep KV cache across requests in the same session
    char active_session[128] = {};
    int session_pos = 0;  // RoPE position after last generation

    fprintf(stderr, "[serve] Ready\n");

    static const char *SSE_HEADERS =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    static const char *CORS_RESPONSE =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "\r\n";

    for (;;) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept"); continue; }

        char *reqbuf = (char *)malloc(1024 * 1024);
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

        char method[16] = {}, path_buf[256] = {};
        sscanf(reqbuf, "%15s %255s", method, path_buf);

        // CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd); continue;
        }

        // GET /health
        if (strcmp(method, "GET") == 0 && strcmp(path_buf, "/health") == 0) {
            http_write_str(client_fd,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                "{\"status\":\"ok\",\"model\":\"qwen3.5-397b-a17b-cuda\"}\n");
            free(reqbuf); close(client_fd); continue;
        }

        // GET /v1/models
        if (strcmp(method, "GET") == 0 && strcmp(path_buf, "/v1/models") == 0) {
            http_write_str(client_fd,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.5-397b-a17b\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}\n");
            free(reqbuf); close(client_fd); continue;
        }

        // POST /v1/chat/completions
        if (strcmp(method, "POST") == 0 && strcmp(path_buf, "/v1/chat/completions") == 0) {
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd, "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            int max_gen = extract_max_tokens(body, 4096);
            if (max_gen > 32768) max_gen = 32768;

            // Extract tools and session_id
            char *tools_json = extract_tools_json(body);
            char req_session[128] = {};
            extract_string_field(body, "session_id", req_session, sizeof(req_session));

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "chatcmpl-%llu", (unsigned long long)++req_counter);

            // Determine if this is a continuation of the active session
            int is_continuation = (req_session[0] && active_session[0] &&
                                   strcmp(req_session, active_session) == 0);

            fprintf(stderr, "[serve] %s max_tokens=%d tools=%s session=%s%s\n",
                    request_id, max_gen, tools_json ? "yes" : "no",
                    req_session[0] ? req_session : "(none)",
                    is_continuation ? " [CONTINUE]" : " [NEW]");

            int pos;
            if (is_continuation) {
                // Continue from existing state — no restore needed
                pos = session_pos;
            } else {
                // New session — restore from system prompt snapshot
                for (int i = 0; i < NUM_LAYERS; i++) {
                    if (model->layers[i].is_full) {
                        size_t sz = snap_kv_len[i] * kv_dim * sizeof(float);
                        if (sz > 0) {
                            CHECK_CUDA(cudaMemcpy(model->kv_k[i], snap_kv_k[i], sz, cudaMemcpyHostToDevice));
                            CHECK_CUDA(cudaMemcpy(model->kv_v[i], snap_kv_v[i], sz, cudaMemcpyHostToDevice));
                        }
                        model->kv_len[i] = snap_kv_len[i];
                    } else {
                        CHECK_CUDA(cudaMemcpy(model->delta_state[i], snap_delta[i], delta_sz, cudaMemcpyHostToDevice));
                        CHECK_CUDA(cudaMemcpy(model->conv_state[i], snap_conv[i], conv_sz, cudaMemcpyHostToDevice));
                    }
                }
                pos = sys_pos;
                if (req_session[0]) {
                    strncpy(active_session, req_session, sizeof(active_session) - 1);
                } else {
                    active_session[0] = '\0';
                }
            }

            // Build per-request prompt (user turn + tools only, system prompt already cached)
            char *prompt = build_chat_prompt(body, tools_json);
            if (tools_json) free(tools_json);

            uint32_t turn_ids[16384];
            int turn_ntokens = bpe_encode(tokenizer, prompt, turn_ids, 16384);
            free(prompt);

            if (turn_ntokens <= 0) {
                http_write_str(client_fd, "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n{\"error\":\"tokenization failed\"}\n");
                free(reqbuf); close(client_fd); continue;
            }

            fprintf(stderr, "[serve] %s prompt=%d tokens, pos=%d\n", request_id, turn_ntokens, pos);

            // Send SSE headers
            http_write_str(client_fd, SSE_HEADERS);

            // Prefill user turn tokens — last forward() return = first generated token
            int next_token = 0;
            for (int i = 0; i < turn_ntokens; i++) {
                next_token = forward(model, turn_ids[i], pos++, K);
            }

            double t_gen = now_ms();
            int gen_count = 0;
            int client_ok = 1;

            // Buffer for detecting tool calls in output
            char gen_buffer[65536] = {};
            int gen_buf_len = 0;
            int in_tool_call = 0;
            int tool_call_count = 0;

            // Special token IDs to suppress from output
            // These are Qwen3.5 special token IDs that should not appear as content
            int suppress_tokens[] = {
                EOS_TOKEN_1,  // <|endoftext|>
                IM_START,     // <|im_start|>
                EOS_TOKEN_2,  // <|im_end|>
            };
            int n_suppress = sizeof(suppress_tokens) / sizeof(suppress_tokens[0]);

            for (int gen = 0; gen < max_gen && client_ok; gen++) {
                // Stop on EOS tokens
                if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) break;

                // Check if this is a suppressed special token
                int is_special = 0;
                for (int s = 0; s < n_suppress; s++) {
                    if (next_token == suppress_tokens[s]) { is_special = 1; break; }
                }
                if (is_special) {
                    // Special token — don't output, just continue generating
                    gen_count++;
                    next_token = forward(model, next_token, pos++, K);
                    continue;
                }

                // Decode token
                char decoded[1024] = {};
                if (vocab_strings[next_token])
                    bpe_decode_token(vocab_strings[next_token], decoded, sizeof(decoded));

                // Accumulate in buffer for tool call detection
                if (gen_buf_len + (int)strlen(decoded) < (int)sizeof(gen_buffer) - 1) {
                    strcpy(gen_buffer + gen_buf_len, decoded);
                    gen_buf_len += strlen(decoded);
                }

                // Check for tool call start
                if (!in_tool_call && strstr(gen_buffer, "<tool_call>")) {
                    in_tool_call = 1;
                    // Flush any content before <tool_call> that was already sent
                    // (the "<tool_call>" text itself was accumulated but not sent)
                }

                // Stop if decoded text contains EOS markers
                if (strstr(decoded, "<|im_end|>") || strstr(decoded, "<|endoftext|>")) break;

                // Filter special text patterns from content
                int is_filtered = (
                    strstr(decoded, "<|im_start|>") ||
                    strstr(decoded, "<|im_end|>") ||
                    strstr(decoded, "<|endoftext|>") ||
                    strcmp(decoded, "<think>") == 0 ||
                    strcmp(decoded, "</think>") == 0 ||
                    strcmp(decoded, "user") == 0 ||     // stray role tokens
                    strcmp(decoded, "assistant") == 0 || // stray role tokens
                    strcmp(decoded, "system") == 0       // stray role tokens
                );

                // If not in a tool call and not filtered, stream content
                if (!in_tool_call && decoded[0] && !is_filtered) {
                    if (sse_send_delta(client_fd, request_id, decoded) < 0) {
                        client_ok = 0; break;
                    }
                }

                // Check for tool call end
                if (in_tool_call && strstr(gen_buffer, "</tool_call>")) {
                    // Parse and send the tool call
                    char func_name[256] = {}, func_args[4096] = {};
                    if (parse_tool_call(gen_buffer, func_name, sizeof(func_name),
                                        func_args, sizeof(func_args))) {
                        char call_id[64];
                        snprintf(call_id, sizeof(call_id), "call_%d", ++tool_call_count);
                        sse_send_tool_call(client_fd, request_id, call_id, func_name, func_args);
                        fprintf(stderr, "[serve] %s tool_call: %s(%s)\n",
                                request_id, func_name, func_args);
                    }
                    // Stop generation after tool call — the client needs to
                    // execute the tool and send results back in a new request
                    break;
                }

                gen_count++;
                next_token = forward(model, next_token, pos++, K);
            }

            if (client_ok) {
                if (tool_call_count > 0) {
                    sse_send_tool_done(client_fd, request_id);
                } else {
                    sse_send_done(client_fd, request_id);
                }
            }

            // Save session position for continuation
            session_pos = pos;

            double gen_ms = now_ms() - t_gen;
            fprintf(stderr, "[serve] %s generated %d tokens in %.0fms (%.2f tok/s)%s pos=%d\n",
                    request_id, gen_count, gen_ms,
                    gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0.0,
                    tool_call_count > 0 ? " [tool_calls]" : "", pos);

            free(reqbuf); close(client_fd); continue;
        }

        // POST /v1/messages (Anthropic Messages API)
        if (strcmp(method, "POST") == 0 && strcmp(path_buf, "/v1/messages") == 0) {
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd, "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            int max_gen = extract_max_tokens(body, 4096);
            if (max_gen > 32768) max_gen = 32768;

            char req_session[128] = {};
            extract_string_field(body, "session_id", req_session, sizeof(req_session));
            // Also check x-session-id header for Anthropic clients
            if (!req_session[0]) {
                const char *hdr = strcasestr(reqbuf, "x-session-id:");
                if (hdr) { hdr += 13; while (*hdr == ' ') hdr++;
                    int i = 0; while (*hdr && *hdr != '\r' && *hdr != '\n' && i < 127) req_session[i++] = *hdr++;
                    req_session[i] = '\0';
                }
            }

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "msg_%llu", (unsigned long long)++req_counter);

            int is_continuation = (req_session[0] && active_session[0] &&
                                   strcmp(req_session, active_session) == 0);

            fprintf(stderr, "[serve] %s (anthropic) max_tokens=%d session=%s%s\n",
                    request_id, max_gen,
                    req_session[0] ? req_session : "(none)",
                    is_continuation ? " [CONTINUE]" : " [NEW]");

            int pos;
            if (is_continuation) {
                pos = session_pos;
            } else {
                // Restore from system prompt snapshot
                for (int i = 0; i < NUM_LAYERS; i++) {
                    if (model->layers[i].is_full) {
                        size_t sz = snap_kv_len[i] * kv_dim * sizeof(float);
                        if (sz > 0) {
                            CHECK_CUDA(cudaMemcpy(model->kv_k[i], snap_kv_k[i], sz, cudaMemcpyHostToDevice));
                            CHECK_CUDA(cudaMemcpy(model->kv_v[i], snap_kv_v[i], sz, cudaMemcpyHostToDevice));
                        }
                        model->kv_len[i] = snap_kv_len[i];
                    } else {
                        CHECK_CUDA(cudaMemcpy(model->delta_state[i], snap_delta[i], delta_sz, cudaMemcpyHostToDevice));
                        CHECK_CUDA(cudaMemcpy(model->conv_state[i], snap_conv[i], conv_sz, cudaMemcpyHostToDevice));
                    }
                }
                pos = sys_pos;
                if (req_session[0])
                    strncpy(active_session, req_session, sizeof(active_session) - 1);
                else
                    active_session[0] = '\0';
            }

            // Build prompt from Anthropic format (user turn only, system prompt cached)
            char *prompt = build_anthropic_prompt(body, "You are a helpful assistant.");
            uint32_t turn_ids[16384];
            int turn_ntokens = bpe_encode(tokenizer, prompt, turn_ids, 16384);
            free(prompt);

            if (turn_ntokens <= 0) {
                http_write_str(client_fd, "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"tokenization failed\"}}\n");
                free(reqbuf); close(client_fd); continue;
            }

            fprintf(stderr, "[serve] %s prompt=%d tokens, pos=%d\n", request_id, turn_ntokens, pos);

            // Send SSE headers
            static const char *ANTH_SSE_HEADERS =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "\r\n";
            http_write_str(client_fd, ANTH_SSE_HEADERS);

            // message_start
            anth_send_message_start(client_fd, request_id, "qwen3.5-397b-a17b");

            // Prefill
            int next_token = 0;
            for (int i = 0; i < turn_ntokens; i++)
                next_token = forward(model, turn_ids[i], pos++, K);

            // Start text content block
            int block_index = 0;
            anth_send_content_block_start(client_fd, block_index, "text");

            double t_gen = now_ms();
            int gen_count = 0;
            int client_ok = 1;
            char gen_buffer[65536] = {};
            int gen_buf_len = 0;
            int in_tool_call = 0;
            int tool_call_count = 0;
            const char *stop_reason = "end_turn";

            // Special tokens to suppress
            int suppress_tokens[] = { 151643, 151644, 151645, 151646, 151647, 151648,
                                      151649, 151650, 151651, 151652, 151653, 151654 };
            int n_suppress = sizeof(suppress_tokens) / sizeof(suppress_tokens[0]);

            for (int gen = 0; gen < max_gen && client_ok; gen++) {
                if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) break;

                int is_special = 0;
                for (int s = 0; s < n_suppress; s++)
                    if (next_token == suppress_tokens[s]) { is_special = 1; break; }
                if (is_special) {
                    gen_count++;
                    next_token = forward(model, next_token, pos++, K);
                    continue;
                }

                char decoded[1024] = {};
                if (vocab_strings[next_token])
                    bpe_decode_token(vocab_strings[next_token], decoded, sizeof(decoded));

                if (strstr(decoded, "<|im_end|>") || strstr(decoded, "<|endoftext|>")) break;

                // Accumulate for tool call detection
                if (gen_buf_len + (int)strlen(decoded) < (int)sizeof(gen_buffer) - 1) {
                    strcpy(gen_buffer + gen_buf_len, decoded);
                    gen_buf_len += strlen(decoded);
                }

                if (!in_tool_call && strstr(gen_buffer, "<tool_call>"))
                    in_tool_call = 1;

                // Stream text content
                if (!in_tool_call && decoded[0]) {
                    int is_filtered = (
                        strstr(decoded, "<|im_start|>") || strstr(decoded, "<|im_end|>") ||
                        strstr(decoded, "<|endoftext|>") ||
                        strcmp(decoded, "<think>") == 0 || strcmp(decoded, "</think>") == 0 ||
                        strcmp(decoded, "user") == 0 || strcmp(decoded, "assistant") == 0 ||
                        strcmp(decoded, "system") == 0
                    );
                    if (!is_filtered) {
                        if (anth_send_text_delta(client_fd, block_index, decoded) < 0) {
                            client_ok = 0; break;
                        }
                    }
                }

                // Tool call detected
                if (in_tool_call && strstr(gen_buffer, "</tool_call>")) {
                    char func_name[256] = {}, func_args[4096] = {};
                    if (parse_tool_call(gen_buffer, func_name, sizeof(func_name),
                                        func_args, sizeof(func_args))) {
                        // Close text block, open tool_use block
                        anth_send_content_block_stop(client_fd, block_index);
                        block_index++;

                        char tool_id[64];
                        snprintf(tool_id, sizeof(tool_id), "toolu_%d", ++tool_call_count);
                        anth_send_content_block_start_tool(client_fd, block_index, tool_id, func_name);
                        anth_send_tool_input_delta(client_fd, block_index, func_args);
                        anth_send_content_block_stop(client_fd, block_index);

                        fprintf(stderr, "[serve] %s tool_use: %s(%s)\n",
                                request_id, func_name, func_args);
                        stop_reason = "tool_use";
                    }
                    break;
                }

                gen_count++;
                next_token = forward(model, next_token, pos++, K);
            }

            if (client_ok) {
                if (tool_call_count == 0)
                    anth_send_content_block_stop(client_fd, block_index);
                anth_send_message_delta(client_fd, stop_reason, gen_count);
                anth_send_message_stop(client_fd);
            }

            // Save session position
            session_pos = pos;

            double gen_ms = now_ms() - t_gen;
            fprintf(stderr, "[serve] %s generated %d tokens in %.0fms (%.2f tok/s)%s pos=%d\n",
                    request_id, gen_count, gen_ms,
                    gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0.0,
                    tool_call_count > 0 ? " [tool_use]" : "", pos);

            free(reqbuf); close(client_fd); continue;
        }

        // Unknown endpoint
        http_write_str(client_fd, "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n{\"error\":\"not found\"}\n");
        free(reqbuf); close(client_fd);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    setbuf(stdout, NULL);  // unbuffered stdout for serve mode
    const char *weights_path = "model_weights.bin";
    const char *manifest_path = "model_weights.json";
    const char *vocab_path = "vocab.bin";
    const char *tokenizer_path = "tokenizer.bin";
    const char *expert_dir = "packed_experts";
    const char *prompt_text = NULL;
    int serve_port = 0;
    int max_tokens = 20;
    int K = 4;
    int timing = 0;

    static struct option long_options[] = {
        {"weights",   required_argument, 0, 'w'},
        {"manifest",  required_argument, 0, 'j'},
        {"vocab",     required_argument, 0, 'v'},
        {"tokenizer", required_argument, 0, 'T'},
        {"experts",   required_argument, 0, 'e'},
        {"prompt",    required_argument, 0, 'P'},
        {"tokens",    required_argument, 0, 't'},
        {"k",         required_argument, 0, 'k'},
        {"serve",     required_argument, 0, 'S'},
        {"timing",    no_argument,       0, 'M'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "w:j:v:T:e:P:t:k:S:Mh", long_options, NULL)) != -1) {
        switch (c) {
            case 'w': weights_path = optarg; break;
            case 'j': manifest_path = optarg; break;
            case 'v': vocab_path = optarg; break;
            case 'T': tokenizer_path = optarg; break;
            case 'e': expert_dir = optarg; break;
            case 'P': prompt_text = optarg; break;
            case 't': max_tokens = atoi(optarg); break;
            case 'k': K = atoi(optarg); break;
            case 'S': serve_port = atoi(optarg); break;
            case 'M': timing = 1; g_timing_enabled = 1; break;
            case 'h':
                printf("Usage: %s --prompt TEXT [options]\n", argv[0]);
                printf("  --weights PATH   model_weights.bin\n");
                printf("  --manifest PATH  model_weights.json\n");
                printf("  --vocab PATH     vocab.bin\n");
                printf("  --tokenizer PATH tokenizer.bin\n");
                printf("  --experts PATH   packed_experts directory\n");
                printf("  --prompt TEXT    input prompt\n");
                printf("  --tokens N       max tokens (default: 20)\n");
                printf("  --k N            active experts (default: 4)\n");
                printf("  --serve PORT     HTTP server (OpenAI-compatible API)\n");
                printf("  --timing         per-layer timing\n");
                return 0;
            default: return 1;
        }
    }

    if (!prompt_text && serve_port == 0) {
        fprintf(stderr, "Error: --prompt or --serve required\n");
        return 1;
    }
    // Initialize CUDA
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, VRAM: %zu MB, SM: %d\n",
           prop.name, prop.totalGlobalMem / (1024*1024), prop.multiProcessorCount);

    // Load weights
    WeightFile *wf = open_weights(weights_path, manifest_path);
    if (!wf) return 1;

    // Load vocab
    // (vocab.bin format: u32 num_entries, u32 max_id, then per entry: u16 len + bytes)
    FILE *vf = fopen(vocab_path, "rb");
    if (!vf) { perror(vocab_path); return 1; }
    uint32_t vocab_n, vocab_max;
    fread(&vocab_n, 4, 1, vf);
    fread(&vocab_max, 4, 1, vf);
    char **vocab_strings = (char **)calloc(vocab_n, sizeof(char *));
    for (uint32_t i = 0; i < vocab_n; i++) {
        uint16_t len;
        fread(&len, 2, 1, vf);
        if (len > 0) {
            vocab_strings[i] = (char *)malloc(len + 1);
            fread(vocab_strings[i], 1, len, vf);
            vocab_strings[i][len] = '\0';
        }
    }
    fclose(vf);
    printf("[vocab] %u tokens\n", vocab_n);

    // Load tokenizer
    bpe_tokenizer tokenizer;
    if (bpe_load(&tokenizer, tokenizer_path) < 0) {
        fprintf(stderr, "Cannot load tokenizer %s\n", tokenizer_path);
        return 1;
    }
    printf("[tokenizer] Loaded (%d vocab, %d merges)\n", tokenizer.vocab_size, tokenizer.num_merges);

    // Initialize model
    Model *model = model_init(wf, expert_dir, K);
    if (!model) return 1;

    // Serve mode
    if (serve_port > 0) {
        serve_loop(model, vocab_strings, &tokenizer, serve_port, K);
        return 0;
    }

    if (!prompt_text) { fprintf(stderr, "Error: --prompt required\n"); return 1; }

    // Tokenize prompt
    uint32_t token_ids_buf[4096];
    int num_tokens = bpe_encode(&tokenizer, prompt_text, token_ids_buf, 4096);
    if (num_tokens < 0) { fprintf(stderr, "Tokenization failed\n"); return 1; }
    printf("[prompt] \"%s\" → %d tokens:", prompt_text, num_tokens);
    for (int i = 0; i < num_tokens && i < 20; i++) printf(" %u", token_ids_buf[i]);
    if (num_tokens > 20) printf(" ...");
    printf("\n");

    printf("\n[generating] %d tokens, K=%d experts\n", max_tokens, K);
    double gen_start = now_ms();

    // Process prompt tokens (prefill)
    for (int i = 0; i < num_tokens; i++) {
        double t0 = now_ms();
        int next = forward(model, token_ids_buf[i], i, K);
        double elapsed = now_ms() - t0;
        if (timing) printf("[prefill %d/%d] token=%d, %.1f ms\n", i+1, num_tokens, token_ids_buf[i], elapsed);
        if (i == num_tokens - 1) {
            // Print first generated token
            if (vocab_strings[next]) print_token(vocab_strings[next]);
            fflush(stdout);
            // Continue generating
            int prev = next;
            for (int t = 0; t < max_tokens - 1; t++) {
                double tt0 = now_ms();
                next = forward(model, prev, num_tokens + t, K);
                double telapsed = now_ms() - tt0;
                if (vocab_strings[next]) print_token(vocab_strings[next]);
                fflush(stdout);
                if (timing) printf(" [%.1fms]", telapsed);
                prev = next;
                // Stop on EOS
                if (next == 151643 || next == 151645) break;  // <|endoftext|>, <|im_end|>
            }
        }
    }

    double gen_elapsed = now_ms() - gen_start;
    printf("\n\n[done] %.1f ms total, %.1f ms/token, %.2f tok/s\n",
           gen_elapsed, gen_elapsed / max_tokens, max_tokens / (gen_elapsed / 1000.0));

    bpe_free(&tokenizer);
    return 0;
}
