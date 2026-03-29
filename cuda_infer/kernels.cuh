/*
 * kernels.cuh — CUDA compute kernels for Flash-MoE inference
 *
 * Port of shaders.metal for NVIDIA GPUs (RTX 4090 target).
 * All kernels operate on the same quantization format:
 *   - 4-bit affine quantization, group_size=64
 *   - Weights: uint32 holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *
 * Kernel list:
 *   1. dequant_matvec_4bit_fma  — FMA-optimized 4-bit dequant matvec
 *   2. swiglu_fused             — SiLU(gate) * up
 *   3. rms_norm                 — Fused sum-of-squares + normalize (single kernel)
 *   4. rms_norm_bf16            — RMS norm with bf16 weights
 *   5. residual_add             — a + b
 *   6. attn_scores              — Q @ K^T (batched over heads)
 *   7. attn_softmax             — Softmax over seq_len per head
 *   8. attn_values              — scores @ V (batched over heads)
 *   9. sigmoid_gate             — x *= sigmoid(gate)
 *  10. gated_delta_net_step     — GatedDeltaNet recurrence
 *  11. conv1d_step              — Depthwise conv1d (kernel=4) + SiLU
 *  12. rms_norm_qk              — Per-head RMS norm for Q and K
 *  13. compute_decay_beta       — GatedDeltaNet decay and beta gate
 *  14. gated_rms_norm           — RMS norm with SiLU gate and bf16 weights
 *  15. moe_combine_residual     — Weighted expert sum + shared expert + residual
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define GROUP_SIZE 64

// ============================================================================
// BFloat16 helper
// ============================================================================

__device__ __forceinline__ float bf16_to_f32(uint16_t bf16) {
    return __uint_as_float((uint32_t)bf16 << 16);
}

// ============================================================================
// Warp reduction helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================================
// 1. 4-bit FMA dequant matvec
// ============================================================================
// blockDim = (32, ROWS_PER_BLOCK), gridDim = ceil(out_dim / ROWS_PER_BLOCK)
// Shared memory: in_dim * sizeof(float)

#define ROWS_PER_BLOCK 8

__global__ void dequant_matvec_4bit_fma(
    const uint32_t* __restrict__ W_packed,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    const float*    __restrict__ x,
    float*          __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    const uint32_t packed_cols = in_dim / 8;
    const uint32_t num_groups = in_dim / GROUP_SIZE;
    const uint32_t packed_per_group = GROUP_SIZE / 8;

    // Cooperative load
    const uint32_t tid = warp_id * 32 + lane;
    const uint32_t total = blockDim.x * blockDim.y;
    for (uint32_t i = tid; i < in_dim; i += total)
        x_shared[i] = x[i];
    __syncthreads();

    if (row >= out_dim) return;

    const uint32_t* w_row = W_packed + row * packed_cols;
    const uint16_t* s_row = scales + row * num_groups;
    const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    for (uint32_t col = lane; col < packed_cols; col += 32) {
        uint32_t g = col / packed_per_group;
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];
        uint32_t xb = col * 8;

        float sx0 = scale * x_shared[xb+0]; float bx0 = bias * x_shared[xb+0];
        float sx1 = scale * x_shared[xb+1]; float bx1 = bias * x_shared[xb+1];
        float sx2 = scale * x_shared[xb+2]; float bx2 = bias * x_shared[xb+2];
        float sx3 = scale * x_shared[xb+3]; float bx3 = bias * x_shared[xb+3];
        float sx4 = scale * x_shared[xb+4]; float bx4 = bias * x_shared[xb+4];
        float sx5 = scale * x_shared[xb+5]; float bx5 = bias * x_shared[xb+5];
        float sx6 = scale * x_shared[xb+6]; float bx6 = bias * x_shared[xb+6];
        float sx7 = scale * x_shared[xb+7]; float bx7 = bias * x_shared[xb+7];

        acc += __fmaf_rn((float)((packed >>  0) & 0xF), sx0, bx0);
        acc += __fmaf_rn((float)((packed >>  4) & 0xF), sx1, bx1);
        acc += __fmaf_rn((float)((packed >>  8) & 0xF), sx2, bx2);
        acc += __fmaf_rn((float)((packed >> 12) & 0xF), sx3, bx3);
        acc += __fmaf_rn((float)((packed >> 16) & 0xF), sx4, bx4);
        acc += __fmaf_rn((float)((packed >> 20) & 0xF), sx5, bx5);
        acc += __fmaf_rn((float)((packed >> 24) & 0xF), sx6, bx6);
        acc += __fmaf_rn((float)((packed >> 28) & 0xF), sx7, bx7);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

// ============================================================================
// 1b. 4-bit FMA dequant matvec with uint4 vectorized loads
// ============================================================================
// Loads 4 × uint32 (128 bits) per instruction instead of 1 × uint32.
// Each uint4 = 32 nibbles = 32 input elements processed per load.
// Reduces instruction count and improves memory throughput.

__global__ void dequant_matvec_4bit_fma_vec4(
    const uint32_t* __restrict__ W_packed,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    const float*    __restrict__ x,
    float*          __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    // All divisions by powers of 2 → shifts. No runtime division.
    const uint32_t packed_cols = in_dim >> 3;        // in_dim / 8
    const uint32_t num_groups = in_dim >> 6;         // in_dim / 64 (GROUP_SIZE=64)
    const uint32_t vec4_cols = packed_cols >> 2;      // packed_cols / 4

    // Cooperative load — all threads must participate before barrier
    const uint32_t tid = warp_id * 32 + lane;
    for (uint32_t i = tid; i < in_dim; i += (32 * ROWS_PER_BLOCK))
        x_shared[i] = x[i];
    __syncthreads();

    if (row >= out_dim) return;

    const uint4* w_row_v = (const uint4*)(W_packed + row * packed_cols);
    const uint16_t* s_row = scales + row * num_groups;
    const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint32_t vi = lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = __ldg(w_row_v + vi);  // read-through L1 cache
        uint32_t base_col = vi << 2;
        uint32_t x_base = base_col << 3;      // base_col * 8

        #pragma unroll
        for (uint32_t w = 0; w < 4; w++) {
            uint32_t packed = ((const uint32_t*)&packed4)[w];
            // group index: (base_col + w) / 8 = (base_col + w) >> 3
            // (packed_per_group = GROUP_SIZE/8 = 8)
            uint32_t g = (base_col + w) >> 3;
            float scale = bf16_to_f32(__ldg(s_row + g));
            float bias  = bf16_to_f32(__ldg(b_row + g));
            uint32_t xb = x_base + (w << 3);  // w * 8

            float sx0 = scale * x_shared[xb+0]; float bx0 = bias * x_shared[xb+0];
            float sx1 = scale * x_shared[xb+1]; float bx1 = bias * x_shared[xb+1];
            float sx2 = scale * x_shared[xb+2]; float bx2 = bias * x_shared[xb+2];
            float sx3 = scale * x_shared[xb+3]; float bx3 = bias * x_shared[xb+3];
            float sx4 = scale * x_shared[xb+4]; float bx4 = bias * x_shared[xb+4];
            float sx5 = scale * x_shared[xb+5]; float bx5 = bias * x_shared[xb+5];
            float sx6 = scale * x_shared[xb+6]; float bx6 = bias * x_shared[xb+6];
            float sx7 = scale * x_shared[xb+7]; float bx7 = bias * x_shared[xb+7];

            acc += __fmaf_rn((float)((packed >>  0) & 0xF), sx0, bx0);
            acc += __fmaf_rn((float)((packed >>  4) & 0xF), sx1, bx1);
            acc += __fmaf_rn((float)((packed >>  8) & 0xF), sx2, bx2);
            acc += __fmaf_rn((float)((packed >> 12) & 0xF), sx3, bx3);
            acc += __fmaf_rn((float)((packed >> 16) & 0xF), sx4, bx4);
            acc += __fmaf_rn((float)((packed >> 20) & 0xF), sx5, bx5);
            acc += __fmaf_rn((float)((packed >> 24) & 0xF), sx6, bx6);
            acc += __fmaf_rn((float)((packed >> 28) & 0xF), sx7, bx7);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

// ============================================================================
// 1c. GGML Q4_K dequant matvec — native GGUF format support
// ============================================================================
// Q4_K super-block: 256 elements, 144 bytes (4.5 bits/weight)
//   d (fp16): super-block scale for quantized scales
//   dmin (fp16): super-block scale for quantized mins
//   scales[12]: packed 6-bit per-sub-block scales and mins (8 sub-blocks of 32)
//   qs[128]: 4-bit quantized values (256 values, 2 per byte)
// Dequant: value = d * sub_scale * nibble - dmin * sub_min

#define QK_K 256
#define Q4_K_BLOCK_SIZE 144  // 2+2+12+128 bytes

// Unpack 6-bit scale and min from the packed scales array
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t *q,
                                                  uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// Q4_K matvec with Q8_K input quantization (matches llama.cpp vec_dot_q4_K_q8_K)
// Shared memory layout: [in_dim floats for x] + [in_dim int8 for q8] + [in_dim/256 floats for q8_scales]
__global__ void dequant_matvec_q4k(
    const uint8_t* __restrict__ W_q4k,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    int8_t *q8_shared = (int8_t *)(x_shared + in_dim);
    float *q8_scales = (float *)(q8_shared + in_dim);

    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    const uint32_t blocks_per_row = in_dim >> 8;  // in_dim / 256
    const uint32_t tid = warp_id * 32 + lane;

    // Load x into shared memory
    for (uint32_t i = tid; i < in_dim; i += (32 * ROWS_PER_BLOCK))
        x_shared[i] = x[i];
    __syncthreads();

    // Quantize x to Q8_K: per-256-element blocks
    for (uint32_t bi = tid; bi < blocks_per_row; bi += (32 * ROWS_PER_BLOCK)) {
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < 256; i++) {
            float v = fabsf(x_shared[bi * 256 + i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;
        q8_scales[bi] = scale;
        for (uint32_t i = 0; i < 256; i++) {
            float v = x_shared[bi * 256 + i] * inv_scale;
            // lroundf: round ties away from zero (matching ggml quantize_row_q8_K)
            int q = (int)(v >= 0 ? (v + 0.5f) : (v - 0.5f));
            q8_shared[bi * 256 + i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
    __syncthreads();

    if (row >= out_dim) return;

    const uint8_t *row_data = W_q4k + (size_t)row * blocks_per_row * Q4_K_BLOCK_SIZE;
    float acc = 0.0f;

    for (uint32_t bi = lane; bi < blocks_per_row; bi += 32) {
        const uint8_t *block = row_data + bi * Q4_K_BLOCK_SIZE;
        const uint8_t *qs = block + 16;
        const uint8_t *sc_bytes = block + 4;
        float d_w = __half2float(__ldg((const __half *)(block)));
        float dmin_w = __half2float(__ldg((const __half *)(block + 2)));
        float d_q8 = q8_scales[bi];
        const int8_t *q8 = q8_shared + bi * 256;

        // Dequant Q4_K into aux8 (matching vec_dot_q4_K_q8_K_generic)
        // 4 pairs of 64: low nibbles then high nibbles
        int8_t aux8[256];
        int a = 0, q4_off = 0;
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 32; l++) aux8[a + l] = (int8_t)(__ldg(&qs[q4_off + l]) & 0xF);
            a += 32;
            for (int l = 0; l < 32; l++) aux8[a + l] = (int8_t)(__ldg(&qs[q4_off + l]) >> 4);
            a += 32;
            q4_off += 32;
        }

        // Unpack scales using utmp/kmask approach (matching vec_dot)
        uint32_t utmp[4];
        utmp[0] = __ldg((const uint32_t *)(sc_bytes + 0));
        utmp[1] = __ldg((const uint32_t *)(sc_bytes + 4));
        utmp[2] = __ldg((const uint32_t *)(sc_bytes + 8));
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0fu) | (((utmp[1] >> 6) & 0x03030303u) << 4);
        uint32_t uaux = utmp[1] & 0x3f3f3f3fu;
        utmp[1] = (utmp[2] & 0x0f0f0f0fu) | (((utmp[0] >> 6) & 0x03030303u) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f3f3fu;
        const uint8_t *scales = (const uint8_t *)&utmp[0];
        const uint8_t *mins = (const uint8_t *)&utmp[2];

        // Compute bsums (sum of q8 per 16-element group)
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            int bsum = 0;
            for (int l = 0; l < 16; l++) bsum += q8[j * 16 + l];
            sumi += bsum * (int)mins[j / 2];
        }

        // Integer dot product with 8 accumulators (matching vec_dot_qX_K_q8_K exactly)
        int32_t aux32[8] = {0,0,0,0,0,0,0,0};
        int ai = 0, is_idx = 0;
        for (int j = 0; j < 8; j++) {
            int32_t sc = scales[is_idx++];
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
        }
        float d_combined = d_w * d_q8;
        for (int l = 0; l < 8; l++) acc += d_combined * (float)aux32[l];
        acc -= dmin_w * d_q8 * (float)sumi;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

// Launch helper for Q4_K format
static inline void launch_dequant_matvec_q4k(
    const uint8_t* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim, cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    // Shared memory: float x[in_dim] + int8 q8[in_dim] + float scales[in_dim/256]
    size_t smem = in_dim * sizeof(float) + in_dim * sizeof(int8_t) + (in_dim / 256) * sizeof(float);
    dequant_matvec_q4k<<<grid, block, smem, stream>>>(W, x, out, out_dim, in_dim);
}

// ============================================================================
// Q5_K dequantized matrix-vector multiply (GGML format)
// ============================================================================
// Q5_K block (176 bytes per 256 elements, 5.5 bits/weight):
//   d (fp16): super-block scale
//   dmin (fp16): super-block min scale
//   scales[12]: packed 6-bit per-sub-block scales and mins (same as Q4_K)
//   qh[32]: high bits (5th bit) for each of 256 elements, packed 8 per byte
//   qs[128]: low 4 bits, packed 2 per byte (same as Q4_K)
// Dequant: value = d * sub_scale * q5_value - dmin * sub_min
//   where q5_value = (low_nibble) | (high_bit << 4), range 0-31

#define Q5_K_BLOCK_SIZE 176  // 2+2+12+32+128 bytes

// Q5_K matvec with Q8_K input quantization (matches llama.cpp vec_dot_q5_K_q8_K)
// Shared memory layout: [in_dim floats for x] + [in_dim int8 for q8] + [in_dim/256 floats for q8_scales]
__global__ void dequant_matvec_q5k(
    const uint8_t* __restrict__ W_q5k,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    // Layout shared mem: [in_dim float] then [in_dim int8] then [in_dim/256 float scales]
    int8_t *q8_shared = (int8_t *)(x_shared + in_dim);
    float *q8_scales = (float *)(q8_shared + in_dim);

    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    const uint32_t blocks_per_row = in_dim >> 8;
    const uint32_t tid = warp_id * 32 + lane;

    // Load x into shared memory
    for (uint32_t i = tid; i < in_dim; i += (32 * ROWS_PER_BLOCK))
        x_shared[i] = x[i];
    __syncthreads();

    // Quantize x to Q8_K: per-256-element blocks
    for (uint32_t bi = tid; bi < blocks_per_row; bi += (32 * ROWS_PER_BLOCK)) {
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < 256; i++) {
            float v = fabsf(x_shared[bi * 256 + i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;
        q8_scales[bi] = scale;
        for (uint32_t i = 0; i < 256; i++) {
            float v = x_shared[bi * 256 + i] * inv_scale;
            // lroundf: round ties away from zero (matching ggml quantize_row_q8_K)
            int q = (int)(v >= 0 ? (v + 0.5f) : (v - 0.5f));
            q8_shared[bi * 256 + i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
    __syncthreads();

    if (row >= out_dim) return;

    const uint8_t *row_data = W_q5k + (size_t)row * blocks_per_row * Q5_K_BLOCK_SIZE;
    float acc = 0.0f;

    for (uint32_t bi = lane; bi < blocks_per_row; bi += 32) {
        const uint8_t *block = row_data + bi * Q5_K_BLOCK_SIZE;
        const uint8_t *qs = block + 48;
        const uint8_t *qh = block + 16;
        const uint8_t *sc_bytes = block + 4;
        float d_w = __half2float(__ldg((const __half *)(block)));
        float dmin_w = __half2float(__ldg((const __half *)(block + 2)));
        float d_q8 = q8_scales[bi];
        const int8_t *q8 = q8_shared + bi * 256;

        // Dequant Q5K into aux8 (matching vec_dot_q5_K_q8_K_generic)
        int8_t aux8[256];
        uint8_t m = 1;
        int a = 0, q4_off = 0;
        for (int j = 0; j < 4; j++) {
            for (int l = 0; l < 32; l++) aux8[a + l] = qs[q4_off + l] & 0xF;
            for (int l = 0; l < 32; l++) aux8[a + l] += (__ldg(&qh[l]) & m) ? 16 : 0;
            a += 32; m <<= 1;
            for (int l = 0; l < 32; l++) aux8[a + l] = qs[q4_off + l] >> 4;
            for (int l = 0; l < 32; l++) aux8[a + l] += (__ldg(&qh[l]) & m) ? 16 : 0;
            a += 32; m <<= 1;
            q4_off += 32;
        }

        // Unpack scales using utmp/kmask approach (matching vec_dot)
        uint32_t utmp[4];
        utmp[0] = __ldg((const uint32_t *)(sc_bytes + 0));
        utmp[1] = __ldg((const uint32_t *)(sc_bytes + 4));
        utmp[2] = __ldg((const uint32_t *)(sc_bytes + 8));
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0fu) | (((utmp[1] >> 6) & 0x03030303u) << 4);
        uint32_t uaux = utmp[1] & 0x3f3f3f3fu;
        utmp[1] = (utmp[2] & 0x0f0f0f0fu) | (((utmp[0] >> 6) & 0x03030303u) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f3f3fu;
        const uint8_t *scales = (const uint8_t *)&utmp[0];
        const uint8_t *mins = (const uint8_t *)&utmp[2];

        // Compute bsums (sum of q8 per 16-element group)
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            int bsum = 0;
            for (int l = 0; l < 16; l++) bsum += q8[j * 16 + l];
            sumi += bsum * (int)mins[j / 2];
        }

        // Integer dot product with 8 accumulators (matching vec_dot_qX_K_q8_K exactly)
        int32_t aux32[8] = {0,0,0,0,0,0,0,0};
        int ai = 0, is_idx = 0;
        for (int j = 0; j < 8; j++) {
            int32_t sc = scales[is_idx++];
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += sc * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
        }
        float d_combined = d_w * d_q8;
        for (int l = 0; l < 8; l++) acc += d_combined * (float)aux32[l];
        acc -= dmin_w * d_q8 * (float)sumi;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

static inline void launch_dequant_matvec_q5k(
    const uint8_t* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim, cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    // Shared memory: float x[in_dim] + int8 q8[in_dim] + float scales[in_dim/256]
    size_t smem = in_dim * sizeof(float) + in_dim * sizeof(int8_t) + (in_dim / 256) * sizeof(float);
    dequant_matvec_q5k<<<grid, block, smem, stream>>>(W, x, out, out_dim, in_dim);
}

// ============================================================================
// 2. SwiGLU: out[i] = SiLU(gate[i]) * up[i]
// ============================================================================

__global__ void swiglu_fused(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    uint32_t dim
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ============================================================================
// 3. RMS Norm (fused: sum_sq + normalize in one kernel)
// ============================================================================
// blockDim = 256 (or 1024), gridDim = 1
// Shared memory: 32 * sizeof(float)

__global__ void rms_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,  // f32 weights
    float* __restrict__ out,
    uint32_t dim,
    float eps
) {
    // Use float for warp reduction but accumulate per-thread in double
    __shared__ float shared[32];
    double acc_d = 0.0;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        double v = (double)x[i];
        acc_d += v * v;
    }
    // Convert to float for warp reduction (small loss, but per-thread sum is precise)
    float acc = (float)acc_d;
    acc = warp_reduce_sum(acc);
    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    if (lane == 0) shared[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        acc = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) shared[0] = acc;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = x[i] * rms * weight[i];
}

// ============================================================================
// 4. RMS Norm with bf16 weights
// ============================================================================

__global__ void rms_norm_bf16(
    const float* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ out,
    uint32_t dim,
    float eps
) {
    __shared__ float shared[32];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        acc += x[i] * x[i];

    acc = warp_reduce_sum(acc);
    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    if (lane == 0) shared[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) shared[0] = acc;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = x[i] * rms * bf16_to_f32(weight[i]);
}

// ============================================================================
// 5. Residual add: out = a + b
// ============================================================================

__global__ void residual_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    uint32_t dim
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = a[i] + b[i];
}

// ============================================================================
// 6. Attention scores: Q @ K^T (batched over heads)
// ============================================================================
// Grid: (seq_len * num_heads), Block: 256
// GQA: heads_per_kv query heads share one KV head

__global__ void attn_scores(
    const float* __restrict__ Q,        // [num_heads, head_dim]
    const float* __restrict__ K_cache,  // [max_seq, kv_dim]
    float* __restrict__ scores,         // [num_heads, seq_stride]
    uint32_t head_dim, uint32_t kv_dim, uint32_t seq_len,
    uint32_t seq_stride, float scale, uint32_t heads_per_kv, uint32_t num_seq_tgs
) {
    __shared__ float shared[32];
    uint32_t pos = blockIdx.x % num_seq_tgs;
    uint32_t h = blockIdx.x / num_seq_tgs;
    if (pos >= seq_len) return;

    uint32_t kv_h = h / heads_per_kv;
    const float* qh = Q + h * head_dim;
    const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
        acc += qh[d] * kp[d];

    acc = warp_reduce_sum(acc);
    uint32_t wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) shared[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) scores[h * seq_stride + pos] = acc * scale;
    }
}

// ============================================================================
// 7. Softmax over seq_len per head
// ============================================================================

__global__ void attn_softmax(
    float* __restrict__ scores,  // [num_heads, seq_stride]
    uint32_t seq_len, uint32_t seq_stride
) {
    __shared__ float s_max, s_sum;
    float* s = scores + blockIdx.x * seq_stride;

    // Max
    float local_max = -1e30f;
    for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_max = fmaxf(local_max, s[i]);

    __shared__ float shared[32];
    float wmax = warp_reduce_max(local_max);
    uint32_t wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) shared[wid] = wmax;
    __syncthreads();
    if (wid == 0) {
        wmax = (lane < (blockDim.x + 31) / 32) ? shared[lane] : -1e30f;
        wmax = warp_reduce_max(wmax);
        if (lane == 0) s_max = wmax;
    }
    __syncthreads();

    // Exp + sum
    float local_sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float v = expf(s[i] - s_max);
        s[i] = v;
        local_sum += v;
    }
    float wsum = warp_reduce_sum(local_sum);
    if (lane == 0) shared[wid] = wsum;
    __syncthreads();
    if (wid == 0) {
        wsum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        wsum = warp_reduce_sum(wsum);
        if (lane == 0) s_sum = wsum;
    }
    __syncthreads();

    // Normalize
    float inv = 1.0f / s_sum;
    for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x)
        s[i] *= inv;
}

// ============================================================================
// 8. Attention values: scores @ V
// ============================================================================

__global__ void attn_values(
    const float* __restrict__ scores,  // [num_heads, seq_stride]
    const float* __restrict__ V_cache, // [max_seq, kv_dim]
    float* __restrict__ out,           // [num_heads, head_dim]
    uint32_t head_dim, uint32_t kv_dim, uint32_t seq_len,
    uint32_t seq_stride, uint32_t heads_per_kv
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t d = tid % head_dim;
    uint32_t h = tid / head_dim;
    uint32_t kv_h = h / heads_per_kv;
    const float* sc = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint32_t p = 0; p < seq_len; p++)
        acc += sc[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    out[h * head_dim + d] = acc;
}

// ============================================================================
// 9. Sigmoid gate: x *= sigmoid(gate)
// ============================================================================

__global__ void sigmoid_gate(
    float* __restrict__ x_out,
    const float* __restrict__ gate,
    uint32_t dim
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        x_out[i] *= g;
    }
}

// ============================================================================
// 10. GatedDeltaNet step (single token, all heads)
// ============================================================================
// Grid: 64 (v-heads), Block: 128 (value_dim)

__global__ void gated_delta_net_step(
    float* __restrict__ state,         // [64 * 128 * 128]
    const float* __restrict__ q,       // [2048]
    const float* __restrict__ k,       // [2048]
    const float* __restrict__ v,       // [8192]
    const float* __restrict__ g_decay, // [64]
    const float* __restrict__ beta_gate, // [64]
    float* __restrict__ output,        // [8192]
    uint32_t k_heads_per_v             // = 4
) {
    uint32_t head_id = blockIdx.x;
    uint32_t vi = threadIdx.x;
    // Key head mapping: modulo (matching llama.cpp iq1 = iv1 % neq1)
    uint32_t num_k_heads = blockDim.x;  // HACK: pass via unused param
    // Actually, num_k_heads = num_v_heads / k_heads_per_v
    uint32_t n_kh = gridDim.x / k_heads_per_v;
    uint32_t kh = head_id % n_kh;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint32_t state_base = head_id * 128 * 128 + vi * 128;
    uint32_t k_base = kh * 128;
    uint32_t v_base = head_id * 128;

    // Decay + memory read
    float kv_mem = 0.0f;
    for (uint32_t ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Delta update
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint32_t ki = 0; ki < 128; ki++)
        state[state_base + ki] += k[k_base + ki] * delta;

    // Output
    float out_val = 0.0f;
    for (uint32_t ki = 0; ki < 128; ki++)
        out_val += state[state_base + ki] * q[k_base + ki];
    output[v_base + vi] = out_val;

    // Debug: print head 1, vi=0 intermediate values
    if (head_id == 1 && vi == 0) {
        printf("[gdn] h1v0: g=%.6f beta=%.6f kv_mem=%.8f v=%.8f delta=%.8f out=%.8f\n",
               g, beta, kv_mem, v[v_base], delta, out_val);
        printf("[gdn] h1v0: q[0:3]=%.6f %.6f %.6f  k[0:3]=%.6f %.6f %.6f\n",
               q[k_base], q[k_base+1], q[k_base+2], k[k_base], k[k_base+1], k[k_base+2]);
        printf("[gdn] h1v0: state[0:3]=%.8f %.8f %.8f\n",
               state[state_base], state[state_base+1], state[state_base+2]);
    }
}

// ============================================================================
// 11. Conv1d step (kernel=4, depthwise, with SiLU)
// ============================================================================

__global__ void conv1d_step(
    float* __restrict__ conv_state,     // [3 * conv_dim]
    const float* __restrict__ input,    // [conv_dim]
    const uint16_t* __restrict__ weights, // [conv_dim * 4] bf16
    float* __restrict__ output,         // [conv_dim]
    uint32_t conv_dim
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) return;

    uint32_t wb = idx * 4;
    float acc = conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[wb + 0])
              + conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[wb + 1])
              + conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[wb + 2]);
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[wb + 3]);

    output[idx] = acc / (1.0f + expf(-acc));  // SiLU

    // Shift history
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}

// ============================================================================
// 12a. Per-head L2 norm for Q and K (GGUF models — matches llama.cpp ggml_l2_norm)
// ============================================================================

__global__ void l2_norm_qk(
    float* __restrict__ q,
    float* __restrict__ k,
    uint32_t key_dim
) {
    uint32_t head = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t base = head * key_dim;

    __shared__ float buf[128];
    __shared__ float norm_scale;

    // Q L2 norm: q = q / max(||q||, eps)  (matching ggml_compute_forward_l2_norm_f32)
    float qv = (tid < key_dim) ? q[base + tid] : 0.0f;
    buf[tid] = qv * qv;
    __syncthreads();
    if (tid == 0) {
        double s = 0.0;  // double precision matching ggml_float
        for (uint32_t i = 0; i < key_dim; i++) s += (double)buf[i];
        float sq = sqrtf((float)s);
        norm_scale = 1.0f / fmaxf(sq, 1e-6f);
    }
    __syncthreads();
    if (tid < key_dim)
        q[base + tid] = qv * norm_scale;

    // K L2 norm
    float kv = (tid < key_dim) ? k[base + tid] : 0.0f;
    buf[tid] = kv * kv;
    __syncthreads();
    if (tid == 0) {
        double s = 0.0;
        for (uint32_t i = 0; i < key_dim; i++) s += (double)buf[i];
        float sq = sqrtf((float)s);
        norm_scale = 1.0f / fmaxf(sq, 1e-6f);
    }
    __syncthreads();
    if (tid < key_dim)
        k[base + tid] = kv * norm_scale;
}

// ============================================================================
// 12b. Per-head RMS norm for Q and K (MLX models — original 397B behavior)
// ============================================================================

__global__ void rms_norm_qk(
    float* __restrict__ q,
    float* __restrict__ k,
    uint32_t key_dim, float inv_scale
) {
    uint32_t head = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t base = head * key_dim;

    // Q norm
    __shared__ float q_sum;
    __shared__ float buf[128];
    float qv = (tid < key_dim) ? q[base + tid] : 0.0f;
    buf[tid] = qv * qv;
    __syncthreads();
    if (tid == 0) { float s = 0; for (uint32_t i = 0; i < key_dim; i++) s += buf[i]; q_sum = s; }
    __syncthreads();
    if (tid < key_dim)
        q[base + tid] = qv * rsqrtf(q_sum / (float)key_dim + 1e-6f) * inv_scale * inv_scale;

    // K norm
    __shared__ float k_sum;
    float kv = (tid < key_dim) ? k[base + tid] : 0.0f;
    buf[tid] = kv * kv;
    __syncthreads();
    if (tid == 0) { float s = 0; for (uint32_t i = 0; i < key_dim; i++) s += buf[i]; k_sum = s; }
    __syncthreads();
    if (tid < key_dim)
        k[base + tid] = kv * rsqrtf(k_sum / (float)key_dim + 1e-6f) * inv_scale;
}

// ============================================================================
// 13. Compute decay and beta gate for GatedDeltaNet
// ============================================================================

// MLX version: A_log stores log(A), compute exp(A_log) first, dt_bias is bf16
__global__ void compute_decay_beta(
    const float* __restrict__ alpha_out,
    const float* __restrict__ beta_out,
    const float* __restrict__ A_log,
    const uint16_t* __restrict__ dt_bias,
    float* __restrict__ g_decay,
    float* __restrict__ beta_gate
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = expf(A_log[idx]);
    float sp = logf(1.0f + expf(a_val + dt_b));
    g_decay[idx] = expf(-A_val * sp);
    beta_gate[idx] = 1.0f / (1.0f + expf(-beta_out[idx]));
}

// GGUF version: ssm_a and dt_bias are both F32
__global__ void compute_decay_beta_gguf(
    const float* __restrict__ alpha_out,
    const float* __restrict__ beta_out,
    const float* __restrict__ ssm_a,       // negative values, used directly
    const float* __restrict__ dt_bias,     // F32 (not converted to bf16)
    float* __restrict__ g_decay,
    float* __restrict__ beta_gate
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a_val = alpha_out[idx];
    float dt_b = dt_bias[idx];
    float sp = logf(1.0f + expf(a_val + dt_b));  // softplus(alpha + dt_bias)
    g_decay[idx] = expf(ssm_a[idx] * sp);         // exp(ssm_a * softplus) — ssm_a is negative
    beta_gate[idx] = 1.0f / (1.0f + expf(-beta_out[idx]));  // sigmoid(beta)
}

// ============================================================================
// 14. Gated RMS norm: rms_norm(values) * SiLU(z) * weight
// ============================================================================

__global__ void gated_rms_norm(
    const float* __restrict__ values,
    const float* __restrict__ z,
    const uint16_t* __restrict__ weight,
    float* __restrict__ output,
    uint32_t value_dim, float eps
) {
    uint32_t head = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t base = head * value_dim;

    __shared__ float buf[128];
    float val = (tid < value_dim) ? values[base + tid] : 0.0f;
    buf[tid] = val * val;
    __syncthreads();
    if (tid == 0) { float s = 0; for (uint32_t i = 0; i < value_dim; i++) s += buf[i]; buf[0] = s; }
    __syncthreads();
    float inv_rms = rsqrtf(buf[0] / (float)value_dim + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zv = z[base + tid];
        float gate = zv / (1.0f + expf(-zv));  // SiLU
        output[base + tid] = normed * gate * bf16_to_f32(weight[tid]);
    }
}

// ============================================================================
// 15. MoE combine + residual + shared expert gate
// ============================================================================
// out[i] = h_mid[i] + sum_k(weight[k] * expert_out[k*dim+i]) + sigmoid(shared_gate) * shared[i]

__global__ void moe_combine_residual(
    const float* __restrict__ h_mid,
    const float* __restrict__ shared_out,
    float* __restrict__ hidden_out,
    const float* __restrict__ expert_outs,  // [K * dim] concatenated
    const float* __restrict__ weights,      // [K]
    float shared_gate_score,
    uint32_t dim, uint32_t K
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float sg = 1.0f / (1.0f + expf(-shared_gate_score));
    float moe = 0.0f;
    for (uint32_t k = 0; k < K; k++)
        moe += weights[k] * expert_outs[k * dim + i];

    hidden_out[i] = h_mid[i] + moe + sg * shared_out[i];
}

// ============================================================================
// Launch helpers
// ============================================================================

static inline void launch_dequant_matvec(
    const uint32_t* W, const uint16_t* scales, const uint16_t* biases,
    const float* x, float* out, uint32_t out_dim, uint32_t in_dim,
    cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    size_t smem = in_dim * sizeof(float);
    dequant_matvec_4bit_fma_vec4<<<grid, block, smem, stream>>>(W, scales, biases, x, out, out_dim, in_dim);
}

// ============================================================================
// Q6_K dequantized matrix-vector multiply (GGML format)
// ============================================================================
// Q6_K block (210 bytes per 256 elements, 6.5625 bits/weight):
//   ql[128]: low 4 bits of each 6-bit value (2 per byte, low nibble first)
//   qh[64]:  high 2 bits of each 6-bit value (4 values per byte)
//   scales[16]: int8 per-sub-block scales (16 sub-blocks of 16 elements)
//   d (fp16): super-block scale
// Dequant: value = d * scale * (q6_value - 32)  where q6_value is 0-63 (6-bit unsigned)

#define Q6_K_BLOCK_SIZE 210  // 128+64+16+2 bytes

// Q6_K matvec with Q8_K input quantization (matches llama.cpp vec_dot_q6_K_q8_K)
// Shared memory layout: [in_dim floats for x] + [in_dim int8 for q8] + [in_dim/256 floats for q8_scales]
__global__ void dequant_matvec_q6k(
    const uint8_t* __restrict__ W_q6k,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    int8_t *q8_shared = (int8_t *)(x_shared + in_dim);
    float *q8_scales = (float *)(q8_shared + in_dim);

    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    const uint32_t blocks_per_row = in_dim >> 8;  // in_dim / 256
    const uint32_t tid = warp_id * 32 + lane;

    // Load x into shared memory
    for (uint32_t i = tid; i < in_dim; i += (32 * ROWS_PER_BLOCK))
        x_shared[i] = x[i];
    __syncthreads();

    // Quantize x to Q8_K: per-256-element blocks
    for (uint32_t bi = tid; bi < blocks_per_row; bi += (32 * ROWS_PER_BLOCK)) {
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < 256; i++) {
            float v = fabsf(x_shared[bi * 256 + i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 0) ? (1.0f / scale) : 0.0f;
        q8_scales[bi] = scale;
        for (uint32_t i = 0; i < 256; i++) {
            float v = x_shared[bi * 256 + i] * inv_scale;
            // lroundf: round ties away from zero (matching ggml quantize_row_q8_K)
            int q = (int)(v >= 0 ? (v + 0.5f) : (v - 0.5f));
            q8_shared[bi * 256 + i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
    __syncthreads();

    if (row >= out_dim) return;

    const uint8_t *row_data = W_q6k + (size_t)row * blocks_per_row * Q6_K_BLOCK_SIZE;
    float acc = 0.0f;

    for (uint32_t bi = lane; bi < blocks_per_row; bi += 32) {
        const uint8_t *block = row_data + bi * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        float d_w = __half2float(__ldg((const __half *)(block + 208)));
        float d_q8 = q8_scales[bi];
        const int8_t *q8 = q8_shared + bi * 256;

        // Dequant Q6_K into aux8 (matching vec_dot_q6_K_q8_K_generic)
        // 2 halves of 128 elements: ql advances by 64, qh by 32 per half
        // qh shifts are always 0,2,4,6 (reset each half)
        int8_t aux8[256];
        {
            const uint8_t *q4 = ql;
            const uint8_t *qh_p = qh;
            int8_t *a = aux8;
            for (int j = 0; j < 256; j += 128) {
                for (int l = 0; l < 32; l++) {
                    uint8_t qlv0  = __ldg(&q4[l]);
                    uint8_t qlv32 = __ldg(&q4[l + 32]);
                    uint8_t qhv   = __ldg(&qh_p[l]);
                    a[l +  0] = (int8_t)(((qlv0  & 0xF) | (((qhv >> 0) & 3) << 4)) - 32);
                    a[l + 32] = (int8_t)(((qlv32 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32);
                    a[l + 64] = (int8_t)(((qlv0  >> 4)  | (((qhv >> 4) & 3) << 4)) - 32);
                    a[l + 96] = (int8_t)(((qlv32 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32);
                }
                a += 128;
                q4 += 64;
                qh_p += 32;
            }
        }

        // Integer dot product with 8 accumulators (matching vec_dot_q6_K_q8_K)
        int32_t aux32[8] = {0,0,0,0,0,0,0,0};
        int ai = 0, is_q6 = 0;
        for (int j = 0; j < 16; j++) {
            int32_t scale = sc[is_q6++];
            for (int l = 0; l < 8; l++) aux32[l] += scale * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
            for (int l = 0; l < 8; l++) aux32[l] += scale * (int)q8[ai+l] * (int)aux8[ai+l]; ai += 8;
        }
        float d_combined = d_w * d_q8;
        for (int l = 0; l < 8; l++) acc += d_combined * (float)aux32[l];
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

static inline void launch_dequant_matvec_q6k(
    const uint8_t* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim, cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    // Shared memory: float x[in_dim] + int8 q8[in_dim] + float scales[in_dim/256]
    size_t smem = in_dim * sizeof(float) + in_dim * sizeof(int8_t) + (in_dim / 256) * sizeof(float);
    dequant_matvec_q6k<<<grid, block, smem, stream>>>(W, x, out, out_dim, in_dim);
}

// ============================================================================
// F32 matrix-vector multiply (for unquantized weights like norms, small tensors)
// ============================================================================

__global__ void matvec_f32(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float*       __restrict__ out,
    uint32_t out_dim,
    uint32_t in_dim
) {
    extern __shared__ float x_shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    const uint32_t tid = warp_id * 32 + lane;
    for (uint32_t i = tid; i < in_dim; i += (32 * ROWS_PER_BLOCK))
        x_shared[i] = x[i];
    __syncthreads();

    if (row >= out_dim) return;

    const float *row_data = W + (size_t)row * in_dim;
    float acc = 0.0f;

    for (uint32_t i = lane; i < in_dim; i += 32)
        acc += row_data[i] * x_shared[i];

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

static inline void launch_matvec_f32(
    const float* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim, cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    size_t smem = in_dim * sizeof(float);
    matvec_f32<<<grid, block, smem, stream>>>(W, x, out, out_dim, in_dim);
}

// ============================================================================
// Format-aware matvec dispatch for GGUF support
// ============================================================================
// GGML quant type IDs (from ggml-common.h)
#define GGUF_TYPE_F32   0
#define GGUF_TYPE_F16   1
#define GGUF_TYPE_Q4_K  12
#define GGUF_TYPE_Q5_K  13
#define GGUF_TYPE_Q6_K  14

static inline void launch_dequant_matvec_gguf(
    const void* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim, int gguf_type, cudaStream_t stream = 0
) {
    switch (gguf_type) {
        case GGUF_TYPE_Q4_K:
            launch_dequant_matvec_q4k((const uint8_t*)W, x, out, out_dim, in_dim, stream);
            break;
        case GGUF_TYPE_Q5_K:
            launch_dequant_matvec_q5k((const uint8_t*)W, x, out, out_dim, in_dim, stream);
            break;
        case GGUF_TYPE_Q6_K:
            launch_dequant_matvec_q6k((const uint8_t*)W, x, out, out_dim, in_dim, stream);
            break;
        case GGUF_TYPE_F32:
            launch_matvec_f32((const float*)W, x, out, out_dim, in_dim, stream);
            break;
        default:
            // Unsupported type — fall back to Q4_K as best guess
            launch_dequant_matvec_q4k((const uint8_t*)W, x, out, out_dim, in_dim, stream);
            break;
    }
}

static inline void launch_swiglu(const float* gate, const float* up, float* out, uint32_t dim, cudaStream_t s = 0) {
    swiglu_fused<<<(dim+255)/256, 256, 0, s>>>(gate, up, out, dim);
}

static inline void launch_rms_norm_bf16(const float* x, const uint16_t* w, float* out, uint32_t dim, float eps, cudaStream_t s = 0) {
    rms_norm_bf16<<<1, 256, 0, s>>>(x, w, out, dim, eps);
}

// In-place vector scale: x[i] *= scale
__global__ void vec_scale(float* __restrict__ x, float scale, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

static inline void launch_residual_add(const float* a, const float* b, float* out, uint32_t dim, cudaStream_t s = 0) {
    residual_add<<<(dim+255)/256, 256, 0, s>>>(a, b, out, dim);
}
