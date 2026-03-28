/*
 * bench_q4k.cu — Compare MLX affine 4-bit vs GGML Q4_K kernel performance
 *
 * Build:
 *   nvcc -O2 -o bench_q4k bench_q4k.cu -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Need ROWS_PER_BLOCK and GROUP_SIZE before including kernels
#define GROUP_SIZE 64
#include "kernels.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static inline float bf16_to_f32_h(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f; memcpy(&f, &tmp, sizeof(f)); return f;
}

int main() {
    // Test dimensions matching expert projections
    struct { int out_dim; int in_dim; const char *name; } tests[] = {
        {1024, 4096, "gate/up_proj"},
        {4096, 1024, "down_proj"},
        {512, 4096, "routing"},
        {248320, 4096, "lm_head"},
    };

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, %d SMs, %.0f GB/s\n", prop.name,
           prop.multiProcessorCount,
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);

    int iters = 200;

    for (int t = 0; t < 4; t++) {
        int out_dim = tests[t].out_dim;
        int in_dim = tests[t].in_dim;
        printf("\n=== %s [%d, %d] ===\n", tests[t].name, out_dim, in_dim);

        // Allocate input vector
        float *h_x = (float *)malloc(in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) h_x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        float *d_x, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, in_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, out_dim * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, in_dim * sizeof(float), cudaMemcpyHostToDevice));

        // ---- MLX format ----
        uint32_t packed_cols = in_dim / 8;
        uint32_t num_groups = in_dim / GROUP_SIZE;
        size_t mlx_w_sz = out_dim * packed_cols * sizeof(uint32_t);
        size_t mlx_s_sz = out_dim * num_groups * sizeof(uint16_t);

        uint32_t *h_W = (uint32_t *)malloc(mlx_w_sz);
        uint16_t *h_S = (uint16_t *)malloc(mlx_s_sz);
        uint16_t *h_B = (uint16_t *)malloc(mlx_s_sz);
        for (size_t i = 0; i < out_dim * packed_cols; i++) h_W[i] = rand();
        for (size_t i = 0; i < out_dim * num_groups; i++) {
            float sv = 0.01f; uint32_t tmp; memcpy(&tmp, &sv, 4); h_S[i] = tmp >> 16;
            float bv = -0.5f; memcpy(&tmp, &bv, 4); h_B[i] = tmp >> 16;
        }

        uint32_t *d_W; uint16_t *d_S, *d_B;
        CHECK_CUDA(cudaMalloc(&d_W, mlx_w_sz));
        CHECK_CUDA(cudaMalloc(&d_S, mlx_s_sz));
        CHECK_CUDA(cudaMalloc(&d_B, mlx_s_sz));
        CHECK_CUDA(cudaMemcpy(d_W, h_W, mlx_w_sz, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_S, h_S, mlx_s_sz, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, mlx_s_sz, cudaMemcpyHostToDevice));

        // Warmup
        launch_dequant_matvec(d_W, d_S, d_B, d_x, d_out, out_dim, in_dim);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            launch_dequant_matvec(d_W, d_S, d_B, d_x, d_out, out_dim, in_dim);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float mlx_ms;
        CHECK_CUDA(cudaEventElapsedTime(&mlx_ms, start, stop));
        mlx_ms /= iters;

        size_t mlx_total = mlx_w_sz + mlx_s_sz * 2;
        printf("  MLX affine 4-bit: %.3f ms (%.1f GB/s, data=%.1f MB)\n",
               mlx_ms, mlx_total / (mlx_ms / 1000.0) / 1e9, mlx_total / 1e6);

        // ---- Q4_K format ----
        uint32_t blocks_per_row = in_dim / QK_K;
        size_t q4k_row_sz = blocks_per_row * Q4_K_BLOCK_SIZE;
        size_t q4k_total = (size_t)out_dim * q4k_row_sz;

        uint8_t *h_Q4K = (uint8_t *)malloc(q4k_total);
        // Fill with synthetic Q4_K data
        for (size_t row = 0; row < (size_t)out_dim; row++) {
            for (uint32_t bi = 0; bi < blocks_per_row; bi++) {
                uint8_t *block = h_Q4K + row * q4k_row_sz + bi * Q4_K_BLOCK_SIZE;
                __half d_val = __float2half(0.01f);
                __half dmin_val = __float2half(0.005f);
                memcpy(block, &d_val, 2);
                memcpy(block + 2, &dmin_val, 2);
                for (int i = 0; i < 12; i++) block[4 + i] = rand() & 0x3F;
                for (int i = 0; i < 128; i++) block[16 + i] = rand();
            }
        }

        uint8_t *d_Q4K;
        CHECK_CUDA(cudaMalloc(&d_Q4K, q4k_total));
        CHECK_CUDA(cudaMemcpy(d_Q4K, h_Q4K, q4k_total, cudaMemcpyHostToDevice));

        // Warmup
        launch_dequant_matvec_q4k(d_Q4K, d_x, d_out, out_dim, in_dim);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            launch_dequant_matvec_q4k(d_Q4K, d_x, d_out, out_dim, in_dim);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float q4k_ms;
        CHECK_CUDA(cudaEventElapsedTime(&q4k_ms, start, stop));
        q4k_ms /= iters;

        printf("  GGML Q4_K:        %.3f ms (%.1f GB/s, data=%.1f MB)\n",
               q4k_ms, q4k_total / (q4k_ms / 1000.0) / 1e9, q4k_total / 1e6);

        float ratio = q4k_ms / mlx_ms;
        printf("  Ratio Q4_K/MLX:   %.2fx %s\n", ratio,
               ratio < 1.05 ? "(comparable)" : ratio < 1.2 ? "(slightly slower)" : "(slower)");

        CHECK_CUDA(cudaFree(d_W)); CHECK_CUDA(cudaFree(d_S)); CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_Q4K));
        free(h_W); free(h_S); free(h_B); free(h_Q4K);
        CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_out));
        free(h_x);
    }

    return 0;
}
