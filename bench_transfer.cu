/*
 * bench_transfer.cu — Benchmark SSD→CPU→GPU and SSD→GPU (GDS) transfer paths
 *
 * Tests the critical data paths for MoE expert streaming on NVIDIA:
 *   1. pread() SSD → CPU RAM (single, cold cache)
 *   2. cudaMemcpy CPU → GPU (PCIe transfer)
 *   3. pread() + cudaMemcpy end-to-end
 *   4. cuFileRead SSD → GPU (GPUDirect Storage)
 *   5. Parallel pread K=4 (cold cache)
 *   6. Parallel pread K=4 + cudaMemcpy (full cold pipeline)
 *   7. Parallel cuFileRead K=4 → GPU (GDS parallel)
 *   8. Warm cache: pread K=4 + cudaMemcpy (page cache hits)
 *   9. 4-bit FMA dequant matvec CUDA kernel (GPU compute benchmark)
 *
 * Build:
 *   nvcc -O2 -o bench_transfer bench_transfer.cu -lcufile -lpthread
 *
 * Run:
 *   ./bench_transfer [--file /path/to/testfile] [--size 7077888] [--iters 50]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <pthread.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <cufile.h>

// Expert layout matching Flash-MoE Qwen3.5-397B
#define EXPERT_SIZE     7077888   // bytes per expert (from packed layout)
#define K_EXPERTS       4         // active experts per layer
#define NUM_LAYERS      60
#define DEFAULT_ITERS   50
#define ALIGN           4096      // O_DIRECT alignment

// Expert projection dimensions
#define HIDDEN_DIM      4096
#define MoE_INTERMEDIATE 1024
#define GROUP_SIZE      64

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

static void drop_caches(void) {
    (void)system("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1");
}

static void create_test_file(const char *path, size_t total_size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("create_test_file"); exit(1); }
    size_t chunk = 1024 * 1024;
    char *buf = (char *)malloc(chunk);
    for (size_t i = 0; i < chunk; i++) buf[i] = (char)(i & 0xFF);
    size_t written = 0;
    while (written < total_size) {
        size_t w = (total_size - written < chunk) ? total_size - written : chunk;
        ssize_t r = write(fd, buf, w);
        if (r < 0) { perror("write"); exit(1); }
        written += r;
    }
    close(fd);
    free(buf);
    drop_caches();
}

// ============================================================================
// BFloat16 helpers (host-side)
// ============================================================================

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &tmp, sizeof(f));
    return f;
}

// ============================================================================
// CUDA Kernel: 4-bit FMA dequant matvec (port of Metal v3 kernel)
// ============================================================================
//
// Quantization format (MLX affine 4-bit, group_size=64):
//   - Weights: uint32, each holding 8 x 4-bit values
//   - Per-group scale and bias in bfloat16
//   - Dequant: value = nibble * scale + bias
//
// FMA optimization: (nibble * scale + bias) * x = fma(nibble, scale*x, bias*x)
//   Pre-compute scale*x and bias*x per element, then use __fmaf_rn.
//
// Thread layout: blockDim.x = 32 (one warp per row), blockDim.y = ROWS_PER_BLOCK
//   Each warp processes one output row. Lane k handles packed columns k, k+32, k+64...
//   Warp shuffle reduction to sum across lanes.

#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float device_bf16_to_f32(uint16_t bf16) {
    return __uint_as_float((uint32_t)bf16 << 16);
}

__global__ void dequant_matvec_4bit_fma(
    const uint32_t* __restrict__ W_packed,   // [out_dim, in_dim/8]
    const uint16_t* __restrict__ scales,     // [out_dim, num_groups] bf16
    const uint16_t* __restrict__ biases,     // [out_dim, num_groups] bf16
    const float*    __restrict__ x,          // [in_dim]
    float*          __restrict__ out,        // [out_dim]
    uint32_t out_dim,
    uint32_t in_dim
) {
    // Shared memory cache for input vector
    extern __shared__ float x_shared[];

    const uint32_t lane = threadIdx.x;            // 0..31 (warp lane)
    const uint32_t warp_id = threadIdx.y;         // 0..ROWS_PER_BLOCK-1
    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    const uint32_t packed_cols = in_dim / 8;
    const uint32_t num_groups = in_dim / GROUP_SIZE;
    const uint32_t packed_per_group = GROUP_SIZE / 8;  // 8

    // Cooperative load of input vector into shared memory
    const uint32_t total_threads = blockDim.x * blockDim.y;
    const uint32_t tid = warp_id * 32 + lane;
    for (uint32_t i = tid; i < in_dim; i += total_threads) {
        x_shared[i] = x[i];
    }
    __syncthreads();

    if (row >= out_dim) return;

    const uint32_t* w_row = W_packed + row * packed_cols;
    const uint16_t* s_row = scales + row * num_groups;
    const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane handles columns: lane, lane+32, lane+64, ...
    // Adjacent lanes read adjacent uint32 words → coalesced
    for (uint32_t col = lane; col < packed_cols; col += 32) {
        uint32_t g = col / packed_per_group;
        float scale = device_bf16_to_f32(s_row[g]);
        float bias  = device_bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint32_t x_base = col * 8;

        // FMA optimization: (nibble * scale + bias) * x = fma(nibble, scale*x, bias*x)
        float sx0 = scale * x_shared[x_base + 0]; float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1]; float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2]; float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3]; float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4]; float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5]; float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6]; float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7]; float bx7 = bias * x_shared[x_base + 7];

        acc += __fmaf_rn((float)((packed >>  0) & 0xF), sx0, bx0);
        acc += __fmaf_rn((float)((packed >>  4) & 0xF), sx1, bx1);
        acc += __fmaf_rn((float)((packed >>  8) & 0xF), sx2, bx2);
        acc += __fmaf_rn((float)((packed >> 12) & 0xF), sx3, bx3);
        acc += __fmaf_rn((float)((packed >> 16) & 0xF), sx4, bx4);
        acc += __fmaf_rn((float)((packed >> 20) & 0xF), sx5, bx5);
        acc += __fmaf_rn((float)((packed >> 24) & 0xF), sx6, bx6);
        acc += __fmaf_rn((float)((packed >> 28) & 0xF), sx7, bx7);
    }

    // Warp reduction (sum across 32 lanes)
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        out[row] = acc;
    }
}

// ============================================================================
// Transfer benchmarks
// ============================================================================

typedef struct {
    int fd;
    size_t size;
    off_t offset;
    void *buf;
} PreadArg;

static void *pread_thread(void *arg) {
    PreadArg *a = (PreadArg *)arg;
    (void)pread(a->fd, a->buf, a->size, a->offset);
    return NULL;
}

// Test 1: pread SSD → CPU
static double bench_pread_cpu(int fd, size_t size, int iters) {
    void *buf;
    (void)posix_memalign(&buf, ALIGN, size);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        off_t offset = (off_t)((i % 128) * size);
        pread(fd, buf, size, offset);
    }
    double elapsed = now_ms() - t0;
    free(buf);
    return elapsed / iters;
}

// Test 2: cudaMemcpy CPU → GPU
static double bench_cudamemcpy(size_t size, int iters) {
    void *h_buf, *d_buf;
    CHECK_CUDA(cudaMallocHost(&h_buf, size));
    CHECK_CUDA(cudaMalloc(&d_buf, size));
    memset(h_buf, 0xAB, size);
    CHECK_CUDA(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice)); // warmup
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_ms() - t0;
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
    return elapsed / iters;
}

// Test 3: pread + cudaMemcpy
static double bench_pread_then_cuda(int fd, size_t size, int iters) {
    void *h_buf, *d_buf;
    CHECK_CUDA(cudaMallocHost(&h_buf, size));
    CHECK_CUDA(cudaMalloc(&d_buf, size));
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        pread(fd, h_buf, size, (off_t)((i % 128) * size));
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_ms() - t0;
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
    return elapsed / iters;
}

// Test 4: GDS cuFileRead (single)
static double bench_gds(const char *path, size_t size, int iters) {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "  cuFileDriverOpen failed: %d\n", status.err);
        return -1.0;
    }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { cuFileDriverClose(); return -1.0; }

    CUfileDescr_t desc = {};
    desc.handle.fd = fd;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t handle;
    status = cuFileHandleRegister(&handle, &desc);
    if (status.err != CU_FILE_SUCCESS) { close(fd); cuFileDriverClose(); return -1.0; }

    void *d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, size));
    cuFileBufRegister(d_buf, size, 0);

    ssize_t ret = cuFileRead(handle, d_buf, size, 0, 0); // warmup
    if (ret < 0) {
        fprintf(stderr, "  cuFileRead failed: %zd\n", ret);
        cuFileBufDeregister(d_buf); CHECK_CUDA(cudaFree(d_buf));
        cuFileHandleDeregister(handle); close(fd); cuFileDriverClose();
        return -1.0;
    }

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        off_t offset = (off_t)(((i % 128) * size) / ALIGN * ALIGN);
        cuFileRead(handle, d_buf, size, offset, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_ms() - t0;

    cuFileBufDeregister(d_buf); CHECK_CUDA(cudaFree(d_buf));
    cuFileHandleDeregister(handle); close(fd); cuFileDriverClose();
    return elapsed / iters;
}

// Test 5: Parallel pread K=4
static double bench_parallel_pread(int fd, size_t size, int k, int iters) {
    void *bufs[8];
    pthread_t threads[8];
    PreadArg args[8];
    for (int i = 0; i < k; i++) (void)posix_memalign(&bufs[i], ALIGN, size);
    double t0 = now_ms();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < k; i++) {
            args[i] = (PreadArg){fd, size, (off_t)(((iter*k+i) % 128) * size), bufs[i]};
            pthread_create(&threads[i], NULL, pread_thread, &args[i]);
        }
        for (int i = 0; i < k; i++) pthread_join(threads[i], NULL);
    }
    double elapsed = now_ms() - t0;
    for (int i = 0; i < k; i++) free(bufs[i]);
    return elapsed / iters;
}

// Test 6: Parallel pread K=4 + cudaMemcpy
static double bench_parallel_pread_cuda(int fd, size_t size, int k, int iters) {
    void *h_bufs[8], *d_bufs[8];
    cudaStream_t streams[8];
    for (int i = 0; i < k; i++) {
        CHECK_CUDA(cudaMallocHost(&h_bufs[i], size));
        CHECK_CUDA(cudaMalloc(&d_bufs[i], size));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    pthread_t threads[8];
    PreadArg args[8];

    double t0 = now_ms();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < k; i++) {
            args[i] = (PreadArg){fd, size, (off_t)(((iter*k+i) % 128) * size), h_bufs[i]};
            pthread_create(&threads[i], NULL, pread_thread, &args[i]);
        }
        for (int i = 0; i < k; i++) pthread_join(threads[i], NULL);
        for (int i = 0; i < k; i++)
            CHECK_CUDA(cudaMemcpyAsync(d_bufs[i], h_bufs[i], size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    double elapsed = now_ms() - t0;
    for (int i = 0; i < k; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_bufs[i]));
        CHECK_CUDA(cudaFreeHost(h_bufs[i]));
    }
    return elapsed / iters;
}

// Test 7: Parallel GDS cuFileRead K=4
typedef struct {
    CUfileHandle_t handle;
    void *d_buf;
    size_t size;
    off_t offset;
} GDSReadArg;

static void *gds_read_thread(void *arg) {
    GDSReadArg *a = (GDSReadArg *)arg;
    cuFileRead(a->handle, a->d_buf, a->size, a->offset, 0);
    return NULL;
}

static double bench_parallel_gds(const char *path, size_t size, int k, int iters) {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) return -1.0;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { cuFileDriverClose(); return -1.0; }

    CUfileDescr_t desc = {};
    desc.handle.fd = fd;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t handle;
    status = cuFileHandleRegister(&handle, &desc);
    if (status.err != CU_FILE_SUCCESS) { close(fd); cuFileDriverClose(); return -1.0; }

    void *d_bufs[8];
    for (int i = 0; i < k; i++) {
        CHECK_CUDA(cudaMalloc(&d_bufs[i], size));
        cuFileBufRegister(d_bufs[i], size, 0);
    }

    // Warmup
    cuFileRead(handle, d_bufs[0], size, 0, 0);

    pthread_t threads[8];
    GDSReadArg args[8];

    double t0 = now_ms();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < k; i++) {
            off_t offset = (off_t)((((iter*k+i) % 128) * size) / ALIGN * ALIGN);
            args[i] = (GDSReadArg){handle, d_bufs[i], size, offset};
            pthread_create(&threads[i], NULL, gds_read_thread, &args[i]);
        }
        for (int i = 0; i < k; i++) pthread_join(threads[i], NULL);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_ms() - t0;

    for (int i = 0; i < k; i++) {
        cuFileBufDeregister(d_bufs[i]);
        CHECK_CUDA(cudaFree(d_bufs[i]));
    }
    cuFileHandleDeregister(handle); close(fd); cuFileDriverClose();
    return elapsed / iters;
}

// Test 8: Warm cache — read same data repeatedly (page cache hits)
static double bench_warm_pread_cuda(int fd, size_t size, int k, int iters) {
    void *h_bufs[8], *d_bufs[8];
    cudaStream_t streams[8];
    for (int i = 0; i < k; i++) {
        CHECK_CUDA(cudaMallocHost(&h_bufs[i], size));
        CHECK_CUDA(cudaMalloc(&d_bufs[i], size));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    pthread_t threads[8];
    PreadArg args[8];

    // Warm up: read the same offsets to populate page cache
    for (int i = 0; i < k; i++) {
        off_t offset = (off_t)(i * size);
        pread(fd, h_bufs[i], size, offset);
    }

    double t0 = now_ms();
    for (int iter = 0; iter < iters; iter++) {
        // Always read from same offsets → page cache hit
        for (int i = 0; i < k; i++) {
            args[i] = (PreadArg){fd, size, (off_t)(i * size), h_bufs[i]};
            pthread_create(&threads[i], NULL, pread_thread, &args[i]);
        }
        for (int i = 0; i < k; i++) pthread_join(threads[i], NULL);
        for (int i = 0; i < k; i++)
            CHECK_CUDA(cudaMemcpyAsync(d_bufs[i], h_bufs[i], size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    double elapsed = now_ms() - t0;
    for (int i = 0; i < k; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_bufs[i]));
        CHECK_CUDA(cudaFreeHost(h_bufs[i]));
    }
    return elapsed / iters;
}

// ============================================================================
// Test 9: CUDA dequant_matvec kernel benchmark
// ============================================================================
static void bench_dequant_kernel(int iters) {
    // Simulate gate_proj: [1024, 4096] (out=1024, in=4096)
    uint32_t out_dim = MoE_INTERMEDIATE;  // 1024
    uint32_t in_dim = HIDDEN_DIM;          // 4096
    uint32_t packed_cols = in_dim / 8;     // 512
    uint32_t num_groups = in_dim / GROUP_SIZE; // 64

    // Allocate and fill with synthetic quantized data
    size_t w_size = out_dim * packed_cols * sizeof(uint32_t);
    size_t s_size = out_dim * num_groups * sizeof(uint16_t);

    uint32_t *h_W = (uint32_t *)malloc(w_size);
    uint16_t *h_s = (uint16_t *)malloc(s_size);
    uint16_t *h_b = (uint16_t *)malloc(s_size);
    float *h_x = (float *)malloc(in_dim * sizeof(float));
    float *h_out = (float *)malloc(out_dim * sizeof(float));

    // Fill with realistic-ish data
    srand(42);
    for (uint32_t i = 0; i < out_dim * packed_cols; i++)
        h_W[i] = rand();
    for (uint32_t i = 0; i < out_dim * num_groups; i++) {
        // bf16 encoding of small floats
        float sv = 0.01f * (rand() % 100) / 100.0f;
        float bv = -0.5f + (rand() % 100) / 100.0f;
        uint32_t tmp;
        memcpy(&tmp, &sv, 4); h_s[i] = (uint16_t)(tmp >> 16);
        memcpy(&tmp, &bv, 4); h_b[i] = (uint16_t)(tmp >> 16);
    }
    for (uint32_t i = 0; i < in_dim; i++)
        h_x[i] = -1.0f + 2.0f * (rand() % 10000) / 10000.0f;

    // Device allocations
    uint32_t *d_W; uint16_t *d_s, *d_b; float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_W, w_size));
    CHECK_CUDA(cudaMalloc(&d_s, s_size));
    CHECK_CUDA(cudaMalloc(&d_b, s_size));
    CHECK_CUDA(cudaMalloc(&d_x, in_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, out_dim * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_W, h_W, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_s, h_s, s_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, s_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, in_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch config: 32 threads/warp × ROWS_PER_BLOCK warps per block
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    size_t shared_mem = in_dim * sizeof(float);

    // Warmup
    dequant_matvec_4bit_fma<<<grid, block, shared_mem>>>(d_W, d_s, d_b, d_x, d_out, out_dim, in_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify: compute on CPU and compare
    float *cpu_out = (float *)calloc(out_dim, sizeof(float));
    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < packed_cols; col++) {
            uint32_t g = col / (GROUP_SIZE / 8);
            float scale = bf16_to_f32(h_s[row * num_groups + g]);
            float bias  = bf16_to_f32(h_b[row * num_groups + g]);
            uint32_t packed = h_W[row * packed_cols + col];
            for (int n = 0; n < 8; n++) {
                float nibble = (float)((packed >> (n * 4)) & 0xF);
                acc += (nibble * scale + bias) * h_x[col * 8 + n];
            }
        }
        cpu_out[row] = acc;
    }
    CHECK_CUDA(cudaMemcpy(h_out, d_out, out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (uint32_t i = 0; i < out_dim; i++) {
        float err = fabsf(h_out[i] - cpu_out[i]);
        float rel = (fabsf(cpu_out[i]) > 1e-6f) ? err / fabsf(cpu_out[i]) : err;
        if (rel > max_err) max_err = rel;
    }
    printf("  Verification: max relative error = %.2e %s\n", max_err,
           max_err < 1e-3 ? "(OK)" : "(WARNING: large error)");
    free(cpu_out);

    // Benchmark gate_proj [1024, 4096]
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        dequant_matvec_4bit_fma<<<grid, block, shared_mem>>>(d_W, d_s, d_b, d_x, d_out, out_dim, in_dim);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gate_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gate_ms, start, stop));
    gate_ms /= iters;

    printf("  gate_proj [%d, %d]: %.3f ms\n", out_dim, in_dim, gate_ms);

    // Benchmark down_proj [4096, 1024]
    uint32_t down_out = HIDDEN_DIM;       // 4096
    uint32_t down_in = MoE_INTERMEDIATE;  // 1024
    uint32_t down_packed = down_in / 8;   // 128
    uint32_t down_groups = down_in / GROUP_SIZE; // 16
    size_t dw_size = down_out * down_packed * sizeof(uint32_t);
    size_t ds_size = down_out * down_groups * sizeof(uint16_t);

    uint32_t *d_dW; uint16_t *d_ds, *d_db; float *d_dx, *d_dout;
    CHECK_CUDA(cudaMalloc(&d_dW, dw_size));
    CHECK_CUDA(cudaMalloc(&d_ds, ds_size));
    CHECK_CUDA(cudaMalloc(&d_db, ds_size));
    CHECK_CUDA(cudaMalloc(&d_dx, down_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dout, down_out * sizeof(float)));

    dim3 down_grid((down_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    size_t down_shared = down_in * sizeof(float);

    // Warmup
    dequant_matvec_4bit_fma<<<down_grid, block, down_shared>>>(d_dW, d_ds, d_db, d_dx, d_dout, down_out, down_in);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        dequant_matvec_4bit_fma<<<down_grid, block, down_shared>>>(d_dW, d_ds, d_db, d_dx, d_dout, down_out, down_in);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float down_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&down_ms, start, stop));
    down_ms /= iters;

    printf("  down_proj [%d, %d]: %.3f ms\n", down_out, down_in, down_ms);

    // Full expert forward: gate + up + SwiGLU + down (K=1)
    float expert_ms = gate_ms * 2 + down_ms;  // gate + up (same dims) + down
    printf("  Full expert (gate+up+down): %.3f ms\n", expert_ms);
    printf("  K=%d experts: %.3f ms\n", K_EXPERTS, expert_ms * K_EXPERTS);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_W)); CHECK_CUDA(cudaFree(d_s)); CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_dW)); CHECK_CUDA(cudaFree(d_ds)); CHECK_CUDA(cudaFree(d_db));
    CHECK_CUDA(cudaFree(d_dx)); CHECK_CUDA(cudaFree(d_dout));
    free(h_W); free(h_s); free(h_b); free(h_x); free(h_out);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
    const char *testfile = "/tmp/flash_moe_bench.dat";
    size_t expert_size = EXPERT_SIZE;
    int iters = DEFAULT_ITERS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i+1 < argc) testfile = argv[++i];
        else if (strcmp(argv[i], "--size") == 0 && i+1 < argc) expert_size = atol(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i+1 < argc) iters = atoi(argv[++i]);
    }

    printf("=== Flash-MoE NVIDIA Transfer + Compute Benchmark ===\n");
    printf("Expert size: %.2f MB, K=%d, %d iterations\n\n",
           expert_size / (1024.0 * 1024.0), K_EXPERTS, iters);

    size_t file_size = expert_size * 256;
    printf("Creating test file (%zu MB)...\n", file_size / (1024*1024));
    create_test_file(testfile, file_size);

    int fd = open(testfile, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, VRAM: %zu MB, SM: %d, Mem BW: %.0f GB/s\n\n",
           prop.name, prop.totalGlobalMem / (1024*1024),
           prop.multiProcessorCount,
           prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);

    double ms;

    // Test 1
    printf("Test 1: pread() SSD → CPU (1x %.1fMB, cold)\n", expert_size/1048576.0);
    drop_caches();
    ms = bench_pread_cpu(fd, expert_size, iters);
    printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (expert_size / (ms/1000.0)) / 1e9);

    // Test 2
    printf("Test 2: cudaMemcpy CPU → GPU (pinned, %.1fMB)\n", expert_size/1048576.0);
    ms = bench_cudamemcpy(expert_size, iters);
    printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (expert_size / (ms/1000.0)) / 1e9);

    // Test 3
    printf("Test 3: pread + cudaMemcpy (cold, %.1fMB)\n", expert_size/1048576.0);
    drop_caches();
    ms = bench_pread_then_cuda(fd, expert_size, iters);
    printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (expert_size / (ms/1000.0)) / 1e9);

    // Test 4
    printf("Test 4: GDS cuFileRead (1x %.1fMB, cold)\n", expert_size/1048576.0);
    close(fd);
    drop_caches();
    ms = bench_gds(testfile, expert_size, iters);
    if (ms > 0) printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (expert_size / (ms/1000.0)) / 1e9);
    else printf("  (not available)\n\n");

    fd = open(testfile, O_RDONLY);

    // Test 5
    printf("Test 5: Parallel pread K=%d → CPU (cold)\n", K_EXPERTS);
    drop_caches();
    ms = bench_parallel_pread(fd, expert_size, K_EXPERTS, iters);
    printf("  %.2f ms  (%.2f GB/s agg)\n\n", ms, (K_EXPERTS*expert_size / (ms/1000.0)) / 1e9);

    // Test 6
    printf("Test 6: Parallel pread K=%d + cudaMemcpy (cold, full pipeline)\n", K_EXPERTS);
    drop_caches();
    ms = bench_parallel_pread_cuda(fd, expert_size, K_EXPERTS, iters);
    double cold_pipeline_ms = ms;
    printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (K_EXPERTS*expert_size / (ms/1000.0)) / 1e9);

    // Test 7
    printf("Test 7: Parallel GDS cuFileRead K=%d → GPU (cold)\n", K_EXPERTS);
    close(fd);
    drop_caches();
    ms = bench_parallel_gds(testfile, expert_size, K_EXPERTS, iters);
    double gds_pipeline_ms = ms;
    if (ms > 0) printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (K_EXPERTS*expert_size / (ms/1000.0)) / 1e9);
    else printf("  (not available)\n\n");

    fd = open(testfile, O_RDONLY);

    // Test 8
    printf("Test 8: Warm cache pread K=%d + cudaMemcpy (page cache hits)\n", K_EXPERTS);
    ms = bench_warm_pread_cuda(fd, expert_size, K_EXPERTS, iters);
    double warm_pipeline_ms = ms;
    printf("  %.2f ms  (%.2f GB/s)\n\n", ms, (K_EXPERTS*expert_size / (ms/1000.0)) / 1e9);

    // Test 9
    printf("Test 9: CUDA 4-bit FMA dequant matvec kernel\n");
    bench_dequant_kernel(iters * 10);  // more iters for GPU kernel
    printf("\n");

    // Summary
    printf("========================================\n");
    printf("=== Per-Token Estimate (60 layers) ===\n");
    printf("========================================\n");
    printf("  Cold cache (pread+cuda):   %.1f ms/layer → %.0f ms/tok → %.2f tok/s\n",
           cold_pipeline_ms, cold_pipeline_ms * NUM_LAYERS, 1000.0 / (cold_pipeline_ms * NUM_LAYERS));
    if (gds_pipeline_ms > 0)
        printf("  Cold cache (GDS K=%d):      %.1f ms/layer → %.0f ms/tok → %.2f tok/s\n",
               K_EXPERTS, gds_pipeline_ms, gds_pipeline_ms * NUM_LAYERS, 1000.0 / (gds_pipeline_ms * NUM_LAYERS));
    printf("  Warm cache (page cache):   %.1f ms/layer → %.0f ms/tok → %.2f tok/s\n",
           warm_pipeline_ms, warm_pipeline_ms * NUM_LAYERS, 1000.0 / (warm_pipeline_ms * NUM_LAYERS));

    // Mixed estimate: ~30% cache hit rate with 64GB RAM
    double mixed_ms = 0.7 * cold_pipeline_ms + 0.3 * warm_pipeline_ms;
    printf("  Mixed (~30%% cache hit):    %.1f ms/layer → %.0f ms/tok → %.2f tok/s\n",
           mixed_ms, mixed_ms * NUM_LAYERS, 1000.0 / (mixed_ms * NUM_LAYERS));

    double best_cold = (gds_pipeline_ms > 0 && gds_pipeline_ms < cold_pipeline_ms) ? gds_pipeline_ms : cold_pipeline_ms;
    double mixed_best = 0.7 * best_cold + 0.3 * warm_pipeline_ms;
    printf("  Mixed (best cold path):    %.1f ms/layer → %.0f ms/tok → %.2f tok/s\n",
           mixed_best, mixed_best * NUM_LAYERS, 1000.0 / (mixed_best * NUM_LAYERS));

    printf("\n  (GPU compute adds ~0.1-0.3ms/layer on RTX 4090 — see Test 9)\n");

    close(fd);
    unlink(testfile);
    return 0;
}
