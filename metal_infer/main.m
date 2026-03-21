/*
 * main.m — Pure C/Metal 4-bit dequantized MoE expert computation engine
 *
 * Standalone benchmark of 4-bit dequant matvec via Metal compute shaders.
 * Loads expert weights from packed binary files, runs the Metal shader, verifies output.
 *
 * Supports single-layer mode (original) and full 60-layer forward pass (--full).
 *
 * This is the foundation for a full llama.cpp-style inference engine for
 * Qwen3.5-397B-A17B running on Apple Silicon with SSD-streamed expert weights.
 *
 * Build: make
 * Run:   ./metal_infer [--layer N] [--expert E] [--benchmark]
 *        ./metal_infer --model <path> --full --k 4 --benchmark
 *
 * What it does:
 *   1. Creates Metal device, command queue, loads compute shaders
 *   2. Opens packed expert files (one per layer, or all 60 for --full)
 *   3. pread()s expert weights into Metal shared buffers (8 pthreads parallel I/O)
 *   4. Runs the full MoE expert forward pass per layer:
 *      - gate_proj matvec (4096 -> 1024)
 *      - up_proj matvec (4096 -> 1024)
 *      - SwiGLU activation
 *      - down_proj matvec (1024 -> 4096)
 *      - weighted_sum to combine K experts
 *   5. In --full mode: pipelines I/O with compute via double buffering
 *   6. Reports timing and throughput
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dirent.h>

// ============================================================================
// Constants matching the Qwen3.5-397B packed expert layout
// ============================================================================

#define HIDDEN_DIM       4096
#define INTERMEDIATE_DIM 1024
#define GROUP_SIZE       64
#define BITS             4
#define NUM_EXPERTS      512
#define NUM_LAYERS       60
#define MAX_ACTIVE_EXPERTS 64

// Expert component sizes (from layout.json)
#define GATE_W_OFFSET    0
#define GATE_W_SIZE      2097152   // [1024, 512] uint32
#define GATE_S_OFFSET    2097152
#define GATE_S_SIZE      131072    // [1024, 64] uint16 (bf16)
#define GATE_B_OFFSET    2228224
#define GATE_B_SIZE      131072

#define UP_W_OFFSET      2359296
#define UP_W_SIZE        2097152
#define UP_S_OFFSET      4456448
#define UP_S_SIZE        131072
#define UP_B_OFFSET      4587520
#define UP_B_SIZE        131072

#define DOWN_W_OFFSET    4718592
#define DOWN_W_SIZE      2097152   // [4096, 128] uint32
#define DOWN_S_OFFSET    6815744
#define DOWN_S_SIZE      131072    // [4096, 16] uint16 (bf16)
#define DOWN_B_OFFSET    6946816
#define DOWN_B_SIZE      131072

#define EXPERT_SIZE      7077888   // Total bytes per expert

// Default model path — resolved at runtime, can be overridden with --model
#define MODEL_PATH NULL

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side, for reference/verification)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ============================================================================
// CPU reference: 4-bit dequant matvec (for verification)
// ============================================================================

static void cpu_dequant_matvec_4bit(
    const uint32_t *W_packed,   // [out_dim, in_dim/8]
    const uint16_t *scales,     // [out_dim, num_groups]
    const uint16_t *biases,     // [out_dim, num_groups]
    const float *x,             // [in_dim]
    float *out,                 // [out_dim]
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size
) {
    uint32_t num_groups = in_dim / group_size;
    uint32_t packed_per_group = group_size / 8;
    uint32_t packed_cols = in_dim / 8;

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W_packed + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (uint32_t g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint32_t base_packed = g * packed_per_group;
            uint32_t base_x = g * group_size;

            for (uint32_t p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                uint32_t x_base = base_x + p * 8;

                for (uint32_t n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    float w_val = (float)nibble * scale + bias;
                    acc += w_val * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// ============================================================================
// CPU reference: SwiGLU
// ============================================================================

static void cpu_swiglu(const float *gate, const float *up, float *out, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// ============================================================================
// Metal setup and shader management
// ============================================================================

// V3 shader: 8 rows per threadgroup (256 threads, 8 SIMD groups of 32)
#define ROWS_PER_TG 8

// Max experts in fused MoE path
#define MAX_K_FUSED 16

typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> matvec_naive;
    id<MTLComputePipelineState> matvec_fast;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v4;
    id<MTLComputePipelineState> matvec_batched;
    id<MTLComputePipelineState> swiglu;
    id<MTLComputePipelineState> swiglu_vec4;
    id<MTLComputePipelineState> swiglu_batched;
    id<MTLComputePipelineState> weighted_sum;
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> fused_gate_up;
} MetalContext;

static MetalContext *metal_init(void) {
    MetalContext *ctx = calloc(1, sizeof(MetalContext));
    if (!ctx) { fprintf(stderr, "ERROR: alloc MetalContext\n"); return NULL; }

    // Get default Metal device
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device found\n");
        free(ctx);
        return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);
    printf("[metal] Unified memory: %s\n", [ctx->device hasUnifiedMemory] ? "YES" : "NO");
    printf("[metal] Max buffer size: %.0f MB\n", [ctx->device maxBufferLength] / (1024.0 * 1024.0));

    // Create command queue
    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: Failed to create command queue\n");
        free(ctx);
        return NULL;
    }

    // Load shader source and compile at runtime
    // (Metal offline compiler may not be available on all systems)
    NSError *error = nil;

    // Try loading pre-compiled metallib first, then fall back to source compilation
    NSString *execPath = [[NSBundle mainBundle] executablePath];
    NSString *execDir = [execPath stringByDeletingLastPathComponent];

    // Search for metallib
    NSArray *metallib_paths = @[
        [execDir stringByAppendingPathComponent:@"shaders.metallib"],
        @"shaders.metallib",
        @"metal_infer/shaders.metallib"
    ];
    for (NSString *libPath in metallib_paths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            ctx->library = [ctx->device newLibraryWithURL:libURL error:&error];
            if (ctx->library) {
                printf("[metal] Loaded pre-compiled shader library: %s\n", [libPath UTF8String]);
                break;
            }
        }
    }

    // Fall back: compile from source
    if (!ctx->library) {
        NSArray *source_paths = @[
            [execDir stringByAppendingPathComponent:@"shaders.metal"],
            @"shaders.metal",
            @"metal_infer/shaders.metal"
        ];
        NSString *shaderSource = nil;
        NSString *foundPath = nil;
        for (NSString *srcPath in source_paths) {
            shaderSource = [NSString stringWithContentsOfFile:srcPath
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
            if (shaderSource) {
                foundPath = srcPath;
                break;
            }
        }
        if (!shaderSource) {
            fprintf(stderr, "ERROR: Could not find shaders.metal or shaders.metallib\n");
            free(ctx);
            return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_1;

        printf("[metal] Compiling shaders from source: %s ...\n", [foundPath UTF8String]);
        double t_compile = now_ms();
        ctx->library = [ctx->device newLibraryWithSource:shaderSource
                                                 options:opts
                                                   error:&error];
        if (!ctx->library) {
            fprintf(stderr, "ERROR: Shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }
        printf("[metal] Shader compilation: %.0f ms\n", now_ms() - t_compile);
    }

    // Create pipeline states for each kernel
    // Helper block to create a pipeline from a function name
    id<MTLComputePipelineState> (^makePipeline)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "ERROR: Shader function '%s' not found\n", [name UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        NSError *pipeError = nil;
        id<MTLComputePipelineState> ps =
            [ctx->device newComputePipelineStateWithFunction:fn error:&pipeError];
        if (!ps) {
            fprintf(stderr, "ERROR: Failed to create pipeline for '%s': %s\n",
                    [name UTF8String], [[pipeError localizedDescription] UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        printf("[metal] Pipeline '%s': maxTotalThreadsPerThreadgroup=%lu\n",
               [name UTF8String], (unsigned long)[ps maxTotalThreadsPerThreadgroup]);
        return ps;
    };

    ctx->matvec_naive   = makePipeline(@"dequant_matvec_4bit");
    ctx->matvec_fast    = makePipeline(@"dequant_matvec_4bit_fast");
    ctx->matvec_v3      = makePipeline(@"dequant_matvec_4bit_v3");
    ctx->matvec_v4      = makePipeline(@"dequant_matvec_4bit_v4");
    ctx->matvec_batched = makePipeline(@"dequant_matvec_4bit_batched");
    ctx->swiglu         = makePipeline(@"swiglu_fused");
    ctx->swiglu_vec4    = makePipeline(@"swiglu_fused_vec4");
    ctx->swiglu_batched = makePipeline(@"swiglu_fused_batched");
    ctx->weighted_sum   = makePipeline(@"weighted_sum");
    ctx->rms_norm_sum   = makePipeline(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipeline(@"rms_norm_apply");
    ctx->fused_gate_up  = makePipeline(@"fused_gate_up_swiglu");

    // Required pipelines (v3 is the primary optimized path)
    if (!ctx->matvec_naive || !ctx->matvec_v3 || !ctx->swiglu ||
        !ctx->weighted_sum || !ctx->rms_norm_sum || !ctx->rms_norm_apply) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

static void metal_destroy(MetalContext *ctx) {
    if (ctx) free(ctx);
}

// ============================================================================
// Metal buffer helpers
// ============================================================================

// Create a shared-memory Metal buffer (CPU and GPU see the same memory)
static id<MTLBuffer> metal_buf_shared(MetalContext *ctx, size_t size) {
    id<MTLBuffer> buf = [ctx->device newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "ERROR: Failed to allocate Metal buffer of %zu bytes\n", size);
    }
    return buf;
}

// Create a shared buffer and fill it with pread from fd
static id<MTLBuffer> metal_buf_pread(MetalContext *ctx, int fd, size_t size, off_t offset) {
    id<MTLBuffer> buf = metal_buf_shared(ctx, size);
    if (!buf) return nil;

    ssize_t nread = pread(fd, [buf contents], size, offset);
    if (nread != (ssize_t)size) {
        fprintf(stderr, "ERROR: pread returned %zd, expected %zu (errno=%d)\n",
                nread, size, errno);
        return nil;
    }
    return buf;
}

// ============================================================================
// Run a single dequant matvec on Metal (supports buffer offsets)
// ============================================================================

static void metal_dequant_matvec_offset(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> W_packed,  NSUInteger w_offset,
    id<MTLBuffer> scales,    NSUInteger s_offset,
    id<MTLBuffer> biases,    NSUInteger b_offset,
    id<MTLBuffer> x,         NSUInteger x_offset,
    id<MTLBuffer> out,       NSUInteger o_offset,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size,
    int use_fast
) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    id<MTLComputePipelineState> pipeline;

    if (use_fast >= 3 && ctx->matvec_v3) {
        pipeline = ctx->matvec_v3;
    } else if (use_fast >= 1 && ctx->matvec_fast) {
        pipeline = ctx->matvec_fast;
    } else {
        pipeline = ctx->matvec_naive;
    }

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:W_packed offset:w_offset atIndex:0];
    [enc setBuffer:scales   offset:s_offset atIndex:1];
    [enc setBuffer:biases   offset:b_offset atIndex:2];
    [enc setBuffer:x        offset:x_offset atIndex:3];
    [enc setBuffer:out      offset:o_offset atIndex:4];
    [enc setBytes:&out_dim    length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&in_dim     length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:7];

    if (use_fast >= 3) {
        // v3: tiled threadgroups, 256 threads, 8 SIMD groups of 32
        uint32_t num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else if (use_fast) {
        NSUInteger tg_size = MIN(64, [pipeline maxTotalThreadsPerThreadgroup]);
        MTLSize numGroups = MTLSizeMake(out_dim, 1, 1);
        MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:tgSize];
    } else {
        NSUInteger tg_size = MIN(256, [pipeline maxTotalThreadsPerThreadgroup]);
        MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
        MTLSize numGroups = MTLSizeMake((out_dim + tg_size - 1) / tg_size, 1, 1);
        [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:tgSize];
    }

    [enc endEncoding];
}

// Convenience wrapper with zero offsets
static void metal_dequant_matvec(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> W_packed,
    id<MTLBuffer> scales,
    id<MTLBuffer> biases,
    id<MTLBuffer> x,
    id<MTLBuffer> out,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size,
    int use_fast
) {
    metal_dequant_matvec_offset(ctx, cmdbuf, W_packed, 0, scales, 0, biases, 0,
                                 x, 0, out, 0, out_dim, in_dim, group_size, use_fast);
}

// ============================================================================
// Run SwiGLU on Metal
// ============================================================================

static void metal_swiglu(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> gate,
    id<MTLBuffer> up,
    id<MTLBuffer> out,
    uint32_t dim
) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up   offset:0 atIndex:1];
    [enc setBuffer:out  offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];

    NSUInteger tg_size = MIN(256, [ctx->swiglu maxTotalThreadsPerThreadgroup]);
    MTLSize numGroups = MTLSizeMake((dim + tg_size - 1) / tg_size, 1, 1);
    [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Full expert forward pass: gate/up -> SwiGLU -> down
// ============================================================================

typedef struct {
    double io_ms;       // time to pread expert weights
    double compute_ms;  // time for Metal compute
    double total_ms;    // end-to-end
    size_t io_bytes;    // bytes read from SSD
} ExpertTiming;

// Original version: 9 separate preads, 9 separate Metal buffers
static ExpertTiming run_expert_forward(
    MetalContext *ctx,
    int packed_fd,
    int expert_idx,
    id<MTLBuffer> x_buf,       // [HIDDEN_DIM] float input
    id<MTLBuffer> out_buf,     // [HIDDEN_DIM] float output
    int use_fast
) {
    ExpertTiming timing = {0};
    double t0 = now_ms();

    // ---- I/O: pread all 9 components for this expert ----
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    double t_io_start = now_ms();

    id<MTLBuffer> gate_w = metal_buf_pread(ctx, packed_fd, GATE_W_SIZE, expert_offset + GATE_W_OFFSET);
    id<MTLBuffer> gate_s = metal_buf_pread(ctx, packed_fd, GATE_S_SIZE, expert_offset + GATE_S_OFFSET);
    id<MTLBuffer> gate_b = metal_buf_pread(ctx, packed_fd, GATE_B_SIZE, expert_offset + GATE_B_OFFSET);
    id<MTLBuffer> up_w   = metal_buf_pread(ctx, packed_fd, UP_W_SIZE,   expert_offset + UP_W_OFFSET);
    id<MTLBuffer> up_s   = metal_buf_pread(ctx, packed_fd, UP_S_SIZE,   expert_offset + UP_S_OFFSET);
    id<MTLBuffer> up_b   = metal_buf_pread(ctx, packed_fd, UP_B_SIZE,   expert_offset + UP_B_OFFSET);
    id<MTLBuffer> down_w = metal_buf_pread(ctx, packed_fd, DOWN_W_SIZE, expert_offset + DOWN_W_OFFSET);
    id<MTLBuffer> down_s = metal_buf_pread(ctx, packed_fd, DOWN_S_SIZE, expert_offset + DOWN_S_OFFSET);
    id<MTLBuffer> down_b = metal_buf_pread(ctx, packed_fd, DOWN_B_SIZE, expert_offset + DOWN_B_OFFSET);

    double t_io_end = now_ms();
    timing.io_ms = t_io_end - t_io_start;
    timing.io_bytes = EXPERT_SIZE;

    if (!gate_w || !gate_s || !gate_b || !up_w || !up_s || !up_b ||
        !down_w || !down_s || !down_b) {
        fprintf(stderr, "ERROR: Failed to load expert %d weights\n", expert_idx);
        timing.total_ms = now_ms() - t0;
        return timing;
    }

    // ---- Compute: gate/up matvecs -> SwiGLU -> down matvec ----

    // Intermediate buffers
    id<MTLBuffer> gate_out = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> up_out   = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> act_out  = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));

    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    double t_compute_start = now_ms();

    // Create a single command buffer for the full expert pipeline
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // gate_proj: [4096] -> [1024]
    metal_dequant_matvec(ctx, cmdbuf, gate_w, gate_s, gate_b, x_buf, gate_out,
                         inter, hidden, gs, use_fast);

    // up_proj: [4096] -> [1024]
    metal_dequant_matvec(ctx, cmdbuf, up_w, up_s, up_b, x_buf, up_out,
                         inter, hidden, gs, use_fast);

    // SwiGLU: silu(gate) * up -> [1024]
    metal_swiglu(ctx, cmdbuf, gate_out, up_out, act_out, inter);

    // down_proj: [1024] -> [4096]
    metal_dequant_matvec(ctx, cmdbuf, down_w, down_s, down_b, act_out, out_buf,
                         hidden, inter, gs, use_fast);

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    double t_compute_end = now_ms();
    timing.compute_ms = t_compute_end - t_compute_start;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// Optimized expert forward: single pread, buffer offsets
// ============================================================================
// One 7.08 MB pread per expert instead of 9 separate reads.
// Uses Metal buffer offsets to point shaders at the right data within the buffer.
// Eliminates 9x Metal buffer allocation overhead per expert.

static ExpertTiming run_expert_forward_fast(
    MetalContext *ctx,
    int packed_fd,
    int expert_idx,
    id<MTLBuffer> expert_buf,  // Pre-allocated buffer for one expert (EXPERT_SIZE bytes)
    id<MTLBuffer> x_buf,       // [HIDDEN_DIM] float input
    id<MTLBuffer> gate_out,    // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> up_out,      // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> act_out,     // [INTERMEDIATE_DIM] float scratch
    id<MTLBuffer> out_buf,     // [HIDDEN_DIM] float output
    int use_fast
) {
    ExpertTiming timing = {0};
    double t0 = now_ms();

    // ---- I/O: single pread for the entire expert ----
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    double t_io_start = now_ms();
    ssize_t nread = pread(packed_fd, [expert_buf contents], EXPERT_SIZE, expert_offset);
    double t_io_end = now_ms();

    timing.io_ms = t_io_end - t_io_start;
    timing.io_bytes = EXPERT_SIZE;

    if (nread != EXPERT_SIZE) {
        fprintf(stderr, "ERROR: pread expert %d: got %zd, expected %d\n",
                expert_idx, nread, EXPERT_SIZE);
        timing.total_ms = now_ms() - t0;
        return timing;
    }

    // ---- Compute using buffer offsets ----
    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    double t_compute_start = now_ms();
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // gate_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, GATE_W_OFFSET,
        expert_buf, GATE_S_OFFSET,
        expert_buf, GATE_B_OFFSET,
        x_buf, 0,
        gate_out, 0,
        inter, hidden, gs, use_fast);

    // up_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, UP_W_OFFSET,
        expert_buf, UP_S_OFFSET,
        expert_buf, UP_B_OFFSET,
        x_buf, 0,
        up_out, 0,
        inter, hidden, gs, use_fast);

    // SwiGLU
    metal_swiglu(ctx, cmdbuf, gate_out, up_out, act_out, inter);

    // down_proj: [1024] -> [4096]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, DOWN_W_OFFSET,
        expert_buf, DOWN_S_OFFSET,
        expert_buf, DOWN_B_OFFSET,
        act_out, 0,
        out_buf, 0,
        hidden, inter, gs, use_fast);

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    double t_compute_end = now_ms();
    timing.compute_ms = t_compute_end - t_compute_start;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// Run full MoE: K experts, weighted combination
// ============================================================================

typedef struct {
    double io_ms;
    double compute_ms;
    double combine_ms;
    double total_ms;
    size_t io_bytes;
} MoETiming;

// ============================================================================
// V3 encode helpers: encode dispatches into an EXISTING command encoder
// ============================================================================

static void encode_matvec_v3(
    MetalContext *ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> W_packed,  NSUInteger w_offset,
    id<MTLBuffer> scales,    NSUInteger s_offset,
    id<MTLBuffer> biases,    NSUInteger b_offset,
    id<MTLBuffer> x,         NSUInteger x_offset,
    id<MTLBuffer> out,       NSUInteger o_offset,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size
) {
    [enc setComputePipelineState:ctx->matvec_v3];
    [enc setBuffer:W_packed offset:w_offset atIndex:0];
    [enc setBuffer:scales   offset:s_offset atIndex:1];
    [enc setBuffer:biases   offset:b_offset atIndex:2];
    [enc setBuffer:x        offset:x_offset atIndex:3];
    [enc setBuffer:out      offset:o_offset atIndex:4];
    [enc setBytes:&out_dim    length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&in_dim     length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:7];

    uint32_t num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_swiglu_v(
    MetalContext *ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> gate, NSUInteger gate_offset,
    id<MTLBuffer> up,   NSUInteger up_offset,
    id<MTLBuffer> out,  NSUInteger out_offset,
    uint32_t dim
) {
    // Use vec4 if available and dim is aligned
    if (ctx->swiglu_vec4 && (dim % 4 == 0)) {
        [enc setComputePipelineState:ctx->swiglu_vec4];
        [enc setBuffer:gate offset:gate_offset atIndex:0];
        [enc setBuffer:up   offset:up_offset   atIndex:1];
        [enc setBuffer:out  offset:out_offset  atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        uint32_t vec_dim = dim / 4;
        uint32_t num_tgs = (vec_dim + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:gate offset:gate_offset atIndex:0];
        [enc setBuffer:up   offset:up_offset   atIndex:1];
        [enc setBuffer:out  offset:out_offset  atIndex:2];
        [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
        uint32_t num_tgs = (dim + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

// ============================================================================
// FUSED MoE forward: parallel I/O + single command buffer for ALL K experts
// ============================================================================
//
// 1. Parallel pread: K pthreads load all K expert weights simultaneously
// 2. Single command buffer with phased encoders:
//    Phase 1: gate_proj + up_proj for ALL K experts (independent, GPU overlaps)
//    Phase 2: SwiGLU for ALL K experts
//    Phase 3: down_proj for ALL K experts
//    Phase 4: blit copy + weighted_sum
// 3. Single commit + wait

typedef struct {
    int fd;
    void *dst;
    size_t size;
    off_t offset;
    ssize_t result;
} FusedPreadTask;

static void *fused_pread_fn(void *arg) {
    FusedPreadTask *t = (FusedPreadTask *)arg;
    t->result = pread(t->fd, t->dst, t->size, t->offset);
    return NULL;
}

static MoETiming run_moe_forward_fused(
    MetalContext *ctx,
    int packed_fd,
    const int *expert_indices,
    const float *expert_weights,
    int K,
    id<MTLBuffer> x_buf,
    id<MTLBuffer> moe_out_buf
) {
    MoETiming timing = {0};
    double t0 = now_ms();

    int K_use = (K > MAX_K_FUSED) ? MAX_K_FUSED : K;

    // Pre-allocate ALL buffers upfront (no per-expert allocation in hot path)
    id<MTLBuffer> expert_bufs[MAX_K_FUSED];
    id<MTLBuffer> gate_outs[MAX_K_FUSED];
    id<MTLBuffer> up_outs[MAX_K_FUSED];
    id<MTLBuffer> act_outs[MAX_K_FUSED];
    id<MTLBuffer> expert_outs[MAX_K_FUSED];
    for (int k = 0; k < K_use; k++) {
        expert_bufs[k] = metal_buf_shared(ctx, EXPERT_SIZE);
        gate_outs[k]   = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        up_outs[k]     = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        act_outs[k]    = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        expert_outs[k] = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));
    }

    // ---- Parallel I/O: load all K experts concurrently ----
    double t_io_start = now_ms();
    pthread_t io_threads[MAX_K_FUSED];
    FusedPreadTask io_tasks[MAX_K_FUSED];
    for (int k = 0; k < K_use; k++) {
        io_tasks[k].fd = packed_fd;
        io_tasks[k].dst = [expert_bufs[k] contents];
        io_tasks[k].size = EXPERT_SIZE;
        io_tasks[k].offset = (off_t)expert_indices[k] * EXPERT_SIZE;
        io_tasks[k].result = 0;
        pthread_create(&io_threads[k], NULL, fused_pread_fn, &io_tasks[k]);
    }
    for (int k = 0; k < K_use; k++) {
        pthread_join(io_threads[k], NULL);
        if (io_tasks[k].result != EXPERT_SIZE) {
            fprintf(stderr, "ERROR: fused pread expert %d: got %zd\n",
                    expert_indices[k], io_tasks[k].result);
        }
    }
    timing.io_ms = now_ms() - t_io_start;
    timing.io_bytes = (size_t)K_use * EXPERT_SIZE;

    // ---- Single fused command buffer ----
    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    double t_compute_start = now_ms();
    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // Phase 1: gate_proj + up_proj for ALL experts (independent, GPU can overlap)
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        for (int k = 0; k < K_use; k++) {
            encode_matvec_v3(ctx, enc,
                expert_bufs[k], GATE_W_OFFSET,
                expert_bufs[k], GATE_S_OFFSET,
                expert_bufs[k], GATE_B_OFFSET,
                x_buf, 0, gate_outs[k], 0,
                inter, hidden, gs);
            encode_matvec_v3(ctx, enc,
                expert_bufs[k], UP_W_OFFSET,
                expert_bufs[k], UP_S_OFFSET,
                expert_bufs[k], UP_B_OFFSET,
                x_buf, 0, up_outs[k], 0,
                inter, hidden, gs);
        }
        [enc endEncoding];
    }

    // Phase 2: SwiGLU for ALL experts (depends on phase 1)
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        for (int k = 0; k < K_use; k++) {
            encode_swiglu_v(ctx, enc,
                gate_outs[k], 0, up_outs[k], 0, act_outs[k], 0,
                inter);
        }
        [enc endEncoding];
    }

    // Phase 3: down_proj for ALL experts (depends on phase 2)
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        for (int k = 0; k < K_use; k++) {
            encode_matvec_v3(ctx, enc,
                expert_bufs[k], DOWN_W_OFFSET,
                expert_bufs[k], DOWN_S_OFFSET,
                expert_bufs[k], DOWN_B_OFFSET,
                act_outs[k], 0, expert_outs[k], 0,
                hidden, inter, gs);
        }
        [enc endEncoding];
    }

    // Phase 4: stack outputs via blit + weighted sum
    {
        id<MTLBuffer> stacked = metal_buf_shared(ctx, K_use * HIDDEN_DIM * sizeof(float));
        id<MTLBlitCommandEncoder> blit = [cmdbuf blitCommandEncoder];
        for (int k = 0; k < K_use; k++) {
            [blit copyFromBuffer:expert_outs[k]
                    sourceOffset:0
                        toBuffer:stacked
               destinationOffset:k * HIDDEN_DIM * sizeof(float)
                            size:HIDDEN_DIM * sizeof(float)];
        }
        [blit endEncoding];

        id<MTLBuffer> w_buf = metal_buf_shared(ctx, K_use * sizeof(float));
        memcpy([w_buf contents], expert_weights, K_use * sizeof(float));

        uint32_t k_val = (uint32_t)K_use;
        uint32_t dim_val = HIDDEN_DIM;

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->weighted_sum];
        [enc setBuffer:stacked     offset:0 atIndex:0];
        [enc setBuffer:w_buf       offset:0 atIndex:1];
        [enc setBuffer:moe_out_buf offset:0 atIndex:2];
        [enc setBytes:&k_val   length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dim_val length:sizeof(uint32_t) atIndex:4];
        uint32_t num_tgs = (HIDDEN_DIM + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    timing.compute_ms = now_ms() - t_compute_start;
    timing.total_ms = now_ms() - t0;

    return timing;
}

static MoETiming run_moe_forward(
    MetalContext *ctx,
    int packed_fd,
    const int *expert_indices,   // [K]
    const float *expert_weights, // [K]
    int K,
    id<MTLBuffer> x_buf,        // [HIDDEN_DIM] input
    id<MTLBuffer> moe_out_buf,  // [HIDDEN_DIM] output
    int use_fast
) {
    MoETiming timing = {0};
    double t0 = now_ms();

    // Pre-allocate reusable buffers for optimized path
    id<MTLBuffer> expert_buf = metal_buf_shared(ctx, EXPERT_SIZE);
    id<MTLBuffer> gate_out = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> up_out   = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
    id<MTLBuffer> act_out  = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));

    // Stacked outputs for weighted combination
    id<MTLBuffer> stacked = metal_buf_shared(ctx, K * HIDDEN_DIM * sizeof(float));
    id<MTLBuffer> expert_out = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

    // Run each expert using optimized single-pread path
    for (int k = 0; k < K; k++) {
        ExpertTiming et = run_expert_forward_fast(ctx, packed_fd, expert_indices[k],
                                                   expert_buf, x_buf,
                                                   gate_out, up_out, act_out,
                                                   expert_out, use_fast);
        timing.io_ms += et.io_ms;
        timing.compute_ms += et.compute_ms;
        timing.io_bytes += et.io_bytes;

        // Copy this expert's output into the stacked buffer
        memcpy((float *)[stacked contents] + k * HIDDEN_DIM,
               [expert_out contents],
               HIDDEN_DIM * sizeof(float));
    }

    // Combine: out = sum(w[k] * expert_out[k])
    double t_combine = now_ms();

    // Upload weights
    id<MTLBuffer> w_buf = metal_buf_shared(ctx, K * sizeof(float));
    memcpy([w_buf contents], expert_weights, K * sizeof(float));

    // Run weighted sum kernel
    uint32_t k_val = (uint32_t)K;
    uint32_t dim_val = HIDDEN_DIM;

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->weighted_sum];
    [enc setBuffer:stacked    offset:0 atIndex:0];
    [enc setBuffer:w_buf      offset:0 atIndex:1];
    [enc setBuffer:moe_out_buf offset:0 atIndex:2];
    [enc setBytes:&k_val   length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim_val length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg_size = MIN(256, [ctx->weighted_sum maxTotalThreadsPerThreadgroup]);
    MTLSize numGroups = MTLSizeMake((HIDDEN_DIM + tg_size - 1) / tg_size, 1, 1);
    [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    timing.combine_ms = now_ms() - t_combine;
    timing.total_ms = now_ms() - t0;

    return timing;
}

// ============================================================================
// Parallel pread: 8 pthreads loading K experts simultaneously
// ============================================================================

#define NUM_IO_THREADS 8

typedef struct {
    int fd;
    void *dst;           // destination buffer (expert_bufs[k] contents)
    off_t offset;        // file offset for this expert
    size_t size;         // EXPERT_SIZE
    ssize_t result;      // bytes actually read
} PreadTask;

typedef struct {
    PreadTask *tasks;
    int num_tasks;
    int thread_id;
} PreadThreadArg;

// ARC-compatible wrapper: array of void* pointing to Metal buffer contents.
// We pass raw pointers (from [buf contents]) to pread threads, avoiding
// ARC ownership issues with id<MTLBuffer>* across threads.
typedef struct {
    void *dst[MAX_ACTIVE_EXPERTS];
    off_t offset[MAX_ACTIVE_EXPERTS];
    int K;
    int fd;
} ExpertIOPlan;

static void *pread_thread_fn(void *arg) {
    PreadThreadArg *ta = (PreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        PreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// Build an I/O plan on the main thread (ARC-safe: extracts void* from id<MTLBuffer>)
static void build_io_plan(ExpertIOPlan *plan, int layer_fd,
                           int *expert_indices,
                           id<MTLBuffer> __strong *expert_bufs, int K) {
    plan->fd = layer_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        plan->dst[k] = [expert_bufs[k] contents];
        plan->offset[k] = (off_t)expert_indices[k] * EXPERT_SIZE;
    }
}

// Execute a pre-built I/O plan: parallel pread using NUM_IO_THREADS pthreads.
// Pure C, no ARC objects. Returns wall-clock time in ms.
static double execute_io_plan(ExpertIOPlan *plan) {
    PreadTask tasks[MAX_ACTIVE_EXPERTS];
    for (int k = 0; k < plan->K; k++) {
        tasks[k].fd = plan->fd;
        tasks[k].dst = plan->dst[k];
        tasks[k].offset = plan->offset[k];
        tasks[k].size = EXPERT_SIZE;
        tasks[k].result = 0;
    }

    int nthreads = (plan->K < NUM_IO_THREADS) ? plan->K : NUM_IO_THREADS;
    pthread_t threads[NUM_IO_THREADS];
    PreadThreadArg args[NUM_IO_THREADS];

    double t0 = now_ms();
    for (int t = 0; t < nthreads; t++) {
        args[t].tasks = tasks;
        args[t].num_tasks = plan->K;
        args[t].thread_id = t;
        pthread_create(&threads[t], NULL, pread_thread_fn, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    double io_ms = now_ms() - t0;

    for (int k = 0; k < plan->K; k++) {
        if (tasks[k].result != (ssize_t)EXPERT_SIZE) {
            fprintf(stderr, "ERROR: pread (offset=%lld): got %zd, expected %d (errno=%d)\n",
                    (long long)tasks[k].offset, tasks[k].result, EXPERT_SIZE, errno);
        }
    }
    return io_ms;
}

// ============================================================================
// Encode one expert's full forward pass into a command buffer (no commit)
// ============================================================================
// All 4 dispatches (gate, up, swiglu, down) go into ONE command buffer.

static void encode_expert_compute(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> expert_buf,      // one expert's packed weights
    id<MTLBuffer> x_buf,           // [HIDDEN_DIM] input
    id<MTLBuffer> gate_out,        // [INTERMEDIATE_DIM] scratch
    id<MTLBuffer> up_out,          // [INTERMEDIATE_DIM] scratch
    id<MTLBuffer> act_out,         // [INTERMEDIATE_DIM] scratch
    id<MTLBuffer> out_buf,         // [HIDDEN_DIM] output
    int use_fast
) {
    uint32_t hidden = HIDDEN_DIM;
    uint32_t inter  = INTERMEDIATE_DIM;
    uint32_t gs     = GROUP_SIZE;

    // gate_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, GATE_W_OFFSET,
        expert_buf, GATE_S_OFFSET,
        expert_buf, GATE_B_OFFSET,
        x_buf, 0, gate_out, 0,
        inter, hidden, gs, use_fast);

    // up_proj: [4096] -> [1024]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, UP_W_OFFSET,
        expert_buf, UP_S_OFFSET,
        expert_buf, UP_B_OFFSET,
        x_buf, 0, up_out, 0,
        inter, hidden, gs, use_fast);

    // SwiGLU
    metal_swiglu(ctx, cmdbuf, gate_out, up_out, act_out, inter);

    // down_proj: [1024] -> [4096]
    metal_dequant_matvec_offset(ctx, cmdbuf,
        expert_buf, DOWN_W_OFFSET,
        expert_buf, DOWN_S_OFFSET,
        expert_buf, DOWN_B_OFFSET,
        act_out, 0, out_buf, 0,
        hidden, inter, gs, use_fast);
}

// ============================================================================
// Encode weighted sum of K expert outputs into a command buffer
// ============================================================================

static void encode_weighted_sum(
    MetalContext *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> stacked,         // [K * HIDDEN_DIM] expert outputs
    id<MTLBuffer> w_buf,           // [K] weights
    id<MTLBuffer> out_buf,         // [HIDDEN_DIM] output
    uint32_t K
) {
    uint32_t dim_val = HIDDEN_DIM;

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->weighted_sum];
    [enc setBuffer:stacked offset:0 atIndex:0];
    [enc setBuffer:w_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:&K       length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&dim_val length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg_size = MIN(256, [ctx->weighted_sum maxTotalThreadsPerThreadgroup]);
    MTLSize numGroups = MTLSizeMake((HIDDEN_DIM + tg_size - 1) / tg_size, 1, 1);
    [enc dispatchThreadgroups:numGroups threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Full 60-layer MoE forward pass with double-buffered I/O + compute pipeline
// ============================================================================
//
// Architecture:
//   - Opens all 60 layer files at startup
//   - For each layer: picks K experts, preads them in parallel, runs GPU compute
//   - Double buffering: while GPU computes layer N, pread layer N+1 into buffer set B
//   - Single command buffer per layer (all K expert computes + weighted sum)
//   - 8 pthreads for parallel pread across K experts
//   - h (hidden state) accumulates: h = h + moe_output per layer (residual)

typedef struct {
    double total_ms;
    double io_ms;
    double compute_ms;
    double overhead_ms;  // total - io - compute (sync, alloc, etc.)
    size_t io_bytes;
} FullForwardTiming;

// I/O thread context for background prefetch (double buffering).
// Uses ExpertIOPlan (pure C, no ObjC objects) to avoid ARC issues across threads.
typedef struct {
    ExpertIOPlan plan;      // pre-built I/O plan (void* pointers, no ARC)
    double io_ms;           // result: time for this prefetch
    int done;               // flag: set to 1 when done
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // flag: set to 1 to start prefetch
    int shutdown;           // flag: set to 1 to exit thread
} PrefetchCtx;

static void *prefetch_thread_fn(void *arg) {
    PrefetchCtx *pf = (PrefetchCtx *)arg;

    while (1) {
        // Wait for start signal
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute the pre-built I/O plan (pure C, no ARC)
        pf->io_ms = execute_io_plan(&pf->plan);

        // Signal done
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build plan on main thread (ARC-safe), then signal prefetch thread.
static void prefetch_start_with_plan(PrefetchCtx *pf, int layer_fd,
                                      int *expert_indices,
                                      id<MTLBuffer> __strong *expert_bufs, int K) {
    pthread_mutex_lock(&pf->mutex);
    // Build the I/O plan here on the main thread where ARC is available
    build_io_plan(&pf->plan, layer_fd, expert_indices, expert_bufs, K);
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

static double prefetch_wait(PrefetchCtx *pf) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    double io_ms = pf->io_ms;
    pthread_mutex_unlock(&pf->mutex);
    return io_ms;
}

static FullForwardTiming run_full_forward(
    MetalContext *ctx,
    int *layer_fds,          // [NUM_LAYERS] open file descriptors
    int K,                   // number of active experts per layer
    int use_fast,
    int verbose
) {
    FullForwardTiming timing = {0};
    double t0 = now_ms();

    // ---- Allocate double-buffered expert weight buffers ----
    // Buffer set A and B, each holds K expert weight buffers
    id<MTLBuffer> expert_bufs_A[MAX_ACTIVE_EXPERTS];
    id<MTLBuffer> expert_bufs_B[MAX_ACTIVE_EXPERTS];
    for (int k = 0; k < K; k++) {
        expert_bufs_A[k] = metal_buf_shared(ctx, EXPERT_SIZE);
        expert_bufs_B[k] = metal_buf_shared(ctx, EXPERT_SIZE);
        if (!expert_bufs_A[k] || !expert_bufs_B[k]) {
            fprintf(stderr, "ERROR: Failed to allocate expert buffer %d\n", k);
            timing.total_ms = now_ms() - t0;
            return timing;
        }
    }

    // Per-expert scratch buffers — K sets so all experts run in ONE command buffer
    id<MTLBuffer> per_k_gate[MAX_ACTIVE_EXPERTS];
    id<MTLBuffer> per_k_up[MAX_ACTIVE_EXPERTS];
    id<MTLBuffer> per_k_act[MAX_ACTIVE_EXPERTS];
    id<MTLBuffer> per_k_out[MAX_ACTIVE_EXPERTS];
    for (int k = 0; k < K; k++) {
        per_k_gate[k] = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        per_k_up[k]   = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        per_k_act[k]  = metal_buf_shared(ctx, INTERMEDIATE_DIM * sizeof(float));
        per_k_out[k]  = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));
    }
    // Keep originals for compatibility
    id<MTLBuffer> gate_out = per_k_gate[0];
    id<MTLBuffer> up_out   = per_k_up[0];
    id<MTLBuffer> act_out  = per_k_act[0];

    // Stacked expert outputs for weighted combination
    id<MTLBuffer> stacked = metal_buf_shared(ctx, K * HIDDEN_DIM * sizeof(float));

    // Per-expert output buffer (unused now — using per_k_out instead)
    id<MTLBuffer> expert_out = per_k_out[0];

    // Hidden state buffer (h): starts with input, accumulates residual per layer
    id<MTLBuffer> h_buf = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));
    float *h_data = (float *)[h_buf contents];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        h_data[i] = 0.1f * sinf((float)i * 0.1f + 0.3f);
    }

    // MoE output buffer per layer
    id<MTLBuffer> moe_out = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

    // Expert routing weights (uniform for benchmarking)
    float expert_weights[MAX_ACTIVE_EXPERTS];
    float wsum = 0.0f;
    for (int k = 0; k < K; k++) {
        expert_weights[k] = 1.0f / (float)(k + 1);
        wsum += expert_weights[k];
    }
    for (int k = 0; k < K; k++) {
        expert_weights[k] /= wsum;
    }
    id<MTLBuffer> w_buf = metal_buf_shared(ctx, K * sizeof(float));
    memcpy([w_buf contents], expert_weights, K * sizeof(float));

    // Pre-generate deterministic expert indices for each layer
    // Uses a simple hash to spread across 512 experts
    int layer_experts[NUM_LAYERS][MAX_ACTIVE_EXPERTS];
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        for (int k = 0; k < K; k++) {
            // Deterministic spread: different experts per layer for realistic benchmark
            layer_experts[layer][k] = ((layer * 7 + k * 31 + 13) % NUM_EXPERTS);
        }
    }

    // ---- Set up prefetch thread for double buffering ----
    PrefetchCtx prefetch;
    memset(&prefetch, 0, sizeof(prefetch));
    pthread_mutex_init(&prefetch.mutex, NULL);
    pthread_cond_init(&prefetch.cond, NULL);
    prefetch.shutdown = 0;

    pthread_t prefetch_tid;
    pthread_create(&prefetch_tid, NULL, prefetch_thread_fn, &prefetch);

    // ---- Layer 0: initial synchronous load into buffer set A ----
    double io_total = 0.0;
    double compute_total = 0.0;

    {
        ExpertIOPlan plan0;
        build_io_plan(&plan0, layer_fds[0], layer_experts[0], expert_bufs_A, K);
        double io_layer0 = execute_io_plan(&plan0);
        io_total += io_layer0;
        if (verbose) {
            printf("  [layer  0] I/O: %.2f ms (sync initial load)\n", io_layer0);
        }
    }

    // ---- Main loop: process layer N, prefetch layer N+1 ----
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // Determine which buffer set has this layer's data
        // Use a flag instead of raw pointers to avoid ARC issues
        int cur_is_A = (layer % 2 == 0);
        id<MTLBuffer> __strong *cur_bufs = cur_is_A ? expert_bufs_A : expert_bufs_B;
        id<MTLBuffer> __strong *next_bufs = cur_is_A ? expert_bufs_B : expert_bufs_A;

        // Start prefetching next layer (if not the last)
        if (layer + 1 < NUM_LAYERS) {
            prefetch_start_with_plan(&prefetch, layer_fds[layer + 1],
                                      layer_experts[layer + 1], next_bufs, K);
        }

        // ---- GPU compute for current layer ----
        // Single command buffer: K expert forwards + weighted sum
        double t_compute = now_ms();

        id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

        // ALL K experts in ONE command buffer (no per-expert sync)
        // Each expert writes directly into its slot in the stacked buffer
        for (int k = 0; k < K; k++) {
            // Use stacked buffer with offset for this expert's output
            id<MTLBuffer> k_gate_out = per_k_gate[k];
            id<MTLBuffer> k_up_out   = per_k_up[k];
            id<MTLBuffer> k_act_out  = per_k_act[k];

            encode_expert_compute(ctx, cmdbuf, cur_bufs[k],
                                  h_buf, k_gate_out, k_up_out, k_act_out,
                                  stacked, use_fast);
            // Override the output offset to write into stacked[k*HIDDEN:]
            // Actually, encode_expert_compute writes to out_buf at offset 0.
            // We need to write to stacked at offset k*HIDDEN_DIM*sizeof(float).
            // Let me use expert_out per-k and then blit.
        }

        // Simpler: just use separate expert_out buffers and blit into stacked
        // Actually the cleanest fix: have encode_expert_compute take an output offset
        // For now: allocate per-k expert_out buffers and copy after ONE commit

        // All K experts in ONE command buffer
        for (int k = 0; k < K; k++) {
            encode_expert_compute(ctx, cmdbuf, cur_bufs[k],
                                  h_buf, per_k_gate[k], per_k_up[k], per_k_act[k],
                                  per_k_out[k], use_fast);
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        // CPU memcpy into stacked (faster than GPU blit for 64KB)
        for (int k = 0; k < K; k++) {
            memcpy((float *)[stacked contents] + k * HIDDEN_DIM,
                   [per_k_out[k] contents],
                   HIDDEN_DIM * sizeof(float));
        }

        // Weighted sum on CPU (4 × 4096 floats = trivial, avoids extra cmd buffer)
        {
            float *moe = (float *)[moe_out contents];
            float *w = (float *)[w_buf contents];
            memset(moe, 0, HIDDEN_DIM * sizeof(float));
            for (int k = 0; k < K; k++) {
                float *ek = (float *)[per_k_out[k] contents];
                float wk = w[k];
                for (int d = 0; d < HIDDEN_DIM; d++) {
                    moe[d] += ek[d] * wk;
                }
            }
        }

        // Accumulate residual: h = h + moe_out
        float *h = (float *)[h_buf contents];
        float *m = (float *)[moe_out contents];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h[i] += m[i];
        }

        double compute_ms = now_ms() - t_compute;
        compute_total += compute_ms;

        // Wait for prefetch of next layer to finish
        if (layer + 1 < NUM_LAYERS) {
            double next_io_ms = prefetch_wait(&prefetch);
            io_total += next_io_ms;
            if (verbose) {
                printf("  [layer %2d] compute: %.2f ms, next I/O: %.2f ms\n",
                       layer, compute_ms, next_io_ms);
            }
        } else {
            if (verbose) {
                printf("  [layer %2d] compute: %.2f ms (last layer)\n",
                       layer, compute_ms);
            }
        }
    }

    // ---- Shut down prefetch thread ----
    pthread_mutex_lock(&prefetch.mutex);
    prefetch.shutdown = 1;
    pthread_cond_signal(&prefetch.cond);
    pthread_mutex_unlock(&prefetch.mutex);
    pthread_join(prefetch_tid, NULL);
    pthread_mutex_destroy(&prefetch.mutex);
    pthread_cond_destroy(&prefetch.cond);

    timing.total_ms = now_ms() - t0;
    timing.io_ms = io_total;
    timing.compute_ms = compute_total;
    timing.overhead_ms = timing.total_ms - timing.io_ms - timing.compute_ms;
    // Note: overhead_ms can be negative when I/O is fully overlapped with compute
    // (the I/O time is measured wall-clock on the prefetch thread, so it overlaps)
    // More accurate: overhead = total - max(io, compute) approximately
    timing.io_bytes = (size_t)NUM_LAYERS * K * EXPERT_SIZE;

    // Print final hidden state sample
    float *h_final = (float *)[h_buf contents];
    printf("\n[full] h[0..7] = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
           h_final[0], h_final[1], h_final[2], h_final[3],
           h_final[4], h_final[5], h_final[6], h_final[7]);

    return timing;
}

// ============================================================================
// CPU reference for full expert forward (for verification)
// ============================================================================

static void cpu_expert_forward(
    int packed_fd,
    int expert_idx,
    const float *x,     // [HIDDEN_DIM]
    float *out          // [HIDDEN_DIM]
) {
    off_t expert_offset = (off_t)expert_idx * EXPERT_SIZE;

    // Read all components
    uint32_t gate_w[GATE_W_SIZE / 4];
    uint16_t gate_s[GATE_S_SIZE / 2];
    uint16_t gate_b[GATE_B_SIZE / 2];
    uint32_t up_w[UP_W_SIZE / 4];
    uint16_t up_s[UP_S_SIZE / 2];
    uint16_t up_b[UP_B_SIZE / 2];
    uint32_t down_w[DOWN_W_SIZE / 4];
    uint16_t down_s[DOWN_S_SIZE / 2];
    uint16_t down_b[DOWN_B_SIZE / 2];

    pread(packed_fd, gate_w, GATE_W_SIZE, expert_offset + GATE_W_OFFSET);
    pread(packed_fd, gate_s, GATE_S_SIZE, expert_offset + GATE_S_OFFSET);
    pread(packed_fd, gate_b, GATE_B_SIZE, expert_offset + GATE_B_OFFSET);
    pread(packed_fd, up_w,   UP_W_SIZE,   expert_offset + UP_W_OFFSET);
    pread(packed_fd, up_s,   UP_S_SIZE,   expert_offset + UP_S_OFFSET);
    pread(packed_fd, up_b,   UP_B_SIZE,   expert_offset + UP_B_OFFSET);
    pread(packed_fd, down_w, DOWN_W_SIZE, expert_offset + DOWN_W_OFFSET);
    pread(packed_fd, down_s, DOWN_S_SIZE, expert_offset + DOWN_S_OFFSET);
    pread(packed_fd, down_b, DOWN_B_SIZE, expert_offset + DOWN_B_OFFSET);

    // gate_proj: [4096] -> [1024]
    float gate_out[INTERMEDIATE_DIM];
    cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, x, gate_out,
                            INTERMEDIATE_DIM, HIDDEN_DIM, GROUP_SIZE);

    // up_proj: [4096] -> [1024]
    float up_out[INTERMEDIATE_DIM];
    cpu_dequant_matvec_4bit(up_w, up_s, up_b, x, up_out,
                            INTERMEDIATE_DIM, HIDDEN_DIM, GROUP_SIZE);

    // SwiGLU
    float act_out[INTERMEDIATE_DIM];
    cpu_swiglu(gate_out, up_out, act_out, INTERMEDIATE_DIM);

    // down_proj: [1024] -> [4096]
    cpu_dequant_matvec_4bit(down_w, down_s, down_b, act_out, out,
                            HIDDEN_DIM, INTERMEDIATE_DIM, GROUP_SIZE);
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --layer N        Layer index (default: 0)\n");
    printf("  --expert E       Expert index (default: 0)\n");
    printf("  --benchmark      Run timing benchmark (10 iterations)\n");
    printf("  --moe            Run full MoE with K experts on one layer\n");
    printf("  --full           Run full 60-layer MoE forward pass\n");
    printf("  --k N            Number of active experts per layer (default: 4)\n");
    printf("  --verify         Verify Metal output against CPU reference\n");
    printf("  --fast           Use threadgroup-optimized shader\n");
    printf("  --model PATH     Model path (default: built-in)\n");
    printf("  --help           This message\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        int layer_idx = 0;
        int expert_idx = 0;
        int do_benchmark = 0;
        int do_moe = 0;
        int do_full = 0;
        int num_active_experts = 4;  // --k flag
        int do_verify = 0;
        int use_fast = 0;
        // Resolve default model path at runtime using $HOME
        static char default_model_path[1024];
        const char *home = getenv("HOME");
        if (!home) home = "/tmp";
        {
            char snapshots_dir[1024];
            snprintf(snapshots_dir, sizeof(snapshots_dir),
                     "%s/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots", home);
            DIR *d = opendir(snapshots_dir);
            if (d) {
                struct dirent *entry;
                while ((entry = readdir(d)) != NULL) {
                    if (entry->d_name[0] != '.') {
                        snprintf(default_model_path, sizeof(default_model_path), "%s/%s", snapshots_dir, entry->d_name);
                        break;
                    }
                }
                closedir(d);
            } else {
                snprintf(default_model_path, sizeof(default_model_path),
                         "%s/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3", home);
            }
        }
        const char *model_path = default_model_path;

        static struct option long_options[] = {
            {"layer",     required_argument, 0, 'l'},
            {"expert",    required_argument, 0, 'e'},
            {"benchmark", no_argument,       0, 'b'},
            {"moe",       no_argument,       0, 'm'},
            {"full",      no_argument,       0, 'F'},
            {"k",         required_argument, 0, 'k'},
            {"verify",    no_argument,       0, 'v'},
            {"fast",      no_argument,       0, 'f'},
            {"model",     required_argument, 0, 'p'},
            {"help",      no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "l:e:bmFk:vfp:h", long_options, NULL)) != -1) {
            switch (c) {
                case 'l': layer_idx = atoi(optarg); break;
                case 'e': expert_idx = atoi(optarg); break;
                case 'b': do_benchmark = 1; break;
                case 'm': do_moe = 1; break;
                case 'F': do_full = 1; break;
                case 'k': num_active_experts = atoi(optarg); break;
                case 'v': do_verify = 1; break;
                case 'f': use_fast = 3; break;  // v3 tiled+SIMD shader
                case 'p': model_path = optarg; break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        // Clamp K to valid range
        if (num_active_experts < 1) num_active_experts = 1;
        if (num_active_experts > MAX_ACTIVE_EXPERTS) {
            fprintf(stderr, "WARNING: --k %d exceeds MAX_ACTIVE_EXPERTS (%d), clamping\n",
                    num_active_experts, MAX_ACTIVE_EXPERTS);
            num_active_experts = MAX_ACTIVE_EXPERTS;
        }
        if (num_active_experts > NUM_EXPERTS) num_active_experts = NUM_EXPERTS;

        const char *shader_name = (use_fast >= 3) ? "v3-tiled" :
                                  (use_fast >= 1) ? "fast-simd" : "naive";

        printf("=== metal_infer: 4-bit dequant MoE engine (v3 optimized) ===\n");
        if (do_full) {
            printf("Mode: FULL 60-layer forward, K=%d, Shader: %s, Benchmark: %s\n",
                   num_active_experts, shader_name,
                   do_benchmark ? "YES" : "NO");
        } else {
            printf("Layer: %d, Expert: %d, Shader: %s, Benchmark: %s, MoE: %s, Verify: %s\n",
                   layer_idx, expert_idx, shader_name,
                   do_benchmark ? "YES" : "NO",
                   do_moe ? "YES" : "NO",
                   do_verify ? "YES" : "NO");
        }

        // ---- Initialize Metal ----
        MetalContext *ctx = metal_init();
        if (!ctx) return 1;

        // ========== Full 60-layer forward pass mode ==========
        if (do_full) {
            // ---- Open ALL 60 packed layer files ----
            int layer_fds[NUM_LAYERS];
            int open_count = 0;
            printf("\n[io] Opening all %d layer files...\n", NUM_LAYERS);
            double t_open = now_ms();
            for (int i = 0; i < NUM_LAYERS; i++) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin", model_path, i);
                layer_fds[i] = open(path, O_RDONLY);
                if (layer_fds[i] < 0) {
                    fprintf(stderr, "ERROR: Cannot open %s: %s\n", path, strerror(errno));
                    // Close already-opened fds
                    for (int j = 0; j < i; j++) close(layer_fds[j]);
                    metal_destroy(ctx);
                    return 1;
                }
                open_count++;
            }
            printf("[io] Opened %d layer files in %.1f ms\n", open_count, now_ms() - t_open);

            // ---- Run full forward pass ----
            int K = num_active_experts;
            size_t total_expert_bytes = (size_t)NUM_LAYERS * K * EXPERT_SIZE;
            printf("\n=== Full %d-layer MoE forward (K=%d) ===\n", NUM_LAYERS, K);
            printf("[config] %d layers x %d experts x %.2f MB = %.1f MB total I/O\n",
                   NUM_LAYERS, K, EXPERT_SIZE / (1024.0 * 1024.0),
                   total_expert_bytes / (1024.0 * 1024.0));
            printf("[config] Double-buffered I/O + compute pipeline\n");
            printf("[config] %d pthreads for parallel pread\n", NUM_IO_THREADS);

            FullForwardTiming ft = run_full_forward(ctx, layer_fds, K, use_fast,
                                                     do_benchmark ? 0 : 1);

            printf("\nFull %d-layer MoE (K=%d):\n", NUM_LAYERS, K);
            printf("  Total:   %.1f ms (%.2f tok/s)\n",
                   ft.total_ms, 1000.0 / ft.total_ms);
            printf("  I/O:     %.1f ms (%.1f GB/s)\n",
                   ft.io_ms, ft.io_bytes / (ft.io_ms * 1e6));
            printf("  Compute: %.1f ms\n", ft.compute_ms);
            printf("  Overhead: %.1f ms\n",
                   ft.total_ms - ft.compute_ms);
            // Note: I/O is overlapped with compute via double buffering,
            // so effective overhead = total - compute (I/O is hidden when faster)

            // ---- Benchmark: run multiple iterations ----
            if (do_benchmark) {
                int N = 3;  // 3 iterations for full forward (each is ~seconds)
                printf("\n--- Full Forward Benchmark (%d iterations) ---\n", N);
                double best_total = 1e9;
                double sum_total = 0, sum_io = 0, sum_compute = 0;

                for (int i = 0; i < N; i++) {
                    FullForwardTiming bt = run_full_forward(ctx, layer_fds, K, use_fast, 0);
                    sum_total += bt.total_ms;
                    sum_io += bt.io_ms;
                    sum_compute += bt.compute_ms;
                    if (bt.total_ms < best_total) best_total = bt.total_ms;

                    printf("  [%d] total=%.1f ms, io=%.1f ms, compute=%.1f ms, "
                           "%.2f tok/s\n",
                           i, bt.total_ms, bt.io_ms, bt.compute_ms,
                           1000.0 / bt.total_ms);
                }

                printf("\n[bench] Average:\n");
                printf("  Total:   %.1f ms (%.2f tok/s)\n",
                       sum_total / N, 1000.0 / (sum_total / N));
                printf("  I/O:     %.1f ms (%.1f GB/s)\n",
                       sum_io / N,
                       (double)total_expert_bytes * N / (sum_io * 1e6));
                printf("  Compute: %.1f ms\n", sum_compute / N);
                printf("[bench] Best: %.1f ms (%.2f tok/s)\n",
                       best_total, 1000.0 / best_total);
            }

            // Cleanup all fds
            for (int i = 0; i < NUM_LAYERS; i++) close(layer_fds[i]);
            metal_destroy(ctx);
            printf("\nDone.\n");
            return 0;
        }

        // ---- Open single packed expert file (original single-layer modes) ----
        char packed_path[1024];
        snprintf(packed_path, sizeof(packed_path),
                 "%s/packed_experts/layer_%02d.bin", model_path, layer_idx);

        printf("[io] Opening: %s\n", packed_path);
        int packed_fd = open(packed_path, O_RDONLY);
        if (packed_fd < 0) {
            fprintf(stderr, "ERROR: Cannot open %s: %s\n", packed_path, strerror(errno));
            metal_destroy(ctx);
            return 1;
        }
        printf("[io] Opened layer %d packed file (fd=%d)\n", layer_idx, packed_fd);

        // ---- Create input vector with deterministic values ----
        // Use realistic magnitude (~1.0) to stress-test numerical accuracy
        id<MTLBuffer> x_buf = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));
        float *x_data = (float *)[x_buf contents];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            x_data[i] = 0.1f * sinf((float)i * 0.1f + 0.3f);
        }
        printf("[init] Input vector: x[0..3] = [%.6f, %.6f, %.6f, %.6f]\n",
               x_data[0], x_data[1], x_data[2], x_data[3]);

        // ---- Output buffer ----
        id<MTLBuffer> out_buf = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

        // ========== Single expert forward ==========
        if (!do_moe) {
            printf("\n--- Single expert forward (expert %d) ---\n", expert_idx);

            ExpertTiming et = run_expert_forward(ctx, packed_fd, expert_idx,
                                                  x_buf, out_buf, use_fast);

            float *out_data = (float *)[out_buf contents];
            printf("[result] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   out_data[0], out_data[1], out_data[2], out_data[3],
                   out_data[4], out_data[5], out_data[6], out_data[7]);
            printf("[timing] I/O: %.2f ms (%.1f GB/s), Compute: %.2f ms, Total: %.2f ms\n",
                   et.io_ms, et.io_bytes / (et.io_ms * 1e6),
                   et.compute_ms, et.total_ms);

            // ---- Verify against CPU ----
            if (do_verify) {
                printf("\n--- CPU verification ---\n");
                float *cpu_out = calloc(HIDDEN_DIM, sizeof(float));
                double t_cpu = now_ms();
                cpu_expert_forward(packed_fd, expert_idx, x_data, cpu_out);
                double cpu_ms = now_ms() - t_cpu;

                printf("[cpu] Time: %.2f ms\n", cpu_ms);
                printf("[cpu] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                       cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3],
                       cpu_out[4], cpu_out[5], cpu_out[6], cpu_out[7]);

                // Compare
                float max_diff = 0.0f;
                float max_rel_diff = 0.0f;
                int worst_idx = 0;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float diff = fabsf(out_data[i] - cpu_out[i]);
                    float rel = (fabsf(cpu_out[i]) > 1e-6f) ? diff / fabsf(cpu_out[i]) : diff;
                    if (diff > max_diff) {
                        max_diff = diff;
                        worst_idx = i;
                    }
                    if (rel > max_rel_diff) max_rel_diff = rel;
                }
                printf("[verify] Max abs diff: %.6f at index %d (GPU=%.6f, CPU=%.6f)\n",
                       max_diff, worst_idx, out_data[worst_idx], cpu_out[worst_idx]);
                printf("[verify] Max rel diff: %.6f\n", max_rel_diff);
                printf("[verify] %s (threshold: 0.01)\n",
                       max_rel_diff < 0.01f ? "PASS" : "FAIL");
                printf("[verify] GPU speedup: %.1fx vs CPU\n", cpu_ms / et.compute_ms);
                free(cpu_out);
            }

            // ---- Benchmark ----
            if (do_benchmark) {
                printf("\n--- Benchmark (10 iterations) ---\n");
                double io_sum = 0, compute_sum = 0, total_sum = 0;
                int N = 10;
                for (int i = 0; i < N; i++) {
                    ExpertTiming bt = run_expert_forward(ctx, packed_fd, expert_idx,
                                                          x_buf, out_buf, use_fast);
                    io_sum += bt.io_ms;
                    compute_sum += bt.compute_ms;
                    total_sum += bt.total_ms;
                    printf("  [%d] io=%.2f ms, compute=%.2f ms, total=%.2f ms\n",
                           i, bt.io_ms, bt.compute_ms, bt.total_ms);
                }
                printf("[bench] Average: io=%.2f ms, compute=%.2f ms, total=%.2f ms\n",
                       io_sum / N, compute_sum / N, total_sum / N);
                printf("[bench] I/O throughput: %.1f GB/s\n",
                       EXPERT_SIZE * N / (io_sum * 1e6));
            }
        }

        // ========== Full MoE forward (K experts, single layer) ==========
        if (do_moe) {
            int K = num_active_experts;
            printf("\n--- Full MoE forward (%d experts, %s) ---\n",
                   K, (use_fast >= 3) ? "FUSED v3" : "legacy");

            // Simulated routing: spread across expert range
            int moe_experts[MAX_ACTIVE_EXPERTS];
            float moe_weights[MAX_ACTIVE_EXPERTS];
            float wsum = 0.0f;
            for (int k = 0; k < K; k++) {
                moe_experts[k] = (k * (NUM_EXPERTS / K)) % NUM_EXPERTS;
                moe_weights[k] = 1.0f / (float)(k + 1);
                wsum += moe_weights[k];
            }
            for (int k = 0; k < K; k++) {
                moe_weights[k] /= wsum;
            }

            printf("[moe] Experts: ");
            for (int k = 0; k < K; k++) printf("%d(%.3f) ", moe_experts[k], moe_weights[k]);
            printf("\n");

            id<MTLBuffer> moe_out = metal_buf_shared(ctx, HIDDEN_DIM * sizeof(float));

            // Use fused path when --fast (v3 shader)
            MoETiming mt;
            if (use_fast >= 3 && K <= MAX_K_FUSED) {
                mt = run_moe_forward_fused(ctx, packed_fd, moe_experts, moe_weights, K,
                                            x_buf, moe_out);
            } else {
                mt = run_moe_forward(ctx, packed_fd, moe_experts, moe_weights, K,
                                      x_buf, moe_out, use_fast);
            }

            float *moe_data = (float *)[moe_out contents];
            printf("[result] out[0..7] = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   moe_data[0], moe_data[1], moe_data[2], moe_data[3],
                   moe_data[4], moe_data[5], moe_data[6], moe_data[7]);
            printf("[timing] I/O: %.2f ms (%.1f GB/s)\n",
                   mt.io_ms, (double)(EXPERT_SIZE * K) / (mt.io_ms * 1e6));
            printf("[timing] Compute: %.2f ms (all %d experts)\n", mt.compute_ms, K);
            printf("[timing] Total: %.2f ms\n", mt.total_ms);
            printf("[timing] Experts/sec: %.0f\n", K / (mt.total_ms / 1000.0));
            printf("[timing] Per-expert compute: %.3f ms\n", mt.compute_ms / K);

            // Benchmark MoE
            if (do_benchmark) {
                printf("\n--- MoE Benchmark (10 iterations, %s) ---\n",
                       (use_fast >= 3) ? "FUSED v3" : "legacy");
                int N = 10;
                double total_time = 0, io_time = 0, compute_time = 0;
                for (int i = 0; i < N; i++) {
                    MoETiming bt;
                    if (use_fast >= 3 && K <= MAX_K_FUSED) {
                        bt = run_moe_forward_fused(ctx, packed_fd, moe_experts, moe_weights, K,
                                                    x_buf, moe_out);
                    } else {
                        bt = run_moe_forward(ctx, packed_fd, moe_experts, moe_weights, K,
                                              x_buf, moe_out, use_fast);
                    }
                    total_time += bt.total_ms;
                    io_time += bt.io_ms;
                    compute_time += bt.compute_ms;
                    printf("  [%d] io=%.2f compute=%.2f total=%.2f ms\n",
                           i, bt.io_ms, bt.compute_ms, bt.total_ms);
                }
                printf("[bench] Average: io=%.2f ms, compute=%.2f ms, total=%.2f ms\n",
                       io_time / N, compute_time / N, total_time / N);
                printf("[bench] Per-expert compute: %.3f ms\n", compute_time / (N * K));
                printf("[bench] If this were the whole token: %.1f tok/s\n",
                       1000.0 / (total_time / N));

                // Run legacy comparison if we used fused
                if (use_fast >= 3 && K <= MAX_K_FUSED) {
                    printf("\n--- Legacy sequential comparison (10 iter) ---\n");
                    total_time = 0;
                    for (int i = 0; i < N; i++) {
                        MoETiming bt = run_moe_forward(ctx, packed_fd, moe_experts, moe_weights, K,
                                                        x_buf, moe_out, use_fast);
                        total_time += bt.total_ms;
                        printf("  [%d] total=%.2f ms\n", i, bt.total_ms);
                    }
                    printf("[bench-legacy] Average: %.2f ms (%.1f tok/s)\n",
                           total_time / N, 1000.0 / (total_time / N));
                }
            }
        }

        // Cleanup
        close(packed_fd);
        metal_destroy(ctx);

        printf("\nDone.\n");
        return 0;
    }
}
