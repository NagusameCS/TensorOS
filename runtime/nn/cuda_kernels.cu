/*
 * TensorOS CUDA Kernels — GPU Implementations
 *
 * Real CUDA kernel implementations for LLM inference.
 * Compile with: nvcc -shared -o cuda_kernels.dll cuda_kernels.cu
 *               -DCUDA_KERNELS_EXPORTS --compiler-options /MD
 *
 * Key optimizations:
 *   - Q4_0 dequant fused into GEMV (one kernel, no intermediate buffer)
 *   - Warp-level reductions for RMSNorm/Softmax (shfl_down_sync)
 *   - Coalesced memory access patterns
 *   - Shared memory for attention score computation
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#define CUDA_KERNELS_EXPORTS
#include "cuda_kernels.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ════════════════════════════════════════════════════════════════════════ */

/* FP16 → FP32 conversion (device) */
__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    __half hv;
    *(uint16_t *)&hv = h;
    return __half2float(hv);
}

/* Q4_0 block structure: fp16 scale + 16 bytes (32 nibbles) = 18 bytes */
struct q4_0_block {
    uint16_t d;      /* FP16 scale */
    uint8_t  qs[16]; /* 32 x 4-bit quants packed into 16 bytes */
};

/* Q8_0 block: fp16 scale + 32 int8 = 34 bytes */
struct q8_0_block {
    uint16_t d;
    int8_t   qs[32];
};

/* Warp-level sum reduction */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/* Warp-level max reduction */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/* Block-level sum using shared memory */
__device__ float block_reduce_sum(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

/* Block-level max using shared memory */
__device__ float block_reduce_max(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? shared[threadIdx.x] : -INFINITY;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMV Kernel — Q4_0 quantized matrix @ float vector
 *
 * Each block computes one output element.
 * Threads cooperatively dot-product rows in Q4_0 format.
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_gemv_q4_0(
    float       *out,
    const void  *W,        /* Q4_0 blocks, row-major */
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q4_0_block *blocks =
        (const struct q4_0_block *)W + (int64_t)row * n_blocks;

    float sum = 0.0f;

    /* Each thread handles a stride of Q4_0 blocks */
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float scale = fp16_to_f32(blocks[b].d);
        const float *xp = x + b * 32;

        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t packed = blocks[b].qs[j];
            int q_lo = (int)(packed & 0x0F) - 8;   /* elements [0..15] */
            int q_hi = (int)(packed >> 4)   - 8;   /* elements [16..31] */
            local += (float)q_lo * xp[j];
            local += (float)q_hi * xp[j + 16];
        }
        sum += scale * local;
    }

    /* Reduce within block */
    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);

    if (threadIdx.x == 0) out[row] = sum;
}

/* GEMV for Q8_0: each block = {fp16 scale, int8[32]} */
__global__ void kernel_gemv_q8_0(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q8_0_block *blocks =
        (const struct q8_0_block *)W + (int64_t)row * n_blocks;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float scale = fp16_to_f32(blocks[b].d);
        const float *xp = x + b * 32;
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            local += (float)blocks[b].qs[j] * xp[j];
        sum += scale * local;
    }

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* GEMV for F32: simple dense matrix-vector */
__global__ void kernel_gemv_f32(
    float       *out,
    const float *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    const float *w_row = W + (int64_t)row * in_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += w_row[i] * x[i];

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* GEMV for F16: fp16 weights, f32 activation */
__global__ void kernel_gemv_f16(
    float       *out,
    const __half *W,
    const float  *x,
    int           out_dim,
    int           in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    const __half *w_row = W + (int64_t)row * in_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += __half2float(w_row[i]) * x[i];

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * RMSNorm Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_rmsnorm(
    float       *out,
    const float *x,
    const float *w,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum_sq += x[i] * x[i];

    sum_sq = block_reduce_sum(sum_sq, smem);

    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = x[i] * inv_rms * w[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * LayerNorm Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_layernorm(
    float       *out,
    const float *x,
    const float *w,
    const float *bias,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];

    /* Compute mean */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += x[i];
    sum = block_reduce_sum(sum, smem);

    __shared__ float mean_val;
    if (threadIdx.x == 0) mean_val = sum / (float)dim;
    __syncthreads();

    /* Compute variance */
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean_val;
        var_sum += d * d;
    }
    var_sum = block_reduce_sum(var_sum, smem);

    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / (float)dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm = (x[i] - mean_val) * inv_std;
        out[i] = norm * w[i] + (bias ? bias[i] : 0.0f);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Softmax Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_softmax(float *x, int n) {
    __shared__ float smem[32];

    /* Pass 1: find max */
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        max_val = fmaxf(max_val, x[i]);
    max_val = block_reduce_max(max_val, smem);

    __shared__ float max_shared;
    if (threadIdx.x == 0) max_shared = max_val;
    __syncthreads();

    /* Pass 2: exp and sum */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - max_shared);
        sum += x[i];
    }
    sum = block_reduce_sum(sum, smem);

    __shared__ float inv_sum;
    if (threadIdx.x == 0) inv_sum = 1.0f / sum;
    __syncthreads();

    /* Pass 3: normalize */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] *= inv_sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Elementwise Kernels
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void kernel_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void kernel_mul(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_add(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_scale(float *out, const float *x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * s;
}

__global__ void kernel_softcap(float *x, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

/* ═══════════════════════════════════════════════════════════════════════
 * RoPE Kernel
 *
 * Applies rotary position encoding to Q and K vectors.
 * Each pair (x[2i], x[2i+1]) is rotated by angle pos * freq_i.
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_rope(
    float       *q,
    float       *k,
    int          head_dim,
    int          n_heads,
    int          n_kv_heads,
    int          pos,
    float        base,
    const float *freqs)  /* precomputed freq table [head_dim/2], or NULL */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total_pairs = n_heads * half_hd;

    if (i < total_pairs) {
        int head = i / half_hd;
        int pair = i % half_hd;

        float freq;
        if (freqs)
            freq = freqs[pair];
        else
            freq = 1.0f / powf(base, (float)(2 * pair) / (float)head_dim);

        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* Rotate Q */
        int qi = head * head_dim + 2 * pair;
        float q0 = q[qi], q1 = q[qi + 1];
        q[qi]     = q0 * cos_a - q1 * sin_a;
        q[qi + 1] = q0 * sin_a + q1 * cos_a;

        /* Rotate K (only for KV heads) */
        if (head < n_kv_heads) {
            int ki = head * head_dim + 2 * pair;
            float k0 = k[ki], k1 = k[ki + 1];
            k[ki]     = k0 * cos_a - k1 * sin_a;
            k[ki + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dequantize Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_dequantize_q4_0(
    float      *out,
    const void *data,
    int         n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = n_elements / 32;

    if (i < n_blocks) {
        const struct q4_0_block *b = (const struct q4_0_block *)data + i;
        float scale = fp16_to_f32(b->d);
        float *o = out + i * 32;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t packed = b->qs[j];
            o[j]      = scale * ((float)(packed & 0x0F) - 8.0f);
            o[j + 16] = scale * ((float)(packed >> 4)   - 8.0f);
        }
    }
}

__global__ void kernel_dequantize_q8_0(
    float      *out,
    const void *data,
    int         n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = n_elements / 32;

    if (i < n_blocks) {
        const struct q8_0_block *b = (const struct q8_0_block *)data + i;
        float scale = fp16_to_f32(b->d);
        float *o = out + i * 32;

        #pragma unroll
        for (int j = 0; j < 32; j++)
            o[j] = scale * (float)b->qs[j];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Attention Kernel
 *
 * Single-query attention with GQA support.
 * For each head: score = Q_h @ K_cache[kv_h], softmax, V_cache[kv_h] @ score
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_attention(
    float       *out,        /* [n_heads * head_dim] */
    const float *Q,          /* [n_heads * head_dim] */
    const float *K_cache,    /* [n_kv_heads * max_seq * head_dim] — transposed */
    const float *V_cache,    /* [n_kv_heads * max_seq * head_dim] */
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    int          seq_len,
    float        scale,
    float        softcap)
{
    /* One block per attention head */
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int kv_head = head * n_kv_heads / n_heads; /* GQA head mapping */

    extern __shared__ float shared[];
    float *scores = shared;                       /* [seq_len] */
    float *smem = shared + seq_len;               /* [32] for reductions */

    /* Phase 1: Compute attention scores = Q_h @ K_cache[kv_h, :seq_len, :] */
    const float *q_h = Q + head * head_dim;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const float *k_t = K_cache + ((int64_t)kv_head * seq_len + t) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_h[d] * k_t[d];
        dot *= scale;

        /* Apply softcap if enabled */
        if (softcap > 0.0f)
            dot = softcap * tanhf(dot / softcap);

        scores[t] = dot;
    }
    __syncthreads();

    /* Phase 2: Softmax over scores */
    /* Find max */
    float max_val = -INFINITY;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val, smem);

    __shared__ float max_shared_val;
    if (threadIdx.x == 0) max_shared_val = max_val;
    __syncthreads();

    /* Exp and sum */
    float sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_shared_val);
        sum += scores[t];
    }
    sum = block_reduce_sum(sum, smem);

    __shared__ float inv_sum_shared;
    if (threadIdx.x == 0) inv_sum_shared = 1.0f / sum;
    __syncthreads();

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)
        scores[t] *= inv_sum_shared;
    __syncthreads();

    /* Phase 3: Weighted sum of V: out_h = scores @ V_cache[kv_h] */
    float *out_h = out + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = V_cache + ((int64_t)kv_head * seq_len + t) * head_dim;
            val += scores[t] * v_t[d];
        }
        out_h[d] = val;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * KV Cache Update
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_kv_update(
    float       *K_cache,    /* [n_kv_heads * max_seq * head_dim] */
    float       *V_cache,
    const float *K_new,      /* [n_kv_heads * head_dim] */
    const float *V_new,
    int          n_kv_heads,
    int          head_dim,
    int          pos,
    int          max_seq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (i >= total) return;

    int kv_head = i / head_dim;
    int d = i % head_dim;

    int64_t cache_idx = ((int64_t)kv_head * max_seq + pos) * head_dim + d;
    K_cache[cache_idx] = K_new[i];
    V_cache[cache_idx] = V_new[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * Embedding Lookup
 * ════════════════════════════════════════════════════════════════════════ */

/* For Q4_0 embedding: dequantize row token_id from table → out */
__global__ void kernel_embed_q4_0(
    float      *out,
    const void *table,
    int         token_id,
    int         dim)
{
    int n_blocks = dim / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const struct q4_0_block *blocks =
        (const struct q4_0_block *)table + (int64_t)token_id * n_blocks + b;

    float scale = fp16_to_f32(blocks->d);
    float *o = out + b * 32;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint8_t packed = blocks->qs[j];
        o[j]      = scale * ((float)(packed & 0x0F) - 8.0f);
        o[j + 16] = scale * ((float)(packed >> 4)   - 8.0f);
    }
}

__global__ void kernel_embed_f32(
    float       *out,
    const float *table,
    int          token_id,
    int          dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = table[(int64_t)token_id * dim + i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dot product Kernel (returns scalar via device-to-host copy)
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_dot(float *result, const float *a, const float *b, int n) {
    __shared__ float smem[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += a[i] * b[i];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) *result = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMM Kernel — F32 (simple tiled)
 * ════════════════════════════════════════════════════════════════════════ */

#define TILE_SIZE 16

__global__ void kernel_gemm(
    float       *C,
    const float *A,
    const float *B,
    int          M,
    int          N,
    int          K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
            A[(int64_t)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
            B[(int64_t)b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[(int64_t)row * N + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * C API Wrappers (extern "C")
 * ════════════════════════════════════════════════════════════════════════ */

/* Type IDs from GGML */
#define TYPE_F32  0
#define TYPE_F16  1
#define TYPE_Q4_0 2
#define TYPE_Q8_0 8

extern "C" {

/* ─── Device management ─── */

CUDA_API int ck_init(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return -1;
    cudaSetDevice(0);
    return 0;
}

CUDA_API void ck_shutdown(void) {
    cudaDeviceReset();
}

CUDA_API int ck_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

CUDA_API uint64_t ck_free_memory(int dev) {
    size_t free_mem = 0, total = 0;
    cudaSetDevice(dev);
    cudaMemGetInfo(&free_mem, &total);
    return (uint64_t)free_mem;
}

/* ─── Memory ops ─── */

CUDA_API void *ck_alloc(uint64_t size) {
    void *ptr = NULL;
    if (cudaMalloc(&ptr, size) != cudaSuccess) return NULL;
    return ptr;
}

CUDA_API void ck_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

CUDA_API int ck_upload(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

CUDA_API int ck_download(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

CUDA_API void ck_sync(void) {
    cudaDeviceSynchronize();
}

/* ─── Compute wrappers ─── */

CUDA_API void ck_gemv(float *out, const void *W, const float *x,
                      int out_dim, int in_dim, int type_id) {
    int threads = 256;
    switch (type_id) {
        case TYPE_Q4_0:
            kernel_gemv_q4_0<<<out_dim, threads>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_Q8_0:
            kernel_gemv_q8_0<<<out_dim, threads>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_F32:
            kernel_gemv_f32<<<out_dim, threads>>>(out, (const float *)W, x, out_dim, in_dim);
            break;
        case TYPE_F16:
            kernel_gemv_f16<<<out_dim, threads>>>(out, (const __half *)W, x, out_dim, in_dim);
            break;
        default:
            break;
    }
}

CUDA_API void ck_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    kernel_gemm<<<blocks, threads>>>(C, A, B, M, N, K);
}

CUDA_API void ck_rmsnorm(float *out, const float *x, const float *w,
                         int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    kernel_rmsnorm<<<1, threads>>>(out, x, w, dim, eps);
}

CUDA_API void ck_layernorm(float *out, const float *x, const float *w,
                           const float *bias, int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    kernel_layernorm<<<1, threads>>>(out, x, w, bias, dim, eps);
}

CUDA_API void ck_rope(float *q, float *k, int head_dim, int n_heads,
                      int n_kv_heads, int pos, float base, const float *freqs) {
    int total = n_heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_rope<<<blocks, threads>>>(q, k, head_dim, n_heads, n_kv_heads,
                                     pos, base, freqs);
}

CUDA_API void ck_softmax(float *x, int n) {
    int threads = (n < 1024) ? n : 1024;
    kernel_softmax<<<1, threads>>>(x, n);
}

CUDA_API void ck_silu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_silu<<<blocks, threads>>>(x, n);
}

CUDA_API void ck_gelu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_gelu<<<blocks, threads>>>(x, n);
}

CUDA_API void ck_mul(float *out, const float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_mul<<<blocks, threads>>>(out, a, b, n);
}

CUDA_API void ck_add(float *out, const float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_add<<<blocks, threads>>>(out, a, b, n);
}

CUDA_API void ck_scale(float *out, const float *x, float s, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_scale<<<blocks, threads>>>(out, x, s, n);
}

CUDA_API float ck_dot(const float *a, const float *b, int n) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    int threads = (n < 1024) ? n : 1024;
    kernel_dot<<<1, threads>>>(d_result, a, b, n);
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}

CUDA_API void ck_dequantize(float *out, const void *data, int n_elements,
                            int type_id) {
    int n_blocks = n_elements / 32;
    int threads = 256;
    int grid = (n_blocks + threads - 1) / threads;
    switch (type_id) {
        case TYPE_Q4_0:
            kernel_dequantize_q4_0<<<grid, threads>>>(out, data, n_elements);
            break;
        case TYPE_Q8_0:
            kernel_dequantize_q8_0<<<grid, threads>>>(out, data, n_elements);
            break;
        case TYPE_F32:
            cudaMemcpy(out, data, n_elements * sizeof(float), cudaMemcpyDeviceToDevice);
            break;
        default:
            break;
    }
}

CUDA_API void ck_attention(float *out, const float *Q, const float *K,
                           const float *V, int n_heads, int n_kv_heads,
                           int head_dim, int seq_len, float scale, float softcap) {
    int threads = 256;
    /* Shared memory: scores[seq_len] + smem[32] */
    int shared_bytes = (seq_len + 32) * sizeof(float);
    kernel_attention<<<n_heads, threads, shared_bytes>>>(
        out, Q, K, V, n_heads, n_kv_heads, head_dim, seq_len, scale, softcap);
}

CUDA_API void ck_kv_update(float *K_cache, float *V_cache,
                           const float *K_new, const float *V_new,
                           int n_kv_heads, int head_dim, int pos,
                           int max_seq, int layer) {
    (void)layer;
    int total = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_kv_update<<<blocks, threads>>>(K_cache, V_cache, K_new, V_new,
                                          n_kv_heads, head_dim, pos, max_seq);
}

CUDA_API void ck_embed_lookup(float *out, const void *table, int token_id,
                              int dim, int type_id) {
    int threads = 256;
    switch (type_id) {
        case TYPE_Q4_0: {
            int n_blocks = dim / 32;
            int grid = (n_blocks + threads - 1) / threads;
            kernel_embed_q4_0<<<grid, threads>>>(out, table, token_id, dim);
            break;
        }
        case TYPE_F32: {
            int grid = (dim + threads - 1) / threads;
            kernel_embed_f32<<<grid, threads>>>(out, (const float *)table, token_id, dim);
            break;
        }
        default:
            break;
    }
}

CUDA_API void ck_softcap(float *x, int n, float cap) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_softcap<<<blocks, threads>>>(x, n, cap);
}

} /* extern "C" */
