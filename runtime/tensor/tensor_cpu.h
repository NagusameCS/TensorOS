/* =============================================================================
 * TensorOS - CPU Tensor Math Operations
 *
 * Real SIMD-accelerated tensor operations for x86_64.
 * Uses SSE2 (128-bit, 4 floats) via GCC vector extensions.
 *
 * All functions operate on flat float arrays with explicit dimensions.
 * Alignment: all pointers should be 16-byte aligned for SIMD.
 * =============================================================================*/

#ifndef TENSOROS_TENSOR_CPU_H
#define TENSOROS_TENSOR_CPU_H

#include <stdint.h>
#include <stddef.h>

/* =============================================================================
 * Fast Math Utilities (no libc dependency)
 * =============================================================================*/

float fast_expf(float x);
float fast_sqrtf(float x);
float fast_rsqrtf(float x);
float fast_tanhf(float x);
float fast_fabsf(float x);
float fast_logf(float x);

/* =============================================================================
 * Core Tensor Operations
 * All operate on contiguous float arrays
 * =============================================================================*/

/* Matrix multiply: C[M×N] = A[M×K] * B[K×N] — tiled SSE2 */
void tensor_cpu_matmul(float *C, const float *A, const float *B,
                       int M, int N, int K);

/* Batched GEMV: out[batch×N] = in[batch×K] * W^T[K×N] + bias[N]
 * Turns inference from memory-bound GEMV into compute-bound GEMM.
 * activation: 0=none, 1=relu */
void tensor_cpu_batch_gemv(float *out, const float *in, const float *W,
                           const float *bias, int batch, int K, int N,
                           int activation);

/* Conv2D: out[OH×OW×OC] = im2col(in) × W via GEMM
 * in:  [H × W × IC] (channels-last)
 * W:   [OC × KH × KW × IC] (filters)
 * bias: [OC] or NULL
 * stride, pad: spatial parameters */
void tensor_cpu_conv2d(float *out, const float *in, const float *W,
                       const float *bias, int H, int W_dim, int IC,
                       int OC, int KH, int KW, int stride, int pad);

/* Winograd F(2,3) convolution — 2.25x fewer multiplications for 3×3 filters.
 * Transforms input and filter into Winograd domain, does element-wise multiply,
 * then transforms back. Only works for KH=KW=3, stride=1.
 * in:  [H × W × IC] (channels-last)
 * W:   [OC × 3 × 3 × IC] (filters)
 * bias: [OC] or NULL */
void tensor_cpu_conv2d_winograd(float *out, const float *in, const float *W,
                                const float *bias, int H, int W_dim, int IC,
                                int OC, int pad);

/* Element-wise binary operations */
void tensor_cpu_add(float *out, const float *a, const float *b, int n);
void tensor_cpu_mul(float *out, const float *a, const float *b, int n);
void tensor_cpu_scale(float *out, const float *a, float scalar, int n);

/* Activation functions (all vectorized with SSE2) */
void tensor_cpu_relu(float *out, const float *x, int n);
void tensor_cpu_gelu(float *out, const float *x, int n);
void tensor_cpu_sigmoid(float *out, const float *x, int n);
void tensor_cpu_silu(float *out, const float *x, int n);

/* Normalization & softmax */
void tensor_cpu_softmax(float *out, const float *x, int n);
void tensor_cpu_layernorm(float *out, const float *x, int n, float eps);

/* Scaled dot-product attention:
 *   out[seq×d] = softmax(Q[seq×d] · K^T[d×seq] / sqrt(d)) · V[seq×d] */
void tensor_cpu_attention(float *out, const float *Q, const float *K,
                          const float *V, int seq_len, int head_dim);

/* Utility operations */
void  tensor_cpu_transpose(float *out, const float *in, int rows, int cols);
void  tensor_cpu_fill(float *out, float val, int n);
void  tensor_cpu_zero(float *out, int n);
float tensor_cpu_reduce_sum(const float *x, int n);
float tensor_cpu_reduce_max(const float *x, int n);
float tensor_cpu_dot(const float *a, const float *b, int n);
int   tensor_cpu_argmax(const float *x, int n);

/* Verify: run a small self-test and return 0 on success */
int tensor_cpu_selftest(void);

/* SMP Parallel GEMM: splits M dimension across CPU cores.
 * Falls back to tensor_cpu_matmul when ncpus <= 1 or M < 64. */
void tensor_cpu_matmul_smp(float *C, const float *A, const float *B,
                           int M, int N, int K);

#endif /* TENSOROS_TENSOR_CPU_H */
