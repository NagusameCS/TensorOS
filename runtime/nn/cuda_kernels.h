/*
 * TensorOS CUDA Kernels — Shared Header
 *
 * Declares the C interface for CUDA kernel launch wrappers.
 * Used by both cuda_kernels.cu (implementation) and backend_cuda.c (consumer).
 *
 * All functions use extern "C" linkage and are exported from cuda_kernels.dll.
 */

#ifndef TENSOROS_CUDA_KERNELS_H
#define TENSOROS_CUDA_KERNELS_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef CUDA_KERNELS_EXPORTS
    #define CUDA_API __declspec(dllexport)
  #else
    #define CUDA_API __declspec(dllimport)
  #endif
#else
  #define CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Device init/shutdown ─── */
CUDA_API int  ck_init(void);
CUDA_API void ck_shutdown(void);
CUDA_API int  ck_device_count(void);
CUDA_API uint64_t ck_free_memory(int dev);

/* ─── Memory ops ─── */
CUDA_API void *ck_alloc(uint64_t size);
CUDA_API void  ck_free(void *ptr);
CUDA_API int   ck_upload(void *dst, const void *src, uint64_t size);
CUDA_API int   ck_download(void *dst, const void *src, uint64_t size);
CUDA_API void  ck_sync(void);

/* ─── Compute kernels ─── */

/* GEMV: out[od] = W[od, id] @ x[id], W is quantized (type_id: 2=Q4_0, 8=Q8_0, etc) */
CUDA_API void ck_gemv(float *out, const void *W, const float *x,
                      int out_dim, int in_dim, int type_id);

/* GEMM: C[M,N] = A[M,K] @ B[K,N], all F32 */
CUDA_API void ck_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K);

/* RMSNorm: out[d] = x[d] / sqrt(mean(x^2) + eps) * w[d] */
CUDA_API void ck_rmsnorm(float *out, const float *x, const float *w,
                         int dim, float eps);

/* LayerNorm: out[d] = (x[d] - mean) / sqrt(var + eps) * w[d] + b[d] */
CUDA_API void ck_layernorm(float *out, const float *x, const float *w,
                           const float *bias, int dim, float eps);

/* RoPE: apply rotary position encoding to q and k */
CUDA_API void ck_rope(float *q, float *k, int head_dim, int n_heads,
                      int n_kv_heads, int pos, float base, const float *freqs);

/* Softmax: in-place softmax over n elements */
CUDA_API void ck_softmax(float *x, int n);

/* SiLU: x = x * sigmoid(x) */
CUDA_API void ck_silu(float *x, int n);

/* GELU: x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
CUDA_API void ck_gelu(float *x, int n);

/* Element-wise multiply: out = a * b */
CUDA_API void ck_mul(float *out, const float *a, const float *b, int n);

/* Element-wise add: out = a + b */
CUDA_API void ck_add(float *out, const float *a, const float *b, int n);

/* Scale: out = x * s */
CUDA_API void ck_scale(float *out, const float *x, float s, int n);

/* Dot product: returns sum(a[i] * b[i]) */
CUDA_API float ck_dot(const float *a, const float *b, int n);

/* Dequantize: quantized data → float, type_id as in GGML */
CUDA_API void ck_dequantize(float *out, const void *data, int n_elements,
                            int type_id);

/* Multi-head attention:
 * out[n_heads * head_dim] = Attention(Q, K_cache, V_cache) */
CUDA_API void ck_attention(float *out, const float *Q, const float *K,
                           const float *V, int n_heads, int n_kv_heads,
                           int head_dim, int seq_len, float scale, float softcap);

/* KV cache update: insert new K,V vectors at position pos */
CUDA_API void ck_kv_update(float *K_cache, float *V_cache,
                           const float *K_new, const float *V_new,
                           int n_kv_heads, int head_dim, int pos,
                           int max_seq, int layer);

/* Embedding lookup: out[dim] = table[token_id * dim] (with dequant) */
CUDA_API void ck_embed_lookup(float *out, const void *table, int token_id,
                              int dim, int type_id);

/* Softcap: x = cap * tanh(x / cap) */
CUDA_API void ck_softcap(float *x, int n, float cap);

#ifdef __cplusplus
}
#endif

#endif /* TENSOROS_CUDA_KERNELS_H */
