/*
 * TensorOS Backend Abstraction Layer
 *
 * Provides a vtable-based dispatch mechanism for tensor operations,
 * enabling multiple backends (CPU, CUDA, MLIR) to share the same
 * model loading/parsing/tokenization code while differing in execution.
 *
 * The CPU backend is the gold-standard reference implementation.
 * Other backends must match CPU output within specified numerical tolerances.
 */

#ifndef TENSOROS_BACKEND_H
#define TENSOROS_BACKEND_H

#include <stdint.h>
#include "runtime/nn/gguf.h"

/* ─── Backend identifiers ─── */
typedef enum {
    BACKEND_CPU  = 0,
    BACKEND_CUDA = 1,
    BACKEND_MLIR = 2,
    BACKEND_COUNT
} backend_id_t;

/* ─── Tensor descriptor ─── */
typedef struct {
    void       *data;       /* Pointer to tensor data (host or device) */
    int         ndim;       /* Number of dimensions */
    int         shape[4];   /* Dimension sizes */
    int         stride[4];  /* Strides in elements */
    ggml_type_t dtype;      /* Data type / quantization format */
    int         on_device;  /* 1 if data lives on accelerator memory */
} tensor_t;

/* ─── Memory management ops ─── */
typedef struct {
    /* Allocate device memory (or host-aligned for CPU). Returns NULL on failure. */
    void *(*alloc)(uint64_t size);

    /* Free previously allocated memory. */
    void  (*free)(void *ptr);

    /* Copy host→device. For CPU backend this is just memcpy. */
    int   (*upload)(void *dst_dev, const void *src_host, uint64_t size);

    /* Copy device→host. */
    int   (*download)(void *dst_host, const void *src_dev, uint64_t size);

    /* Synchronize: wait for all pending operations to complete. */
    void  (*sync)(void);
} backend_mem_ops_t;

/* ─── Compute kernel ops ─── */
typedef struct {
    /* GEMV: out[out_dim] = weight[out_dim × in_dim] × x[in_dim]
     * Weight may be quantized (Q4_0, Q8_0, etc). Input x is always float. */
    void (*gemv)(float *out, const void *weight, const float *x,
                 int out_dim, int in_dim, ggml_type_t weight_type);

    /* GEMM: C[M×N] += A[M×K] × B[K×N]   (row-major, float) */
    void (*gemm)(float *C, const float *A, const float *B,
                 int M, int N, int K);

    /* RMSNorm: out[dim] = x[dim] * w[dim] / rms(x) */
    void (*rmsnorm)(float *out, const float *x, const float *w,
                    int dim, float eps);

    /* LayerNorm: out[dim] = (x - mean) / sqrt(var + eps) * w + b */
    void (*layernorm)(float *out, const float *x, const float *w,
                      const float *bias, int dim, float eps);

    /* RoPE: apply rotary position embeddings in-place */
    void (*rope)(float *q, float *k, int head_dim, int n_heads,
                 int n_kv_heads, int pos, float base,
                 const float *freq_factors);

    /* Softmax: in-place softmax over n elements */
    void (*softmax)(float *x, int n);

    /* SiLU: element-wise x * sigmoid(x), in-place */
    void (*silu)(float *x, int n);

    /* GELU: element-wise Gaussian error linear unit, in-place */
    void (*gelu)(float *x, int n);

    /* Element-wise multiply: out[i] = a[i] * b[i] */
    void (*mul)(float *out, const float *a, const float *b, int n);

    /* Element-wise add: out[i] = a[i] + b[i] */
    void (*add)(float *out, const float *a, const float *b, int n);

    /* Scale: out[i] = x[i] * s */
    void (*scale)(float *out, const float *x, float s, int n);

    /* Dot product: returns sum(a[i] * b[i]) */
    float (*dot)(const float *a, const float *b, int n);

    /* Dequantize: convert quantized data to float */
    void (*dequantize)(float *out, const void *data, int n, ggml_type_t type);

    /* Attention: compute scaled dot-product attention
     * Q[n_heads × head_dim], K/V from cache, output to out */
    void (*attention)(float *out, const float *Q,
                      const float *K_cache, const float *V_cache,
                      int n_heads, int n_kv_heads, int head_dim,
                      int seq_len, float scale, float softcap);

    /* KV cache update: write new K/V vectors to cache at position pos */
    void (*kv_update)(float *K_cache, float *V_cache,
                      const float *K_new, const float *V_new,
                      int n_kv_heads, int head_dim, int pos,
                      int max_seq, int layer);

    /* Embedding lookup: out = embd_table[token_id] */
    void (*embed_lookup)(float *out, const void *embd_table,
                         int token_id, int dim, ggml_type_t type);

    /* Logit softcapping: out = cap * tanh(x / cap) */
    void (*softcap)(float *x, int n, float cap);
} backend_compute_ops_t;

/* ─── Full backend interface ─── */
typedef struct {
    backend_id_t         id;
    const char          *name;

    /* Initialize backend. Returns 0 on success. */
    int (*init)(void);

    /* Shutdown backend, free resources. */
    void (*shutdown)(void);

    /* Query device capabilities. */
    int (*get_device_count)(void);
    uint64_t (*get_free_memory)(int device);

    /* Operation vtables. */
    backend_mem_ops_t     mem;
    backend_compute_ops_t compute;
} backend_t;

/* ─── Backend registry ─── */

/* Get the currently active backend. */
const backend_t *backend_get(void);

/* Set the active backend by ID. Returns 0 on success. */
int backend_set(backend_id_t id);

/* Get a specific backend by ID (may return NULL if not available). */
const backend_t *backend_get_by_id(backend_id_t id);

/* Register a backend (called during init). */
void backend_register(const backend_t *be);

/* Initialize all available backends. */
void backend_init_all(void);

/* ─── CPU backend (always available) ─── */
extern const backend_t backend_cpu;

/* ─── Optional backends (linked when available) ─── */
#ifdef ENABLE_CUDA
extern const backend_t backend_cuda;
#endif

#ifdef ENABLE_MLIR
extern const backend_t backend_mlir;
#endif

#endif /* TENSOROS_BACKEND_H */
