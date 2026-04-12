/*
 * TensorOS CUDA Backend Skeleton
 *
 * Provides the CUDA backend vtable with stub implementations.
 * Real kernels will be added in P3 when CUDA compilation is enabled.
 *
 * Build with: -DENABLE_CUDA -lcuda -lcudart
 */

#ifdef ENABLE_CUDA

#include "runtime/nn/backend.h"
#include <string.h>

/* ─── Forward declarations for CUDA runtime (avoid header dependency) ─── */
typedef int cudaError_t;
typedef void *cudaStream_t;
extern cudaError_t cudaMalloc(void **devPtr, uint64_t size);
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaMemcpy(void *dst, const void *src, uint64_t count, int kind);
extern cudaError_t cudaDeviceSynchronize(void);
extern cudaError_t cudaGetDeviceCount(int *count);
extern cudaError_t cudaMemGetInfo(uint64_t *free, uint64_t *total);
extern cudaError_t cudaSetDevice(int device);

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaSuccess 0

/* ═══════════════════════════════════════════════════════════════════════
 * Memory ops
 * ════════════════════════════════════════════════════════════════════════ */

static void *cuda_alloc(uint64_t size) {
    void *ptr = (void *)0;
    if (cudaMalloc(&ptr, size) != cudaSuccess) return (void *)0;
    return ptr;
}

static void cuda_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

static int cuda_upload(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

static int cuda_download(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

static void cuda_sync(void) {
    cudaDeviceSynchronize();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Compute stubs (P3: replace with real CUDA kernels)
 * ════════════════════════════════════════════════════════════════════════ */

static void cuda_gemv(float *out, const void *w, const float *x,
                      int od, int id, ggml_type_t t)       { (void)out;(void)w;(void)x;(void)od;(void)id;(void)t; }
static void cuda_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K)                  { (void)C;(void)A;(void)B;(void)M;(void)N;(void)K; }
static void cuda_rmsnorm(float *o, const float *x, const float *w,
                         int d, float e)                    { (void)o;(void)x;(void)w;(void)d;(void)e; }
static void cuda_layernorm(float *o, const float *x, const float *w,
                           const float *b, int d, float e)  { (void)o;(void)x;(void)w;(void)b;(void)d;(void)e; }
static void cuda_rope(float *q, float *k, int hd, int nh,
                      int nkv, int p, float b, const float *f) { (void)q;(void)k;(void)hd;(void)nh;(void)nkv;(void)p;(void)b;(void)f; }
static void cuda_softmax(float *x, int n)                   { (void)x;(void)n; }
static void cuda_silu(float *x, int n)                      { (void)x;(void)n; }
static void cuda_gelu(float *x, int n)                      { (void)x;(void)n; }
static void cuda_mul(float *o, const float *a, const float *b, int n) { (void)o;(void)a;(void)b;(void)n; }
static void cuda_add(float *o, const float *a, const float *b, int n) { (void)o;(void)a;(void)b;(void)n; }
static void cuda_scale(float *o, const float *x, float s, int n) { (void)o;(void)x;(void)s;(void)n; }
static float cuda_dot(const float *a, const float *b, int n) { (void)a;(void)b;(void)n; return 0.0f; }
static void cuda_dequant(float *o, const void *d, int n, ggml_type_t t) { (void)o;(void)d;(void)n;(void)t; }
static void cuda_attention(float *o, const float *Q, const float *K, const float *V,
                           int nh, int nkv, int hd, int sl, float sc, float cap) {
    (void)o;(void)Q;(void)K;(void)V;(void)nh;(void)nkv;(void)hd;(void)sl;(void)sc;(void)cap; }
static void cuda_kv_update(float *K, float *V, const float *Kn, const float *Vn,
                           int nkv, int hd, int p, int ms, int l) {
    (void)K;(void)V;(void)Kn;(void)Vn;(void)nkv;(void)hd;(void)p;(void)ms;(void)l; }
static void cuda_embed(float *o, const void *t, int id, int d, ggml_type_t ty) {
    (void)o;(void)t;(void)id;(void)d;(void)ty; }
static void cuda_softcap(float *x, int n, float c) { (void)x;(void)n;(void)c; }

/* ═══════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ════════════════════════════════════════════════════════════════════════ */

static int cuda_init(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return -1;
    cudaSetDevice(0);
    return 0;
}

static void cuda_shutdown(void) { }

static int cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

static uint64_t cuda_free_memory(int dev) {
    uint64_t free_mem = 0, total = 0;
    cudaSetDevice(dev);
    cudaMemGetInfo(&free_mem, &total);
    return free_mem;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backend Definition
 * ════════════════════════════════════════════════════════════════════════ */

const backend_t backend_cuda = {
    .id   = BACKEND_CUDA,
    .name = "cuda",
    .init = cuda_init,
    .shutdown = cuda_shutdown,
    .get_device_count = cuda_device_count,
    .get_free_memory  = cuda_free_memory,
    .mem = {
        .alloc    = cuda_alloc,
        .free     = cuda_free,
        .upload   = cuda_upload,
        .download = cuda_download,
        .sync     = cuda_sync,
    },
    .compute = {
        .gemv         = cuda_gemv,
        .gemm         = cuda_gemm,
        .rmsnorm      = cuda_rmsnorm,
        .layernorm    = cuda_layernorm,
        .rope         = cuda_rope,
        .softmax      = cuda_softmax,
        .silu         = cuda_silu,
        .gelu         = cuda_gelu,
        .mul          = cuda_mul,
        .add          = cuda_add,
        .scale        = cuda_scale,
        .dot          = cuda_dot,
        .dequantize   = cuda_dequant,
        .attention    = cuda_attention,
        .kv_update    = cuda_kv_update,
        .embed_lookup = cuda_embed,
        .softcap      = cuda_softcap,
    },
};

#endif /* ENABLE_CUDA */
