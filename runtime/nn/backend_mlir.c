/*
 * TensorOS MLIR Backend Skeleton
 *
 * Provides the MLIR backend vtable with stub implementations.
 * Real lowering pipeline will be added in P4.
 *
 * Build with: -DENABLE_MLIR
 */

#ifdef ENABLE_MLIR

#include "runtime/nn/backend.h"

/* MLIR compute stubs — will be replaced with MLIR dialect lowering in P4 */

static void mlir_gemv(float *o, const void *w, const float *x,
                      int od, int id, ggml_type_t t)       { (void)o;(void)w;(void)x;(void)od;(void)id;(void)t; }
static void mlir_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K)                  { (void)C;(void)A;(void)B;(void)M;(void)N;(void)K; }
static void mlir_rmsnorm(float *o, const float *x, const float *w,
                         int d, float e)                    { (void)o;(void)x;(void)w;(void)d;(void)e; }
static void mlir_layernorm(float *o, const float *x, const float *w,
                           const float *b, int d, float e)  { (void)o;(void)x;(void)w;(void)b;(void)d;(void)e; }
static void mlir_rope(float *q, float *k, int hd, int nh,
                      int nkv, int p, float b, const float *f) { (void)q;(void)k;(void)hd;(void)nh;(void)nkv;(void)p;(void)b;(void)f; }
static void mlir_softmax(float *x, int n)                   { (void)x;(void)n; }
static void mlir_silu(float *x, int n)                      { (void)x;(void)n; }
static void mlir_gelu(float *x, int n)                      { (void)x;(void)n; }
static void mlir_mul(float *o, const float *a, const float *b, int n) { (void)o;(void)a;(void)b;(void)n; }
static void mlir_add(float *o, const float *a, const float *b, int n) { (void)o;(void)a;(void)b;(void)n; }
static void mlir_scale(float *o, const float *x, float s, int n) { (void)o;(void)x;(void)s;(void)n; }
static float mlir_dot(const float *a, const float *b, int n) { (void)a;(void)b;(void)n; return 0.0f; }
static void mlir_dequant(float *o, const void *d, int n, ggml_type_t t) { (void)o;(void)d;(void)n;(void)t; }
static void mlir_attention(float *o, const float *Q, const float *K, const float *V,
                           int nh, int nkv, int hd, int sl, float sc, float cap) {
    (void)o;(void)Q;(void)K;(void)V;(void)nh;(void)nkv;(void)hd;(void)sl;(void)sc;(void)cap; }
static void mlir_kv_update(float *K, float *V, const float *Kn, const float *Vn,
                           int nkv, int hd, int p, int ms, int l) {
    (void)K;(void)V;(void)Kn;(void)Vn;(void)nkv;(void)hd;(void)p;(void)ms;(void)l; }
static void mlir_embed(float *o, const void *t, int id, int d, ggml_type_t ty) {
    (void)o;(void)t;(void)id;(void)d;(void)ty; }
static void mlir_softcap(float *x, int n, float c) { (void)x;(void)n;(void)c; }

/* Memory ops: MLIR operates on host memory (like CPU) */
static void *mlir_alloc(uint64_t size) { return backend_cpu.mem.alloc(size); }
static void  mlir_free(void *p)        { backend_cpu.mem.free(p); }
static int   mlir_upload(void *d, const void *s, uint64_t sz)  { return backend_cpu.mem.upload(d, s, sz); }
static int   mlir_download(void *d, const void *s, uint64_t sz){ return backend_cpu.mem.download(d, s, sz); }
static void  mlir_sync(void) {}

static int  mlir_init(void) { return 0; }
static void mlir_shutdown(void) {}
static int  mlir_device_count(void) { return 1; }
static uint64_t mlir_free_memory(int dev) { (void)dev; return backend_cpu.get_free_memory(0); }

const backend_t backend_mlir = {
    .id   = BACKEND_MLIR,
    .name = "mlir",
    .init = mlir_init,
    .shutdown = mlir_shutdown,
    .get_device_count = mlir_device_count,
    .get_free_memory  = mlir_free_memory,
    .mem = {
        .alloc    = mlir_alloc,
        .free     = mlir_free,
        .upload   = mlir_upload,
        .download = mlir_download,
        .sync     = mlir_sync,
    },
    .compute = {
        .gemv         = mlir_gemv,
        .gemm         = mlir_gemm,
        .rmsnorm      = mlir_rmsnorm,
        .layernorm    = mlir_layernorm,
        .rope         = mlir_rope,
        .softmax      = mlir_softmax,
        .silu         = mlir_silu,
        .gelu         = mlir_gelu,
        .mul          = mlir_mul,
        .add          = mlir_add,
        .scale        = mlir_scale,
        .dot          = mlir_dot,
        .dequantize   = mlir_dequant,
        .attention    = mlir_attention,
        .kv_update    = mlir_kv_update,
        .embed_lookup = mlir_embed,
        .softcap      = mlir_softcap,
    },
};

#endif /* ENABLE_MLIR */
