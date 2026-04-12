/*
 * TensorOS CUDA Backend — Dynamic Loading
 *
 * Loads cuda_kernels.dll at runtime (compiled separately by nvcc).
 * Falls back gracefully if the DLL is not found.
 *
 * Build: compile with main project (no CUDA deps needed at compile time)
 * Runtime: needs cuda_kernels.dll next to the executable
 */

#ifdef ENABLE_CUDA

#include "runtime/nn/backend.h"

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle_t;
#define LIB_OPEN(path)       LoadLibraryA(path)
#define LIB_SYM(h, name)    GetProcAddress(h, name)
#define LIB_CLOSE(h)        FreeLibrary(h)
#else
#include <dlfcn.h>
typedef void *lib_handle_t;
#define LIB_OPEN(path)       dlopen(path, RTLD_NOW)
#define LIB_SYM(h, name)    dlsym(h, name)
#define LIB_CLOSE(h)        dlclose(h)
#endif

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#endif

/* ─── Function pointer typedefs matching cuda_kernels.h ─── */
typedef int      (*fn_init)(void);
typedef void     (*fn_shutdown)(void);
typedef int      (*fn_device_count)(void);
typedef uint64_t (*fn_free_memory)(int);
typedef void    *(*fn_alloc)(uint64_t);
typedef void     (*fn_free)(void *);
typedef int      (*fn_upload)(void *, const void *, uint64_t);
typedef int      (*fn_download)(void *, const void *, uint64_t);
typedef void     (*fn_sync)(void);
typedef void     (*fn_gemv)(float *, const void *, const float *, int, int, int);
typedef void     (*fn_gemm)(float *, const float *, const float *, int, int, int);
typedef void     (*fn_rmsnorm)(float *, const float *, const float *, int, float);
typedef void     (*fn_layernorm)(float *, const float *, const float *, const float *, int, float);
typedef void     (*fn_rope)(float *, float *, int, int, int, int, float, const float *);
typedef void     (*fn_softmax)(float *, int);
typedef void     (*fn_silu)(float *, int);
typedef void     (*fn_gelu)(float *, int);
typedef void     (*fn_mul)(float *, const float *, const float *, int);
typedef void     (*fn_add)(float *, const float *, const float *, int);
typedef void     (*fn_scale)(float *, const float *, float, int);
typedef float    (*fn_dot)(const float *, const float *, int);
typedef void     (*fn_dequantize)(float *, const void *, int, int);
typedef void     (*fn_attention)(float *, const float *, const float *, const float *,
                                 int, int, int, int, float, float);
typedef void     (*fn_kv_update)(float *, float *, const float *, const float *,
                                  int, int, int, int, int);
typedef void     (*fn_embed)(float *, const void *, int, int, int);
typedef void     (*fn_softcap)(float *, int, float);

/* ─── Dynamic dispatch table ─── */
static struct {
    lib_handle_t    lib;
    fn_init         init;
    fn_shutdown     shutdown;
    fn_device_count device_count;
    fn_free_memory  free_memory;
    fn_alloc        alloc;
    fn_free         free;
    fn_upload       upload;
    fn_download     download;
    fn_sync         sync;
    fn_gemv         gemv;
    fn_gemm         gemm;
    fn_rmsnorm      rmsnorm;
    fn_layernorm    layernorm;
    fn_rope         rope;
    fn_softmax      softmax;
    fn_silu         silu;
    fn_gelu         gelu;
    fn_mul          mul;
    fn_add          add;
    fn_scale        scale;
    fn_dot          dot;
    fn_dequantize   dequantize;
    fn_attention    attention;
    fn_kv_update    kv_update;
    fn_embed        embed;
    fn_softcap      softcap;
} ck;

/* Load a symbol, return 0 on success */
#define LOAD_SYM(field, sym_name) do { \
    ck.field = (typeof(ck.field))LIB_SYM(ck.lib, sym_name); \
    if (!ck.field) { kprintf("[CUDA] Missing symbol: %s\n", sym_name); return -1; } \
} while (0)

static int cuda_load_library(void) {
    /* Try several paths */
    const char *paths[] = {
        "cuda_kernels.dll",
        "build_host/cuda_kernels.dll",
        "./cuda_kernels.dll",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        ck.lib = LIB_OPEN(paths[i]);
        if (ck.lib) break;
    }

    if (!ck.lib) {
        kprintf("[CUDA] cuda_kernels.dll not found\n");
        return -1;
    }

    LOAD_SYM(init,         "ck_init");
    LOAD_SYM(shutdown,     "ck_shutdown");
    LOAD_SYM(device_count, "ck_device_count");
    LOAD_SYM(free_memory,  "ck_free_memory");
    LOAD_SYM(alloc,        "ck_alloc");
    LOAD_SYM(free,         "ck_free");
    LOAD_SYM(upload,       "ck_upload");
    LOAD_SYM(download,     "ck_download");
    LOAD_SYM(sync,         "ck_sync");
    LOAD_SYM(gemv,         "ck_gemv");
    LOAD_SYM(gemm,         "ck_gemm");
    LOAD_SYM(rmsnorm,      "ck_rmsnorm");
    LOAD_SYM(layernorm,    "ck_layernorm");
    LOAD_SYM(rope,         "ck_rope");
    LOAD_SYM(softmax,      "ck_softmax");
    LOAD_SYM(silu,         "ck_silu");
    LOAD_SYM(gelu,         "ck_gelu");
    LOAD_SYM(mul,          "ck_mul");
    LOAD_SYM(add,          "ck_add");
    LOAD_SYM(scale,        "ck_scale");
    LOAD_SYM(dot,          "ck_dot");
    LOAD_SYM(dequantize,   "ck_dequantize");
    LOAD_SYM(attention,    "ck_attention");
    LOAD_SYM(kv_update,    "ck_kv_update");
    LOAD_SYM(embed,        "ck_embed_lookup");
    LOAD_SYM(softcap,      "ck_softcap");

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backend vtable wrappers — delegate to loaded DLL
 * ════════════════════════════════════════════════════════════════════════ */

static void *cuda_alloc(uint64_t size)                    { return ck.alloc(size); }
static void  cuda_free(void *ptr)                         { ck.free(ptr); }
static int   cuda_upload(void *d, const void *s, uint64_t sz) { return ck.upload(d, s, sz); }
static int   cuda_download(void *d, const void *s, uint64_t sz) { return ck.download(d, s, sz); }
static void  cuda_sync(void)                              { ck.sync(); }

static void cuda_gemv(float *o, const void *w, const float *x,
                      int od, int id, ggml_type_t t) {
    ck.gemv(o, w, x, od, id, (int)t);
}

static void cuda_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K) {
    ck.gemm(C, A, B, M, N, K);
}

static void cuda_rmsnorm(float *o, const float *x, const float *w,
                         int d, float e) {
    ck.rmsnorm(o, x, w, d, e);
}

static void cuda_layernorm(float *o, const float *x, const float *w,
                           const float *b, int d, float e) {
    ck.layernorm(o, x, w, b, d, e);
}

static void cuda_rope(float *q, float *k, int hd, int nh,
                      int nkv, int p, float b, const float *f) {
    ck.rope(q, k, hd, nh, nkv, p, b, f);
}

static void cuda_softmax(float *x, int n)  { ck.softmax(x, n); }
static void cuda_silu(float *x, int n)     { ck.silu(x, n); }
static void cuda_gelu(float *x, int n)     { ck.gelu(x, n); }

static void cuda_mul(float *o, const float *a, const float *b, int n) {
    ck.mul(o, a, b, n);
}

static void cuda_add(float *o, const float *a, const float *b, int n) {
    ck.add(o, a, b, n);
}

static void cuda_scale(float *o, const float *x, float s, int n) {
    ck.scale(o, x, s, n);
}

static float cuda_dot(const float *a, const float *b, int n) {
    return ck.dot(a, b, n);
}

static void cuda_dequant(float *o, const void *d, int n, ggml_type_t t) {
    ck.dequantize(o, d, n, (int)t);
}

static void cuda_attention(float *o, const float *Q, const float *K, const float *V,
                           int nh, int nkv, int hd, int sl, float sc, float cap) {
    ck.attention(o, Q, K, V, nh, nkv, hd, sl, sc, cap);
}

static void cuda_kv_update(float *K, float *V, const float *Kn, const float *Vn,
                           int nkv, int hd, int p, int ms, int l) {
    ck.kv_update(K, V, Kn, Vn, nkv, hd, p, ms, l);
}

static void cuda_embed(float *o, const void *t, int id, int d, ggml_type_t ty) {
    ck.embed(o, t, id, d, (int)ty);
}

static void cuda_softcap_fn(float *x, int n, float c) {
    ck.softcap(x, n, c);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ════════════════════════════════════════════════════════════════════════ */

static int cuda_init(void) {
    if (cuda_load_library() != 0) return -1;
    int ret = ck.init();
    if (ret == 0) {
        int count = ck.device_count();
        uint64_t free_mb = ck.free_memory(0) / (1024 * 1024);
        kprintf("[CUDA] Initialized: %d device(s), %llu MB free\n",
                count, (unsigned long long)free_mb);
    }
    return ret;
}

static void cuda_shutdown(void) {
    if (ck.shutdown) ck.shutdown();
    if (ck.lib) { LIB_CLOSE(ck.lib); ck.lib = 0; }
}

static int cuda_device_count(void) {
    return ck.device_count ? ck.device_count() : 0;
}

static uint64_t cuda_free_memory_fn(int dev) {
    return ck.free_memory ? ck.free_memory(dev) : 0;
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
    .get_free_memory  = cuda_free_memory_fn,
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
        .softcap      = cuda_softcap_fn,
    },
};

#endif /* ENABLE_CUDA */
