/*
 * TensorOS Backend Registry + CPU Reference Implementation
 *
 * The CPU backend provides gold-standard reference implementations
 * of all tensor operations. Other backends must match CPU output
 * within specified numerical tolerances.
 */

#include "runtime/nn/backend.h"
#include <string.h>
#include <math.h>

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Backend Registry
 * ════════════════════════════════════════════════════════════════════════ */

static const backend_t *registered_backends[BACKEND_COUNT] = {0};
static backend_id_t active_backend_id = BACKEND_CPU;

void backend_register(const backend_t *be) {
    if (be && be->id < BACKEND_COUNT)
        registered_backends[be->id] = be;
}

const backend_t *backend_get(void) {
    return registered_backends[active_backend_id];
}

int backend_set(backend_id_t id) {
    if (id >= BACKEND_COUNT) return -1;
    if (!registered_backends[id]) return -2;
    active_backend_id = id;
    return 0;
}

const backend_t *backend_get_by_id(backend_id_t id) {
    if (id >= BACKEND_COUNT) return (void *)0;
    return registered_backends[id];
}

void backend_init_all(void) {
    /* CPU is always available */
    backend_register(&backend_cpu);
    if (backend_cpu.init) backend_cpu.init();

#ifdef ENABLE_CUDA
    backend_register(&backend_cuda);
    if (backend_cuda.init) backend_cuda.init();
#endif

#ifdef ENABLE_MLIR
    backend_register(&backend_mlir);
    if (backend_mlir.init) backend_mlir.init();
#endif
}

/* ═══════════════════════════════════════════════════════════════════════
 * CPU Backend Implementation
 * ════════════════════════════════════════════════════════════════════════ */

/* ─── FP16 helper ─── */
static float cpu_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 1; while (!(mant&0x400)){mant<<=1;exp--;} mant&=0x3FF;
               bits = sign | ((exp+127-15)<<23) | (mant<<13); }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant<<13);
    else bits = sign | ((exp+127-15)<<23) | (mant<<13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* ─── Memory ops ─── */
static void *cpu_alloc(uint64_t size) { return tensor_alloc(size); }
static void  cpu_free(void *ptr)      { tensor_free(ptr); }
static int   cpu_upload(void *d, const void *s, uint64_t sz) {
    kmemcpy(d, s, sz); return 0;
}
static int   cpu_download(void *d, const void *s, uint64_t sz) {
    kmemcpy(d, s, sz); return 0;
}
static void  cpu_sync(void) { /* CPU is synchronous */ }

/* ─── Dequantize ─── */
static void cpu_dequantize(float *out, const void *data, int n, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        kmemcpy(out, data, n * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)data;
        for (int i = 0; i < n; i++) out[i] = cpu_fp16_to_f32(h[i]);
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_t;
        int nb = n / 32;
        const q4_0_t *blocks = (const q4_0_t *)data;
        for (int b = 0; b < nb; b++) {
            float d = cpu_fp16_to_f32(blocks[b].d);
            for (int j = 0; j < 16; j++) {
                int lo = (blocks[b].qs[j] & 0x0F) - 8;
                int hi = (blocks[b].qs[j] >> 4) - 8;
                out[b*32 + j]      = d * (float)lo;
                out[b*32 + j + 16] = d * (float)hi;
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        typedef struct { uint16_t d; int8_t qs[32]; } q8_0_t;
        int nb = n / 32;
        const q8_0_t *blocks = (const q8_0_t *)data;
        for (int b = 0; b < nb; b++) {
            float d = cpu_fp16_to_f32(blocks[b].d);
            for (int j = 0; j < 32; j++)
                out[b*32 + j] = d * (float)blocks[b].qs[j];
        }
        return;
    }
    /* Unsupported type: zero fill */
    kmemset(out, 0, n * sizeof(float));
}

/* ─── Dot product ─── */
static float cpu_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ─── GEMV (scalar reference) ─── */
static void cpu_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t weight_type) {
    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)weight;
        for (int i = 0; i < out_dim; i++) {
            float s = 0.0f;
            for (int j = 0; j < in_dim; j++)
                s += w[i * in_dim + j] * x[j];
            out[i] = s;
        }
        return;
    }
    /* Quantized: dequant row then dot */
    float *row_buf = (float *)cpu_alloc(in_dim * sizeof(float));
    if (!row_buf) return;
    uint64_t row_bytes = 0;
    if (weight_type == GGML_TYPE_Q4_0) row_bytes = (in_dim / 32) * 18;
    else if (weight_type == GGML_TYPE_Q8_0) row_bytes = (in_dim / 32) * 34;
    else if (weight_type == GGML_TYPE_F16) row_bytes = in_dim * 2;
    else row_bytes = in_dim * 4;

    const uint8_t *base = (const uint8_t *)weight;
    for (int i = 0; i < out_dim; i++) {
        cpu_dequantize(row_buf, base + (uint64_t)i * row_bytes, in_dim, weight_type);
        float s = 0.0f;
        for (int j = 0; j < in_dim; j++) s += row_buf[j] * x[j];
        out[i] = s;
    }
    cpu_free(row_buf);
}

/* ─── GEMM ─── */
static void cpu_gemm(float *C, const float *A, const float *B,
                     int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] += s;
        }
}

/* ─── RMSNorm ─── */
static void cpu_rmsnorm(float *out, const float *x, const float *w,
                        int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * ss * w[i];
}

/* ─── LayerNorm ─── */
static void cpu_layernorm(float *out, const float *x, const float *w,
                          const float *bias, int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv * w[i];
        if (bias) out[i] += bias[i];
    }
}

/* ─── RoPE ─── */
static void cpu_rope(float *q, float *k, int head_dim, int n_heads,
                     int n_kv_heads, int pos, float base,
                     const float *freq_factors) {
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]     = q0 * cs - q1 * sn;
            qh[i + 1] = q0 * sn + q1 * cs;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]     = k0 * cs - k1 * sn;
            kh[i + 1] = k0 * sn + k1 * cs;
        }
    }
}

/* ─── Softmax ─── */
static void cpu_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ─── SiLU ─── */
static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ─── GELU ─── */
static void cpu_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* ─── Element-wise ops ─── */
static void cpu_mul(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
static void cpu_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void cpu_scale(float *out, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
}

/* ─── Attention ─── */
static void cpu_attention(float *out, const float *Q,
                          const float *K_cache, const float *V_cache,
                          int n_heads, int n_kv_heads, int head_dim,
                          int seq_len, float scale, float softcap) {
    int kv_group = n_heads / n_kv_heads;
    /* Allocate scratch for attention scores */
    float *scores = (float *)cpu_alloc(seq_len * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_group;
        const float *qh = Q + h * head_dim;

        /* Compute attention scores: Q·K^T / sqrt(d) */
        for (int t = 0; t < seq_len; t++) {
            const float *kh = K_cache + (kv_h * seq_len + t) * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += qh[d] * kh[d];
            s *= scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            scores[t] = s;
        }

        /* Softmax */
        cpu_softmax(scores, seq_len);

        /* Weighted sum of V */
        float *oh = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float v = 0.0f;
            for (int t = 0; t < seq_len; t++)
                v += scores[t] * V_cache[(kv_h * seq_len + t) * head_dim + d];
            oh[d] = v;
        }
    }
    cpu_free(scores);
}

/* ─── KV cache update ─── */
static void cpu_kv_update(float *K_cache, float *V_cache,
                          const float *K_new, const float *V_new,
                          int n_kv_heads, int head_dim, int pos,
                          int max_seq, int layer) {
    (void)layer;
    for (int h = 0; h < n_kv_heads; h++) {
        kmemcpy(K_cache + (h * max_seq + pos) * head_dim,
                K_new + h * head_dim, head_dim * sizeof(float));
        kmemcpy(V_cache + (h * max_seq + pos) * head_dim,
                V_new + h * head_dim, head_dim * sizeof(float));
    }
}

/* ─── Embedding lookup ─── */
static void cpu_embed_lookup(float *out, const void *embd_table,
                             int token_id, int dim, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        const float *t = (const float *)embd_table;
        kmemcpy(out, t + (uint64_t)token_id * dim, dim * sizeof(float));
    } else {
        /* Quantized embedding: dequant the row */
        uint64_t row_bytes = 0;
        if (type == GGML_TYPE_Q4_0) row_bytes = (dim / 32) * 18;
        else if (type == GGML_TYPE_Q8_0) row_bytes = (dim / 32) * 34;
        else if (type == GGML_TYPE_F16) row_bytes = dim * 2;
        else row_bytes = dim * 4;
        const uint8_t *base = (const uint8_t *)embd_table;
        cpu_dequantize(out, base + (uint64_t)token_id * row_bytes, dim, type);
    }
}

/* ─── Softcap ─── */
static void cpu_softcap(float *x, int n, float cap) {
    for (int i = 0; i < n; i++)
        x[i] = cap * tanhf(x[i] / cap);
}

/* ─── Init/Shutdown ─── */
static int  cpu_init(void) { return 0; }
static void cpu_shutdown(void) {}
static int  cpu_device_count(void) { return 1; }
static uint64_t cpu_free_memory(int dev) {
    (void)dev;
    return tensor_mm_free_bytes();
}

/* ═══════════════════════════════════════════════════════════════════════
 * CPU Backend Definition
 * ════════════════════════════════════════════════════════════════════════ */

const backend_t backend_cpu = {
    .id   = BACKEND_CPU,
    .name = "cpu",
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .get_device_count = cpu_device_count,
    .get_free_memory  = cpu_free_memory,
    .mem = {
        .alloc    = cpu_alloc,
        .free     = cpu_free,
        .upload   = cpu_upload,
        .download = cpu_download,
        .sync     = cpu_sync,
    },
    .compute = {
        .gemv         = cpu_gemv,
        .gemm         = cpu_gemm,
        .rmsnorm      = cpu_rmsnorm,
        .layernorm    = cpu_layernorm,
        .rope         = cpu_rope,
        .softmax      = cpu_softmax,
        .silu         = cpu_silu,
        .gelu         = cpu_gelu,
        .mul          = cpu_mul,
        .add          = cpu_add,
        .scale        = cpu_scale,
        .dot          = cpu_dot,
        .dequantize   = cpu_dequantize,
        .attention    = cpu_attention,
        .kv_update    = cpu_kv_update,
        .embed_lookup = cpu_embed_lookup,
        .softcap      = cpu_softcap,
    },
};
