/* =============================================================================
 * TensorOS - Real LLM Inference Engine Implementation
 *
 * Complete pipeline: disk → GGUF parse → tensor map → tokenize → forward →
 * generate text → evaluate mathematical reasoning.
 *
 * Supported architectures (all LLaMA-family):
 *   - Qwen2 / Qwen2.5  (qwen2)
 *   - LLaMA / LLaMA2/3  (llama)
 *   - Gemma / Gemma2    (gemma)
 *   - SmolLM / SmolLM2  (llama)
 *   - TinyLlama         (llama)
 *   - Mistral           (llama)
 *   - Phi-3             (phi3)
 *
 * Quantization formats: Q4_0, Q8_0, F16, F32
 * All running bare-metal with SSE2 SIMD, zero OS overhead.
 * =============================================================================*/

#include "runtime/nn/llm.h"
#include "runtime/nn/backend.h"
#include "runtime/nn/model_meta.h"
#include "runtime/nn/tensor_bridge.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/perf.h"
#include "runtime/nn/gguf.h"
#include "runtime/jit/x86_jit.h"
#include "kernel/security/crypto.h"
#ifndef __aarch64__
#include "kernel/drivers/blk/virtio_blk.h"
#include "kernel/core/smp.h"
#endif

/* ─────────────────────────────────────────────────────────────────────────── */
/*  SSE2 SIMD type                                                             */
/* ─────────────────────────────────────────────────────────────────────────── */
typedef float v4f __attribute__((vector_size(16)));

#ifndef __aarch64__
/* AVX2 8-wide float vector */
typedef float v8f __attribute__((vector_size(32)));
#include "kernel/core/cpu_features.h"
#endif

/* Forward declarations */
static void llm_build_hash_table(const llm_model_t *m);
static uint64_t llm_row_bytes(int in_dim, ggml_type_t type);

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Static Allocations                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/* GGUF parsing context */
static gguf_ctx_t llm_gguf_ctx;

/* Model descriptor */
static llm_model_t llm_model;

/* Inference serialization: prevent concurrent use of static buffers */
static volatile int llm_inference_active = 0;

/* Streaming callback: called per-token during generation (NULL = disabled) */
typedef void (*llm_token_cb_t)(const char *text, int len, void *userdata);
static llm_token_cb_t llm_stream_cb   = (llm_token_cb_t)0;
static void          *llm_stream_cb_ud = (void *)0;
static llm_backend_t  llm_backend      = LLM_BACKEND_CPU;

/* Tensor bridge for hidden-state injection / daisy-chaining */
static tensor_bridge_t llm_bridge;

void llm_set_stream_cb(llm_token_cb_t cb, void *userdata) {
    llm_stream_cb    = cb;
    llm_stream_cb_ud = userdata;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Dynamic scratch arena (allocated from tensor heap after model load)         */
/* ─────────────────────────────────────────────────────────────────────────── */

static float *llm_kv_k;        /* [n_layers * max_seq * n_kv_heads * head_dim] */
static float *llm_kv_v;

static float *llm_x;           /* [dim]    hidden state */
static float *llm_xn;          /* [dim]    normalized */
static float *llm_q;           /* [dim]    query */
static float *llm_k_buf;       /* [dim]    key (current token) */
static float *llm_v_buf;       /* [dim]    value (current token) */
static float *llm_attn_out;    /* [dim]    attention output */
static float *llm_ffn_g;       /* [ff_dim] FFN gate */
static float *llm_ffn_u;       /* [ff_dim] FFN up */
static float *llm_ffn_d;       /* [dim]    FFN down */
static float *llm_head_buf;    /* [dim]    LM head scratch */
static float *llm_attn_scores; /* [max_seq] attention scores */
static float *llm_logits;      /* [vocab_size] output logits */
static int   *llm_tokens;      /* [max_tokens] token buffer */
static float *llm_rope_freqs_buf; /* [head_dim/2] precomputed RoPE */
static float *llm_iswa_per_layer; /* [iswa_n_embd * n_layers] per-layer embeddings */
static float *llm_iswa_tmp;       /* [iswa_n_embd] ISWA gating scratch */

/* Sizes cached from the loaded model (for bounds checking) */
static int llm_alloc_dim;
static int llm_alloc_ff;
static int llm_alloc_seq;
static int llm_alloc_vocab;
static int llm_alloc_tokens;
static int llm_alloc_kv_floats;

/* Allocate all inference scratch buffers from the tensor heap.
 * Called once after the model's dimensions are known. */
static int llm_alloc_scratch(const llm_model_t *m)
{
    int dim      = m->dim;
    int ff       = m->ff_dim;
    /* Find max ff_dim across all layers (Gemma4: layers 15+ have doubled FFN) */
    for (int i = 0; i < m->n_layers; i++) {
        if (m->layers[i].ff_dim_layer > ff) ff = m->layers[i].ff_dim_layer;
    }
    int seq      = m->max_seq;
    int vocab    = m->vocab_size;
    int hd       = m->head_dim;
    int q_dim    = m->n_heads * hd;     /* Q/attn output dimension (may differ from dim) */
    int kv_dim   = m->n_kv_heads * hd;  /* K/V dimension per token */
    int kv_total = m->n_layers * seq * m->n_kv_heads * hd;
    int max_tok  = seq * 2;  /* generous token buffer */

    /* Compute total bytes needed (64-byte aligned per buffer) */
    #define ALIGN64(x) (((x) + 63) & ~63ULL)
    uint64_t total = 0;
    total += ALIGN64((uint64_t)kv_total * sizeof(float)) * 2; /* k+v cache */
    total += ALIGN64((uint64_t)dim * sizeof(float)) * 3;      /* x, xn, ffn_d */
    total += ALIGN64((uint64_t)q_dim * sizeof(float)) * 2;    /* q, attn_out */
    total += ALIGN64((uint64_t)kv_dim * sizeof(float)) * 2;   /* k_buf, v_buf */
    total += ALIGN64((uint64_t)hd * sizeof(float));            /* head_buf */
    total += ALIGN64((uint64_t)ff * sizeof(float)) * 2;       /* ffn_g, ffn_u */
    total += ALIGN64((uint64_t)seq * sizeof(float));           /* attn_scores */
    total += ALIGN64((uint64_t)vocab * sizeof(float));         /* logits */
    total += ALIGN64((uint64_t)max_tok * sizeof(int));         /* tokens */
    total += ALIGN64((uint64_t)(hd / 2) * sizeof(float));     /* rope freqs */
    /* ISWA scratch */
    int iswa_n_ = m->iswa_n_embd > 0 ? m->iswa_n_embd : 1;
    total += ALIGN64((uint64_t)(iswa_n_ * m->n_layers) * sizeof(float)); /* per_layer */
    total += ALIGN64((uint64_t)iswa_n_ * sizeof(float));                  /* tmp */

    kprintf("[LLM] Allocating %lu MB scratch arena\n",
            (unsigned long)(total / (1024 * 1024)));

    uint8_t *arena = (uint8_t *)tensor_alloc(total);
    if (!arena) {
        kprintf("[LLM] ERROR: Failed to allocate %lu MB for inference buffers\n",
                (unsigned long)(total / (1024 * 1024)));
        return -1;
    }
    kmemset(arena, 0, total);

    /* Carve out buffers from the arena */
    #define ARENA_NEXT(ptr, type, count) do { \
        ptr = (type *)arena; \
        arena += ALIGN64((uint64_t)(count) * sizeof(type)); \
    } while (0)

    ARENA_NEXT(llm_kv_k,        float, kv_total);
    ARENA_NEXT(llm_kv_v,        float, kv_total);
    ARENA_NEXT(llm_x,           float, dim);
    ARENA_NEXT(llm_xn,          float, dim);
    ARENA_NEXT(llm_q,           float, q_dim);
    ARENA_NEXT(llm_k_buf,       float, kv_dim);
    ARENA_NEXT(llm_v_buf,       float, kv_dim);
    ARENA_NEXT(llm_attn_out,    float, q_dim);
    ARENA_NEXT(llm_ffn_g,       float, ff);
    ARENA_NEXT(llm_ffn_u,       float, ff);
    ARENA_NEXT(llm_ffn_d,       float, dim);
    ARENA_NEXT(llm_head_buf,    float, hd);
    ARENA_NEXT(llm_attn_scores, float, seq);
    ARENA_NEXT(llm_logits,      float, vocab);
    ARENA_NEXT(llm_tokens,      int,   max_tok);
    ARENA_NEXT(llm_rope_freqs_buf, float, hd / 2);
    /* ISWA scratch (Gemma4) */
    int iswa_n = m->iswa_n_embd > 0 ? m->iswa_n_embd : 1;
    int iswa_total = iswa_n * m->n_layers;
    ARENA_NEXT(llm_iswa_per_layer, float, iswa_total);
    ARENA_NEXT(llm_iswa_tmp,       float, iswa_n);

    #undef ARENA_NEXT
    #undef ALIGN64

    /* Cache sizes for runtime bounds checks */
    llm_alloc_dim      = dim;
    llm_alloc_ff       = ff;
    llm_alloc_seq      = seq;
    llm_alloc_vocab    = vocab;
    llm_alloc_tokens   = max_tok;
    llm_alloc_kv_floats = kv_total;

    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Float16 (half precision) conversion                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Positive/negative zero */
            float f;
            kmemcpy(&f, &sign, 4);
            return f;
        }
        /* Denormalized — normalize it */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* Inf / NaN */
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f;
        kmemcpy(&f, &bits, 4);
        return f;
    }

    exp = exp + 127 - 15;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

/* BF16 to FP32: just shift left by 16 (BF16 is upper 16 bits of FP32) */
static float bf16_to_fp32(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Math helpers                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

static float llm_fabsf(float x) { return x < 0 ? -x : x; }

static float llm_sqrtf(float x)
{
    float r;
#if defined(__aarch64__)
    __asm__("fsqrt %s0, %s1" : "=w"(r) : "w"(x));
#else
    __asm__("sqrtss %1, %0" : "=x"(r) : "x"(x));
#endif
    return r;
}

static float llm_expf(float x)
{
    if (x > 88.7f) return 3.4028235e38f;
    if (x < -88.7f) return 0.0f;
    /* Range reduction: exp(x) = 2^n * exp(r), r in [-ln2/2, ln2/2]
     * n = round(x / ln2), r = x - n * ln2 */
    float n = (float)(int)(x * 1.4426950408f + (x >= 0 ? 0.5f : -0.5f)); /* round(x/ln2) */
    float r = x - n * 0.693145751953125f;   /* Cody-Waite high part */
    r -= n * 1.428606765330187e-06f;         /* Cody-Waite low part  */
    /* Minimax polynomial for exp(r), r in [-ln2/2, ln2/2], max err < 2e-7 */
    float r2 = r * r;
    float r3 = r2 * r;
    float p = 1.0f + r + r2 * 0.5f + r3 * 0.16666666f
            + r2 * r2 * 0.041666666f + r2 * r3 * 0.008333333f
            + r3 * r3 * 0.001388889f;
    /* Reconstruct: multiply by 2^n via IEEE 754 bit manipulation */
    int ni = (int)n;
    union { float f; unsigned int u; } scale;
    scale.u = (unsigned int)(127 + ni) << 23;
    return p * scale.f;
}

/* Accurate sinf/cosf using Cody-Waite range reduction + minimax polynomials.
 * Reduces to [-pi/4, pi/4] using quadrant decomposition where Taylor series
 * converges rapidly (max |x| = 0.785, so x^8/8! ~ 5e-7). */
static float llm_sinf(float x)
{
    /* Range reduction to [-pi, pi] */
    float pi  = 3.14159265358979f;
    float hpi = 1.57079632679490f;
    float itp = 0.15915494309190f; /* 1/(2*pi) */
    x = x - (float)(int)(x * itp + (x >= 0 ? 0.5f : -0.5f)) * (2.0f * pi);

    /* Quadrant decomposition: reduce to [-pi/4, pi/4] */
    int negate = 0;
    if (x < 0) { x = -x; negate = 1; }
    /* x now in [0, pi] */
    int use_cos = 0;
    if (x > hpi) { x = pi - x; }       /* mirror: sin(pi-x) = sin(x) */
    if (x > hpi * 0.5f) { x = hpi - x; use_cos = 1; } /* sin(pi/2-x) = cos(x) */

    float x2 = x * x;
    float r;
    if (use_cos) {
        /* cos(x) = 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 */
        r = 1.0f - x2 * (0.5f - x2 * (0.041666666f - x2 * (0.001388889f - x2 * 2.48016e-05f)));
    } else {
        /* sin(x) = x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 */
        r = x * (1.0f - x2 * (0.16666667f - x2 * (0.008333333f - x2 * (0.000198413f - x2 * 2.7558e-06f))));
    }
    return negate ? -r : r;
}

static float llm_cosf(float x)
{
    /* cos(x) = sin(x + pi/2) */
    return llm_sinf(x + 1.57079632679490f);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Quantized Tensor Operations (GGML format)                                  */
/*                                                                             */
/*  Q4_0: 32 elements → 2 bytes (fp16 scale) + 16 bytes (nibbles) = 18 bytes  */
/*  Q8_0: 32 elements → 2 bytes (fp16 scale) + 32 bytes (int8)    = 34 bytes  */
/*  F16:  1 element  → 2 bytes (half float)                                    */
/*  F32:  1 element  → 4 bytes (float)                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/* GGML Q4_0 block: {fp16 scale, uint8[16]} = 18 bytes for 32 elements */
typedef struct { uint16_t d; uint8_t qs[16]; } __attribute__((packed)) ggml_q4_0_t;

/* GGML Q4_1 block: {fp16 scale, fp16 min, uint8[16]} = 20 bytes for 32 elements
 * Dequant: val = d * nibble + m  (unsigned nibble 0-15, m is minimum value) */
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } __attribute__((packed)) ggml_q4_1_t;

/* GGML Q8_0 block: {fp16 scale, int8[32]}  = 34 bytes for 32 elements */
typedef struct { uint16_t d; int8_t  qs[32]; } __attribute__((packed)) ggml_q8_0_t;

/* GGML Q6_K super-block: 256 elements in 210 bytes
 * ql[128] — lower 4 bits of each 6-bit quant (2 quants per byte for first half,
 *           then 2 quants per byte for second half)
 * qh[64]  — upper 2 bits of each quant (4 quants packed per byte)
 * scales[16] — int8 scale per 16-element sub-block
 * d — fp16 super-block scale
 */
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} __attribute__((packed)) ggml_q6_k_t;

/* Dot product: Q4_0 block · float[32] — SSE2 optimized */
static float q4_0_dot32(const ggml_q4_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    const uint8_t *qs = block->qs;

#ifndef __aarch64__
    /* SSE2: standard GGML layout — lo nibbles map to x[0..15], hi to x[16..31] */
    v4f acc_lo0 = {0, 0, 0, 0};
    v4f acc_lo1 = {0, 0, 0, 0};
    v4f acc_hi0 = {0, 0, 0, 0};
    v4f acc_hi1 = {0, 0, 0, 0};

    for (int j = 0; j < 16; j += 4) {
        uint8_t b0 = qs[j], b1 = qs[j+1], b2 = qs[j+2], b3 = qs[j+3];
        v4f qlo = {(float)((int)(b0 & 0xF) - 8), (float)((int)(b1 & 0xF) - 8),
                   (float)((int)(b2 & 0xF) - 8), (float)((int)(b3 & 0xF) - 8)};
        v4f qhi = {(float)((int)(b0 >> 4) - 8), (float)((int)(b1 >> 4) - 8),
                   (float)((int)(b2 >> 4) - 8), (float)((int)(b3 >> 4) - 8)};
        v4f xlo = *(const v4f *)(x + j);       /* x[j..j+3] */
        v4f xhi = *(const v4f *)(x + j + 16);  /* x[j+16..j+19] */
        acc_lo0 += qlo * xlo;
        acc_hi0 += qhi * xhi;
    }

    v4f acc = acc_lo0 + acc_hi0;
    union { v4f v; float f[4]; } u = { .v = acc };
    return (u.f[0] + u.f[1] + u.f[2] + u.f[3]) * d;
#else
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        uint8_t packed = qs[j];
        int lo = (int)(packed & 0x0F) - 8;
        int hi = (int)(packed >> 4)   - 8;
        sum += (float)lo * x[j] + (float)hi * x[j + 16];
    }
    return sum * d;
#endif
}

/* Dot product: Q4_1 block · float[32]
 * Each nibble is unsigned (0-15), dequant: d * nibble + m
 * dot = sum_i (d * nibble_i + m) * x_i = d * sum(nibble_i * x_i) + m * sum(x_i) */
static float q4_1_dot32(const ggml_q4_1_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float m = fp16_to_fp32(block->m);
    const uint8_t *qs = block->qs;

#ifndef __aarch64__
    /* Standard GGML layout: lo nibbles → x[0..15], hi nibbles → x[16..31] */
    v4f qxacc_lo = {0, 0, 0, 0};
    v4f qxacc_hi = {0, 0, 0, 0};
    v4f xacc_lo = {0, 0, 0, 0};
    v4f xacc_hi = {0, 0, 0, 0};

    for (int j = 0; j < 16; j += 4) {
        uint8_t b0 = qs[j], b1 = qs[j+1], b2 = qs[j+2], b3 = qs[j+3];
        v4f qlo = {(float)(b0 & 0xF), (float)(b1 & 0xF),
                   (float)(b2 & 0xF), (float)(b3 & 0xF)};
        v4f qhi = {(float)(b0 >> 4), (float)(b1 >> 4),
                   (float)(b2 >> 4), (float)(b3 >> 4)};
        v4f xlo = *(const v4f *)(x + j);       /* x[j..j+3] */
        v4f xhi = *(const v4f *)(x + j + 16);  /* x[j+16..j+19] */
        qxacc_lo += qlo * xlo;
        qxacc_hi += qhi * xhi;
        xacc_lo += xlo;
        xacc_hi += xhi;
    }

    v4f qx = qxacc_lo + qxacc_hi;
    v4f xs = xacc_lo + xacc_hi;
    union { v4f v; float f[4]; } uq = { .v = qx };
    union { v4f v; float f[4]; } ux = { .v = xs };
    float qxsum = uq.f[0] + uq.f[1] + uq.f[2] + uq.f[3];
    float xsum  = ux.f[0] + ux.f[1] + ux.f[2] + ux.f[3];
    return d * qxsum + m * xsum;
#else
    float qxsum = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 16; j++) {
        float lo = (float)(qs[j] & 0xF);
        float hi = (float)(qs[j] >> 4);
        qxsum += lo * x[j] + hi * x[j + 16];
        xsum  += x[j] + x[j + 16];
    }
    return d * qxsum + m * xsum;
#endif
}

/* Dot product: Q8_0 block · float[32] — SSE2 optimized */
static float q8_0_dot32(const ggml_q8_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    const int8_t *qs = block->qs;

#ifndef __aarch64__
    /* SSE2: dual v4f accumulators, 4 iterations of 8 elements */
    v4f acc0 = {0, 0, 0, 0};
    v4f acc1 = {0, 0, 0, 0};

    for (int j = 0; j < 32; j += 8) {
        v4f q0 = {(float)qs[j], (float)qs[j+1], (float)qs[j+2], (float)qs[j+3]};
        v4f q1 = {(float)qs[j+4], (float)qs[j+5], (float)qs[j+6], (float)qs[j+7]};
        v4f x0 = *(const v4f *)(x + j);
        v4f x1 = *(const v4f *)(x + j + 4);
        acc0 += q0 * x0;
        acc1 += q1 * x1;
    }

    v4f acc = acc0 + acc1;
    union { v4f v; float f[4]; } u = { .v = acc };
    return (u.f[0] + u.f[1] + u.f[2] + u.f[3]) * d;
#else
    float sum = 0.0f;
    for (int j = 0; j < 32; j++)
        sum += (float)qs[j] * x[j];
    return sum * d;
#endif
}

/* Q6_K super-block dot product: 256 elements
 * Dequant: val = d * scales[sub] * q6 where q6 is 6-bit signed (-32..31)
 * Lower 4 bits in ql[], upper 2 bits in qh[]
 * Layout follows ggml reference: 2 halves of 128, each with 4 groups of 32.
 * ql[0..63]: lower nibbles for quants at offsets 0..31 (lo nibble) and 64..95 (hi nibble)
 *            in the first half, ql[32..63] for offsets 32..63 and 96..127
 * ql[64..127]: same pattern for second half (quants 128..255)
 * qh[0..31]: 2-bit high parts for first half (4 quants per byte, shifts 0/2/4/6)
 * qh[32..63]: same for second half
 * scales[0..15]: int8 scales, 2 per group of 32: sc[0]=grp0, sc[2]=grp1, etc. */
static float q6_k_dot256(const ggml_q6_k_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    /* Process in 2 halves of 128 quants each */
    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_h = ql + half * 64;
        const uint8_t *qh_h = qh + half * 32;
        const int8_t *sc_h = sc + half * 8;
        const float *x_h = x + half * 128;

        for (int l = 0; l < 32; l++) {
            /* 4 quants per iteration from the 4 groups of 32 in this half */
            int q0 = (int)( ql_h[l]      & 0xF) | (int)(((qh_h[l] >> 0) & 3) << 4);
            int q1 = (int)( ql_h[l + 32] & 0xF) | (int)(((qh_h[l] >> 2) & 3) << 4);
            int q2 = (int)( ql_h[l]      >>  4) | (int)(((qh_h[l] >> 4) & 3) << 4);
            int q3 = (int)( ql_h[l + 32] >>  4) | (int)(((qh_h[l] >> 6) & 3) << 4);

            /* Scale mapping: each group of 32 has 2 sub-blocks of 16.
             * sc_h[0] for l<16 in group 0, sc_h[1] for l>=16 in group 0, etc. */
            int si0 = l / 16;        /* 0 or 1 → scale index within group 0 */
            sum += (float)sc_h[0 + si0] * (float)(q0 - 32) * x_h[l];
            sum += (float)sc_h[2 + si0] * (float)(q1 - 32) * x_h[l + 32];
            sum += (float)sc_h[4 + si0] * (float)(q2 - 32) * x_h[l + 64];
            sum += (float)sc_h[6 + si0] * (float)(q3 - 32) * x_h[l + 96];
        }
    }

    return sum * d;
}

/* Generic vector dot product: quantized weight row · float input
 * Returns the dot product of n elements.
 * The weight pointer must be to the start of the row's block data. */
static float llm_vec_dot(const void *weight, const float *x, int n, ggml_type_t type)
{
    float sum = 0.0f;
    int nb = n / 32;  /* number of blocks (both Q4_0 and Q8_0 use block_size=32) */

    switch (type) {
    case GGML_TYPE_Q4_0: {
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q4_0_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_Q4_1: {
        const ggml_q4_1_t *blocks = (const ggml_q4_1_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q4_1_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q8_0_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *f16 = (const uint16_t *)weight;
#ifndef __aarch64__
        v4f acc = {0, 0, 0, 0};
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            v4f w = {fp16_to_fp32(f16[i]), fp16_to_fp32(f16[i+1]),
                     fp16_to_fp32(f16[i+2]), fp16_to_fp32(f16[i+3])};
            v4f vx = *(const v4f *)(x + i);
            acc += w * vx;
        }
        union { v4f v; float f[4]; } u = { .v = acc };
        sum += u.f[0] + u.f[1] + u.f[2] + u.f[3];
        for (; i < n; i++)
            sum += fp16_to_fp32(f16[i]) * x[i];
#else
        for (int i = 0; i < n; i++)
            sum += fp16_to_fp32(f16[i]) * x[i];
#endif
        break;
    }
    case GGML_TYPE_F32: {
        const float *f32 = (const float *)weight;
#ifndef __aarch64__
        v4f acc0 = {0, 0, 0, 0};
        v4f acc1 = {0, 0, 0, 0};
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v4f w0 = *(const v4f *)(f32 + i);
            v4f w1 = *(const v4f *)(f32 + i + 4);
            v4f x0 = *(const v4f *)(x + i);
            v4f x1 = *(const v4f *)(x + i + 4);
            acc0 += w0 * x0;
            acc1 += w1 * x1;
        }
        v4f a = acc0 + acc1;
        union { v4f v; float f[4]; } u = { .v = a };
        sum += u.f[0] + u.f[1] + u.f[2] + u.f[3];
        for (; i < n; i++)
            sum += f32[i] * x[i];
#else
        for (int i = 0; i < n; i++)
            sum += f32[i] * x[i];
#endif
        break;
    }
    case GGML_TYPE_Q6_K: {
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)weight;
        int nsb = n / 256;  /* Q6_K uses super-blocks of 256 elements */
        for (int b = 0; b < nsb; b++)
            sum += q6_k_dot256(&blocks[b], x + b * 256);
        break;
    }
    case GGML_TYPE_BF16: {
        const uint16_t *bf = (const uint16_t *)weight;
        for (int i = 0; i < n; i++)
            sum += bf16_to_fp32(bf[i]) * x[i];
        break;
    }
    default:
        break;
    }
    return sum;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  AVX2+FMA Optimized Dot Products (2× throughput vs SSE2)                    */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__

__attribute__((target("avx2,fma")))
static inline v8f v8f_setzero(void) { return (v8f){0,0,0,0,0,0,0,0}; }

__attribute__((target("avx2,fma")))
static inline v8f v8f_set1(float x) { return (v8f){x,x,x,x,x,x,x,x}; }

__attribute__((target("avx2,fma")))
static inline float v8f_reduce(v8f v) {
    union { v8f vec; float f[8]; } u;
    u.vec = v;
    return (u.f[0]+u.f[1]+u.f[2]+u.f[3])+(u.f[4]+u.f[5]+u.f[6]+u.f[7]);
}

/* AVX2 min/max per-element */
__attribute__((target("avx2,fma")))
static inline v8f v8f_min(v8f a, v8f b) {
    v8f r; __asm__("vminps %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
    return r;
}
__attribute__((target("avx2,fma")))
static inline v8f v8f_max(v8f a, v8f b) {
    v8f r; __asm__("vmaxps %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
    return r;
}

/* AVX2 vectorized exp: range reduction exp(x)=2^n * exp(r), minimax poly */
__attribute__((target("avx2,fma")))
static inline v8f v8f_expf(v8f x) {
    x = v8f_max(x, v8f_set1(-86.0f));
    x = v8f_min(x, v8f_set1(86.0f));
    /* n = round(x / ln2) */
    v8f nf;
    __asm__("vroundps $0, %1, %0" : "=x"(nf) : "x"(x * v8f_set1(1.4426950408f)));
    /* r = x - n*ln2 (Cody-Waite two-part) */
    v8f r = x - nf * v8f_set1(0.693145751953125f);
    r = r - nf * v8f_set1(1.428606765330187e-06f);
    /* Polynomial: exp(r) for r in [-ln2/2, ln2/2] */
    v8f r2 = r * r;
    v8f r3 = r2 * r;
    v8f p = v8f_set1(1.0f) + r + r2 * v8f_set1(0.5f)
          + r3 * v8f_set1(0.16666666f) + r2 * r2 * v8f_set1(0.041666666f)
          + r2 * r3 * v8f_set1(0.008333333f) + r3 * r3 * v8f_set1(0.001388889f);
    /* 2^n: convert n to int, add 127 bias, shift into IEEE754 exponent field */
    union { int i[8]; v8f v; } ni, scale;
    __asm__("vcvtps2dq %1, %0" : "=x"(ni.v) : "x"(nf));
    for (int j = 0; j < 8; j++)
        scale.i[j] = (ni.i[j] + 127) << 23;
    return p * scale.v;
}

/* SIMD int8→float: vpmovsxbd (sign-extend 8×i8→8×i32) + vcvtdq2ps
 * Replaces 8 scalar (float)cast operations with 2 SIMD instructions */
__attribute__((target("avx2,fma")))
static inline v8f v8f_load_i8(const int8_t *src)
{
    v8f result;
    __asm__ volatile(
        "vpmovsxbd (%[s]), %[r]\n\t"
        "vcvtdq2ps %[r], %[r]"
        : [r] "=x"(result) : [s] "r"(src)
    );
    return result;
}

/* Fast branchless fp16→fp32 for quantization scales (normal values only).
 * ~8 ops vs ~20 for the full conversion. Safe for Q8_0/Q4_0 scale values
 * which are always normal fp16 (never denorm/inf/nan in practice). */
static inline float fp16_to_fp32_fast(uint16_t h)
{
    uint32_t bits = ((uint32_t)(h & 0x8000) << 16) |
                    ((((uint32_t)(h >> 10) & 0x1F) + 112u) << 23) |
                    ((uint32_t)(h & 0x3FF) << 13);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

/* ─── Integer Q8×Q8 dot product (avoids ALL int→float conversion) ─── */

/* Temporary quantized input block (float scale, not fp16, for speed) */
typedef struct { float d; int32_t isum; int8_t qs[32]; } q8_input_t;

/* Pre-quantized input buffer — max 512 blocks = 16384 elements (covers 9216-dim FFN) */
#define Q8_MAX_BLOCKS 512
static q8_input_t llm_xq_buf[Q8_MAX_BLOCKS];

/* AVX2 quantize 32 floats → Q8_0 block (8-wide SIMD abs-max + rounding) */
__attribute__((target("avx2,fma")))
static void q8_quantize_block(q8_input_t *out, const float *x)
{
    /* Find absmax with 8-wide AVX2 */
    v8f sign_mask = v8f_set1(-0.0f);
    v8f mx = v8f_setzero();
    for (int i = 0; i < 32; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f av;
        __asm__("vandnps %2, %1, %0" : "=x"(av) : "x"(sign_mask), "x"(v)); /* abs */
        mx = v8f_max(mx, av);
    }
    /* Horizontal max of 8 lanes */
    union { v8f vec; float f[8]; } mu = { .vec = mx };
    float amax = mu.f[0];
    for (int j = 1; j < 8; j++) if (mu.f[j] > amax) amax = mu.f[j];

    if (amax < 1e-30f) {
        out->d = 0.0f;
        out->isum = 0;
        __builtin_memset(out->qs, 0, 32);
        return;
    }
    out->d = amax / 127.0f;
    float id = 127.0f / amax;
    v8f vid = v8f_set1(id);
    /* Quantize 32 floats with rounding via vcvtps2dq (banker's rounding) */
    for (int i = 0; i < 32; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f scaled = v * vid;
        /* vcvtps2dq rounds to nearest-even (same as llama.cpp) */
        typedef int v8i __attribute__((vector_size(32)));
        v8i iv;
        __asm__("vcvtps2dq %1, %0" : "=x"(iv) : "x"(scaled));
        /* Pack int32→int16→int8: vpackssdw + vpacksswb */
        /* For simplicity, extract and clamp (compiler optimizes well) */
        union { v8i vec; int i[8]; } u = { .vec = iv };
        for (int j = 0; j < 8; j++) {
            int q = u.i[j];
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out->qs[i + j] = (int8_t)q;
        }
    }
    /* Cache sum of quantized values for Q4×Q8 bias correction */
    {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += out->qs[j];
        out->isum = s;
    }
}

/* Quantize a full row of n floats to q8 blocks */
__attribute__((target("avx2,fma")))
static void q8_quantize_row(q8_input_t *out, const float *x, int n)
{
    int nb = n / 32;
    for (int b = 0; b < nb; b++)
        q8_quantize_block(&out[b], x + b * 32);
}

/* Integer dot: Q8_0 weight block x q8_input block
 * Uses v8f_load_i8 (vpmovsxbd+vcvtdq2ps) for BOTH weight and input
 * Avoids float input loads — input stays as int8, converted in-place */
__attribute__((target("avx2,fma")))
static float q8_0_dot_q8_avx2(const ggml_q8_0_t *w, const q8_input_t *xq)
{
    /* Load both sides from int8 using SIMD dequant (proven to work) */
    v8f w0 = v8f_load_i8(w->qs);      v8f x0 = v8f_load_i8(xq->qs);
    v8f w1 = v8f_load_i8(w->qs + 8);  v8f x1 = v8f_load_i8(xq->qs + 8);
    v8f w2 = v8f_load_i8(w->qs + 16); v8f x2 = v8f_load_i8(xq->qs + 16);
    v8f w3 = v8f_load_i8(w->qs + 24); v8f x3 = v8f_load_i8(xq->qs + 24);

    v8f acc = w0 * x0;
    acc += w1 * x1;
    acc += w2 * x2;
    acc += w3 * x3;

    return v8f_reduce(acc) * fp16_to_fp32(w->d) * xq->d;
}

/* Integer Q8 vec dot: weight row × pre-quantized input */
__attribute__((target("avx2,fma")))
static float llm_vec_dot_avx2_q8(const void *weight, const q8_input_t *xq, int n)
{
    int nb = n / 32;
    const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++)
        sum += q8_0_dot_q8_avx2(&blocks[b], &xq[b]);
    return sum;
}

/* AVX2 Q8_0 dot: process 32 elements per block using SIMD dequant */
__attribute__((target("avx2,fma")))
static float q8_0_dot32_avx2(const ggml_q8_0_t *block, const float *x)
{
    float d = fp16_to_fp32_fast(block->d);
    const int8_t *qs = block->qs;

    /* SIMD int8→float: 2 ops per 8 elements instead of 8 scalar casts */
    v8f q0 = v8f_load_i8(qs);
    v8f q1 = v8f_load_i8(qs + 8);
    v8f q2 = v8f_load_i8(qs + 16);
    v8f q3 = v8f_load_i8(qs + 24);

    v8f x0, x1, x2, x3;
    __builtin_memcpy(&x0, x, 32);
    __builtin_memcpy(&x1, x+8, 32);
    __builtin_memcpy(&x2, x+16, 32);
    __builtin_memcpy(&x3, x+24, 32);

    /* FMA-friendly accumulation */
    v8f acc = q0 * x0;
    acc += q1 * x1;
    acc += q2 * x2;
    acc += q3 * x3;

    return v8f_reduce(acc) * d;
}

/* AVX2 Q4_0 dot: process 32 elements per block using 8-wide */
__attribute__((target("avx2,fma")))
static float q4_0_dot32_avx2(const ggml_q4_0_t *block, const float *x)
{
    float d = fp16_to_fp32_fast(block->d);
    const uint8_t *qs = block->qs;

    /* Standard GGML layout: lo nibbles → x[0..15], hi nibbles → x[16..31] */
    v8f acc_lo = v8f_setzero();
    v8f acc_hi = v8f_setzero();

    for (int j = 0; j < 16; j += 8) {
        uint8_t b0=qs[j], b1=qs[j+1], b2=qs[j+2], b3=qs[j+3];
        uint8_t b4=qs[j+4], b5=qs[j+5], b6=qs[j+6], b7=qs[j+7];
        v8f qlo = {(float)((int)(b0&0xF)-8), (float)((int)(b1&0xF)-8),
                   (float)((int)(b2&0xF)-8), (float)((int)(b3&0xF)-8),
                   (float)((int)(b4&0xF)-8), (float)((int)(b5&0xF)-8),
                   (float)((int)(b6&0xF)-8), (float)((int)(b7&0xF)-8)};
        v8f qhi = {(float)((int)(b0>>4)-8), (float)((int)(b1>>4)-8),
                   (float)((int)(b2>>4)-8), (float)((int)(b3>>4)-8),
                   (float)((int)(b4>>4)-8), (float)((int)(b5>>4)-8),
                   (float)((int)(b6>>4)-8), (float)((int)(b7>>4)-8)};
        v8f xlo; __builtin_memcpy(&xlo, x + j, 32);       /* x[j..j+7] */
        v8f xhi; __builtin_memcpy(&xhi, x + j + 16, 32);  /* x[j+16..j+23] */
        acc_lo += qlo * xlo;
        acc_hi += qhi * xhi;
    }

    return v8f_reduce(acc_lo + acc_hi) * d;
}

/* AVX2 generic vec_dot: dispatches to AVX2 block kernels */
__attribute__((target("avx2,fma")))
static float llm_vec_dot_avx2(const void *weight, const float *x, int n, ggml_type_t type)
{
    int nb = n / 32;
    switch (type) {
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
        v8f acc = v8f_setzero();
        /* Unroll 2 blocks at a time for ILP */
        int b = 0;
        float sum = 0.0f;
        for (; b + 1 < nb; b += 2) {
            sum += q8_0_dot32_avx2(&blocks[b], x + b * 32);
            sum += q8_0_dot32_avx2(&blocks[b+1], x + (b+1) * 32);
        }
        for (; b < nb; b++)
            sum += q8_0_dot32_avx2(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_Q4_0: {
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)weight;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++)
            sum += q4_0_dot32_avx2(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_Q4_1: {
        /* Q4_1: d * nibble + m — use scalar path for now (AVX2 TODO) */
        const ggml_q4_1_t *blocks = (const ggml_q4_1_t *)weight;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++)
            sum += q4_1_dot32(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_F32: {
        const float *f32 = (const float *)weight;
        v8f s0 = v8f_setzero(), s1 = v8f_setzero();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            v8f a0, a1, b0, b1;
            __builtin_memcpy(&a0, f32+i, 32);
            __builtin_memcpy(&a1, f32+i+8, 32);
            __builtin_memcpy(&b0, x+i, 32);
            __builtin_memcpy(&b1, x+i+8, 32);
            s0 = s0 + a0 * b0;
            s1 = s1 + a1 * b1;
        }
        float sum = v8f_reduce(s0 + s1);
        for (; i < n; i++)
            sum += f32[i] * x[i];
        return sum;
    }
    case GGML_TYPE_Q6_K: {
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)weight;
        int nsb = n / 256;
        float sum = 0.0f;
        for (int b = 0; b < nsb; b++)
            sum += q6_k_dot256(&blocks[b], x + b * 256);
        return sum;
    }
    default:
        return llm_vec_dot(weight, x, n, type);
    }
}

/* ─── Fused Q8_0 GEMV: eliminates per-block horizontal reduction ─── *
 * Key optimizations vs the generic path:
 * 1. Single v8f accumulator per row — v8f_reduce called ONCE per row
 *    instead of once per block (saves ~15 ops × nb per row)
 * 2. Shared x loads across 4 rows (load x once, reuse for 4 weight rows)
 * 3. Branchless fp16_to_fp32_fast (~8 ops vs ~20)
 * 4. Scale broadcast + v8f multiply instead of scalar mul after reduce
 * 5. FMA-friendly instruction ordering
 */
__attribute__((target("avx2,fma")))
static void llm_gemv_q8_fused_avx2(float *out, const void *weight,
                                     const float *x, int out_dim, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 34;
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    /* 4-row batched: shared x loads, fused accumulation */
    for (; i + 3 < out_dim; i += 4) {
        const ggml_q8_0_t *r0 = (const ggml_q8_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q8_0_t *r1 = (const ggml_q8_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q8_0_t *r2 = (const ggml_q8_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q8_0_t *r3 = (const ggml_q8_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            /* Load x ONCE — shared across all 4 rows */
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            /* Row 0 */
            {
                v8f d0 = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = v8f_load_i8(r0[b].qs) * x0;
                s += v8f_load_i8(r0[b].qs + 8)  * x1;
                s += v8f_load_i8(r0[b].qs + 16) * x2;
                s += v8f_load_i8(r0[b].qs + 24) * x3;
                a0 += s * d0;
            }
            /* Row 1 */
            {
                v8f d1 = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = v8f_load_i8(r1[b].qs) * x0;
                s += v8f_load_i8(r1[b].qs + 8)  * x1;
                s += v8f_load_i8(r1[b].qs + 16) * x2;
                s += v8f_load_i8(r1[b].qs + 24) * x3;
                a1 += s * d1;
            }
            /* Row 2 */
            {
                v8f d2 = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = v8f_load_i8(r2[b].qs) * x0;
                s += v8f_load_i8(r2[b].qs + 8)  * x1;
                s += v8f_load_i8(r2[b].qs + 16) * x2;
                s += v8f_load_i8(r2[b].qs + 24) * x3;
                a2 += s * d2;
            }
            /* Row 3 */
            {
                v8f d3 = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = v8f_load_i8(r3[b].qs) * x0;
                s += v8f_load_i8(r3[b].qs + 8)  * x1;
                s += v8f_load_i8(r3[b].qs + 16) * x2;
                s += v8f_load_i8(r3[b].qs + 24) * x3;
                a3 += s * d3;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    /* Tail rows */
    for (; i < out_dim; i++) {
        const ggml_q8_0_t *row = (const ggml_q8_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = v8f_load_i8(row[b].qs) * x0;
            s += v8f_load_i8(row[b].qs + 8)  * x1;
            s += v8f_load_i8(row[b].qs + 16) * x2;
            s += v8f_load_i8(row[b].qs + 24) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* Fused Q8_0 GEMV for a range of rows (used by parallel workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q8_fused_range_avx2(float *out, const void *weight,
                                           const float *x, int row_start,
                                           int row_end, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 34;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q8_0_t *r0 = (const ggml_q8_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q8_0_t *r1 = (const ggml_q8_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q8_0_t *r2 = (const ggml_q8_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q8_0_t *r3 = (const ggml_q8_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = v8f_load_i8(r0[b].qs) * x0;
                s += v8f_load_i8(r0[b].qs + 8)  * x1;
                s += v8f_load_i8(r0[b].qs + 16) * x2;
                s += v8f_load_i8(r0[b].qs + 24) * x3;
                a0 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = v8f_load_i8(r1[b].qs) * x0;
                s += v8f_load_i8(r1[b].qs + 8)  * x1;
                s += v8f_load_i8(r1[b].qs + 16) * x2;
                s += v8f_load_i8(r1[b].qs + 24) * x3;
                a1 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = v8f_load_i8(r2[b].qs) * x0;
                s += v8f_load_i8(r2[b].qs + 8)  * x1;
                s += v8f_load_i8(r2[b].qs + 16) * x2;
                s += v8f_load_i8(r2[b].qs + 24) * x3;
                a2 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = v8f_load_i8(r3[b].qs) * x0;
                s += v8f_load_i8(r3[b].qs + 8)  * x1;
                s += v8f_load_i8(r3[b].qs + 16) * x2;
                s += v8f_load_i8(r3[b].qs + 24) * x3;
                a3 += s * d;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    for (; i < row_end; i++) {
        const ggml_q8_0_t *row = (const ggml_q8_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = v8f_load_i8(row[b].qs) * x0;
            s += v8f_load_i8(row[b].qs + 8)  * x1;
            s += v8f_load_i8(row[b].qs + 16) * x2;
            s += v8f_load_i8(row[b].qs + 24) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* ─── Q4_0 × Q8_0 integer dot product (llama.cpp approach) ─── */
/* Uses vpmaddubsw (unsigned×signed→int16) + vpmaddwd (int16→int32).
 * 32 MACs in ~4 SIMD instructions vs ~16 for the float path.
 * Q4 nibbles are treated as unsigned [0,15], bias of 8 is folded into sum. */
typedef int   v8i  __attribute__((vector_size(32)));
typedef short v16s __attribute__((vector_size(32)));
typedef char  v32b __attribute__((vector_size(32)));

__attribute__((target("avx2,fma")))
static inline float q4_0_q8_dot_avx2(const ggml_q4_0_t *w, const q8_input_t *xq)
{
    /* Scalar nibble unpack: 16 iterations, well-predicted by OOO hardware.
     * Main improvement over old path: avoids the 32-iteration scalar xsum loop
     * by using the precomputed xq->isum from q8_quantize_block. */
    uint8_t q4u[32];
    for (int j = 0; j < 16; j++) {
        q4u[j]      = w->qs[j] & 0x0F;
        q4u[j + 16] = w->qs[j] >> 4;
    }

    v32b q4_unsigned; __builtin_memcpy(&q4_unsigned, q4u, 32);
    v32b xq_bytes;    __builtin_memcpy(&xq_bytes,    xq->qs, 32);

    /* vpmaddubsw: unsigned Q4 [0,15] × signed Q8 → 16 × int16 */
    v16s prod16;
    __asm__("vpmaddubsw %2, %1, %0" : "=x"(prod16) : "x"(q4_unsigned), "x"(xq_bytes));

    /* vpmaddwd with ones → 8 × int32 */
    const v16s ones = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    v8i sum32;
    __asm__("vpmaddwd %2, %1, %0" : "=x"(sum32) : "x"(prod16), "x"(ones));

    /* Horizontal sum of 8 int32s */
    union { v8i vec; int i[8]; } su = { .vec = sum32 };
    int dot = (su.i[0]+su.i[1]+su.i[2]+su.i[3]) + (su.i[4]+su.i[5]+su.i[6]+su.i[7]);

    /* Bias correction using cached isum — avoids the old 32-scalar-op xsum loop */
    float wd = fp16_to_fp32_fast(w->d);
    return wd * xq->d * (float)(dot - 8 * xq->isum);
}

/* Fused Q4_0×Q8_0 integer GEMV: 4 rows at a time with shared Q8 input */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_q8_fused_avx2(float *out, const void *weight,
                                       const q8_input_t *xq, int out_dim,
                                       int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18; /* Q4_0 row bytes */
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    for (; i + 3 < out_dim; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int b = 0; b < nb; b++) {
            s0 += q4_0_q8_dot_avx2(&r0[b], &xq[b]);
            s1 += q4_0_q8_dot_avx2(&r1[b], &xq[b]);
            s2 += q4_0_q8_dot_avx2(&r2[b], &xq[b]);
            s3 += q4_0_q8_dot_avx2(&r3[b], &xq[b]);
        }
        out[i]   = s0;
        out[i+1] = s1;
        out[i+2] = s2;
        out[i+3] = s3;
    }
    for (; i < out_dim; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        float s = 0.0f;
        for (int b = 0; b < nb; b++)
            s += q4_0_q8_dot_avx2(&row[b], &xq[b]);
        out[i] = s;
    }
}

/* Fused Q4_0×Q8_0 integer GEMV for a range of rows (SMP workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_q8_fused_range_avx2(float *out, const void *weight,
                                             const q8_input_t *xq,
                                             int row_start, int row_end,
                                             int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int b = 0; b < nb; b++) {
            s0 += q4_0_q8_dot_avx2(&r0[b], &xq[b]);
            s1 += q4_0_q8_dot_avx2(&r1[b], &xq[b]);
            s2 += q4_0_q8_dot_avx2(&r2[b], &xq[b]);
            s3 += q4_0_q8_dot_avx2(&r3[b], &xq[b]);
        }
        out[i]   = s0;
        out[i+1] = s1;
        out[i+2] = s2;
        out[i+3] = s3;
    }
    for (; i < row_end; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        float s = 0.0f;
        for (int b = 0; b < nb; b++)
            s += q4_0_q8_dot_avx2(&row[b], &xq[b]);
        out[i] = s;
    }
}

/* ─── Q4_0 nibble unpacking helpers for standard GGML layout ─── */
/* Extract 8 lo nibbles from 8 consecutive bytes → 8 floats.
 * Uses VPMOVZXBD (8-byte load → 8×int32) + AND + SUB + VCVTDQ2PS (1 cycle each). */
__attribute__((target("avx2,fma")))
static inline v8f q4_unpack_lo_v8f(const uint8_t *qs)
{
    typedef unsigned char v16ub __attribute__((vector_size(16)));
    typedef int v8i __attribute__((vector_size(32)));
    v16ub tmp = {0}; __builtin_memcpy(&tmp, qs, 8);
    v8i i32; __asm__("vpmovzxbd %1, %0" : "=x"(i32) : "x"(tmp));
    v8i mask = (v8i){0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF};
    v8i bias = (v8i){8,8,8,8,8,8,8,8};
    v8i signed_n = (i32 & mask) - bias;
    v8f result; __asm__("vcvtdq2ps %1, %0" : "=x"(result) : "x"(signed_n));
    return result;
}

/* Extract 8 hi nibbles from 8 consecutive bytes → 8 floats. */
__attribute__((target("avx2,fma")))
static inline v8f q4_unpack_hi_v8f(const uint8_t *qs)
{
    typedef unsigned char v16ub __attribute__((vector_size(16)));
    typedef int v8i __attribute__((vector_size(32)));
    v16ub tmp = {0}; __builtin_memcpy(&tmp, qs, 8);
    v8i i32; __asm__("vpmovzxbd %1, %0" : "=x"(i32) : "x"(tmp));
    v8i mask = (v8i){0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF};
    v8i bias = (v8i){8,8,8,8,8,8,8,8};
    v8i shifted;
    /* VPSRLD $4: logical right shift each int32 element by 4 bits */
    __asm__("vpsrld $4, %1, %0" : "=x"(shifted) : "x"(i32));
    v8i signed_n = (shifted & mask) - bias;
    v8f result; __asm__("vcvtdq2ps %1, %0" : "=x"(result) : "x"(signed_n));
    return result;
}

/* ─── Fused Q4_0 GEMV: 4-row batched with shared x loads ───
 * Standard GGML layout: lo nibbles → elements 0..15, hi → 16..31
 * For each block: qs[0..7] lo → x[0..7], qs[8..15] lo → x[8..15],
 *                 qs[0..7] hi → x[16..23], qs[8..15] hi → x[24..31]
 */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_fused_avx2(float *out, const void *weight,
                                     const float *x, int out_dim, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    for (; i + 3 < out_dim; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);       /* x[0..7]   → lo qs[0..7] */
            __builtin_memcpy(&x1, xp + 8, 32);   /* x[8..15]  → lo qs[8..15] */
            __builtin_memcpy(&x2, xp + 16, 32);  /* x[16..23] → hi qs[0..7] */
            __builtin_memcpy(&x3, xp + 24, 32);  /* x[24..31] → hi qs[8..15] */

            /* Row 0 */
            {
                v8f d0 = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = q4_unpack_lo_v8f(r0[b].qs) * x0;
                s += q4_unpack_lo_v8f(r0[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r0[b].qs) * x2;
                s += q4_unpack_hi_v8f(r0[b].qs + 8) * x3;
                a0 += s * d0;
            }
            /* Row 1 */
            {
                v8f d1 = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = q4_unpack_lo_v8f(r1[b].qs) * x0;
                s += q4_unpack_lo_v8f(r1[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r1[b].qs) * x2;
                s += q4_unpack_hi_v8f(r1[b].qs + 8) * x3;
                a1 += s * d1;
            }
            /* Row 2 */
            {
                v8f d2 = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = q4_unpack_lo_v8f(r2[b].qs) * x0;
                s += q4_unpack_lo_v8f(r2[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r2[b].qs) * x2;
                s += q4_unpack_hi_v8f(r2[b].qs + 8) * x3;
                a2 += s * d2;
            }
            /* Row 3 */
            {
                v8f d3 = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = q4_unpack_lo_v8f(r3[b].qs) * x0;
                s += q4_unpack_lo_v8f(r3[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r3[b].qs) * x2;
                s += q4_unpack_hi_v8f(r3[b].qs + 8) * x3;
                a3 += s * d3;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    /* Tail rows */
    for (; i < out_dim; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = q4_unpack_lo_v8f(row[b].qs) * x0;
            s += q4_unpack_lo_v8f(row[b].qs + 8) * x1;
            s += q4_unpack_hi_v8f(row[b].qs) * x2;
            s += q4_unpack_hi_v8f(row[b].qs + 8) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* Fused Q4_0 GEMV for a range of rows (used by parallel workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_fused_range_avx2(float *out, const void *weight,
                                           const float *x, int row_start,
                                           int row_end, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = q4_unpack_lo_v8f(r0[b].qs) * x0;
                s += q4_unpack_lo_v8f(r0[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r0[b].qs) * x2;
                s += q4_unpack_hi_v8f(r0[b].qs + 8) * x3;
                a0 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = q4_unpack_lo_v8f(r1[b].qs) * x0;
                s += q4_unpack_lo_v8f(r1[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r1[b].qs) * x2;
                s += q4_unpack_hi_v8f(r1[b].qs + 8) * x3;
                a1 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = q4_unpack_lo_v8f(r2[b].qs) * x0;
                s += q4_unpack_lo_v8f(r2[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r2[b].qs) * x2;
                s += q4_unpack_hi_v8f(r2[b].qs + 8) * x3;
                a2 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = q4_unpack_lo_v8f(r3[b].qs) * x0;
                s += q4_unpack_lo_v8f(r3[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r3[b].qs) * x2;
                s += q4_unpack_hi_v8f(r3[b].qs + 8) * x3;
                a3 += s * d;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    for (; i < row_end; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = q4_unpack_lo_v8f(row[b].qs) * x0;
            s += q4_unpack_lo_v8f(row[b].qs + 8) * x1;
            s += q4_unpack_hi_v8f(row[b].qs) * x2;
            s += q4_unpack_hi_v8f(row[b].qs + 8) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* AVX2 GEMV: 4-row batched for better cache utilization */
__attribute__((target("avx2,fma")))
static void llm_gemv_avx2(float *out, const void *weight, const float *x,
                           int out_dim, int in_dim, ggml_type_t type)
{
    /* Q8_0 fast path: fully fused GEMV (no per-block reduce, shared x loads) */
    if (type == GGML_TYPE_Q8_0) {
        llm_gemv_q8_fused_avx2(out, weight, x, out_dim, in_dim);
        return;
    }
    /* Q4_0 fast path: float fused GEMV */
    if (type == GGML_TYPE_Q4_0) {
        llm_gemv_q4_fused_avx2(out, weight, x, out_dim, in_dim);
        return;
    }
    uint64_t rb = llm_row_bytes(in_dim, type);
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    /* Q8_0 pre-quantized path disabled (slower under QEMU TCG) */
    if (0) {
        int nb = in_dim / 32;
        q8_input_t xq[Q8_MAX_BLOCKS];
        q8_quantize_row(xq, x, in_dim);

        for (; i + 3 < out_dim; i += 4) {
            if (i + 7 < out_dim) {
                __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 3);
                __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 3);
            }
            out[i]   = llm_vec_dot_avx2_q8(base + (uint64_t)(i)   * rb, xq, in_dim);
            out[i+1] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+1) * rb, xq, in_dim);
            out[i+2] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+2) * rb, xq, in_dim);
            out[i+3] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+3) * rb, xq, in_dim);
        }
        for (; i < out_dim; i++)
            out[i] = llm_vec_dot_avx2_q8(base + (uint64_t)i * rb, xq, in_dim);
        return;
    }

    /* Generic float path for other types */
    for (; i + 3 < out_dim; i += 4) {
        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 3);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 3);
        }
        const void *r0 = base + (uint64_t)(i)   * rb;
        const void *r1 = base + (uint64_t)(i+1) * rb;
        const void *r2 = base + (uint64_t)(i+2) * rb;
        const void *r3 = base + (uint64_t)(i+3) * rb;
        out[i]   = llm_vec_dot_avx2(r0, x, in_dim, type);
        out[i+1] = llm_vec_dot_avx2(r1, x, in_dim, type);
        out[i+2] = llm_vec_dot_avx2(r2, x, in_dim, type);
        out[i+3] = llm_vec_dot_avx2(r3, x, in_dim, type);
    }
    for (; i < out_dim; i++) {
        const void *row = base + (uint64_t)i * rb;
        out[i] = llm_vec_dot_avx2(row, x, in_dim, type);
    }
}

#endif /* !__aarch64__ */

/* Bytes per row for a quantized matrix [out_dim × in_dim] */
static uint64_t llm_row_bytes(int in_dim, ggml_type_t type)
{
    switch (type) {
    case GGML_TYPE_Q4_0: return (uint64_t)(in_dim / 32) * 18;
    case GGML_TYPE_Q4_1: return (uint64_t)(in_dim / 32) * 20;
    case GGML_TYPE_Q8_0: return (uint64_t)(in_dim / 32) * 34;
    case GGML_TYPE_Q6_K: return (uint64_t)(in_dim / 256) * 210;
    case GGML_TYPE_F16:  return (uint64_t)in_dim * 2;
    case GGML_TYPE_BF16: return (uint64_t)in_dim * 2;
    case GGML_TYPE_F32:  return (uint64_t)in_dim * 4;
    default:             return (uint64_t)in_dim * 4;
    }
}

/* GEMV: out[out_dim] = weight[out_dim x in_dim] . x[in_dim]
 * weight is in quantized GGML format (row-major)
 * Tries JIT-compiled Q8_0 kernel first, falls back to vec_dot loop. */
static jit_gemv_q8_fn llm_jit_gemv_cache[8] = {0};
static int llm_jit_gemv_rows[8] = {0};
static int llm_jit_gemv_cols[8] = {0};
static int llm_jit_gemv_n = 0;

/* AVX2 JIT GEMV cache (preferred when AVX2+FMA available) */
static jit_gemv_q8_fn llm_jit_gemv_avx2_cache[8] = {0};
static int llm_jit_gemv_avx2_rows[8] = {0};
static int llm_jit_gemv_avx2_cols[8] = {0};
static int llm_jit_gemv_avx2_n = 0;

static jit_gemv_q8_fn llm_get_jit_gemv_avx2(int rows, int cols)
{
    for (int i = 0; i < llm_jit_gemv_avx2_n; i++) {
        if (llm_jit_gemv_avx2_rows[i] == rows && llm_jit_gemv_avx2_cols[i] == cols)
            return llm_jit_gemv_avx2_cache[i];
    }
    jit_gemv_q8_fn fn = jit_compile_q8_gemv_avx2(rows, cols);
    if (fn && llm_jit_gemv_avx2_n < 8) {
        llm_jit_gemv_avx2_rows[llm_jit_gemv_avx2_n] = rows;
        llm_jit_gemv_avx2_cols[llm_jit_gemv_avx2_n] = cols;
        llm_jit_gemv_avx2_cache[llm_jit_gemv_avx2_n] = fn;
        llm_jit_gemv_avx2_n++;
    }
    return fn;
}

static jit_gemv_q8_fn llm_get_jit_gemv(int rows, int cols)
{
    /* Check local cache first (fast path) */
    for (int i = 0; i < llm_jit_gemv_n; i++) {
        if (llm_jit_gemv_rows[i] == rows && llm_jit_gemv_cols[i] == cols)
            return llm_jit_gemv_cache[i];
    }
    /* Try to compile */
    jit_gemv_q8_fn fn = jit_compile_q8_gemv(rows, cols);
    if (fn && llm_jit_gemv_n < 8) {
        llm_jit_gemv_rows[llm_jit_gemv_n] = rows;
        llm_jit_gemv_cols[llm_jit_gemv_n] = cols;
        llm_jit_gemv_cache[llm_jit_gemv_n] = fn;
        llm_jit_gemv_n++;
    }
    return fn;
}

/* AVX2 JIT cache for Q4_0×Q8_0 integer GEMV */
static jit_gemv_q8_fn llm_jit_gemv_q4q8_cache[8] = {0};
static int llm_jit_gemv_q4q8_rows[8] = {0};
static int llm_jit_gemv_q4q8_cols[8] = {0};
static int llm_jit_gemv_q4q8_n = 0;

static jit_gemv_q8_fn llm_get_jit_gemv_q4q8(int rows, int cols)
{
    for (int i = 0; i < llm_jit_gemv_q4q8_n; i++) {
        if (llm_jit_gemv_q4q8_rows[i] == rows && llm_jit_gemv_q4q8_cols[i] == cols)
            return llm_jit_gemv_q4q8_cache[i];
    }
    jit_gemv_q8_fn fn = jit_compile_q4_q8_gemv_avx2(rows, cols);
    if (fn && llm_jit_gemv_q4q8_n < 8) {
        llm_jit_gemv_q4q8_rows[llm_jit_gemv_q4q8_n] = rows;
        llm_jit_gemv_q4q8_cols[llm_jit_gemv_q4q8_n] = cols;
        llm_jit_gemv_q4q8_cache[llm_jit_gemv_q4q8_n] = fn;
        llm_jit_gemv_q4q8_n++;
    }
    return fn;
}

/* ─── Parallel GEMV work item for SMP dispatch ─── */
#ifndef __aarch64__
typedef struct {
    float       *out;
    const void  *weight;
    const float *x;
    const q8_input_t *xq; /* Pre-quantized input for Q8_0 integer path */
    int          out_dim;
    int          in_dim;
    ggml_type_t  type;
    int          row_start;
    int          row_end;
} gemv_work_t;

static gemv_work_t gemv_work_items[MAX_CPUS];

__attribute__((target("avx2,fma")))
static void gemv_worker_avx2(void *arg)
{
    gemv_work_t *w = (gemv_work_t *)arg;

    /* Q8_0 fast path: fully fused (no per-block reduce, shared x loads) */
    if (w->type == GGML_TYPE_Q8_0) {
        llm_gemv_q8_fused_range_avx2(w->out, w->weight, w->x,
                                      w->row_start, w->row_end, w->in_dim);
        return;
    }

    /* Q4_0 integer path: use pre-quantized Q8 input if available */
    if (w->type == GGML_TYPE_Q4_0 && w->xq) {
        llm_gemv_q4_q8_fused_range_avx2(w->out, w->weight, w->xq,
                                         w->row_start, w->row_end, w->in_dim);
        return;
    }

    /* Q4_0 float fallback */
    if (w->type == GGML_TYPE_Q4_0) {
        llm_gemv_q4_fused_range_avx2(w->out, w->weight, w->x,
                                      w->row_start, w->row_end, w->in_dim);
        return;
    }

    uint64_t rb = llm_row_bytes(w->in_dim, w->type);
    const uint8_t *base = (const uint8_t *)w->weight;

    for (int i = w->row_start; i < w->row_end; i++) {
        if (i + 1 < w->row_end)
            __builtin_prefetch(base + (uint64_t)(i + 1) * rb, 0, 3);
        const void *row = base + (uint64_t)i * rb;
        w->out[i] = llm_vec_dot_avx2(row, w->x, w->in_dim, w->type);
    }
}
#endif

static void llm_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t type)
{
#ifndef __aarch64__
    /* JIT AVX2+FMA GEMV: best single-core Q8_0 performance */
    if (cpu_features.avx2_usable && type == GGML_TYPE_Q8_0) {
        jit_gemv_q8_fn jfn = llm_get_jit_gemv_avx2(out_dim, in_dim);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim);
            return;
        }
    }

    /* Q4×Q8 integer GEMV: pre-quantize input to Q8, then use integer dot */
    if (cpu_features.avx2_usable && type == GGML_TYPE_Q4_0) {
        int nb = in_dim / 32;
        if (nb <= Q8_MAX_BLOCKS) {
            /* Pre-quantize input vector to Q8 for integer dot path */
            q8_quantize_row(llm_xq_buf, x, in_dim);

            /* Parallel Q4xQ8 integer GEMV across all CPUs */
            uint32_t ncpu = smp.ap_started + 1;
            if (ncpu > 1 && out_dim >= 64) {
                int rows_per_cpu = out_dim / ncpu;
                int remainder    = out_dim % ncpu;
                int row = 0;
                int bsp_chunk = rows_per_cpu + (remainder > 0 ? 1 : 0);
                int bsp_start = 0;
                int bsp_end   = bsp_chunk;
                row = bsp_end;

                for (uint32_t c = 1; c < ncpu; c++) {
                    int chunk = rows_per_cpu + ((int)c < remainder ? 1 : 0);
                    gemv_work_items[c].out      = out;
                    gemv_work_items[c].weight   = weight;
                    gemv_work_items[c].x        = x;
                    gemv_work_items[c].xq       = llm_xq_buf;
                    gemv_work_items[c].out_dim  = out_dim;
                    gemv_work_items[c].in_dim   = in_dim;
                    gemv_work_items[c].type     = type;
                    gemv_work_items[c].row_start = row;
                    gemv_work_items[c].row_end   = row + chunk;
                    smp_dispatch(c, gemv_worker_avx2, &gemv_work_items[c]);
                    row += chunk;
                }
                /* BSP does its share using integer path */
                llm_gemv_q4_q8_fused_range_avx2(out, weight, llm_xq_buf,
                                                 bsp_start, bsp_end, in_dim);
                smp_wait_all();
                return;
            }

            /* Single-core fallback: fused integer GEMV */
            llm_gemv_q4_q8_fused_avx2(out, weight, llm_xq_buf, out_dim, in_dim);
            return;
        }
    }

    /* Use AVX2+FMA GEMV when available */
    if (cpu_features.avx2_usable) {
        uint32_t ncpu = smp.ap_started + 1; /* Only use actually-running CPUs */
        /* Parallel GEMV: split rows across all CPUs when worth it */
        if (ncpu > 1 && out_dim >= 64) {
            int rows_per_cpu = out_dim / ncpu;
            int remainder    = out_dim % ncpu;
            int row = 0;
            /* BSP gets chunk 0, APs get chunks 1..ncpu-1 */
            int bsp_chunk = rows_per_cpu + (remainder > 0 ? 1 : 0);
            int bsp_start = 0;
            int bsp_end   = bsp_chunk;
            row = bsp_end;

            /* Q4×Q8 integer path pending correctness fix — use float path */
            const q8_input_t *xq_ptr = (void *)0;

            /* Dispatch to APs (cpu 1..ncpu-1) */
            for (uint32_t c = 1; c < ncpu; c++) {
                int chunk = rows_per_cpu + ((int)c < remainder ? 1 : 0);
                gemv_work_items[c].out      = out;
                gemv_work_items[c].weight   = weight;
                gemv_work_items[c].x        = x;
                gemv_work_items[c].xq       = xq_ptr;
                gemv_work_items[c].out_dim  = out_dim;
                gemv_work_items[c].in_dim   = in_dim;
                gemv_work_items[c].type     = type;
                gemv_work_items[c].row_start = row;
                gemv_work_items[c].row_end   = row + chunk;
                smp_dispatch(c, gemv_worker_avx2, &gemv_work_items[c]);
                row += chunk;
            }
            /* BSP does its share — use fused path for Q8_0/Q4_0 */
            if (type == GGML_TYPE_Q8_0) {
                llm_gemv_q8_fused_range_avx2(out, weight, x,
                                              bsp_start, bsp_end, in_dim);
            } else if (type == GGML_TYPE_Q4_0) {
                llm_gemv_q4_fused_range_avx2(out, weight, x,
                                              bsp_start, bsp_end, in_dim);
            } else {
                uint64_t rb = llm_row_bytes(in_dim, type);
                const uint8_t *base = (const uint8_t *)weight;
                for (int i = bsp_start; i < bsp_end; i++) {
                    if (i + 1 < bsp_end)
                        __builtin_prefetch(base + (uint64_t)(i + 1) * rb, 0, 3);
                    const void *r = base + (uint64_t)i * rb;
                    out[i] = llm_vec_dot_avx2(r, x, in_dim, type);
                }
            }
            /* Wait for all APs */
            smp_wait_all();
            return;
        }
        llm_gemv_avx2(out, weight, x, out_dim, in_dim, type);
        return;
    }

    /* Try JIT-compiled Q8_0 GEMV kernel */
    if (type == GGML_TYPE_Q8_0) {
        jit_gemv_q8_fn jfn = llm_get_jit_gemv(out_dim, in_dim);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim);
            return;
        }
    }
#endif

    uint64_t rb = llm_row_bytes(in_dim, type);

    for (int i = 0; i < out_dim; i++) {
        const void *row = (const uint8_t *)weight + (uint64_t)i * rb;
        out[i] = llm_vec_dot(row, x, in_dim, type);
    }
}

/* Get a single float from a (possibly quantized) 1D vector at index idx */
static float llm_get_f(const void *data, int idx, ggml_type_t type)
{
    switch (type) {
    case GGML_TYPE_F32:
        return ((const float *)data)[idx];
    case GGML_TYPE_F16:
        return fp16_to_fp32(((const uint16_t *)data)[idx]);
    case GGML_TYPE_BF16:
        return bf16_to_fp32(((const uint16_t *)data)[idx]);
    default:
        return 0.0f;
    }
}

/* Embedding lookup: copy the embedding vector for token_id into out[dim] */
static void llm_embed(float *out, const llm_model_t *m, int token_id)
{
    int dim = m->dim;
    uint64_t rb = llm_row_bytes(dim, m->token_embd_type);
    const uint8_t *row = (const uint8_t *)m->token_embd + (uint64_t)token_id * rb;

    switch (m->token_embd_type) {
    case GGML_TYPE_F32: {
        const float *f = (const float *)row;
        for (int i = 0; i < dim; i++) out[i] = f[i];
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *h = (const uint16_t *)row;
        for (int i = 0; i < dim; i++) out[i] = fp16_to_fp32(h[i]);
        break;
    }
    case GGML_TYPE_Q4_0: {
        /* Dequantize Q4_0 blocks (standard GGML layout: lo→[0..15], hi→[16..31]) */
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)row;
        int nb = dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(blocks[b].d);
            for (int j = 0; j < 16; j++) {
                uint8_t packed = blocks[b].qs[j];
                out[b * 32 + j]      = (float)((int)(packed & 0x0F) - 8) * d;
                out[b * 32 + j + 16] = (float)((int)(packed >> 4)   - 8) * d;
            }
        }
        break;
    }
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)row;
        int nb = dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(blocks[b].d);
            for (int j = 0; j < 32; j++)
                out[b * 32 + j] = (float)blocks[b].qs[j] * d;
        }
        break;
    }
    case GGML_TYPE_Q6_K: {
        /* Q6_K: 256 elements per super-block, 210 bytes each */
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)row;
        int nsb = dim / 256;
        for (int sb = 0; sb < nsb; sb++) {
            const ggml_q6_k_t *b = &blocks[sb];
            float d = fp16_to_fp32(b->d);
            const uint8_t *ql = b->ql;
            const uint8_t *qh = b->qh;
            const int8_t *sc = b->scales;
            float *o = out + sb * 256;
            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_h = ql + half * 64;
                const uint8_t *qh_h = qh + half * 32;
                const int8_t *sc_h = sc + half * 8;
                for (int l = 0; l < 32; l++) {
                    int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                    int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                    int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                    int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                    int si = l / 16;
                    o[half*128 + l]      = d * (float)sc_h[0 + si] * (float)(q0 - 32);
                    o[half*128 + l + 32] = d * (float)sc_h[2 + si] * (float)(q1 - 32);
                    o[half*128 + l + 64] = d * (float)sc_h[4 + si] * (float)(q2 - 32);
                    o[half*128 + l + 96] = d * (float)sc_h[6 + si] * (float)(q3 - 32);
                }
            }
        }
        break;
    }
    default:
        for (int i = 0; i < dim; i++) out[i] = 0.0f;
        break;
    }

    /* Gemma models scale embeddings by sqrt(dim) */
    if (m->embed_scale != 1.0f) {
        float s = m->embed_scale;
        for (int i = 0; i < dim; i++) out[i] *= s;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  RMSNorm                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* AVX2 RMSNorm for F32 weights (8-wide, 2× unroll) */
__attribute__((target("avx2,fma")))
static void llm_rmsnorm_avx2_f32(float *out, const float *x, const float *wf, int dim, float eps)
{
    v8f ss0 = v8f_setzero(), ss1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        ss0 = ss0 + v0 * v0;
        ss1 = ss1 + v1 * v1;
    }
    float ss = v8f_reduce(ss0 + ss1);
    for (; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    v8f ssv = v8f_set1(ss);
    for (i = 0; i + 8 <= dim; i += 8) {
        v8f xv, wv;
        __builtin_memcpy(&xv, x + i, 32);
        __builtin_memcpy(&wv, wf + i, 32);
        v8f r = xv * ssv * wv;
        __builtin_memcpy(out + i, &r, 32);
    }
    for (; i < dim; i++)
        out[i] = x[i] * ss * wf[i];
}

static void llm_rmsnorm(float *out, const float *x, const void *w,
                        int dim, ggml_type_t wtype, float eps)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable && wtype == GGML_TYPE_F32) {
        llm_rmsnorm_avx2_f32(out, x, (const float *)w, dim, eps);
        return;
    }

    /* SSE2 fallback for non-AVX2 or non-F32 weights */
    v4f ss_vec = {0, 0, 0, 0};
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        v4f v = *(const v4f *)(x + i);
        ss_vec += v * v;
    }
    union { v4f v; float f[4]; } u = { .v = ss_vec };
    float ss = u.f[0] + u.f[1] + u.f[2] + u.f[3];
    for (; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    /* Vectorized normalize + weight multiply */
    v4f ss_v = {ss, ss, ss, ss};
    for (i = 0; i + 4 <= dim; i += 4) {
        v4f xv = *(const v4f *)(x + i);
        v4f wv = {llm_get_f(w, i, wtype), llm_get_f(w, i+1, wtype),
                  llm_get_f(w, i+2, wtype), llm_get_f(w, i+3, wtype)};
        *(v4f *)(out + i) = xv * ss_v * wv;
    }
    for (; i < dim; i++)
        out[i] = x[i] * ss * llm_get_f(w, i, wtype);
#else
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    for (int i = 0; i < dim; i++)
        out[i] = x[i] * ss * llm_get_f(w, i, wtype);
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  LayerNorm (Phi-2): subtract mean, normalize, scale + bias                  */
/*  out[i] = w[i] * (x[i] - mean) / sqrt(var + eps) + b[i]                   */
/* ─────────────────────────────────────────────────────────────────────────── */

__attribute__((target("avx2,fma")))
static void llm_layernorm_avx2_f32(float *out, const float *x, const float *wf,
                                    const float *bf, int dim)
{
    /* Compute mean */
    v8f sum0 = v8f_setzero(), sum1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        sum0 = sum0 + v0;
        sum1 = sum1 + v1;
    }
    float mean = v8f_reduce(sum0 + sum1);
    for (; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    /* Compute variance */
    v8f mv = v8f_set1(mean);
    v8f var0 = v8f_setzero(), var1 = v8f_setzero();
    for (i = 0; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        v8f d0 = v0 - mv, d1 = v1 - mv;
        var0 = var0 + d0 * d0;
        var1 = var1 + d1 * d1;
    }
    float var = v8f_reduce(var0 + var1);
    for (; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / llm_sqrtf(var / (float)dim + 1e-5f);

    /* Normalize + scale + bias */
    v8f inv_v = v8f_set1(inv_std);
    for (i = 0; i + 8 <= dim; i += 8) {
        v8f xv, wv, bv;
        __builtin_memcpy(&xv, x + i, 32);
        __builtin_memcpy(&wv, wf + i, 32);
        v8f r = (xv - mv) * inv_v * wv;
        if (bf) {
            __builtin_memcpy(&bv, bf + i, 32);
            r = r + bv;
        }
        __builtin_memcpy(out + i, &r, 32);
    }
    for (; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * wf[i];
        if (bf) out[i] += bf[i];
    }
}

static void llm_layernorm(float *out, const float *x, const void *w,
                           const void *bias, int dim, ggml_type_t wtype)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable && wtype == GGML_TYPE_F32) {
        llm_layernorm_avx2_f32(out, x, (const float *)w,
                                bias ? (const float *)bias : NULL, dim);
        return;
    }
#endif
    /* Scalar fallback */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / llm_sqrtf(var / (float)dim + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * llm_get_f(w, i, wtype);
        if (bias) out[i] += llm_get_f(bias, i, GGML_TYPE_F32);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GELU activation (Phi-2 FFN uses GELU instead of SiLU/SwiGLU)              */
/*  GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))       */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__
__attribute__((target("avx2,fma")))
static void llm_gelu_avx2(float *x, int n)
{
    v8f half   = v8f_set1(0.5f);
    v8f one    = v8f_set1(1.0f);
    v8f coeff  = v8f_set1(0.044715f);
    v8f sqrt2p = v8f_set1(0.7978845608f); /* sqrt(2/pi) */
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f t = sqrt2p * (v + coeff * v * v * v);
        /* tanh(t) ≈ 1 - 2/(1+exp(2t)) */
        v8f e2t = v8f_expf(t + t);
        v8f tanh_t = (e2t - one) / (e2t + one);
        v8f r = v * half * (one + tanh_t);
        __builtin_memcpy(x + i, &r, 32);
    }
    for (; i < n; i++) {
        float v = x[i];
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        float e = llm_expf(2.0f * t);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        x[i] = v * 0.5f * (1.0f + tanh_val);
    }
}
#endif

static void llm_gelu(float *x, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_gelu_avx2(x, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        float e = llm_expf(2.0f * t);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        x[i] = v * 0.5f * (1.0f + tanh_val);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Bias addition helper — add F32 bias to output vector                       */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_add_bias(float *out, const void *bias, int n)
{
    if (!bias) return;
    const float *b = (const float *)bias;
    for (int i = 0; i < n; i++)
        out[i] += b[i];
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Rotary Position Embeddings (RoPE) — Optimized                              */
/*  Precomputed frequency table + fast sin/cos via range-reduced polynomial.   */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Precomputed inverse-frequency table: freq[i] = base^(-2i/dim)
 * Uses the dynamically allocated llm_rope_freqs_buf from scratch arena. */
static int   llm_rope_freqs_ready = 0;
static float llm_rope_base_cached = 0.0f;
static int   llm_rope_hdim_cached = 0;

static float llm_log_approx(float x)
{
    /* log(x) via IEEE754: extract exponent + polynomial on mantissa.
     * x = 2^e * m where m in [1,2). log(x) = e*ln2 + log(m).
     * log(m) ≈ Remez polynomial on [1,2). */
    union { float f; unsigned int u; } u = { .f = x };
    int e = (int)((u.u >> 23) & 0xFF) - 127;
    u.u = (u.u & 0x007FFFFFu) | 0x3F800000u; /* m in [1,2) */
    float m = u.f;
    float m1 = m - 1.0f;
    /* Minimax log(1+m1) for m1 in [0,1): 4th order Remez */
    float logm = m1 * (0.9999964f - m1 * (0.4999899f - m1 * (0.3334508f - m1 * 0.2414937f)));
    return (float)e * 0.69314718f + logm;
}

static void llm_rope_precompute(float base, int head_dim, const float *factors)
{
    if (llm_rope_freqs_ready && llm_rope_base_cached == base &&
        llm_rope_hdim_cached == head_dim)
        return;
    if (!llm_rope_freqs_buf) return; /* not yet allocated */
    float log_base = llm_log_approx(base);
    for (int i = 0; i < head_dim / 2; i++) {
        float exponent = -(float)(2 * i) / (float)head_dim;
        float freq = llm_expf(exponent * log_base);
        /* Longrope: divide frequency by scaling factor */
        if (factors)
            freq /= factors[i];
        llm_rope_freqs_buf[i] = freq;
    }
    llm_rope_base_cached = base;
    llm_rope_hdim_cached = head_dim;
    llm_rope_freqs_ready = 1;
}

static void llm_rope(float *vec, int pos, int head_dim, float base,
                     const float *factors)
{
    llm_rope_precompute(base, head_dim, factors);

    for (int i = 0; i < head_dim; i += 2) {
        float theta = (float)pos * llm_rope_freqs_buf[i / 2];
        float cos_t = llm_cosf(theta);
        float sin_t = llm_sinf(theta);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_t - v1 * sin_t;
        vec[i + 1] = v0 * sin_t + v1 * cos_t;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Softmax                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* AVX2 softmax: 8-wide max, exp, and divide */
__attribute__((target("avx2,fma")))
static void llm_softmax_avx2(float *x, int n)
{
    /* 8-wide find-max */
    v8f mv = v8f_set1(x[0]);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        mv = v8f_max(mv, v);
    }
    union { v8f vec; float f[8]; } mu = { .vec = mv };
    float max_val = mu.f[0];
    for (int j = 1; j < 8; j++)
        if (mu.f[j] > max_val) max_val = mu.f[j];
    for (; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    /* Vectorized exp + sum */
    v8f sv = v8f_setzero();
    v8f mv8 = v8f_set1(max_val);
    for (i = 0; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f e = v8f_expf(v - mv8);
        __builtin_memcpy(x + i, &e, 32);
        sv = sv + e;
    }
    float sum = v8f_reduce(sv);
    for (; i < n; i++) {
        x[i] = llm_expf(x[i] - max_val);
        sum += x[i];
    }

    /* Vectorized divide */
    float inv = 1.0f / (sum + 1e-10f);
    v8f inv_v = v8f_set1(inv);
    for (i = 0; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v = v * inv_v;
        __builtin_memcpy(x + i, &v, 32);
    }
    for (; i < n; i++)
        x[i] *= inv;
}

static void llm_softmax(float *x, int n)
{
    if (n <= 0) return;

#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_softmax_avx2(x, n);
        return;
    }
#endif

    /* Scalar fallback */
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = llm_expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  SiLU (Sigmoid Linear Unit) — used in SwiGLU FFN — Vectorized              */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Fast vectorized exp for small batches (Padé with better range reduction) */
static inline float llm_fast_sigmoid(float x)
{
    /* Clamp to avoid overflow */
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    float e = llm_expf(-x);
    return 1.0f / (1.0f + e);
}

#ifndef __aarch64__
__attribute__((target("avx2,fma")))
static void llm_silu_avx2(float *x, int n)
{
    v8f one = v8f_set1(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
        v8f e = v8f_expf(v8f_setzero() - v);
        v8f r = v / (one + e);
        __builtin_memcpy(x + i, &r, 32);
    }
    for (; i < n; i++) {
        float s = llm_fast_sigmoid(x[i]);
        x[i] *= s;
    }
}
#endif

static void llm_silu(float *x, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_silu_avx2(x, n);
        return;
    }
    /* SSE2 4-wide unrolled */
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float s0 = llm_fast_sigmoid(x[i]);
        float s1 = llm_fast_sigmoid(x[i+1]);
        float s2 = llm_fast_sigmoid(x[i+2]);
        float s3 = llm_fast_sigmoid(x[i+3]);
        x[i] *= s0; x[i+1] *= s1; x[i+2] *= s2; x[i+3] *= s3;
    }
    for (; i < n; i++) {
        float s = llm_fast_sigmoid(x[i]);
        x[i] *= s;
    }
#else
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + llm_expf(-x[i]));
        x[i] = x[i] * s;
    }
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Vectorized helpers for attention inner loops (SSE2 + AVX2 dispatch)        */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__
/* AVX2 dot product: 8-wide with 2× unroll */
__attribute__((target("avx2,fma")))
static float llm_dot_f32_avx2(const float *a, const float *b, int n)
{
    v8f s0 = v8f_setzero(), s1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        v8f a0, a1, b0, b1;
        __builtin_memcpy(&a0, a+i, 32);
        __builtin_memcpy(&a1, a+i+8, 32);
        __builtin_memcpy(&b0, b+i, 32);
        __builtin_memcpy(&b1, b+i+8, 32);
        s0 = s0 + a0 * b0;
        s1 = s1 + a1 * b1;
    }
    float sum = v8f_reduce(s0 + s1);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* AVX2 axpy: dst += scale * src */
__attribute__((target("avx2,fma")))
static void llm_axpy_f32_avx2(float *dst, float scale, const float *src, int n)
{
    v8f sv = v8f_set1(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f d, s;
        __builtin_memcpy(&d, dst+i, 32);
        __builtin_memcpy(&s, src+i, 32);
        d = d + sv * s;
        __builtin_memcpy(dst+i, &d, 32);
    }
    for (; i < n; i++)
        dst[i] += scale * src[i];
}
#endif /* !__aarch64__ */

/* Dot product of two float arrays */
static float llm_dot_f32(const float *a, const float *b, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable)
        return llm_dot_f32_avx2(a, b, n);
    v4f acc0 = {0, 0, 0, 0};
    v4f acc1 = {0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v4f a0 = *(const v4f *)(a + i);
        v4f a1 = *(const v4f *)(a + i + 4);
        v4f b0 = *(const v4f *)(b + i);
        v4f b1 = *(const v4f *)(b + i + 4);
        acc0 += a0 * b0;
        acc1 += a1 * b1;
    }
    v4f acc = acc0 + acc1;
    union { v4f v; float f[4]; } u = { .v = acc };
    float sum = u.f[0] + u.f[1] + u.f[2] + u.f[3];
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

/* Scaled add: dst[i] += scale * src[i] */
static void llm_axpy_f32(float *dst, float scale, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_axpy_f32_avx2(dst, scale, src, n);
        return;
    }
    v4f sv = {scale, scale, scale, scale};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d + sv * s;
    }
    for (; i < n; i++)
        dst[i] += scale * src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] += scale * src[i];
#endif
}

/* Vector add: dst[i] += src[i] */
static void llm_vadd_f32(float *dst, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v8f d, s;
            __builtin_memcpy(&d, dst+i, 32);
            __builtin_memcpy(&s, src+i, 32);
            d = d + s;
            __builtin_memcpy(dst+i, &d, 32);
        }
        for (; i < n; i++) dst[i] += src[i];
        return;
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d + s;
    }
    for (; i < n; i++)
        dst[i] += src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
#endif
}

/* Element-wise multiply: dst[i] *= src[i] */
static void llm_vmul_f32(float *dst, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v8f d, s;
            __builtin_memcpy(&d, dst+i, 32);
            __builtin_memcpy(&s, src+i, 32);
            d = d * s;
            __builtin_memcpy(dst+i, &d, 32);
        }
        for (; i < n; i++) dst[i] *= src[i];
        return;
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d * s;
    }
    for (; i < n; i++)
        dst[i] *= src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] *= src[i];
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Transformer Forward Pass — Single Token                                    */
/*                                                                             */
/*  Process one token through the full transformer stack (incremental decode).  */
/*  Updates KV-cache and returns logits[vocab_size].                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/* === JIT kernel cache for the forward pass (lazy-compiled once) === */
static jit_fused_silu_mul_fn jit_fwd_fused_silu = NULL;
static jit_ewise_fn          jit_fwd_vadd_dim   = NULL;
static jit_ewise_fn          jit_fwd_vmul_ff    = NULL;
static jit_dot_fn            jit_fwd_dot_hd     = NULL;
static jit_axpy_fn           jit_fwd_axpy_hd    = NULL;
static jit_rope_fn           jit_fwd_rope_hd    = NULL;
static jit_rmsnorm_fn        jit_fwd_rmsnorm    = NULL;
static jit_softmax_fn        jit_fwd_softmax    = NULL;
static int                   jit_fwd_ready      = 0;

static void llm_forward_token(llm_model_t *m, float *logits, int token_id, int pos)
{
    int dim = m->dim;
    int n_heads = m->n_heads;
    int n_kv = m->n_kv_heads;
    int hd = m->head_dim;
    int ff = m->ff_dim;
    int kv_dim = n_kv * hd;

    /* Lazy-compile JIT kernels on first call (dimensions fixed per model) */
#ifndef __aarch64__
    if (!jit_fwd_ready) {
        /* Compile ALL JIT kernels — baked dimensions eliminate loop overhead.
         * SSE2 JIT with constants often beats generic AVX2 C for small ops. */
        jit_fwd_vadd_dim   = jit_compile_vadd_kernel(dim);
        jit_fwd_dot_hd     = jit_compile_dot_kernel(hd);
        jit_fwd_axpy_hd    = jit_compile_axpy_kernel(hd);
        jit_fwd_fused_silu  = jit_compile_fused_silu_mul_kernel(ff);
        jit_fwd_rope_hd    = jit_compile_rope_kernel(hd);
        jit_fwd_rmsnorm    = jit_compile_rmsnorm_kernel(dim);
        jit_fwd_softmax    = NULL; /* softmax has variable length (seq_len) — skip */
        jit_fwd_ready = 1;
    }
#endif

    /* 1. Embedding lookup */
    llm_embed(llm_x, m, token_id);

    /* 2. Process each transformer layer */
    int rope_dim = m->rope_dim > 0 ? m->rope_dim : hd; /* partial RoPE for Phi-2 */

    /* === Gemma4 ISWA: precompute per-layer embeddings === */
    if (m->is_gemma4 && m->iswa_tok_embd && m->iswa_model_proj) {
        int iswa_d = m->iswa_n_embd;   /* 256 */
        int iswa_total = iswa_d * m->n_layers; /* 8960 */
        float inv_sqrt_dim = 1.0f / llm_sqrtf((float)dim);
        float sqrt_iswa_d  = llm_sqrtf((float)iswa_d);
        float inv_sqrt_2   = 1.0f / llm_sqrtf(2.0f);

        /* Step 1: token embedding lookup from per_layer_token_embd → [8960] */
        uint64_t emb_rb = llm_row_bytes(iswa_total, m->iswa_tok_embd_type);
        const uint8_t *emb_row = (const uint8_t *)m->iswa_tok_embd
                                 + (uint64_t)token_id * emb_rb;
        /* Dequant Q6_K row into llm_iswa_per_layer */
        if (m->iswa_tok_embd_type == GGML_TYPE_Q6_K) {
            const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)emb_row;
            int nsb = iswa_total / 256;
            for (int sb = 0; sb < nsb; sb++) {
                const ggml_q6_k_t *b = &blocks[sb];
                float d_val = fp16_to_fp32(b->d);
                const uint8_t *ql = b->ql;
                const uint8_t *qh = b->qh;
                const int8_t *sc = b->scales;
                float *o = llm_iswa_per_layer + sb * 256;
                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql + half * 64;
                    const uint8_t *qh_h = qh + half * 32;
                    const int8_t *sc_h = sc + half * 8;
                    for (int l = 0; l < 32; l++) {
                        int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                        int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                        int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                        int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                        int si = l / 16;
                        o[half*128 + l]      = d_val * (float)sc_h[0 + si] * (float)(q0 - 32);
                        o[half*128 + l + 32] = d_val * (float)sc_h[2 + si] * (float)(q1 - 32);
                        o[half*128 + l + 64] = d_val * (float)sc_h[4 + si] * (float)(q2 - 32);
                        o[half*128 + l + 96] = d_val * (float)sc_h[6 + si] * (float)(q3 - 32);
                    }
                }
            }
        } else {
            /* F32/F16/BF16 fallback */
            for (int i = 0; i < iswa_total; i++)
                llm_iswa_per_layer[i] = llm_get_f(m->iswa_tok_embd, i, m->iswa_tok_embd_type);
        }
        /* Scale by sqrt(iswa_d) */
        for (int i = 0; i < iswa_total; i++)
            llm_iswa_per_layer[i] *= sqrt_iswa_d;

        /* Step 2: project model input to per-layer space */
        /* per_layer_proj[8960] = per_layer_model_proj[1536, 8960] · inpL[1536] */
        /* We use llm_iswa_tmp as temp for the projection (need 8960 floats — reuse ffn_g) */
        float *proj_buf = llm_ffn_g; /* reuse FFN scratch (size >= 12288 > 8960) */
        llm_gemv(proj_buf, m->iswa_model_proj, llm_x, iswa_total, dim,
                 m->iswa_model_proj_type);
        /* Scale by 1/sqrt(dim) */
        for (int i = 0; i < iswa_total; i++)
            proj_buf[i] *= inv_sqrt_dim;

        /* Step 3: RMSNorm each [iswa_d] slice with per_layer_proj_norm */
        if (m->iswa_proj_norm) {
            const float *norm_w = (const float *)m->iswa_proj_norm;
            for (int l = 0; l < m->n_layers; l++) {
                float *sl = proj_buf + l * iswa_d;
                float ss = 0.0f;
                for (int i = 0; i < iswa_d; i++) ss += sl[i] * sl[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)iswa_d + m->rms_eps);
                for (int i = 0; i < iswa_d; i++) sl[i] = sl[i] * rms * norm_w[i];
            }
        }

        /* Step 4: Add token embedding + projection, scale by 1/sqrt(2) */
        for (int i = 0; i < iswa_total; i++)
            llm_iswa_per_layer[i] = (llm_iswa_per_layer[i] + proj_buf[i]) * inv_sqrt_2;
    }

    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];

        /* Bridge: inject hidden state before this layer */
        if (llm_bridge.mode & BRIDGE_MODE_INJECT) {
            int inj_layer = llm_bridge.inject_layer < 0 ? 0 : llm_bridge.inject_layer;
            if (L == inj_layer)
                tensor_bridge_inject(&llm_bridge, llm_x, dim, pos);
        }

        /* === Self-Attention === */

        /* 2a. Pre-attention norm (LayerNorm for Phi-2, RMSNorm for others) */
        if (m->use_layernorm)
            llm_layernorm(llm_xn, llm_x, layer->attn_norm,
                          layer->attn_norm_bias, dim, layer->attn_norm_type);
        else
            llm_rmsnorm(llm_xn, llm_x, layer->attn_norm, dim, layer->attn_norm_type, m->rms_eps);

        /* Per-layer head_dim for Gemma4 (SWA=256, full=512) */
        int lhd = layer->head_dim_layer ? layer->head_dim_layer : hd;
        int lkv_dim = n_kv * lhd;
        int lq_dim = n_heads * lhd;

        /* 2b. Q projection (always computed) */
        llm_gemv(llm_q,     layer->q_weight, llm_xn, lq_dim, dim, layer->q_type);

        /* K/V projection: skip if reusing another layer's KV cache */
        int has_own_kv = (layer->kv_reuse_layer < 0);
        if (has_own_kv) {
            llm_gemv(llm_k_buf, layer->k_weight, llm_xn, lkv_dim, dim, layer->k_type);
            llm_gemv(llm_v_buf, layer->v_weight, llm_xn, lkv_dim, dim, layer->v_type);
        }

        /* Add bias if present (Phi-2) */
        llm_add_bias(llm_q,     layer->q_bias, lq_dim);
        if (has_own_kv) {
            llm_add_bias(llm_k_buf, layer->k_bias, lkv_dim);
            llm_add_bias(llm_v_buf, layer->v_bias, lkv_dim);
        }

        /* Gemma4: per-head Q/K normalization */
        if (layer->q_norm) {
            const float *qnw = (const float *)layer->q_norm;
            for (int h = 0; h < n_heads; h++) {
                float *qh = llm_q + h * lhd;
                /* RMSNorm per-head: norm then scale by weights */
                float ss = 0.0f;
                for (int i = 0; i < lhd; i++) ss += qh[i] * qh[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                for (int i = 0; i < lhd; i++) qh[i] = qh[i] * rms * qnw[i];
            }
        }
        if (layer->k_norm && has_own_kv) {
            const float *knw = (const float *)layer->k_norm;
            for (int h = 0; h < n_kv; h++) {
                float *kh = llm_k_buf + h * lhd;
                float ss = 0.0f;
                for (int i = 0; i < lhd; i++) ss += kh[i] * kh[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                for (int i = 0; i < lhd; i++) kh[i] = kh[i] * rms * knw[i];
            }
            /* V bare RMSNorm: normalize per-head without learned weights */
            for (int h = 0; h < n_kv; h++) {
                float *vh = llm_v_buf + h * lhd;
                float ss = 0.0f;
                for (int i = 0; i < lhd; i++) ss += vh[i] * vh[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                for (int i = 0; i < lhd; i++) vh[i] *= rms;
            }
        }

        /* 2c. Apply RoPE to Q and K */
        /* Gemma4: different RoPE base for SWA vs full attention layers */
        float layer_rope_base = m->rope_base;
        int   layer_rope_dim  = lhd; /* Gemma4 uses full head_dim for RoPE */
        if (m->is_gemma4 && lhd == m->head_dim_swa)
            layer_rope_base = m->rope_base_swa;

        /* Select longrope factors: short if pos < orig_ctx, long otherwise */
        const float *rope_f = (const float *)0;
        if (m->rope_factors_short || m->rope_factors_long) {
            rope_f = (pos < m->rope_orig_ctx) ? m->rope_factors_short : m->rope_factors_long;
        }
        /* Gemma4: full-attention layers use precomputed rope_freqs as factors */
        if (m->is_gemma4 && m->rope_freqs && lhd != m->head_dim_swa) {
            rope_f = m->rope_freqs;
        }
        /* Use JIT only when head_dim matches compiled kernel */
        if (jit_fwd_rope_hd && lhd == hd && layer_rope_dim == hd && !m->is_gemma4) {
            llm_rope_precompute(layer_rope_base, lhd, rope_f);
            for (int h = 0; h < n_heads; h++)
                jit_fwd_rope_hd(llm_q + h * lhd, pos, lhd, llm_rope_freqs_buf);
            if (has_own_kv)
                for (int h = 0; h < n_kv; h++)
                    jit_fwd_rope_hd(llm_k_buf + h * lhd, pos, lhd, llm_rope_freqs_buf);
        } else {
            int rdim = m->rope_dim > 0 ? m->rope_dim : layer_rope_dim;
            for (int h = 0; h < n_heads; h++)
                llm_rope(llm_q + h * lhd, pos, rdim, layer_rope_base, rope_f);
            if (has_own_kv)
                for (int h = 0; h < n_kv; h++)
                    llm_rope(llm_k_buf + h * lhd, pos, rdim, layer_rope_base, rope_f);
        }

        /* 2d. Store K,V in cache */
        /* KV cache layout: [layer][position][kv_head][max_head_dim]
         * All layers use the same stride (max head_dim = hd) even if
         * this layer's actual lhd < hd — we just zero-pad. */
        int kv_stride = m->max_seq * kv_dim;
        int kv_src_layer = has_own_kv ? L : layer->kv_reuse_layer;
        if (has_own_kv) {
            /* Write K/V to this layer's cache slot.
             * If lhd < hd, we write lkv_dim floats into a kv_dim slot — pad rest with 0. */
            float *kc = m->k_cache + L * kv_stride + pos * kv_dim;
            float *vc = m->v_cache + L * kv_stride + pos * kv_dim;
            if (lkv_dim < kv_dim) {
                kmemset(kc, 0, kv_dim * sizeof(float));
                kmemset(vc, 0, kv_dim * sizeof(float));
            }
            /* For GQA with potentially smaller head_dim, copy per-head */
            for (int h = 0; h < n_kv; h++) {
                kmemcpy(kc + h * hd, llm_k_buf + h * lhd, lhd * sizeof(float));
                kmemcpy(vc + h * hd, llm_v_buf + h * lhd, lhd * sizeof(float));
            }
        }

        /* 2e. Multi-head attention with GQA */
        kmemset(llm_attn_out, 0, lq_dim * sizeof(float));

        int heads_per_kv = n_heads / n_kv;

        /* Gemma4: attention scaling = 1.0 (pre-normalized by Q/K norms) */
        float attn_scale = m->is_gemma4 ? 1.0f : (1.0f / llm_sqrtf((float)lhd));

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;
            float *qh = llm_q + h * lhd;

            int seq_len = pos + 1;
            for (int t = 0; t < seq_len; t++) {
                float *kt = m->k_cache + kv_src_layer * kv_stride + t * kv_dim + kv_h * hd;
                /* Dot product uses lhd (actual head dim for this layer) */
                float d = llm_dot_f32(qh, kt, lhd);
                llm_attn_scores[t] = d * attn_scale;
            }

            /* Softmax over scores */
            llm_softmax(llm_attn_scores, seq_len);

            /* Weighted sum of V */
            kmemset(llm_head_buf, 0, lhd * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                float s = llm_attn_scores[t];
                float *vt = m->v_cache + kv_src_layer * kv_stride + t * kv_dim + kv_h * hd;
                llm_axpy_f32(llm_head_buf, s, vt, lhd);
            }

            kmemcpy(llm_attn_out + h * lhd, llm_head_buf, lhd * sizeof(float));
        }

        /* 2f. Output projection + residual */
        llm_gemv(llm_ffn_d, layer->o_weight, llm_attn_out, dim, lq_dim, layer->o_type);

        /* Gemma4: post-attention RMSNorm */
        if (layer->post_attn_norm) {
            llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->post_attn_norm, dim,
                        GGML_TYPE_F32, m->rms_eps);
        }

        if (jit_fwd_vadd_dim)
            jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
        else
            llm_vadd_f32(llm_x, llm_ffn_d, dim);

        /* === Feed-Forward Network === */

        /* Per-layer FFN dim (Gemma4: 6144 for early layers, 12288 for later) */
        int lff = layer->ff_dim_layer ? layer->ff_dim_layer : ff;

        /* 2g. Pre-FFN norm (LayerNorm for Phi-2, RMSNorm for others) */
        if (m->use_layernorm)
            llm_layernorm(llm_xn, llm_x, layer->ffn_norm,
                          layer->ffn_norm_bias, dim, layer->ffn_norm_type);
        else
            llm_rmsnorm(llm_xn, llm_x, layer->ffn_norm, dim, layer->ffn_norm_type, m->rms_eps);

        if (m->use_gelu || !layer->ffn_gate) {
            /* 2h-alt. GELU FFN (Phi-2): hidden = GELU(W_up · x); out = W_down · hidden */
            llm_gemv(llm_ffn_u, layer->ffn_up, llm_xn, lff, dim, layer->up_type);
            llm_add_bias(llm_ffn_u, layer->ffn_up_bias, lff);
            llm_gelu(llm_ffn_u, lff);
            llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_u, dim, lff, layer->down_type);
            llm_add_bias(llm_ffn_d, layer->ffn_down_bias, dim);
        } else if (m->use_geglu) {
            /* 2h-geglu. GeGLU (Gemma): hidden = GELU(W_gate · x) ⊙ (W_up · x) */
            llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
            llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, lff, dim, layer->up_type);
            llm_gelu(llm_ffn_g, lff);
            llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);
            llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
        } else {
            /* 2h. SwiGLU: hidden = SiLU(W_gate · x) ⊙ (W_up · x) */
            llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
            llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, lff, dim, layer->up_type);
            if (jit_fwd_fused_silu && lff == ff) {
                jit_fwd_fused_silu(llm_ffn_g, llm_ffn_u, lff);
            } else {
                llm_silu(llm_ffn_g, lff);
                llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);
            }
            llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
        }

        /* Gemma4: post-FFW RMSNorm */
        if (layer->post_ffw_norm) {
            llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->post_ffw_norm, dim,
                        GGML_TYPE_F32, m->rms_eps);
        }

        /* Residual: x = x + ffn_output */
        if (jit_fwd_vadd_dim)
            jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
        else
            llm_vadd_f32(llm_x, llm_ffn_d, dim);

        /* === Gemma4 ISWA: Per-layer embedding injection === */
        if (m->is_gemma4 && layer->iswa_inp_gate) {
            int iswa_d = m->iswa_n_embd; /* 256 */

            /* 2j-a. Gate: tmp[256] = GELU(inp_gate[1536,256] · x[1536]) */
            llm_gemv(llm_iswa_tmp, layer->iswa_inp_gate, llm_x, iswa_d, dim,
                     layer->iswa_inp_gate_type);
            llm_gelu(llm_iswa_tmp, iswa_d);

            /* 2j-b. Element-wise multiply with per-layer embedding */
            float *pl = llm_iswa_per_layer + L * iswa_d;
            for (int i = 0; i < iswa_d; i++)
                llm_iswa_tmp[i] *= pl[i];

            /* 2j-c. Project back: ffn_d[1536] = proj[256,1536] · tmp[256] */
            llm_gemv(llm_ffn_d, layer->iswa_proj, llm_iswa_tmp, dim, iswa_d,
                     layer->iswa_proj_type);

            /* 2j-d. Post-norm on projected output */
            if (layer->iswa_post_norm) {
                llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->iswa_post_norm, dim,
                            GGML_TYPE_F32, m->rms_eps);
            }

            /* 2j-e. Residual: x = x + iswa_output */
            llm_vadd_f32(llm_x, llm_ffn_d, dim);
        }

        /* === Gemma4: Per-layer output scaling === */
        if (layer->iswa_out_scale) {
            float s = ((const float *)layer->iswa_out_scale)[0];
            for (int i = 0; i < dim; i++)
                llm_x[i] *= s;
        }

        /* Bridge: capture hidden state after this layer */
        if (llm_bridge.mode & BRIDGE_MODE_CAPTURE) {
            int cap_layer = llm_bridge.capture_layer < 0 ? m->n_layers - 1 : llm_bridge.capture_layer;
            if (L == cap_layer)
                tensor_bridge_capture(&llm_bridge, llm_x, dim, pos);
        }
    }

    /* 3. Final norm (LayerNorm for Phi-2, RMSNorm for others) */
    if (m->use_layernorm)
        llm_layernorm(llm_xn, llm_x, m->output_norm,
                      m->output_norm_bias, dim, m->output_norm_type);
    else
        llm_rmsnorm(llm_xn, llm_x, m->output_norm, dim, m->output_norm_type, m->rms_eps);

    /* 4. LM head: logits = output_weight · x */
    const void *lm_head = m->output_weight ? m->output_weight : m->token_embd;
    ggml_type_t lm_type = m->output_weight ? m->output_type : m->token_embd_type;

    llm_gemv(logits, lm_head, llm_xn, m->vocab_size, dim, lm_type);

    /* 5. Logit softcapping: logits = cap * tanh(logits / cap) */
    if (m->logit_softcap > 0.0f) {
        float cap = m->logit_softcap;
        float inv_cap = 1.0f / cap;
        for (int i = 0; i < m->vocab_size; i++) {
            float x = logits[i] * inv_cap;
            /* tanh approximation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) */
            float e2x = llm_expf(2.0f * x);
            logits[i] = cap * (e2x - 1.0f) / (e2x + 1.0f);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Tokenizer: GGUF Vocabulary Parsing and BPE Encoding                        */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Walk a GGUF string array and extract the i-th string */
static int gguf_array_string_at(const gguf_kv_t *kv, int idx,
                                const char **out_str, int *out_len)
{
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->value.array.elem_type != GGUF_TYPE_STRING)
        return -1;
    if ((uint64_t)idx >= kv->value.array.count) return -1;

    const uint8_t *p = (const uint8_t *)kv->value.array.data;
    for (int i = 0; i <= idx; i++) {
        uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                       ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                       ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
        p += 8;
        if (i == idx) {
            *out_str = (const char *)p;
            *out_len = (int)len;
            return 0;
        }
        p += len;
    }
    return -1;
}

/* Walk a GGUF float32 array and get the i-th element */
static float gguf_array_f32_at(const gguf_kv_t *kv, int idx)
{
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->value.array.elem_type != GGUF_TYPE_FLOAT32)
        return 0.0f;
    if ((uint64_t)idx >= kv->value.array.count) return 0.0f;

    const float *f = (const float *)kv->value.array.data;
    return f[idx];
}

/* Build vocab lookup from GGUF metadata */
static int llm_build_vocab(llm_model_t *m, gguf_ctx_t *ctx)
{
    const gguf_kv_t *tok_kv = gguf_find_kv(ctx, "tokenizer.ggml.tokens");
    if (!tok_kv || tok_kv->type != GGUF_TYPE_ARRAY) {
        kprintf("[LLM] No tokenizer.ggml.tokens found\n");
        return -1;
    }

    int n_vocab = (int)tok_kv->value.array.count;
    if (m->vocab_size > 0 && n_vocab > m->vocab_size)
        n_vocab = m->vocab_size;
    m->n_vocab = n_vocab;

    /* Allocate vocab array */
    m->vocab = (llm_vocab_entry_t *)tensor_alloc(
        (uint64_t)n_vocab * sizeof(llm_vocab_entry_t));
    if (!m->vocab) {
        kprintf("[LLM] ERROR: Failed to allocate vocab (%d entries)\n", n_vocab);
        return -1;
    }

    /* Walk the string array and populate vocab entries */
    const uint8_t *p = (const uint8_t *)tok_kv->value.array.data;
    for (int i = 0; i < n_vocab; i++) {
        uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                       ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                       ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
        p += 8;
        m->vocab[i].str = (const char *)p;
        m->vocab[i].len = (uint16_t)(len > 65535 ? 65535 : len);
        p += len;
    }

    /* Get merge scores */
    const gguf_kv_t *scores_kv = gguf_find_kv(ctx, "tokenizer.ggml.scores");
    if (scores_kv && scores_kv->type == GGUF_TYPE_ARRAY &&
        scores_kv->value.array.elem_type == GGUF_TYPE_FLOAT32) {
        m->vocab_scores = (float *)scores_kv->value.array.data;
    } else {
        m->vocab_scores = NULL;
    }

    /* Get special token IDs */
    m->bos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
    m->eos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.eos_token_id", 2);

    /* Build byte fallback table */
    for (int b = 0; b < 256; b++) m->byte_tokens[b] = -1;

    /* Check for byte tokens like <0x00>, <0x01>, ... or raw byte entries */
    for (int i = 0; i < n_vocab; i++) {
        if (m->vocab[i].len == 6 && m->vocab[i].str[0] == '<' &&
            m->vocab[i].str[1] == '0' && m->vocab[i].str[2] == 'x' &&
            m->vocab[i].str[5] == '>') {
            /* Parse <0xHH> */
            int hi = 0, lo = 0;
            char c3 = m->vocab[i].str[3];
            char c4 = m->vocab[i].str[4];
            if (c3 >= '0' && c3 <= '9') hi = c3 - '0';
            else if (c3 >= 'A' && c3 <= 'F') hi = c3 - 'A' + 10;
            else if (c3 >= 'a' && c3 <= 'f') hi = c3 - 'a' + 10;
            if (c4 >= '0' && c4 <= '9') lo = c4 - '0';
            else if (c4 >= 'A' && c4 <= 'F') lo = c4 - 'A' + 10;
            else if (c4 >= 'a' && c4 <= 'f') lo = c4 - 'a' + 10;
            m->byte_tokens[hi * 16 + lo] = i;
        } else if (m->vocab[i].len == 1) {
            /* Single-byte token */
            unsigned char ch = (unsigned char)m->vocab[i].str[0];
            if (m->byte_tokens[ch] < 0)
                m->byte_tokens[ch] = i;
        }
    }

    kprintf("[LLM] Vocab: %d tokens, BOS=%d, EOS=%d\n",
            n_vocab, m->bos_id, m->eos_id);

    /* Detect tokenizer type: SentencePiece if vocab contains U+2581 (▁) tokens */
    m->use_spm = 0;
    {
        int check_max = n_vocab < 1000 ? n_vocab : 1000;
        for (int i = 0; i < check_max; i++) {
            if (m->vocab[i].len >= 3) {
                const uint8_t *s = (const uint8_t *)m->vocab[i].str;
                for (int j = 0; j + 2 < m->vocab[i].len; j++) {
                    if (s[j] == 0xE2 && s[j+1] == 0x96 && s[j+2] == 0x81) {
                        m->use_spm = 1;
                        break;
                    }
                }
                if (m->use_spm) break;
            }
        }
        kprintf("[LLM] Tokenizer: %s\n", m->use_spm ? "SentencePiece" : "BPE");
    }

    /* Build hash table for O(1) token lookups */
    llm_build_hash_table(m);

    return 0;
}

/* Check if two strings match (str1 may not be null-terminated) */
static int llm_str_match(const char *a, int alen, const char *b, int blen)
{
    if (alen != blen) return 0;
    for (int i = 0; i < alen; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

/* FNV-1a hash for (string, length) pairs */
static uint32_t llm_hash_str(const char *str, int len)
{
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)str[i];
        h *= 0x01000193u;
    }
    return h;
}

/* Hash-table backed token lookup — O(1) average */
static int32_t llm_vocab_ht[LLM_HASH_SIZE];
static int llm_ht_ready = 0;

static void llm_build_hash_table(const llm_model_t *m)
{
    /* Initialize all slots to -1 (empty) */
    for (int i = 0; i < LLM_HASH_SIZE; i++)
        llm_vocab_ht[i] = -1;  /* int32_t: supports vocab up to 2^31 */

    /* Insert all vocab entries using open addressing (linear probing) */
    for (int i = 0; i < m->n_vocab; i++) {
        uint32_t h = llm_hash_str(m->vocab[i].str, m->vocab[i].len);
        uint32_t slot = h & (LLM_HASH_SIZE - 1);
        int probes = 0;
        while (llm_vocab_ht[slot] != -1 && probes < LLM_HASH_SIZE) {
            slot = (slot + 1) & (LLM_HASH_SIZE - 1);
            probes++;
        }
        if (probes < LLM_HASH_SIZE)
            llm_vocab_ht[slot] = (int32_t)i;
    }
    llm_ht_ready = 1;
}

/* Find exact token ID for a string. Hash table O(1) with fallback. */
static int llm_find_token(const llm_model_t *m, const char *str, int len)
{
    if (llm_ht_ready) {
        uint32_t h = llm_hash_str(str, len);
        uint32_t slot = h & (LLM_HASH_SIZE - 1);
        int probes = 0;
        while (probes < 64) { /* limit probe length */
            int32_t idx = llm_vocab_ht[slot];
            if (idx == -1) return -1; /* empty slot = not found */
            if (llm_str_match(m->vocab[idx].str, m->vocab[idx].len, str, len))
                return idx;
            slot = (slot + 1) & (LLM_HASH_SIZE - 1);
            probes++;
        }
        return -1;
    }
    /* Fallback: linear scan (before hash table is built) */
    for (int i = 0; i < m->n_vocab; i++) {
        if (llm_str_match(m->vocab[i].str, m->vocab[i].len, str, len))
            return i;
    }
    return -1;
}

/* Encode raw bytes to GPT-2 byte-level encoding (reverse of bpe_decode).
 * Maps control chars, space, etc. to their U+0100..U+0143 Unicode forms. */
static int llm_bpe_encode(const char *src, int slen, char *dst, int dmax)
{
    int di = 0;
    for (int si = 0; si < slen && di < dmax - 2; si++) {
        uint8_t b = (uint8_t)src[si];
        /* Passthrough bytes: ASCII printable (except space) + Latin-1 parts */
        if ((b >= 0x21 && b <= 0x7E) ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            dst[di++] = (char)b;
        } else {
            /* Remap to U+0100..U+0143 as UTF-8 */
            int idx;
            if (b <= 0x20)       idx = b;              /* 0x00..0x20 → 0..32 */
            else if (b == 0x7F)  idx = 33;
            else if (b <= 0xA0)  idx = 34 + (b - 0x80); /* 0x80..0xA0 → 34..66 */
            else                 idx = 67;              /* 0xAD → 67 */
            int cp = 0x100 + idx;
            dst[di++] = (char)(0xC0 | (cp >> 6));
            dst[di++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    dst[di] = '\0';
    return di;
}

/* Tokenize text using greedy longest-match with BPE merge fallback.
 * Returns number of tokens produced. */
/* Helper: tokenize a text segment (no special tokens) */
static int llm_tokenize_segment(const llm_model_t *m, const char *text, int text_len,
                                 int *tokens, int max_tokens, int is_first_segment)
{
    int n = 0;
    static char enc_buf[4096];
    int enc_len;

    if (m->use_spm) {
        int di = 0;
        int dmax = (int)sizeof(enc_buf) - 4;
        /* Prepend ▁ for SentencePiece (only for first segment or after special token) */
        if (is_first_segment && text_len > 0 && text[0] != '\n') {
            enc_buf[di++] = (char)0xE2;
            enc_buf[di++] = (char)0x96;
            enc_buf[di++] = (char)0x81;
        }
        for (int si = 0; si < text_len && di < dmax; si++) {
            if (text[si] == ' ') {
                enc_buf[di++] = (char)0xE2;
                enc_buf[di++] = (char)0x96;
                enc_buf[di++] = (char)0x81;
            } else {
                enc_buf[di++] = text[si];
            }
        }
        enc_buf[di] = '\0';
        enc_len = di;
    } else {
        enc_len = llm_bpe_encode(text, text_len, enc_buf, sizeof(enc_buf));
    }

    /* Greedy longest-match tokenization */
    int pos = 0;
    while (pos < enc_len && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;
        int max_try = enc_len - pos;
        if (max_try > 128) max_try = 128;
        for (int try_len = max_try; try_len >= 1; try_len--) {
            int id = llm_find_token(m, enc_buf + pos, try_len);
            if (id >= 0) { best_len = try_len; best_id = id; break; }
        }
        if (best_id >= 0) {
            tokens[n++] = best_id;
            pos += best_len;
        } else {
            uint8_t b0 = (uint8_t)enc_buf[pos];
            if (b0 >= 0xC0 && b0 < 0xE0 && pos + 1 < enc_len) {
                int id = llm_find_token(m, enc_buf + pos, 2);
                if (id >= 0) { tokens[n++] = id; pos += 2; continue; }
            }
            if (m->byte_tokens[b0] >= 0) {
                tokens[n++] = m->byte_tokens[b0];
            } else {
                tokens[n++] = 0;
            }
            pos++;
        }
    }

    /* BPE merge pass */
    if (m->vocab_scores && n > 1) {
        int changed = 1;
        while (changed) {
            changed = 0;
            float best_score = -1e30f;
            int best_idx = -1, best_merged = -1;
            for (int i = 0; i < n - 1; i++) {
                int len1 = m->vocab[tokens[i]].len;
                int len2 = m->vocab[tokens[i + 1]].len;
                if (len1 + len2 > 128) continue;
                char merged[128];
                kmemcpy(merged, m->vocab[tokens[i]].str, len1);
                kmemcpy(merged + len1, m->vocab[tokens[i + 1]].str, len2);
                int mid = llm_find_token(m, merged, len1 + len2);
                if (mid >= 0 && m->vocab_scores[mid] > best_score) {
                    best_score = m->vocab_scores[mid];
                    best_idx = i;
                    best_merged = mid;
                }
            }
            if (best_idx >= 0) {
                tokens[best_idx] = best_merged;
                for (int i = best_idx + 1; i < n - 1; i++)
                    tokens[i] = tokens[i + 1];
                n--;
                changed = 1;
            }
        }
    }
    return n;
}

static int llm_tokenize(const llm_model_t *m, const char *text,
                        int *tokens, int max_tokens)
{
    int n = 0;
    int text_len = 0;
    for (const char *p = text; *p; p++) text_len++;

    /* Pre-split at special token boundaries.
     * Detect both <|...|> patterns (GPT/Phi) and <..._of_...> patterns (Gemma). */
    int pos = 0;
    int first_seg = 1;
    while (pos < text_len && n < max_tokens) {
        /* Scan for next special token pattern */
        int spec_start = -1, spec_end = -1;
        for (int i = pos; i < text_len - 2; i++) {
            if (text[i] == '<') {
                /* Try <|...|> pattern */
                if (i + 3 < text_len && text[i+1] == '|') {
                    for (int j = i + 2; j < text_len - 1 && j < i + 32; j++) {
                        if (text[j] == '|' && text[j+1] == '>') {
                            spec_start = i;
                            spec_end = j + 2;
                            break;
                        }
                    }
                    if (spec_start >= 0) break;
                }
                /* Try <...> pattern — look up in vocab to confirm it's a special token */
                for (int j = i + 1; j < text_len && j < i + 32; j++) {
                    if (text[j] == '>') {
                        int cand_len = j + 1 - i;
                        int cand_id = llm_find_token(m, text + i, cand_len);
                        if (cand_id >= 0) {
                            spec_start = i;
                            spec_end = j + 1;
                        }
                        break;
                    }
                }
                if (spec_start >= 0) break;
            }
        }

        if (spec_start < 0) {
            /* No more special tokens — tokenize remaining text */
            int seg_n = llm_tokenize_segment(m, text + pos, text_len - pos,
                                              tokens + n, max_tokens - n, first_seg);
            n += seg_n;
            break;
        }

        /* Tokenize text before special token */
        if (spec_start > pos) {
            int seg_n = llm_tokenize_segment(m, text + pos, spec_start - pos,
                                              tokens + n, max_tokens - n, first_seg);
            n += seg_n;
            first_seg = 0;
        }

        /* Look up the special token directly in vocab */
        int spec_len = spec_end - spec_start;
        int spec_id = llm_find_token(m, text + spec_start, spec_len);
        if (spec_id >= 0 && n < max_tokens) {
            tokens[n++] = spec_id;
        } else if (n < max_tokens) {
            /* Special token not in vocab — tokenize as regular text */
            int seg_n = llm_tokenize_segment(m, text + spec_start, spec_len,
                                              tokens + n, max_tokens - n, 0);
            n += seg_n;
        }
        first_seg = 0;
        pos = spec_end;
    }

    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GPT-2 BPE Byte-Level Decode                                                */
/*  Reverses the byte_encoder mapping used by GPT-2/SmolLM/LLaMA tokenizers    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Reverse table: index i → original byte for unicode U+0100+i */
static const uint8_t bpe_rev[68] = {
    /* 0-32: U+0100..U+0120 → byte 0x00..0x20 */
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x20, /* space */
    /* 33: U+0121 → 0x7F */
    0x7F,
    /* 34-66: U+0122..U+0142 → 0x80..0xA0 */
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F,
    0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,
    0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F,
    0xA0,
    /* 67: U+0143 → 0xAD */
    0xAD
};

static int llm_bpe_decode(const char *src, int slen, char *dst, int dmax)
{
    int di = 0;
    for (int si = 0; si < slen && di < dmax - 1; ) {
        uint8_t c = (uint8_t)src[si];
        if (c == 0xC4 && si + 1 < slen) {
            uint8_t c2 = (uint8_t)src[si + 1];
            if (c2 >= 0x80 && c2 <= 0xBF) {
                int idx = c2 - 0x80;
                if (idx < 68) { dst[di++] = (char)bpe_rev[idx]; si += 2; continue; }
            }
        } else if (c == 0xC5 && si + 1 < slen) {
            uint8_t c2 = (uint8_t)src[si + 1];
            if (c2 >= 0x80 && c2 <= 0x83) {
                int idx = 64 + (c2 - 0x80);
                if (idx < 68) { dst[di++] = (char)bpe_rev[idx]; si += 2; continue; }
            }
        }
        dst[di++] = src[si++];
    }
    dst[di] = '\0';
    return di;
}

/* Decode token to text with GPT-2 BPE byte-level reverse mapping */
static int llm_decode_token(const llm_model_t *m, int token_id,
                            char *buf, int max_len)
{
    if (token_id < 0 || token_id >= m->n_vocab) return 0;
    int len = m->vocab[token_id].len;
    if (len > 255) len = 255;
    char tmp[256];
    kmemcpy(tmp, m->vocab[token_id].str, len);
    tmp[len] = '\0';
    return llm_bpe_decode(tmp, len, buf, max_len);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Text Generation                                                            */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Argmax over logits */
static int llm_argmax(const float *logits, int n)
{
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* Sort indices by descending logit value (insertion sort for top-k) */
static void llm_partial_sort_desc(int *indices, const float *vals, int n, int k)
{
    /* Initialize first k indices */
    for (int i = 0; i < k && i < n; i++)
        indices[i] = i;

    /* Sort first k by insertion sort */
    for (int i = 1; i < k && i < n; i++) {
        int key = indices[i];
        float key_val = vals[key];
        int j = i - 1;
        while (j >= 0 && vals[indices[j]] < key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    /* For remaining elements, insert if larger than smallest in top-k */
    for (int i = k; i < n; i++) {
        int last = (k < n) ? k - 1 : n - 1;
        if (vals[i] > vals[indices[last]]) {
            indices[last] = i;
            /* Bubble up */
            int j = last - 1;
            while (j >= 0 && vals[indices[j]] < vals[indices[j + 1]]) {
                int tmp = indices[j]; indices[j] = indices[j + 1]; indices[j + 1] = tmp;
                j--;
            }
        }
    }
}

/* Top-k + top-p (nucleus) sampling with temperature */
static int llm_sample(const float *logits, int vocab_size, float temperature)
{
    if (temperature <= 0.001f)
        return llm_argmax(logits, vocab_size);

    /* Top-k filtering: keep only top 40 candidates */
    #define LLM_TOP_K 40
    static int top_indices[LLM_TOP_K];
    static float top_probs[LLM_TOP_K];

    int k = LLM_TOP_K;
    if (k > vocab_size) k = vocab_size;

    llm_partial_sort_desc(top_indices, logits, vocab_size, k);

    /* Apply temperature and compute softmax over top-k */
    float max_val = logits[top_indices[0]];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        top_probs[i] = llm_expf((logits[top_indices[i]] - max_val) / temperature);
        sum += top_probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < k; i++)
        top_probs[i] *= inv_sum;

    /* Top-p (nucleus) filtering: keep tokens until cumulative prob >= 0.9 */
    float top_p = 0.9f;
    float cum = 0.0f;
    int p_cutoff = k;
    for (int i = 0; i < k; i++) {
        cum += top_probs[i];
        if (cum >= top_p) {
            p_cutoff = i + 1;
            break;
        }
    }

    /* Re-normalize after top-p cutoff */
    if (p_cutoff < k) {
        sum = 0.0f;
        for (int i = 0; i < p_cutoff; i++)
            sum += top_probs[i];
        inv_sum = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < p_cutoff; i++)
            top_probs[i] *= inv_sum;
    }

    /* Sample using CSPRNG */
    uint32_t rng_val;
    crypto_random(&rng_val, sizeof(rng_val));
    float r = (float)(rng_val % 10000) / 10000.0f;

    float cdf = 0.0f;
    for (int i = 0; i < p_cutoff; i++) {
        cdf += top_probs[i];
        if (cdf >= r) return top_indices[i];
    }
    return top_indices[p_cutoff - 1];
    #undef LLM_TOP_K
}

/* Generate text tokens autoregressively.
 * prompt_tokens: input token IDs (n_prompt tokens)
 * output_text: buffer for generated text
 * max_gen: maximum tokens to generate
 * temperature: sampling temperature (0.0 = greedy)
 * continue_cache: if nonzero, don't reset KV cache (multi-turn)
 * Returns: number of generated tokens */
static int llm_generate(llm_model_t *m, const int *prompt_tokens, int n_prompt,
                        char *output_text, int max_text_len,
                        int max_gen, float temperature, int continue_cache)
{
    int start_pos = 0;

    if (!continue_cache) {
        /* Reset KV cache — fast memset instead of scalar loop */
        m->cache_len = 0;
        int kv_total = m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim;
        if (kv_total > llm_alloc_kv_floats) kv_total = llm_alloc_kv_floats;
        kmemset(m->k_cache, 0, (uint64_t)kv_total * sizeof(float));
        kmemset(m->v_cache, 0, (uint64_t)kv_total * sizeof(float));
    } else {
        start_pos = m->cache_len;
    }

    int out_pos = 0;
    int gen_count = 0;
    output_text[0] = '\0';

    /* Process prompt tokens (prefill) */
    uint64_t t_prefill_start = rdtsc_fenced();
    for (int i = 0; i < n_prompt && (start_pos + i) < m->max_seq - 1; i++) {
        llm_forward_token(m, llm_logits, prompt_tokens[i], start_pos + i);
    }
    uint64_t t_prefill_end = rdtsc_fenced();

    /* Generate new tokens */
    int last_token = (n_prompt > 0) ? prompt_tokens[n_prompt - 1] : m->bos_id;
    int pos = start_pos + n_prompt;

    /* Get next token from the last forward pass */
    int next = llm_sample(llm_logits, m->vocab_size, temperature);

    uint64_t t_gen_start = rdtsc_fenced();
    int consec_nl = 0;  /* consecutive newline counter */

    for (int g = 0; g < max_gen && pos < m->max_seq; g++) {
        /* Check for EOS BEFORE decoding its text */
        if (next == m->eos_id) break;
        /* Gemma turn marker (106 = <turn|>) always signals end of response */
        if (next == 106) break;

        /* Decode token to text */
        char tok_buf[128];
        int tok_len = llm_decode_token(m, next, tok_buf, sizeof(tok_buf));
        for (int i = 0; i < tok_len && out_pos < max_text_len - 1; i++)
            output_text[out_pos++] = tok_buf[i];
        output_text[out_pos] = '\0';
        gen_count++;

        /* Streaming callback: emit token, replacing UTF-8 ▁ (E2 96 81) with space */
        if (llm_stream_cb) {
            char stream_buf[128];
            int si = 0;
            for (int i = 0; i < tok_len; ) {
                unsigned char c = (unsigned char)tok_buf[i];
                if (c == 0xE2 && i + 2 < tok_len &&
                    (unsigned char)tok_buf[i+1] == 0x96 &&
                    (unsigned char)tok_buf[i+2] == 0x81) {
                    stream_buf[si++] = ' ';
                    i += 3;
                } else {
                    stream_buf[si++] = tok_buf[i++];
                }
            }
            if (si > 0) llm_stream_cb(stream_buf, si, llm_stream_cb_ud);
        }

        /* Check for <|im_end|> or <|endoftext|> in recent text */
        if (out_pos >= 10) {
            const char *tail = output_text + out_pos - 10;
            int stop = 0;
            for (int s = 0; s < 10 && !stop; s++) {
                if (tail[s] == '<' && tail[s+1] == '|') {
                    stop = 1;
                    int trunc_at = (int)(tail - output_text) + s;
                    output_text[trunc_at] = '\0';
                    out_pos = trunc_at;
                }
            }
            if (stop) break;
        }

        /* Track consecutive newlines — stop after 3 in a row (model is done) */
        {
            int has_nl = 0;
            for (int i = 0; i < tok_len; i++)
                if (tok_buf[i] == '\n') has_nl = 1;
            if (has_nl) consec_nl++;
            else consec_nl = 0;
            if (consec_nl >= 3) break;
        }

        /* Forward the new token */
        llm_forward_token(m, llm_logits, next, pos);
        pos++;

        /* Sample next token */
        last_token = next;
        next = llm_sample(llm_logits, m->vocab_size, temperature);
    }
    uint64_t t_gen_end = rdtsc_fenced();

    /* Final cleanup: strip any <|...|> artifacts from output */
    {
        for (int i = 0; i + 1 < out_pos; i++) {
            if (output_text[i] == '<' && output_text[i+1] == '|') {
                output_text[i] = '\0';
                out_pos = i;
                break;
            }
        }
        /* Trim trailing whitespace */
        while (out_pos > 0 && (output_text[out_pos-1] == ' ' || output_text[out_pos-1] == '\n'))
            output_text[--out_pos] = '\0';
    }

    /* Update cache length for multi-turn */
    m->cache_len = pos;

    /* Print timing stats */
    if (gen_count > 0) {
        uint64_t prefill_us = perf_cycles_to_us(t_prefill_end - t_prefill_start);
        uint64_t gen_us = perf_cycles_to_us(t_gen_end - t_gen_start);
        uint64_t ms_per_tok = gen_us / (uint64_t)gen_count / 1000;
        kprintf("\n[%d tok, %lu ms/tok, prefill %lu ms, %d cpus]\n",
                gen_count, (unsigned long)ms_per_tok,
                (unsigned long)(prefill_us / 1000),
                smp.ap_started + 1);
    }

    return gen_count;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GGUF Model Loading and Tensor Mapping                                      */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Map GGUF tensor names to model layer pointers */
static int llm_map_tensors(llm_model_t *m, gguf_ctx_t *ctx)
{
    /* Token embeddings */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "token_embd.weight");
        if (!t) {
            kprintf("[LLM] ERROR: token_embd.weight not found\n");
            return -1;
        }
        m->token_embd = t->data;
        m->token_embd_type = t->type;
        kprintf("[LLM] token_embd.weight: type=%d dims=[%llu,%llu]\n",
                (int)t->type, t->dims[0], t->dims[1]);
    }

    /* Output norm */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output_norm.weight");
        if (!t) {
            kprintf("[LLM] ERROR: output_norm.weight not found\n");
            return -1;
        }
        m->output_norm = t->data;
        m->output_norm_type = t->type;
    }

    /* Output norm bias (Phi-2 LayerNorm) */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output_norm.bias");
        m->output_norm_bias = t ? t->data : NULL;
    }

    /* Output / LM head — may be absent if tied to embeddings */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output.weight");
        if (t) {
            m->output_weight = t->data;
            m->output_type = t->type;
        } else {
            m->output_weight = NULL; /* will use token_embd */
            m->output_type = m->token_embd_type;
            kprintf("[LLM] Note: output.weight tied to token_embd\n");
        }
    }

    /* RoPE scaling factors (Longrope / Su scaling for Phi-3.5) */
    {
        const gguf_tensor_info_t *ts = gguf_find_tensor(ctx, "rope_factors_short.weight");
        const gguf_tensor_info_t *tl = gguf_find_tensor(ctx, "rope_factors_long.weight");
        m->rope_factors_short = ts ? (const float *)ts->data : (const float *)0;
        m->rope_factors_long  = tl ? (const float *)tl->data : (const float *)0;
        if (ts || tl)
            kprintf("[LLM] Longrope factors: short=%p long=%p\n",
                    (const void *)m->rope_factors_short, (const void *)m->rope_factors_long);
    }

    /* Gemma4 ISWA global weights */
    if (m->is_gemma4 && m->iswa_n_embd > 0) {
        const gguf_tensor_info_t *t;
        t = gguf_find_tensor(ctx, "per_layer_token_embd.weight");
        if (t) { m->iswa_tok_embd = t->data; m->iswa_tok_embd_type = t->type;
                 kprintf("[LLM] ISWA: per_layer_token_embd [%llu x %llu]\n", t->dims[0], t->dims[1]); }
        t = gguf_find_tensor(ctx, "per_layer_model_proj.weight");
        if (t) { m->iswa_model_proj = t->data; m->iswa_model_proj_type = t->type; }
        t = gguf_find_tensor(ctx, "per_layer_proj_norm.weight");
        if (t) { m->iswa_proj_norm = t->data; }
        t = gguf_find_tensor(ctx, "rope_freqs.weight");
        if (t) { m->rope_freqs = (const float *)t->data;
                 kprintf("[LLM] ISWA: rope_freqs [%llu]\n", t->dims[0]); }
    }

    /* Per-layer weights */
    char name_buf[128];
    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];

        #define MAP_TENSOR(field, suffix, type_field) do { \
            kstrcpy(name_buf, "blk."); \
            /* Append layer number */ \
            { \
                char num[8]; int n = L, j = 0; \
                if (n == 0) { num[j++] = '0'; } \
                else { char tmp[8]; int k = 0; while (n > 0) { tmp[k++] = '0' + (n % 10); n /= 10; } \
                       while (k > 0) num[j++] = tmp[--k]; } \
                num[j] = '\0'; \
                kstrcpy(name_buf + kstrlen(name_buf), num); \
            } \
            kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf); \
            if (t) { layer->field = t->data; layer->type_field = t->type; } \
            else { layer->field = NULL; } \
        } while(0)

        MAP_TENSOR(attn_norm,  "attn_norm.weight",   attn_norm_type);
        MAP_TENSOR(q_weight,   "attn_q.weight",      q_type);
        MAP_TENSOR(k_weight,   "attn_k.weight",      k_type);
        MAP_TENSOR(v_weight,   "attn_v.weight",      v_type);
        MAP_TENSOR(o_weight,   "attn_output.weight",  o_type);
        MAP_TENSOR(ffn_norm,   "ffn_norm.weight",     ffn_norm_type);
        MAP_TENSOR(ffn_gate,   "ffn_gate.weight",     gate_type);
        MAP_TENSOR(ffn_up,     "ffn_up.weight",       up_type);
        MAP_TENSOR(ffn_down,   "ffn_down.weight",     down_type);

        /* Fused QKV tensor (Phi-3: blk.N.attn_qkv.weight) */
        if (!layer->q_weight || !layer->k_weight || !layer->v_weight) {
            #define BUILD_BLK_NAME(suffix) do { \
                kstrcpy(name_buf, "blk."); \
                { char num[8]; int n_ = L, j_ = 0; \
                  if (n_ == 0) { num[j_++] = '0'; } \
                  else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; } \
                         while (k_ > 0) num[j_++] = tmp[--k_]; } \
                  num[j_] = '\0'; \
                  kstrcpy(name_buf + kstrlen(name_buf), num); } \
                kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            } while(0)

            BUILD_BLK_NAME("attn_qkv.weight");
            const gguf_tensor_info_t *qkv = gguf_find_tensor(ctx, name_buf);
            if (qkv) {
                uint64_t row_bytes = ggml_tensor_size(qkv->type, (uint64_t)m->dim);
                const uint8_t *base = (const uint8_t *)qkv->data;
                int q_rows = m->n_heads * m->head_dim;
                int k_rows = m->n_kv_heads * m->head_dim;
                layer->q_weight = base;
                layer->k_weight = base + (uint64_t)q_rows * row_bytes;
                layer->v_weight = base + (uint64_t)(q_rows + k_rows) * row_bytes;
                layer->q_type = layer->k_type = layer->v_type = qkv->type;
            }
            #undef BUILD_BLK_NAME
        }

        /* Fused gate+up tensor (Phi-3: ffn_up.weight has 2×ff_dim rows) */
        if (!layer->ffn_gate && layer->ffn_up && m->ff_dim > 0) {
            kstrcpy(name_buf, "blk.");
            { char num[8]; int n_ = L, j_ = 0;
              if (n_ == 0) { num[j_++] = '0'; }
              else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; }
                     while (k_ > 0) num[j_++] = tmp[--k_]; }
              num[j_] = '\0';
              kstrcpy(name_buf + kstrlen(name_buf), num); }
            kstrcpy(name_buf + kstrlen(name_buf), ".ffn_up.weight");
            const gguf_tensor_info_t *up_t = gguf_find_tensor(ctx, name_buf);
            if (up_t && up_t->n_dims >= 2 && (int)up_t->dims[1] == 2 * m->ff_dim) {
                uint64_t row_bytes = ggml_tensor_size(up_t->type, (uint64_t)m->dim);
                const uint8_t *base = (const uint8_t *)up_t->data;
                layer->ffn_gate = base;
                layer->gate_type = up_t->type;
                layer->ffn_up = base + (uint64_t)m->ff_dim * row_bytes;
                layer->up_type = up_t->type;
            }
        }

        /* Bias tensors (Phi-2; silently NULL for architectures without bias) */
        #define MAP_BIAS(field, suffix) do { \
            kstrcpy(name_buf, "blk."); \
            { \
                char num[8]; int n = L, j = 0; \
                if (n == 0) { num[j++] = '0'; } \
                else { char tmp[8]; int k = 0; while (n > 0) { tmp[k++] = '0' + (n % 10); n /= 10; } \
                       while (k > 0) num[j++] = tmp[--k]; } \
                num[j] = '\0'; \
                kstrcpy(name_buf + kstrlen(name_buf), num); \
            } \
            kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf); \
            layer->field = t ? t->data : NULL; \
        } while(0)

        MAP_BIAS(attn_norm_bias,  "attn_norm.bias");
        MAP_BIAS(q_bias,          "attn_q.bias");
        MAP_BIAS(k_bias,          "attn_k.bias");
        MAP_BIAS(v_bias,          "attn_v.bias");
        MAP_BIAS(o_bias,          "attn_output.bias");
        MAP_BIAS(ffn_norm_bias,   "ffn_norm.bias");
        MAP_BIAS(ffn_up_bias,     "ffn_up.bias");
        MAP_BIAS(ffn_down_bias,   "ffn_down.bias");

        /* Fused QKV bias fallback (Phi-3: blk.N.attn_qkv.bias) */
        if (!layer->q_bias && !layer->k_bias && !layer->v_bias) {
            kstrcpy(name_buf, "blk.");
            { char num[8]; int n_ = L, j_ = 0;
              if (n_ == 0) { num[j_++] = '0'; }
              else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; }
                     while (k_ > 0) num[j_++] = tmp[--k_]; }
              num[j_] = '\0';
              kstrcpy(name_buf + kstrlen(name_buf), num); }
            kstrcpy(name_buf + kstrlen(name_buf), ".attn_qkv.bias");
            const gguf_tensor_info_t *qkv_b = gguf_find_tensor(ctx, name_buf);
            if (qkv_b) {
                const float *bb = (const float *)qkv_b->data;
                int q_dim = m->n_heads * m->head_dim;
                int k_dim = m->n_kv_heads * m->head_dim;
                layer->q_bias = bb;
                layer->k_bias = bb + q_dim;
                layer->v_bias = bb + q_dim + k_dim;
            }
        }

        #undef MAP_BIAS

        /* Gemma4: per-layer Q/K norm, post-attention norm, post-FFW norm, ISWA */
        layer->q_norm = (void *)0;
        layer->k_norm = (void *)0;
        layer->post_attn_norm = (void *)0;
        layer->post_ffw_norm = (void *)0;
        layer->iswa_inp_gate = (void *)0;
        layer->iswa_proj = (void *)0;
        layer->iswa_post_norm = (void *)0;
        layer->iswa_out_scale = (void *)0;
        layer->head_dim_layer = m->head_dim;
        layer->kv_reuse_layer = -1;
        layer->ff_dim_layer = m->ff_dim;
        if (m->is_gemma4) {
            #define BUILD_BLK_NAME(suffix) do { \
                kstrcpy(name_buf, "blk."); \
                { char num[8]; int n_ = L, j_ = 0; \
                  if (n_ == 0) { num[j_++] = '0'; } \
                  else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; } \
                         while (k_ > 0) num[j_++] = tmp[--k_]; } \
                  num[j_] = '\0'; \
                  kstrcpy(name_buf + kstrlen(name_buf), num); } \
                kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            } while(0)
            /* Q/K norms: blk.N.attn_q_norm.weight, blk.N.attn_k_norm.weight */
            BUILD_BLK_NAME("attn_q_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->q_norm = t->data; } }
            BUILD_BLK_NAME("attn_k_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->k_norm = t->data; } }
            /* Post-attention and post-FFW norms */
            BUILD_BLK_NAME("post_attention_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->post_attn_norm = t->data; } }
            BUILD_BLK_NAME("post_ffw_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->post_ffw_norm = t->data; } }
            /* ISWA per-layer injection tensors */
            BUILD_BLK_NAME("inp_gate.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_inp_gate = t->data; layer->iswa_inp_gate_type = t->type; } }
            BUILD_BLK_NAME("proj.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_proj = t->data; layer->iswa_proj_type = t->type; } }
            BUILD_BLK_NAME("post_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_post_norm = t->data; } }
            BUILD_BLK_NAME("layer_output_scale.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_out_scale = t->data; } }

            /* Determine per-layer head_dim from Q weight shape */
            BUILD_BLK_NAME("attn_q.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t && t->n_dims >= 2) {
                  layer->head_dim_layer = (int)(t->dims[1]) / m->n_heads;
              } }

            /* Determine per-layer ff_dim from ffn_gate shape */
            BUILD_BLK_NAME("ffn_gate.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t && t->n_dims >= 2) {
                  layer->ff_dim_layer = (int)(t->dims[1]);
              } }

            /* KV cache reuse for layers >= n_layer_kv_start */
            if (L >= m->n_layer_kv_start) {
                /* SWA layers reuse kv_start-2, full attn layers reuse kv_start-1 */
                int is_swa_layer = (layer->head_dim_layer == m->head_dim_swa);
                layer->kv_reuse_layer = m->n_layer_kv_start - (is_swa_layer ? 2 : 1);
                if (layer->kv_reuse_layer < 0) layer->kv_reuse_layer = 0;
            }

            if (L == 0 || L == 4 || L == 15)
                kprintf("[LLM] L%d: hd=%d ff=%d kv_reuse=%d q_norm=%p proj=%p scale=%p\n",
                        L, layer->head_dim_layer, layer->ff_dim_layer,
                        layer->kv_reuse_layer, layer->q_norm, layer->iswa_proj,
                        layer->iswa_out_scale);
        }

        #undef BUILD_BLK_NAME
        #undef MAP_TENSOR

        /* Validate critical tensors are mapped (after fused fallback) */
        if (L == 0) {
            if (!layer->q_weight) kprintf("[LLM] WARN: blk.0 Q weight missing\n");
            if (!layer->k_weight) kprintf("[LLM] WARN: blk.0 K weight missing\n");
            if (!layer->v_weight) kprintf("[LLM] WARN: blk.0 V weight missing\n");
            if (!layer->ffn_down) kprintf("[LLM] WARN: blk.0 ffn_down missing\n");
        }
    }

    kprintf("[LLM] Tensor mapping complete: %d layers\n", m->n_layers);
    return 0;
}

/* Load model from virtio-blk disk into model_cache memory region */
static int llm_init_parsed_model(llm_model_t *m);

static int llm_load_from_disk(llm_model_t *m)
{
#ifdef __aarch64__
    kprintf("[LLM] Disk loading not supported on ARM64 yet\n");
    return -1;
#else
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("[LLM] No block device or zero capacity\n");
        return -1;
    }

    /* Check if capacity is reasonable for a GGUF file (at least 1MB) */
    if (capacity < 1024 * 1024) {
        kprintf("[LLM] Disk too small for a model: %lu bytes\n", capacity);
        return -1;
    }

    /* Cap at available model cache size */
    uint64_t cache_size = tensor_mm_model_cache_max();
    if (capacity > cache_size) {
        kprintf("[LLM] Model (%lu MB) exceeds cache (%lu MB)\n",
                capacity / (1024 * 1024), cache_size / (1024 * 1024));
        return -1;
    }

    kprintf("[LLM] Loading model from disk: %lu MB\n", capacity / (1024 * 1024));

    /* Read the entire disk into model cache region */
    m->data_buf = tensor_mm_model_cache_base();
    m->data_size = capacity;

    uint64_t n_sectors = (capacity + 511) / 512;
    uint64_t chunk_sectors = 2048; /* 1MB chunks */
    uint64_t sectors_done = 0;

    uint64_t t0 = rdtsc_fenced();
    while (sectors_done < n_sectors) {
        uint64_t remain = n_sectors - sectors_done;
        uint32_t chunk = (remain > chunk_sectors) ? (uint32_t)chunk_sectors : (uint32_t)remain;
        uint8_t *dst = (uint8_t *)m->data_buf + sectors_done * 512;

        int rc = virtio_blk_read(sectors_done, chunk, dst);
        if (rc != 0) {
            kprintf("[LLM] Disk read error at sector %lu: %d\n", sectors_done, rc);
            return -1;
        }
        sectors_done += chunk;

        /* Progress every 64MB */
        if ((sectors_done * 512) % (64 * 1024 * 1024) == 0) {
            kprintf("[LLM] ... %lu MB loaded\n", (sectors_done * 512) / (1024 * 1024));
        }
    }
    uint64_t t1 = rdtsc_fenced();

    uint64_t load_ms = perf_cycles_to_us(t1 - t0) / 1000;
    uint64_t mbps = (load_ms > 0) ? (capacity / 1024 * 1000 / load_ms) : 0;
    kprintf("[LLM] Loaded %lu MB in %lu ms (%lu KB/s)\n",
            capacity / (1024 * 1024), load_ms, mbps);

    return llm_init_parsed_model(m);
#endif /* __aarch64__ */
}

/* Hosted-mode loader: load model from a pre-mapped memory buffer */
int llm_load_from_buffer(void *data, uint64_t size)
{
    if (!data || size < 1024) {
        kprintf("[LLM] Invalid buffer (ptr=%p, size=%lu)\n", data, (unsigned long)size);
        return -1;
    }

    model_format_t fmt = model_detect_format(data, size);
    if (fmt == MODEL_FMT_UNKNOWN) {
        kprintf("[LLM] Unrecognized model format\n");
        return -1;
    }
    if (fmt != MODEL_FMT_GGUF) {
        static const char *fmt_names[] = {"unknown","gguf","safetensors","onnx","pytorch"};
        kprintf("[LLM] Detected format: %s (not yet supported for inference, use GGUF)\n",
                fmt_names[fmt]);
        return -1;
    }

    kmemset(&llm_model, 0, sizeof(llm_model));
    llm_model.data_buf = data;
    llm_model.data_size = size;
    kprintf("[LLM] Loading model from buffer: %lu MB\n",
            (unsigned long)(size / (1024 * 1024)));
    return llm_init_parsed_model(&llm_model);
}

/* Shared model initialization after data is in memory */
static int llm_init_parsed_model(llm_model_t *m)
{
    /* Parse GGUF */
    kprintf("[LLM] Parsing GGUF...\n");
    int rc = gguf_parse(&llm_gguf_ctx, m->data_buf, m->data_size);
    if (rc != 0) {
        kprintf("[LLM] GGUF parse error: %d\n", rc);
        return -1;
    }
    m->gguf = &llm_gguf_ctx;

    /* Extract architecture parameters */
    kstrcpy(m->arch, llm_gguf_ctx.arch);
    m->dim = (int)llm_gguf_ctx.n_embd;
    m->n_layers = (int)llm_gguf_ctx.n_layers;
    m->n_heads = (int)llm_gguf_ctx.n_heads;
    m->n_kv_heads = (int)llm_gguf_ctx.n_kv_heads;
    m->ff_dim = (int)llm_gguf_ctx.n_ff;
    m->vocab_size = (int)llm_gguf_ctx.n_vocab;
    m->rope_base = llm_gguf_ctx.rope_freq_base;

    /* If vocab_size is 0, infer from token_embd dimensions or tokenizer */
    if (m->vocab_size == 0) {
        const gguf_tensor_info_t *embd = gguf_find_tensor(&llm_gguf_ctx, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            /* GGUF dims: [ne0=n_embd, ne1=n_vocab] for embedding matrix */
            m->vocab_size = (int)embd->dims[1];
        }
    }
    /* Cross-check with tokenizer array count (most reliable source) */
    {
        const gguf_kv_t *tok_kv = gguf_find_kv(&llm_gguf_ctx, "tokenizer.ggml.tokens");
        if (tok_kv && tok_kv->type == GGUF_TYPE_ARRAY && tok_kv->value.array.count > 0) {
            int tv = (int)tok_kv->value.array.count;
            if (tv > m->vocab_size) {
                kprintf("[LLM] vocab_size corrected: %d -> %d (from tokenizer)\n",
                        m->vocab_size, tv);
                m->vocab_size = tv;
            }
        }
    }

    /* Compute derived parameters */
    if (llm_gguf_ctx.n_embd_head_k > 0)
        m->head_dim = (int)llm_gguf_ctx.n_embd_head_k;
    else if (m->n_heads > 0)
        m->head_dim = m->dim / m->n_heads;
    else
        m->head_dim = 64;
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;
    if (m->rope_base < 1.0f) m->rope_base = 10000.0f;

    /* Gemma-family detection (gemma, gemma2, gemma4, etc.) */
    int is_gemma = (kstrlen(m->arch) >= 5 &&
        m->arch[0] == 'g' && m->arch[1] == 'e' &&
        m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a');

    /* Embedding scaling: Gemma models scale embeddings by sqrt(dim) */
    m->embed_scale = is_gemma ? llm_sqrtf((float)m->dim) : 1.0f;

    /* GeGLU: Gemma models use GELU activation with gating (not SiLU) */
    m->use_geglu = is_gemma;

    /* Logit softcapping from GGUF metadata */
    m->logit_softcap = llm_gguf_ctx.final_logit_softcap;

    /* Gemma4 specific: per-layer head_dim, KV sharing, SWA RoPE */
    m->is_gemma4 = 0;
    m->head_dim_swa = 0;
    m->rope_base_swa = 0.0f;
    m->n_layer_kv_start = m->n_layers; /* default: all layers have own KV */
    m->iswa_tok_embd = (void *)0;
    m->iswa_model_proj = (void *)0;
    m->iswa_proj_norm = (void *)0;
    m->iswa_n_embd = 0;
    m->rope_freqs = (const float *)0;
    if (is_gemma && kstrlen(m->arch) >= 6 && m->arch[5] == '4') {
        m->is_gemma4 = 1;
        m->head_dim_swa = (int)llm_gguf_ctx.n_embd_head_k_swa;
        if (m->head_dim_swa == 0) m->head_dim_swa = m->head_dim; /* fallback */
        m->rope_base_swa = llm_gguf_ctx.rope_freq_base_swa;
        if (m->rope_base_swa < 1.0f) m->rope_base_swa = 10000.0f;
        /* KV sharing: n_layer_kv_start = n_layers - shared_kv_layers */
        if (llm_gguf_ctx.shared_kv_layers > 0)
            m->n_layer_kv_start = m->n_layers - (int)llm_gguf_ctx.shared_kv_layers;
        m->iswa_n_embd = (int)llm_gguf_ctx.n_embd_per_layer;
        kprintf("[LLM] Gemma4: head_dim_swa=%d rope_base_swa=%.0f kv_start=%d iswa_embd=%d\n",
                m->head_dim_swa, (double)m->rope_base_swa, m->n_layer_kv_start, m->iswa_n_embd);
    }

    /* Determine max sequence length from GGUF (already parsed by arch prefix) */
    int ctx_len = (int)llm_gguf_ctx.n_ctx;
    if (ctx_len < 128) ctx_len = 2048;  /* sensible default */
    if (ctx_len > 8192) ctx_len = 8192; /* hard cap */

    /* Dynamically reduce context length if KV cache would exceed available memory */
    {
        uint64_t avail = tensor_mm_free_bytes();
        /* KV cache per token: 2 (K+V) * n_layers * n_kv_heads * head_dim * 4 bytes */
        uint64_t kv_per_token = 2ULL * m->n_layers * m->n_kv_heads * m->head_dim * sizeof(float);
        /* Reserve 256 MB for scratch buffers (logits, FFN, etc.) */
        uint64_t reserve = 256ULL * 1024 * 1024;
        if (kv_per_token > 0 && avail > reserve) {
            int max_fit = (int)((avail - reserve) / kv_per_token);
            if (max_fit < ctx_len) {
                kprintf("[LLM] Capping context %d -> %d tokens (%lu MB heap free)\n",
                        ctx_len, max_fit, (unsigned long)(avail / (1024*1024)));
                ctx_len = max_fit;
            }
        }
        if (ctx_len < 128) ctx_len = 128;
    }
    m->max_seq = ctx_len;

    /* Detect Phi-2 architecture (LayerNorm + GELU + partial RoPE) */
    m->use_layernorm = 0;
    m->use_gelu = 0;
    m->rope_dim = 0; /* 0 = full head_dim */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || (m->arch[3] == '\0')) {
            /* Phi-2: LayerNorm + GELU + partial RoPE */
            m->use_layernorm = 1;
            m->use_gelu = 1;
            /* Phi-2 partial RoPE: typically 32 out of 80 head_dim */
            char keybuf[128];
            kstrcpy(keybuf, m->arch);
            kstrcpy(keybuf + kstrlen(keybuf), ".rope.dimension_count");
            m->rope_dim = (int)gguf_get_u32(&llm_gguf_ctx, keybuf, 0);
            if (m->rope_dim == 0) m->rope_dim = m->head_dim; /* fallback */
            kprintf("[LLM] Phi-2 mode: LayerNorm + GELU, rope_dim=%d\n", m->rope_dim);
        } else {
            /* Phi-3/Phi-3.5: LLaMA-compatible (RMSNorm + SwiGLU + full RoPE) */
            kprintf("[LLM] Phi-3 mode: LLaMA-compatible architecture\n");
        }
    }

    /* Longrope: load original context length for factor selection */
    m->rope_orig_ctx = 0;
    {
        char keybuf[128];
        kstrcpy(keybuf, m->arch);
        kstrcpy(keybuf + kstrlen(keybuf), ".rope.scaling.original_context_length");
        m->rope_orig_ctx = (int)gguf_get_u32(&llm_gguf_ctx, keybuf, 0);
        if (m->rope_orig_ctx > 0)
            kprintf("[LLM] Longrope: original_context_length=%d\n", m->rope_orig_ctx);
    }

    /* RMSNorm epsilon from model metadata (default 1e-5) */
    {
        char keybuf[128];
        kstrcpy(keybuf, m->arch);
        kstrcpy(keybuf + kstrlen(keybuf), ".attention.layer_norm_rms_epsilon");
        m->rms_eps = gguf_get_f32(&llm_gguf_ctx, keybuf, 1e-5f);
        kprintf("[LLM] rms_epsilon=%f\n", (double)m->rms_eps);
    }

    /* Get model name */
    const char *name = gguf_get_str(&llm_gguf_ctx, "general.name");
    if (name) {
        /* Copy up to 127 chars (name is GGUF string, not null-terminated) */
        const gguf_kv_t *kv = gguf_find_kv(&llm_gguf_ctx, "general.name");
        if (kv && kv->type == GGUF_TYPE_STRING) {
            int n = (int)kv->value.str.len;
            if (n > 127) n = 127;
            kmemcpy(m->name, kv->value.str.data, n);
            m->name[n] = '\0';
        }
    } else {
        kstrcpy(m->name, m->arch);
    }

    /* Print model info (concise for boot) */
    kprintf("[LLM] Model: %s (%s)\n", m->name, m->arch);
    kprintf("[LLM] %d layers, %d-dim, %d vocab, %d heads (%d KV), head_dim=%d\n",
            m->n_layers, m->dim, m->vocab_size, m->n_heads, m->n_kv_heads, m->head_dim);
    if (is_gemma)
        kprintf("[LLM] Gemma mode: embed_scale=%.1f geglu=%d softcap=%.1f\n",
                (double)m->embed_scale, m->use_geglu, (double)m->logit_softcap);

    /* Validate basic sanity */
    if (m->dim <= 0 || m->n_layers <= 0 || m->n_heads <= 0) {
        kprintf("[LLM] ERROR: invalid model dims (dim=%d layers=%d heads=%d)\n",
                m->dim, m->n_layers, m->n_heads);
        return -1;
    }

    /* Allocate layers array */
    m->layers = (llm_layer_t *)tensor_alloc(
        (uint64_t)m->n_layers * sizeof(llm_layer_t));
    if (!m->layers) {
        kprintf("[LLM] ERROR: failed to allocate %d layer descriptors\n", m->n_layers);
        return -1;
    }
    kmemset(m->layers, 0, (uint64_t)m->n_layers * sizeof(llm_layer_t));
    m->n_layers_alloc = m->n_layers;

    /* Map tensors */
    kprintf("[LLM] About to map tensors...\n");
    rc = llm_map_tensors(m, &llm_gguf_ctx);
    if (rc != 0) return rc;

    /* Gemma4: feed_forward_length is stored as an array in GGUF, so n_ff=0.
       Derive m->ff_dim from the per-layer ff_dim_layer values set by tensor loading. */
    if (m->ff_dim == 0 && m->n_layers > 0) {
        int max_ff = 0;
        for (int i = 0; i < m->n_layers; i++) {
            if (m->layers[i].ff_dim_layer > max_ff)
                max_ff = m->layers[i].ff_dim_layer;
        }
        if (max_ff > 0) {
            m->ff_dim = max_ff;
            kprintf("[LLM] Derived ff_dim=%d from per-layer tensor shapes\n", max_ff);
        }
    }

    kprintf("[LLM] Tensors mapped, building vocab...\n");

    /* Build vocabulary */
    rc = llm_build_vocab(m, &llm_gguf_ctx);
    if (rc != 0) return rc;
    kprintf("[LLM] Vocab built (%d tokens), allocating scratch...\n", m->n_vocab);

    /* Final vocab_size sanity: prefer tokenizer count over GGUF metadata */
    if (m->n_vocab > 0 && m->n_vocab != m->vocab_size) {
        kprintf("[LLM] Vocab: metadata=%d, tokenizer=%d, using %d\n",
                m->vocab_size, m->n_vocab,
                m->n_vocab > m->vocab_size ? m->n_vocab : m->vocab_size);
        if (m->n_vocab > m->vocab_size)
            m->vocab_size = m->n_vocab;
    }

    /* Allocate inference scratch arena (KV cache + all buffers) */
    if (llm_alloc_scratch(m) != 0)
        return -1;

    m->k_cache = llm_kv_k;
    m->v_cache = llm_kv_v;
    m->cache_len = 0;

    kprintf("[LLM] Model loaded successfully! Ready for inference.\n");

    /* Initialize backend registry (CPU always, CUDA/MLIR if compiled) */
    backend_init_all();
    kprintf("[LLM] Backend: %s\n", backend_get()->name);

    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Custom float printing (bare-metal: kprintf doesn't support %f)             */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_print_float(float v)
{
    if (v < 0) { kprintf("-"); v = -v; }
    int integer = (int)v;
    int frac = (int)((v - (float)integer) * 100.0f + 0.5f);
    if (frac >= 100) { integer++; frac -= 100; }
    kprintf("%d.%02d", integer, frac);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Math Evaluation Suite                                                       */
/*                                                                             */
/*  Tests the loaded LLM on mathematical reasoning problems.                   */
/*  Categories: Arithmetic, Algebra, Sequences, Multi-step, Challenge          */
/* ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    const char *prompt;     /* Math problem */
    const char *expected;   /* Expected answer substring */
    int         category;   /* 0=arith, 1=algebra, 2=sequence, 3=multi, 4=challenge */
} math_problem_t;

static const math_problem_t math_problems[] = {
    /* Arithmetic (Category 0) */
    { "Calculate: 25 + 17 = ",                  "42",    0 },
    { "What is 144 / 12? ",                     "12",    0 },
    { "Solve: 7 * 8 = ",                        "56",    0 },

    /* Algebra (Category 1) */
    { "If x + 5 = 12, then x = ",              "7",     1 },
    { "If 2x = 18, then x = ",                 "9",     1 },

    /* Sequences (Category 2) */
    { "What is the next number: 2, 4, 6, 8, ", "10",    2 },

    /* Multi-step (Category 3) */
    { "What is 2^8? ",                          "256",   3 },

    /* Challenge (Category 4) */
    { "What is the square root of 144? ",       "12",    4 },
};
#define N_MATH_PROBLEMS (int)(sizeof(math_problems) / sizeof(math_problems[0]))

static const char *category_names[] = {
    "Arithmetic", "Algebra", "Sequences", "Multi-step", "Challenge"
};

/* Build a chat-formatted prompt based on model architecture */
static int llm_format_prompt(const llm_model_t *m, const char *question,
                             char *buf, int max_len)
{
    int pos = 0;
    /* Detect prompt format from architecture and model name */
    int is_chatml = 0;
    int is_phi3   = 0;

    /* Check for Qwen (ChatML native) */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;

    /* Check for Phi-3/3.5 */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i' &&
        m->arch[3] != '2' && m->arch[3] != '\0')
        is_phi3 = 1;

    /* Check for SmolLM / smollm in model name — SmolLM uses ChatML but
     * the special tokens may not be in base vocab; use simple prompt */
    /* (ChatML disabled for SmolLM until vocab includes <|im_start|>) */

    if (is_phi3) {
        /* Phi-3 format */
        { const char *t = "<|user|>\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "<|end|>\n<|assistant|>\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else if (is_chatml) {
        /* ChatML format (Qwen, SmolLM, Mistral-Instruct-v0.3+, etc.) */
        { const char *t = "<|im_start|>system\nYou are a helpful math assistant. Give only the numeric answer.<|im_end|>\n<|im_start|>user\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "<|im_end|>\n<|im_start|>assistant\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else if (kstrlen(m->arch) >= 5 &&
               m->arch[0] == 'g' && m->arch[1] == 'e' &&
               m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a') {
        /* Gemma format */
        { const char *t = "<start_of_turn>user\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "<end_of_turn>\n<start_of_turn>model\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else {
        /* Generic: simple text prompt (works with any model) */
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
    }
    buf[pos < max_len ? pos : max_len - 1] = '\0';
    return pos;
}

/* Check if expected answer appears in the generated text */
static int llm_check_answer(const char *generated, const char *expected)
{
    int gen_len = kstrlen(generated);
    int exp_len = kstrlen(expected);
    if (exp_len > gen_len) return 0;

    for (int i = 0; i <= gen_len - exp_len; i++) {
        int match = 1;
        for (int j = 0; j < exp_len; j++) {
            if (generated[i + j] != expected[j]) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

/* Run full math evaluation on a loaded model */
static void llm_run_math_eval(llm_model_t *m)
{
    kprintf("\n");
    kprintf("==========================================================\n");
    kprintf("  MATH REASONING EVALUATION\n");
    kprintf("  Model: %s (%s)\n", m->name, m->arch);
    kprintf("  %d problems across 5 categories\n", N_MATH_PROBLEMS);
    kprintf("==========================================================\n\n");

    int total_correct = 0;
    int cat_correct[5] = {0, 0, 0, 0, 0};
    int cat_total[5]   = {0, 0, 0, 0, 0};

    char prompt_buf[512];
    char output_buf[256];

    for (int p = 0; p < N_MATH_PROBLEMS; p++) {
        const math_problem_t *prob = &math_problems[p];
        cat_total[prob->category]++;

        /* Format the prompt */
        llm_format_prompt(m, prob->prompt, prompt_buf, sizeof(prompt_buf));

        /* Tokenize */
        int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens, llm_alloc_tokens - 32);

        kprintf("  [%d/%d] %s\n", p + 1, N_MATH_PROBLEMS, prob->prompt);
        kprintf("         Tokens: %d, generating...\n", n_tokens);

        /* Generate */
        uint64_t t0 = rdtsc_fenced();
        int n_gen = llm_generate(m, llm_tokens, n_tokens, output_buf, sizeof(output_buf),
                                 16, 0.0f, 0); /* greedy, max 16 tokens, fresh cache */
        uint64_t t1 = rdtsc_fenced();

        uint64_t gen_ms = perf_cycles_to_us(t1 - t0) / 1000;
        uint64_t ms_per_tok = (n_gen > 0) ? (gen_ms / (uint64_t)n_gen) : 0;

        /* Check answer */
        int correct = llm_check_answer(output_buf, prob->expected);
        if (correct) {
            total_correct++;
            cat_correct[prob->category]++;
        }

        kprintf("         Answer: %s\n", output_buf);
        kprintf("         Expected: %s | %s | %lu ms (%lu ms/tok)\n",
                prob->expected,
                correct ? "CORRECT" : "WRONG",
                gen_ms, ms_per_tok);
    }

    /* Summary */
    kprintf("\n  ─── Results ─────────────────────────────────────────\n");
    for (int c = 0; c < 5; c++) {
        if (cat_total[c] > 0) {
            kprintf("  %-12s: %d/%d", category_names[c], cat_correct[c], cat_total[c]);
            int pct = (cat_correct[c] * 100) / cat_total[c];
            kprintf(" (%d%%)\n", pct);
        }
    }
    kprintf("  ──────────────────────────────────────────────────────\n");
    kprintf("  TOTAL: %d/%d correct", total_correct, N_MATH_PROBLEMS);
    int total_pct = (total_correct * 100) / N_MATH_PROBLEMS;
    kprintf(" (%d%%)\n", total_pct);
    kprintf("==========================================================\n");
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Inference Speed Benchmark                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_run_benchmark(llm_model_t *m)
{
    kprintf("\n  --- Inference Speed Benchmark ---\n");

    /* Reset KV cache */
    m->cache_len = 0;

    /* Time a single forward pass */
    uint64_t t0 = rdtsc_fenced();
    llm_forward_token(m, llm_logits, m->bos_id, 0);
    uint64_t t1 = rdtsc_fenced();

    uint64_t single_ms = perf_cycles_to_us(t1 - t0) / 1000;
    kprintf("  Single token forward: %lu ms\n", single_ms);

    /* Estimate tokens per second */
    if (single_ms > 0) {
        double tps = 1000.0 / (double)single_ms;
        kprintf("  Estimated throughput: ~%.1f tokens/s\n", tps);
    }

    /* Memory usage */
    int kv_kbytes = (m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim * 4 * 2) / 1024;
    kprintf("  KV cache: %d KB (for %d seq len)\n", kv_kbytes, m->max_seq);
    kprintf("  Model data: %lu MB\n", m->data_size / (1024 * 1024));

    /* Compute total parameters */
    uint64_t total_params = m->gguf->total_param_count;
    kprintf("  Parameters: %lu M\n", total_params / 1000000);

    /* Compute FLOPS estimate per token */
    /* ~2 * params FLOPs per inference token */
    uint64_t flops_per_token = 2 * total_params;
    if (single_ms > 0) {
        double gf = (double)flops_per_token / ((double)single_ms * 1000000.0);
        kprintf("  Effective: ~%.2f GFLOPS\n", gf);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Main Entry Point                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Boot-time model loader: load GGUF from disk into RAM, but don't run eval.
 * This is called during boot for fast startup. */
void llm_boot_load(void)
{
#ifdef __aarch64__
    kprintf("  [--] ARM64 disk loading not yet available\n");
    return;
#else
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("  [--] No model disk (use 'llm' in shell for instructions)\n");
        return;
    }

    /* Verify GGUF magic */
    {
        static uint8_t hdr_buf[512];
        int rc = virtio_blk_read(0, 1, hdr_buf);
        if (rc != 0) {
            kprintf("  [--] Failed to read disk\n");
            return;
        }
        uint32_t magic = (uint32_t)hdr_buf[0] | ((uint32_t)hdr_buf[1] << 8) |
                         ((uint32_t)hdr_buf[2] << 16) | ((uint32_t)hdr_buf[3] << 24);
        if (magic != GGUF_MAGIC) {
            kprintf("  [--] Disk is not a GGUF model file\n");
            return;
        }
    }

    kprintf("  Loading %lu MB model from disk...\n", capacity / (1024 * 1024));

    kmemset(&llm_model, 0, sizeof(llm_model));
    int rc = llm_load_from_disk(&llm_model);
    if (rc != 0) {
        kprintf("  [FAIL] Model load error: %d\n", rc);
        return;
    }

    kprintf("  [OK] %s (%d layers, %lu params)\n",
            llm_model.name, llm_model.n_layers,
            (uint64_t)llm_model.vocab_size * llm_model.dim);
#endif
}

/* Full eval (benchmarks + math) — callable from shell via 'demo llm' */
void llm_run_full_eval(void)
{
    if (!llm_is_loaded()) {
        kprintf("[LLM] No model loaded.\n");
        return;
    }
    llm_run_benchmark(&llm_model);
    llm_run_math_eval(&llm_model);
    kprintf("\n[LLM] Evaluation complete.\n");
}

void llm_run_eval(void)
{
    kprintf("\n============================================================\n");
    kprintf("  REAL LLM INFERENCE ENGINE\n");
    kprintf("  Bare-Metal Model Loading & Math Evaluation\n");
    kprintf("============================================================\n");
    kprintf("  Supported: Qwen, Gemma, LLaMA, SmolLM, Mistral, ...\n");
    kprintf("  Formats:   GGUF v2/v3 with Q4_0, Q8_0, F16, F32\n");
    kprintf("  Features:  GQA, RoPE, KV-Cache, SwiGLU, RMSNorm\n");
    kprintf("============================================================\n\n");

#ifdef __aarch64__
    kprintf("[LLM] ARM64 disk loading not implemented yet.\n");
    kprintf("[LLM] Use x86_64 QEMU with: -drive file=model.gguf,format=raw,if=virtio\n");
    return;
#else
    /* Check if virtio-blk is available with a model */
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("[LLM] No model disk detected.\n\n");
        kprintf("  To run real LLM inference, attach a GGUF model:\n\n");
        kprintf("  1. Download a model (e.g., Qwen2.5-0.5B or SmolLM2-135M):\n");
        kprintf("     .\\tools\\download_model.ps1 -Model qwen2.5-0.5b\n\n");
        kprintf("  2. Rebuild and run with the model attached:\n");
        kprintf("     .\\build.ps1 -Run\n");
        kprintf("     (the build script auto-detects models\\*.gguf)\n\n");
        kprintf("  Or manually:\n");
        kprintf("     qemu-system-x86_64 -kernel build\\tensoros.elf \\\n");
        kprintf("       -drive file=model.gguf,format=raw,if=virtio ...\n\n");
        kprintf("  Supported models:\n");
        kprintf("    - Qwen2.5-0.5B-Instruct  (Q4_0, ~350 MB)\n");
        kprintf("    - SmolLM2-135M-Instruct   (Q8_0, ~145 MB)\n");
        kprintf("    - TinyLlama-1.1B-Chat     (Q4_0, ~600 MB)\n");
        kprintf("    - Gemma-2-2B-IT           (Q4_0, ~1.2 GB)\n");
        kprintf("    - Any GGUF model with LLaMA-style architecture\n\n");
        return;
    }

    /* Verify it looks like a GGUF file (check magic at sector 0) */
    {
        static uint8_t hdr_buf[512];
        int rc = virtio_blk_read(0, 1, hdr_buf);
        if (rc != 0) {
            kprintf("[LLM] Failed to read first sector\n");
            return;
        }
        uint32_t magic = (uint32_t)hdr_buf[0] | ((uint32_t)hdr_buf[1] << 8) |
                         ((uint32_t)hdr_buf[2] << 16) | ((uint32_t)hdr_buf[3] << 24);
        if (magic != GGUF_MAGIC) {
            kprintf("[LLM] Disk does not contain a GGUF file (magic=%x, expected %x)\n",
                    magic, GGUF_MAGIC);
            kprintf("[LLM] Attach a .gguf file as raw virtio disk.\n");
            return;
        }
        kprintf("[LLM] GGUF file detected on virtio-blk (%lu MB)\n",
                capacity / (1024 * 1024));
    }

    /* Load the model */
    kmemset(&llm_model, 0, sizeof(llm_model));
    int rc = llm_load_from_disk(&llm_model);
    if (rc != 0) {
        kprintf("[LLM] Model loading failed: %d\n", rc);
        return;
    }

    /* Run benchmark */
    llm_run_benchmark(&llm_model);

    /* Run math evaluation */
    llm_run_math_eval(&llm_model);

    kprintf("\n[LLM] Evaluation complete.\n");
#endif
}

/* =============================================================================
 * Interactive LLM prompt — called from the shell
 * =============================================================================*/
int llm_is_loaded(void)
{
    return (llm_model.vocab_size > 0 && llm_model.data_buf != NULL);
}

const char *llm_model_name(void)
{
    if (!llm_is_loaded()) return "(none)";
    return llm_model.name;
}

uint64_t llm_param_count(void)
{
    if (!llm_is_loaded()) return 0;
    /* Approximate: dim² × 12 × n_layers (QKV+O+gate+up+down + norms + embd) */
    llm_model_t *m = &llm_model;
    uint64_t per_layer = (uint64_t)m->dim * (uint64_t)m->dim * 4  /* QKV+O */
                       + (uint64_t)m->dim * (uint64_t)m->ff_dim * 3; /* gate+up+down */
    uint64_t total = per_layer * (uint64_t)m->n_layers
                   + (uint64_t)m->vocab_size * (uint64_t)m->dim; /* embeddings */
    return total;
}

int llm_set_backend(llm_backend_t backend)
{
    switch (backend) {
    case LLM_BACKEND_CPU:
        llm_backend = backend;
        return 0;
    case LLM_BACKEND_CUDA:
    case LLM_BACKEND_MLIR:
        /* Backend scaffolding is in place; kernels/runtime are pending. */
        return -2;
    default:
        return -1;
    }
}

llm_backend_t llm_get_backend(void)
{
    return llm_backend;
}

const char *llm_backend_name(void)
{
    switch (llm_backend) {
    case LLM_BACKEND_CPU:  return "cpu";
    case LLM_BACKEND_CUDA: return "cuda";
    case LLM_BACKEND_MLIR: return "mlir";
    default:               return "unknown";
    }
}

int llm_prompt(const char *user_text, char *output, int max_output)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }

    /* Serialize: only one inference at a time (static buffers not reentrant) */
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy — try again]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;

    /* Build chat prompt using the model's preferred format */
    static char prompt_buf[2048];
    const int buf_max = (int)sizeof(prompt_buf) - 1;
    int pos = 0;

    /* Safe append: copies at most (buf_max - pos) bytes */
    #define PROMPT_APPEND(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(prompt_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    /* Detect prompt format */
    int is_chatml = 0;
    int is_phi3   = 0;
    int is_phi2   = 0;
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;
    if (kstrlen(m->arch) >= 3 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || m->arch[3] == '\0')
            is_phi2 = 1;
        else
            is_phi3 = 1;
    }

    if (is_phi3) {
        PROMPT_APPEND("<|user|>\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        PROMPT_APPEND("Instruct: ");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\nOutput: ");
    } else if (is_chatml) {
        PROMPT_APPEND("<|im_start|>system\nYou are a helpful AI assistant running on TensorOS, a bare-metal AI operating system.<|im_end|>\n<|im_start|>user\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("<|im_end|>\n<|im_start|>assistant\n");
    } else if (kstrlen(m->arch) >= 5 &&
               m->arch[0] == 'g' && m->arch[1] == 'e' &&
               m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a') {
        PROMPT_APPEND("<turn|>user\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\n<turn|>model\n");
    } else {
        /* Generic / LLaMA / SmolLM — simple prompt */
        PROMPT_APPEND("User: ");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\nAssistant: ");
    }
    prompt_buf[pos] = '\0';
    #undef PROMPT_APPEND

    /* Tokenize */
    int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    /* Prepend BOS token */
    if (n_tokens > 0 && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        for (int i = 0; i < n_tokens; i++)
            llm_tokens[i] = llm_tokens[i + 1];
    }
    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    /* Generate: max 1024 tokens, temperature 0.7 with top-k/top-p sampling */
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             1024, 0.7f, 0);
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* Like llm_prompt but with an explicit max-token cap */
int llm_prompt_n(const char *user_text, char *output, int max_output, int max_tokens)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;

    /* Build prompt with chat template (same as llm_prompt) */
    static char prompt_buf[512];
    int pos = 0;
    const int buf_max = (int)sizeof(prompt_buf) - 1;
    #define PA(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(prompt_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    /* Detect prompt format from architecture */
    int is_phi3 = 0, is_phi2 = 0, is_chatml = 0, is_gemma = 0;
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;
    if (kstrlen(m->arch) >= 3 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || m->arch[3] == '\0')
            is_phi2 = 1;
        else
            is_phi3 = 1;
    }
    if (kstrlen(m->arch) >= 5 &&
        m->arch[0] == 'g' && m->arch[1] == 'e' &&
        m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a')
        is_gemma = 1;

    if (is_gemma) {
        PA("<turn|>user\n");
        PA(user_text);
        PA("\n<turn|>model\n");
    } else if (is_phi3) {
        PA("<|user|>\n");
        PA(user_text);
        PA("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        PA("Instruct: ");
        PA(user_text);
        PA("\nOutput: ");
    } else if (is_chatml) {
        PA("<|im_start|>user\n");
        PA(user_text);
        PA("<|im_end|>\n<|im_start|>assistant\n");
    } else {
        PA(user_text);
    }
    prompt_buf[pos] = '\0';
    #undef PA

    int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    /* Prepend BOS token */
    if (n_tokens > 0 && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        /* No BOS: shift tokens to start of array */
        for (int i = 0; i < n_tokens; i++)
            llm_tokens[i] = llm_tokens[i + 1];
    }
    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    if (max_tokens < 1) max_tokens = 16;
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             max_tokens, 0.0f, 0); /* greedy for smoke test */
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* Reset KV cache for starting a new conversation */
void llm_reset_cache(void)
{
    if (!llm_is_loaded()) return;
    llm_model_t *m = &llm_model;
    m->cache_len = 0;
    int kv_total = m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim;
    if (kv_total > llm_alloc_kv_floats) kv_total = llm_alloc_kv_floats;
    kmemset(m->k_cache, 0, (uint64_t)kv_total * sizeof(float));
    kmemset(m->v_cache, 0, (uint64_t)kv_total * sizeof(float));
}

/* Reset chat KV context */
void llm_chat_reset(void) { llm_reset_cache(); }

/* How many KV positions are currently occupied */
int llm_chat_context_tokens(void) {
    if (!llm_is_loaded()) return 0;
    return (int)llm_model.cache_len;
}

/* Maximum context window for the loaded model */
int llm_chat_context_max(void) {
    if (!llm_is_loaded()) return 0;
    return (int)llm_model.max_seq;
}

/**
 * Multi-turn chat: maintains KV cache across calls.
 * Each call appends this user turn to the existing context.
 * Call llm_chat_reset() to start a new conversation.
 */
int llm_chat_turn(const char *user_text, char *output, int max_output,
                  int max_tokens, float temperature)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;
    int first_turn = (m->cache_len == 0);

    /* Build only this turn's formatted prompt;
     * previous turns are already baked into the KV cache. */
    static char turn_buf[4096];
    int pos = 0;
    const int buf_max = (int)sizeof(turn_buf) - 1;
    #define TA(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(turn_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    int is_gemma = kstrlen(m->arch) >= 5 &&
        m->arch[0]=='g' && m->arch[1]=='e' && m->arch[2]=='m' &&
        m->arch[3]=='m' && m->arch[4]=='a';
    int is_phi3 = kstrlen(m->arch) >= 3 &&
        m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
        m->arch[3] != '2' && m->arch[3] != '\0';
    int is_phi2 = kstrlen(m->arch) >= 3 &&
        m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
        (m->arch[3] == '2' || m->arch[3] == '\0');
    int is_chatml = kstrlen(m->arch) >= 4 &&
        m->arch[0]=='q' && m->arch[1]=='w' && m->arch[2]=='e' && m->arch[3]=='n';

    if (is_gemma) {
        TA("<turn|>user\n");
        TA(user_text);
        TA("\n<turn|>model\n");
    } else if (is_phi3) {
        TA("<|user|>\n");
        TA(user_text);
        TA("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        TA("Instruct: ");
        TA(user_text);
        TA("\nOutput: ");
    } else if (is_chatml) {
        if (first_turn) {
            TA("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n");
        }
        TA("<|im_start|>user\n");
        TA(user_text);
        TA("<|im_end|>\n<|im_start|>assistant\n");
    } else {
        TA(user_text);
        TA("\n");
    }
    turn_buf[pos] = '\0';
    #undef TA

    /* Tokenize: prepend BOS only on the first turn */
    int n_tokens = llm_tokenize(m, turn_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    if (first_turn && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        for (int i = 0; i < n_tokens; i++) llm_tokens[i] = llm_tokens[i + 1];
    }

    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    if (max_tokens < 1) max_tokens = 256;
    /* continue_cache=0 for first turn (resets KV cache), 1 for subsequent turns */
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             max_tokens, temperature, first_turn ? 0 : 1);
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Tensor Bridge API (exposes bridge for daisy-chaining LLMs)                */
/* ─────────────────────────────────────────────────────────────────────────── */

tensor_bridge_t *llm_get_bridge(void) {
    return &llm_bridge;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Test-facing wrapper API (exposes internal tokenizer for unit tests)        */
/* ─────────────────────────────────────────────────────────────────────────── */

int llm_test_tokenize(const char *text, int text_len, int *tokens, int max_tokens)
{
    (void)text_len;
    if (!llm_is_loaded()) return -1;
    return llm_tokenize(&llm_model, text, tokens, max_tokens);
}

int llm_test_decode_token(int token_id, char *buf, int max_len)
{
    if (!llm_is_loaded()) return -1;
    return llm_decode_token(&llm_model, token_id, buf, max_len);
}
