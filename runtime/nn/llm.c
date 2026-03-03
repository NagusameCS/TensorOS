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
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "runtime/nn/gguf.h"
#ifndef __aarch64__
#include "kernel/drivers/blk/virtio_blk.h"
#endif

/* ─────────────────────────────────────────────────────────────────────────── */
/*  SSE2 SIMD type                                                             */
/* ─────────────────────────────────────────────────────────────────────────── */
typedef float v4f __attribute__((vector_size(16)));

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Static Allocations                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/* GGUF parsing context */
static gguf_ctx_t llm_gguf_ctx;

/* Model descriptor */
static llm_model_t llm_model;

/* KV Cache — 32MB each (K and V), supports models up to ~2B params */
static float llm_kv_k[LLM_KV_FLOATS] __attribute__((aligned(64)));
static float llm_kv_v[LLM_KV_FLOATS] __attribute__((aligned(64)));

/* Scratch buffers for forward pass */
static float llm_x[LLM_MAX_DIM]     __attribute__((aligned(16)));   /* hidden state */
static float llm_xn[LLM_MAX_DIM]    __attribute__((aligned(16)));   /* normalized */
static float llm_q[LLM_MAX_DIM]     __attribute__((aligned(16)));   /* query */
static float llm_k_buf[LLM_MAX_DIM] __attribute__((aligned(16)));   /* key (current) */
static float llm_v_buf[LLM_MAX_DIM] __attribute__((aligned(16)));   /* value (current) */
static float llm_attn_out[LLM_MAX_DIM] __attribute__((aligned(16)));/* attention output */
static float llm_ffn_g[LLM_MAX_FF]  __attribute__((aligned(16)));   /* FFN gate */
static float llm_ffn_u[LLM_MAX_FF]  __attribute__((aligned(16)));   /* FFN up */
static float llm_ffn_d[LLM_MAX_DIM] __attribute__((aligned(16)));   /* FFN down */
static float llm_head_buf[LLM_MAX_DIM] __attribute__((aligned(16)));
static float llm_attn_scores[LLM_MAX_SEQ] __attribute__((aligned(16)));

/* Logits buffer — up to 160K vocab × 4 bytes = 640KB */
static float llm_logits[LLM_MAX_VOCAB] __attribute__((aligned(16)));

/* Token buffer for generation */
static int llm_tokens[LLM_MAX_TOKENS];

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
    if (x > 88.0f) return 3.4e38f;
    if (x < -88.0f) return 0.0f;
    /* Pade approximant for exp(x) near 0, range-reduced */
    float y = 1.0f + x / 256.0f;
    for (int i = 0; i < 8; i++) y = y * y; /* y^256 */
    return y;
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

/* GGML Q8_0 block: {fp16 scale, int8[32]}  = 34 bytes for 32 elements */
typedef struct { uint16_t d; int8_t  qs[32]; } __attribute__((packed)) ggml_q8_0_t;

/* Dot product: Q4_0 block · float[32] */
static float q4_0_dot32(const ggml_q4_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;

    for (int j = 0; j < 16; j++) {
        uint8_t packed = block->qs[j];
        int lo = (int)(packed & 0x0F) - 8;
        int hi = (int)(packed >> 4)   - 8;
        sum += (float)lo * x[2 * j]     +
               (float)hi * x[2 * j + 1];
    }
    return sum * d;
}

/* Dot product: Q8_0 block · float[32] */
static float q8_0_dot32(const ggml_q8_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;

    for (int j = 0; j < 32; j++) {
        sum += (float)block->qs[j] * x[j];
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
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q8_0_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *f16 = (const uint16_t *)weight;
        for (int i = 0; i < n; i++)
            sum += fp16_to_fp32(f16[i]) * x[i];
        break;
    }
    case GGML_TYPE_F32: {
        const float *f32 = (const float *)weight;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            v4f vw = *(const v4f *)(f32 + i);
            v4f vx = *(const v4f *)(x + i);
            v4f p = vw * vx;
            union { v4f v; float f[4]; } u = { .v = p };
            sum += u.f[0] + u.f[1] + u.f[2] + u.f[3];
        }
        for (; i < n; i++)
            sum += f32[i] * x[i];
        break;
    }
    default:
        break;
    }
    return sum;
}

/* Bytes per row for a quantized matrix [out_dim × in_dim] */
static uint64_t llm_row_bytes(int in_dim, ggml_type_t type)
{
    switch (type) {
    case GGML_TYPE_Q4_0: return (uint64_t)(in_dim / 32) * 18;
    case GGML_TYPE_Q8_0: return (uint64_t)(in_dim / 32) * 34;
    case GGML_TYPE_F16:  return (uint64_t)in_dim * 2;
    case GGML_TYPE_F32:  return (uint64_t)in_dim * 4;
    default:             return (uint64_t)in_dim * 4;
    }
}

/* GEMV: out[out_dim] = weight[out_dim × in_dim] · x[in_dim]
 * weight is in quantized GGML format (row-major) */
static void llm_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t type)
{
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
        /* Dequantize Q4_0 blocks */
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)row;
        int nb = dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(blocks[b].d);
            for (int j = 0; j < 16; j++) {
                uint8_t packed = blocks[b].qs[j];
                out[b * 32 + 2 * j]     = (float)((int)(packed & 0x0F) - 8) * d;
                out[b * 32 + 2 * j + 1] = (float)((int)(packed >> 4)   - 8) * d;
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
    default:
        for (int i = 0; i < dim; i++) out[i] = 0.0f;
        break;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  RMSNorm                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_rmsnorm(float *out, const float *x, const void *w,
                        int dim, ggml_type_t wtype)
{
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + 1e-6f);

    for (int i = 0; i < dim; i++)
        out[i] = x[i] * ss * llm_get_f(w, i, wtype);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Rotary Position Embeddings (RoPE)                                          */
/*  Applies complex rotation to each pair of elements in Q and K.              */
/*  This is the key innovation behind modern LLMs like LLaMA/Qwen.            */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_rope(float *vec, int pos, int head_dim, float base)
{
    for (int i = 0; i < head_dim; i += 2) {
        /* Compute rotation angle: theta_i = pos * base^(-2i/dim) */
        float freq = 1.0f;
        /* base^(-2i/dim) ≈ exp(-2i/dim * ln(base)) */
        float exponent = -(float)i / (float)head_dim;
        /* Compute base^exponent via repeated squaring approximation */
        /* For base=10000: ln(10000)=9.21, for base=1000000: ln(1e6)=13.82 */
        float log_base = 0.0f;
        {
            /* Approximate ln(base) */
            float b = base;
            /* ln(x) ≈ series for large x: decompose as 2^k * m */
            int k = 0;
            while (b > 2.0f) { b *= 0.5f; k++; }
            /* ln(2^k * m) = k*ln(2) + ln(m), m in [1,2] */
            /* ln(m) ≈ (m-1) - (m-1)^2/2 + (m-1)^3/3 for m near 1 */
            float m1 = b - 1.0f;
            log_base = (float)k * 0.6931472f +
                       m1 * (1.0f - m1 * (0.5f - m1 * 0.3333f));
        }
        freq = llm_expf(exponent * log_base);
        float theta = (float)pos * freq;

        /* sin/cos via Taylor series */
        /* Reduce theta to [-pi, pi] */
        float pi = 3.14159265f;
        while (theta > pi) theta -= 2.0f * pi;
        while (theta < -pi) theta += 2.0f * pi;

        float t2 = theta * theta;
        float cos_t = 1.0f - t2 * (0.5f - t2 * (0.04166667f - t2 * 0.001388889f));
        float sin_t = theta * (1.0f - t2 * (0.1666667f - t2 * (0.008333333f - t2 * 0.000198413f)));

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_t - v1 * sin_t;
        vec[i + 1] = v0 * sin_t + v1 * cos_t;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Softmax                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_softmax(float *x, int n)
{
    /* Find max for numerical stability */
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
/*  SiLU (Sigmoid Linear Unit) — used in SwiGLU FFN                           */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_silu(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + llm_expf(-x[i]));
        x[i] = x[i] * s;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Transformer Forward Pass — Single Token                                    */
/*                                                                             */
/*  Process one token through the full transformer stack (incremental decode).  */
/*  Updates KV-cache and returns logits[vocab_size].                           */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_forward_token(llm_model_t *m, float *logits, int token_id, int pos)
{
    int dim = m->dim;
    int n_heads = m->n_heads;
    int n_kv = m->n_kv_heads;
    int hd = m->head_dim;
    int ff = m->ff_dim;
    int kv_dim = n_kv * hd;

    /* 1. Embedding lookup */
    llm_embed(llm_x, m, token_id);

    /* 2. Process each transformer layer */
    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];

        /* === Self-Attention === */

        /* 2a. Pre-attention RMSNorm */
        llm_rmsnorm(llm_xn, llm_x, layer->attn_norm, dim, layer->attn_norm_type);

        /* 2b. Q/K/V projections */
        llm_gemv(llm_q,     layer->q_weight, llm_xn, n_heads * hd, dim, layer->q_type);
        llm_gemv(llm_k_buf, layer->k_weight, llm_xn, kv_dim,       dim, layer->k_type);
        llm_gemv(llm_v_buf, layer->v_weight, llm_xn, kv_dim,       dim, layer->v_type);

        /* 2c. Apply RoPE to Q and K */
        for (int h = 0; h < n_heads; h++)
            llm_rope(llm_q + h * hd, pos, hd, m->rope_base);
        for (int h = 0; h < n_kv; h++)
            llm_rope(llm_k_buf + h * hd, pos, hd, m->rope_base);

        /* 2d. Store K,V in cache */
        /* KV cache layout: [layer][position][kv_head][head_dim] */
        int kv_stride = m->max_seq * kv_dim;
        float *kc = m->k_cache + L * kv_stride + pos * kv_dim;
        float *vc = m->v_cache + L * kv_stride + pos * kv_dim;
        for (int i = 0; i < kv_dim; i++) {
            kc[i] = llm_k_buf[i];
            vc[i] = llm_v_buf[i];
        }

        /* 2e. Multi-head attention with GQA (Grouped Query Attention) */
        for (int i = 0; i < dim; i++) llm_attn_out[i] = 0.0f;

        int heads_per_kv = n_heads / n_kv;  /* GQA group size */

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;  /* which KV head this Q head attends to */
            float *qh = llm_q + h * hd;
            float scale = 1.0f / llm_sqrtf((float)hd);

            /* Compute attention scores: score[t] = Q_h · K_t / sqrt(d) */
            int seq_len = pos + 1;
            for (int t = 0; t < seq_len; t++) {
                float *kt = m->k_cache + L * kv_stride + t * kv_dim + kv_h * hd;
                float dot = 0.0f;
                for (int d = 0; d < hd; d++)
                    dot += qh[d] * kt[d];
                llm_attn_scores[t] = dot * scale;
            }

            /* Softmax over scores */
            llm_softmax(llm_attn_scores, seq_len);

            /* Weighted sum of V: head_out = sum_t(score[t] * V_t) */
            for (int d = 0; d < hd; d++) llm_head_buf[d] = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                float s = llm_attn_scores[t];
                float *vt = m->v_cache + L * kv_stride + t * kv_dim + kv_h * hd;
                for (int d = 0; d < hd; d++)
                    llm_head_buf[d] += s * vt[d];
            }

            /* Copy head output to concat buffer */
            for (int d = 0; d < hd; d++)
                llm_attn_out[h * hd + d] = llm_head_buf[d];
        }

        /* 2f. Output projection + residual */
        llm_gemv(llm_ffn_d, layer->o_weight, llm_attn_out, dim, dim, layer->o_type);
        for (int i = 0; i < dim; i++)
            llm_x[i] += llm_ffn_d[i];

        /* === Feed-Forward Network (SwiGLU) === */

        /* 2g. Pre-FFN RMSNorm */
        llm_rmsnorm(llm_xn, llm_x, layer->ffn_norm, dim, layer->ffn_norm_type);

        /* 2h. SwiGLU: hidden = SiLU(W_gate · x) ⊙ (W_up · x) */
        llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, ff, dim, layer->gate_type);
        llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, ff, dim, layer->up_type);
        llm_silu(llm_ffn_g, ff);
        for (int i = 0; i < ff; i++)
            llm_ffn_g[i] *= llm_ffn_u[i];

        /* 2i. Down projection + residual */
        llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, ff, layer->down_type);
        for (int i = 0; i < dim; i++)
            llm_x[i] += llm_ffn_d[i];
    }

    /* 3. Final RMSNorm */
    llm_rmsnorm(llm_xn, llm_x, m->output_norm, dim, m->output_norm_type);

    /* 4. LM head: logits = output_weight · x */
    const void *lm_head = m->output_weight ? m->output_weight : m->token_embd;
    ggml_type_t lm_type = m->output_weight ? m->output_type : m->token_embd_type;
    llm_gemv(logits, lm_head, llm_xn, m->vocab_size, dim, lm_type);
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
    if (n_vocab > LLM_MAX_VOCAB) n_vocab = LLM_MAX_VOCAB;
    m->n_vocab = n_vocab;

    /* Allocate vocab array in the model cache region (after GGUF data) */
    uint8_t *alloc_ptr = (uint8_t *)m->data_buf + m->data_size;
    /* Align to 8 bytes */
    alloc_ptr = (uint8_t *)(((uint64_t)alloc_ptr + 7) & ~7ULL);
    m->vocab = (llm_vocab_entry_t *)alloc_ptr;
    alloc_ptr += n_vocab * sizeof(llm_vocab_entry_t);

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
    for (int i = 0; i < n_vocab && i < LLM_MAX_VOCAB; i++) {
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

/* Find exact token ID for a string. Returns -1 if not found. */
static int llm_find_token(const llm_model_t *m, const char *str, int len)
{
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
static int llm_tokenize(const llm_model_t *m, const char *text,
                        int *tokens, int max_tokens)
{
    int n = 0;
    int text_len = 0;
    for (const char *p = text; *p; p++) text_len++;

    /* BPE-encode the input text so it matches GPT-2 byte-level vocab */
    static char enc_buf[4096];
    int enc_len = llm_bpe_encode(text, text_len, enc_buf, sizeof(enc_buf));

    /* Step 1: Greedy longest-match tokenization on encoded text */
    int pos = 0;
    while (pos < enc_len && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;

        /* Try progressively shorter substrings starting at pos */
        int max_try = enc_len - pos;
        if (max_try > 128) max_try = 128; /* limit match length */

        for (int try_len = max_try; try_len >= 1; try_len--) {
            int id = llm_find_token(m, enc_buf + pos, try_len);
            if (id >= 0) {
                best_len = try_len;
                best_id = id;
                break;
            }
        }

        if (best_id >= 0) {
            tokens[n++] = best_id;
            pos += best_len;
        } else {
            /* Fall back to byte token:
             * look up the BPE-encoded byte representation in vocab */
            uint8_t b0 = (uint8_t)enc_buf[pos];
            if (b0 >= 0xC0 && b0 < 0xE0 && pos + 1 < enc_len) {
                /* 2-byte UTF-8: try matching as token */
                int id = llm_find_token(m, enc_buf + pos, 2);
                if (id >= 0) { tokens[n++] = id; pos += 2; continue; }
            }
            /* Single-byte fallback */
            if (m->byte_tokens[b0] >= 0) {
                tokens[n++] = m->byte_tokens[b0];
            } else {
                tokens[n++] = 0; /* unknown */
            }
            pos++;
        }
    }

    /* Step 2: BPE merge pass (if scores available) */
    if (m->vocab_scores && n > 1) {
        int changed = 1;
        while (changed) {
            changed = 0;
            float best_score = -1e30f;
            int best_idx = -1;
            int best_merged = -1;

            for (int i = 0; i < n - 1; i++) {
                /* Concatenate token strings and look up */
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
                /* Remove the merged token */
                for (int i = best_idx + 1; i < n - 1; i++)
                    tokens[i] = tokens[i + 1];
                n--;
                changed = 1;
            }
        }
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

/* Simple temperature sampling with top-1 (greedy when temp=0) */
static int llm_sample(const float *logits, int vocab_size, float temperature)
{
    if (temperature <= 0.001f)
        return llm_argmax(logits, vocab_size);

    /* Apply temperature */
    static float probs[LLM_MAX_VOCAB];
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = llm_expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < vocab_size; i++)
        probs[i] *= inv_sum;

    /* Sample using pseudo-random (deterministic from TSC) */
    uint32_t lo, hi;
#if defined(__aarch64__)
    uint64_t cnt;
    __asm__ volatile ("mrs %0, cntpct_el0" : "=r"(cnt));
    lo = (uint32_t)cnt;
#else
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
#endif
    float r = (float)(lo % 10000) / 10000.0f;

    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (cdf >= r) return i;
    }
    return vocab_size - 1;
}

/* Generate text tokens autoregressively.
 * prompt_tokens: input token IDs (n_prompt tokens)
 * output_text: buffer for generated text
 * max_gen: maximum tokens to generate
 * temperature: sampling temperature (0.0 = greedy)
 * Returns: number of generated tokens */
static int llm_generate(llm_model_t *m, const int *prompt_tokens, int n_prompt,
                        char *output_text, int max_text_len,
                        int max_gen, float temperature)
{
    /* Reset KV cache */
    m->cache_len = 0;
    int kv_total = m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim;
    if (kv_total > LLM_KV_FLOATS) kv_total = LLM_KV_FLOATS;
    for (int i = 0; i < kv_total; i++) {
        m->k_cache[i] = 0.0f;
        m->v_cache[i] = 0.0f;
    }

    int out_pos = 0;
    int gen_count = 0;
    output_text[0] = '\0';

    /* Process prompt tokens (prefill) */
    for (int i = 0; i < n_prompt && i < m->max_seq - 1; i++) {
        llm_forward_token(m, llm_logits, prompt_tokens[i], i);
    }

    /* Generate new tokens */
    int last_token = (n_prompt > 0) ? prompt_tokens[n_prompt - 1] : m->bos_id;
    int pos = n_prompt;

    /* Get next token from the last forward pass */
    int next = llm_sample(llm_logits, m->vocab_size, temperature);

    for (int g = 0; g < max_gen && pos < m->max_seq; g++) {
        /* Check for EOS BEFORE decoding its text */
        if (next == m->eos_id) break;

        /* Decode token to text */
        char tok_buf[128];
        int tok_len = llm_decode_token(m, next, tok_buf, sizeof(tok_buf));
        for (int i = 0; i < tok_len && out_pos < max_text_len - 1; i++)
            output_text[out_pos++] = tok_buf[i];
        output_text[out_pos] = '\0';
        gen_count++;

        /* Check for <|im_end|> or <|endoftext|> in generated text */
        if (out_pos >= 10) {
            const char *tail = output_text + out_pos - 10;
            int stop = 0;
            for (int s = 0; s <= 10 && !stop; s++) {
                if (tail[s] == '<' && tail[s+1] == '|') {
                    stop = 1;
                    /* Truncate at the '<' */
                    for (int t = (int)(tail - output_text) + s; t < out_pos; t++)
                        output_text[t] = '\0';
                    out_pos = (int)(tail - output_text) + s;
                }
            }
            if (stop) break;
        }

        /* Stop after 2 newlines (model is rambling) */
        {
            int nl_count = 0;
            for (int i = 0; i < out_pos; i++)
                if (output_text[i] == '\n') nl_count++;
            if (nl_count >= 2) {
                /* Truncate at second newline */
                int nl_seen = 0;
                for (int i = 0; i < out_pos; i++) {
                    if (output_text[i] == '\n') {
                        nl_seen++;
                        if (nl_seen >= 2) {
                            output_text[i] = '\0';
                            out_pos = i;
                            break;
                        }
                    }
                }
                break;
            }
        }

        /* Forward the new token */
        llm_forward_token(m, llm_logits, next, pos);
        pos++;

        /* Sample next token */
        last_token = next;
        next = llm_sample(llm_logits, m->vocab_size, temperature);
    }

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
            else { kprintf("[LLM] WARN: %s not found\n", name_buf); layer->field = NULL; } \
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

        #undef MAP_TENSOR
    }

    kprintf("[LLM] Tensor mapping complete: %d layers\n", m->n_layers);
    return 0;
}

/* Load model from virtio-blk disk into model_cache memory region */
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
    uint64_t cache_size = (uint64_t)(__model_cache_end - __model_cache_start);
    if (capacity > cache_size) {
        kprintf("[LLM] Model (%lu MB) exceeds cache (%lu MB)\n",
                capacity / (1024 * 1024), cache_size / (1024 * 1024));
        return -1;
    }

    kprintf("[LLM] Loading model from disk: %lu MB\n", capacity / (1024 * 1024));

    /* Read the entire disk into model_cache_start */
    m->data_buf = (void *)__model_cache_start;
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

    /* If vocab_size is 0, try to infer from token_embd dimensions */
    if (m->vocab_size == 0) {
        const gguf_tensor_info_t *embd = gguf_find_tensor(&llm_gguf_ctx, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            m->vocab_size = (int)embd->dims[0];
        }
    }

    /* Compute derived parameters */
    if (m->n_heads > 0)
        m->head_dim = m->dim / m->n_heads;
    else
        m->head_dim = 64;
    m->max_seq = LLM_MAX_SEQ;
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;
    if (m->rope_base < 1.0f) m->rope_base = 10000.0f;

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

    /* Print model info */
    kprintf("[LLM] Model: %s (%s)\n", m->name, m->arch);
    kprintf("[LLM] Parameters: dim=%d, layers=%d, heads=%d, kv_heads=%d\n",
            m->dim, m->n_layers, m->n_heads, m->n_kv_heads);
    kprintf("[LLM] FFN=%d, vocab=%d, head_dim=%d, RoPE base=",
            m->ff_dim, m->vocab_size, m->head_dim);
    /* Print rope_base as integer if it's a round number */
    if (m->rope_base == (float)(int)m->rope_base && m->rope_base < 1e7f)
        kprintf("%d\n", (int)m->rope_base);
    else
        kprintf("~%d\n", (int)m->rope_base);

    /* Validate dimensions */
    if (m->dim > LLM_MAX_DIM) {
        kprintf("[LLM] ERROR: dim=%d exceeds LLM_MAX_DIM=%d\n", m->dim, LLM_MAX_DIM);
        return -1;
    }
    if (m->n_layers > LLM_MAX_LAYERS) {
        kprintf("[LLM] ERROR: layers=%d exceeds LLM_MAX_LAYERS=%d\n",
                m->n_layers, LLM_MAX_LAYERS);
        return -1;
    }
    if (m->ff_dim > LLM_MAX_FF) {
        kprintf("[LLM] ERROR: ff_dim=%d exceeds LLM_MAX_FF=%d\n", m->ff_dim, LLM_MAX_FF);
        return -1;
    }
    if (m->vocab_size > LLM_MAX_VOCAB) {
        kprintf("[LLM] WARN: vocab=%d exceeds max=%d, capping\n",
                m->vocab_size, LLM_MAX_VOCAB);
        m->vocab_size = LLM_MAX_VOCAB;
    }

    /* Map tensors */
    rc = llm_map_tensors(m, &llm_gguf_ctx);
    if (rc != 0) return rc;

    /* Build vocabulary */
    rc = llm_build_vocab(m, &llm_gguf_ctx);
    if (rc != 0) return rc;

    /* Allocate KV cache */
    int kv_size_needed = m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim;
    if (kv_size_needed > LLM_KV_FLOATS) {
        kprintf("[LLM] WARNING: KV cache too small (%d > %d), reducing max_seq\n",
                kv_size_needed, LLM_KV_FLOATS);
        m->max_seq = LLM_KV_FLOATS / (m->n_layers * m->n_kv_heads * m->head_dim);
        if (m->max_seq < 32) m->max_seq = 32;
        kprintf("[LLM] Adjusted max_seq to %d\n", m->max_seq);
    }
    m->k_cache = llm_kv_k;
    m->v_cache = llm_kv_v;
    m->cache_len = 0;

    kprintf("[LLM] Model loaded successfully! Ready for inference.\n");
    gguf_print_info(&llm_gguf_ctx);

    return 0;
#endif /* __aarch64__ */
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

    /* Check for Qwen (ChatML native) */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;

    /* Check for SmolLM / smollm in model name — SmolLM uses ChatML but
     * the special tokens may not be in base vocab; use simple prompt */
    /* (ChatML disabled for SmolLM until vocab includes <|im_start|>) */

    if (is_chatml) {
        /* ChatML format (Qwen, SmolLM, Mistral-Instruct-v0.3+, etc.) */
        { const char *t = "<|im_start|>system\nYou are a helpful math assistant. Give only the numeric answer.<|im_end|>\n<|im_start|>user\n"; kstrcpy(buf + pos, t); pos += kstrlen(t); }
        { kstrcpy(buf + pos, question); pos += kstrlen(question); }
        { const char *t = "<|im_end|>\n<|im_start|>assistant\n"; kstrcpy(buf + pos, t); pos += kstrlen(t); }
    } else if (kstrlen(m->arch) >= 5 &&
               m->arch[0] == 'g' && m->arch[1] == 'e' &&
               m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a') {
        /* Gemma format */
        { const char *t = "<start_of_turn>user\n"; kstrcpy(buf + pos, t); pos += kstrlen(t); }
        { kstrcpy(buf + pos, question); pos += kstrlen(question); }
        { const char *t = "<end_of_turn>\n<start_of_turn>model\n"; kstrcpy(buf + pos, t); pos += kstrlen(t); }
    } else {
        /* Generic: simple text prompt (works with any model) */
        { kstrcpy(buf + pos, question); pos += kstrlen(question); }
    }
    (void)max_len;
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
        int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens, LLM_MAX_TOKENS - 32);

        kprintf("  [%d/%d] %s\n", p + 1, N_MATH_PROBLEMS, prob->prompt);
        kprintf("         Tokens: %d, generating...\n", n_tokens);

        /* Generate */
        uint64_t t0 = rdtsc_fenced();
        int n_gen = llm_generate(m, llm_tokens, n_tokens, output_buf, sizeof(output_buf),
                                 16, 0.0f); /* greedy, max 16 tokens */
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
