/* =============================================================================
 * TensorOS - LLM JIT Kernels
 *
 * Mature JIT-compiled native x86_64 kernels for every operation in the LLM
 * inference pipeline. Each kernel is SSE2-vectorized with 4-wide processing,
 * proper register allocation, loop unrolling, and tail handling.
 *
 * Kernels:
 *   - Fast vectorized exp (building block for SiLU/softmax/sigmoid/GELU)
 *   - SiLU activation (SwiGLU FFN)
 *   - RMSNorm (pre-attention and pre-FFN normalization)
 *   - Softmax (attention score normalization)
 *   - RoPE (rotary position encoding)
 *   - Element-wise multiply (SwiGLU gate⊙up)
 *   - Element-wise add (residual connections)
 *   - Dot product (attention scores)
 *   - Scaled add / axpy (weighted V accumulation)
 *   - Fused SiLU*multiply (SwiGLU single-pass)
 *   - GELU activation
 *   - LayerNorm
 *
 * All generated code follows System V AMD64 calling convention.
 * =============================================================================*/

#ifndef __aarch64__

#include "runtime/jit/x86_jit.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Constants for JIT exp/sigmoid/SiLU
 *
 * We embed float constants directly in JIT buffers and reference them via
 * absolute address loads (mov r64, imm64; movaps xmm, [r64]).
 * =============================================================================*/

/* Aligned constant pools (16-byte aligned for movaps) */
static const float __attribute__((aligned(16)))
    jit_const_log2e[4]   = {1.4426950408f, 1.4426950408f, 1.4426950408f, 1.4426950408f};
static const float __attribute__((aligned(16)))
    jit_const_ln2[4]     = {0.6931471805f, 0.6931471805f, 0.6931471805f, 0.6931471805f};
static const float __attribute__((aligned(16)))
    jit_const_one[4]     = {1.0f, 1.0f, 1.0f, 1.0f};
static const float __attribute__((aligned(16)))
    jit_const_half[4]    = {0.5f, 0.5f, 0.5f, 0.5f};
static const float __attribute__((aligned(16)))
    jit_const_zero[4]    = {0.0f, 0.0f, 0.0f, 0.0f};
static const float __attribute__((aligned(16)))
    jit_const_neg_zero[4] = {-0.0f, -0.0f, -0.0f, -0.0f};
static const int32_t __attribute__((aligned(16)))
    jit_const_127i[4]    = {127, 127, 127, 127};
static const float __attribute__((aligned(16)))
    jit_const_exp_hi[4]  = {88.3762626647f, 88.3762626647f, 88.3762626647f, 88.3762626647f};
static const float __attribute__((aligned(16)))
    jit_const_exp_lo[4]  = {-88.3762626647f, -88.3762626647f, -88.3762626647f, -88.3762626647f};

/* Polynomial coefficients for exp: 2^f ≈ c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))) */
static const float __attribute__((aligned(16)))
    jit_const_exp_c1[4]  = {0.6931471806f, 0.6931471806f, 0.6931471806f, 0.6931471806f};
static const float __attribute__((aligned(16)))
    jit_const_exp_c2[4]  = {0.2402265070f, 0.2402265070f, 0.2402265070f, 0.2402265070f};
static const float __attribute__((aligned(16)))
    jit_const_exp_c3[4]  = {0.0555041086f, 0.0555041086f, 0.0555041086f, 0.0555041086f};
static const float __attribute__((aligned(16)))
    jit_const_exp_c4[4]  = {0.0096181291f, 0.0096181291f, 0.0096181291f, 0.0096181291f};
static const float __attribute__((aligned(16)))
    jit_const_exp_c5[4]  = {0.0013333558f, 0.0013333558f, 0.0013333558f, 0.0013333558f};

/* RMSNorm epsilon */
static const float __attribute__((aligned(16)))
    jit_const_eps[4]     = {1e-6f, 1e-6f, 1e-6f, 1e-6f};

/* GELU constant: sqrt(2/pi) ≈ 0.7978845608 */
static const float __attribute__((aligned(16)))
    jit_const_gelu_a[4]  = {0.7978845608f, 0.7978845608f, 0.7978845608f, 0.7978845608f};
static const float __attribute__((aligned(16)))
    jit_const_gelu_b[4]  = {0.0356774f, 0.0356774f, 0.0356774f, 0.0356774f};
static const float __attribute__((aligned(16)))
    jit_const_three[4]   = {3.0f, 3.0f, 3.0f, 3.0f};

/* Absolute value mask */
static const uint32_t __attribute__((aligned(16)))
    jit_const_abs_mask[4] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};

/* =============================================================================
 * Helper: Emit inline vectorized exp(xmm_in) → xmm_out
 *
 * Uses Cephes-style range reduction + polynomial evaluation:
 *   exp(x) = 2^n * 2^f where n = round(x * log2e), f = x*log2e - n
 *   2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
 *   2^n = bitcast((n+127) << 23)
 *
 * Clobbers: xmm_out, xmm_tmp1, xmm_tmp2, xmm_tmp3
 * Requires: RAX available for loading constants
 * =============================================================================*/

static void jit_emit_exp4(jit_buf_t *b,
                           int xmm_in,   /* input: x values */
                           int xmm_out,  /* output: exp(x) */
                           int t1, int t2, int t3) /* temporaries */
{
    /* Clamp x to [-88, 88] to avoid overflow/underflow */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_hi);
    jit_movaps_load(b, t1, RAX, 0);
    jit_movaps_reg(b, xmm_out, xmm_in);
    jit_minps(b, xmm_out, t1);     /* min(x, 88) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_lo);
    jit_movaps_load(b, t1, RAX, 0);
    jit_maxps(b, xmm_out, t1);     /* max(min(x, 88), -88) */

    /* t1 = x * log2(e) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_log2e);
    jit_movaps_load(b, t1, RAX, 0);
    jit_mulps(b, t1, xmm_out);     /* t1 = x * log2e */

    /* n = round(t1) — using cvttps2dq (truncate) + adjustment */
    /* Round to nearest integer n via floor(x*log2e + 0.5).
     * SSE2 floor: truncate then correct for negative values where
     * truncation rounds toward zero instead of toward -infinity. */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_half);
    jit_movaps_load(b, t2, RAX, 0);
    jit_movaps_reg(b, t3, t1);
    jit_addps(b, t3, t2);          /* t3 = x*log2e + 0.5 */

    jit_cvttps2dq(b, t2, t3);      /* t2 = (int)trunc(t3) */
    jit_cvtdq2ps(b, xmm_out, t2);  /* xmm_out = (float)trunc for comparison */
    jit_cmpps(b, xmm_out, t3, 6);  /* NLE mask: where float_trunc > biased */
    jit_paddd(b, t2, xmm_out);     /* correct: n -= 1 where truncation overshot */
    jit_cvtdq2ps(b, t3, t2);       /* t3 = corrected (float)n */

    /* f = x*log2e - n (fractional part in [-0.5, 0.5]) */
    jit_movaps_reg(b, xmm_out, t1);
    jit_subps(b, xmm_out, t3);     /* f = x*log2e - n */

    /* Evaluate polynomial: 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))) */
    /* Horner's method: result = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + 1 */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_c5);
    jit_movaps_load(b, t1, RAX, 0);
    jit_mulps(b, t1, xmm_out);     /* c5*f */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_c4);
    jit_movaps_load(b, t3, RAX, 0);
    jit_addps(b, t1, t3);          /* c5*f + c4 */
    jit_mulps(b, t1, xmm_out);     /* (c5*f + c4)*f */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_c3);
    jit_movaps_load(b, t3, RAX, 0);
    jit_addps(b, t1, t3);          /* + c3 */
    jit_mulps(b, t1, xmm_out);     /* *f */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_c2);
    jit_movaps_load(b, t3, RAX, 0);
    jit_addps(b, t1, t3);          /* + c2 */
    jit_mulps(b, t1, xmm_out);     /* *f */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_exp_c1);
    jit_movaps_load(b, t3, RAX, 0);
    jit_addps(b, t1, t3);          /* + c1 */
    jit_mulps(b, t1, xmm_out);     /* *f */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movaps_load(b, t3, RAX, 0);
    jit_addps(b, t1, t3);          /* + 1.0 = poly result */

    /* Construct 2^n: int bits = (n + 127) << 23 → reinterpret as float */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_127i);
    jit_movaps_load(b, t3, RAX, 0); /* t3 = {127,127,127,127} as int */
    jit_paddd(b, t2, t3);          /* t2 = n + 127 */
    jit_pslld_imm(b, t2, 23);     /* t2 = (n+127) << 23 = float bits of 2^n */

    /* result = poly * 2^n */
    jit_mulps(b, t1, t2);          /* t1 is poly, t2 is 2^n (as float bits) */
    jit_movaps_reg(b, xmm_out, t1);
}

/* =============================================================================
 * Kernel cache (shared across all LLM JIT kernels)
 * =============================================================================*/

#define LLM_JIT_CACHE_MAX 32

/* Generic cache entry */
typedef struct {
    int type;       /* kernel type ID */
    int dim;        /* primary dimension */
    void *fn;       /* compiled function pointer */
} llm_jit_cache_entry_t;

static llm_jit_cache_entry_t llm_jit_cache[LLM_JIT_CACHE_MAX];
static int llm_jit_cache_count = 0;

enum {
    JIT_K_SILU = 1, JIT_K_RMSNORM, JIT_K_SOFTMAX, JIT_K_ROPE,
    JIT_K_VMUL, JIT_K_VADD, JIT_K_DOT, JIT_K_AXPY,
    JIT_K_FUSED_SILU_MUL, JIT_K_GELU, JIT_K_LAYERNORM
};

static void *llm_jit_cache_lookup(int type, int dim)
{
    for (int i = 0; i < llm_jit_cache_count; i++)
        if (llm_jit_cache[i].type == type && llm_jit_cache[i].dim == dim)
            return llm_jit_cache[i].fn;
    return NULL;
}

static void llm_jit_cache_store(int type, int dim, void *fn)
{
    if (llm_jit_cache_count < LLM_JIT_CACHE_MAX) {
        llm_jit_cache[llm_jit_cache_count].type = type;
        llm_jit_cache[llm_jit_cache_count].dim = dim;
        llm_jit_cache[llm_jit_cache_count].fn = fn;
        llm_jit_cache_count++;
    }
}

/* =============================================================================
 * JIT Kernel: SiLU — x[i] = x[i] * sigmoid(x[i])
 *
 * SiLU(x) = x / (1 + exp(-x))
 *
 * void silu(float *x, int n)
 * Args: x=rdi (n is compile-time constant)
 *
 * Register allocation:
 *   R12 = x pointer, RBX = loop counter (byte offset)
 *   XMM0 = input x, XMM6 = output, XMM1-5 = temporaries for exp
 *   XMM7 = constant {1,1,1,1}
 * =============================================================================*/

jit_silu_fn jit_compile_silu_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_SILU, n);
    if (cached) return (jit_silu_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);   /* R12 = x */

    /* Preload constant 1.0 into xmm7 */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movaps_load(b, XMM7, RAX, 0);

    jit_xor_reg_reg(b, RBX, RBX);   /* byte offset = 0 */

    int loop_top = b->len;

    /* Load 4 floats: xmm0 = x[i..i+3] */
    jit_movups_load(b, XMM0, R12, 0);

    /* Compute exp(-x): negate xmm0 into xmm6 */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_neg_zero);
    jit_movaps_load(b, XMM6, RAX, 0);
    jit_xorps(b, XMM6, XMM0);      /* xmm6 = -x (flip sign bits) */

    /* xmm6 = exp(-x) using inline exp (clobbers xmm1-5, uses RAX) */
    jit_emit_exp4(b, XMM6, XMM6, XMM1, XMM2, XMM3);

    /* sigmoid = 1 / (1 + exp(-x)) */
    jit_addps(b, XMM6, XMM7);      /* xmm6 = 1 + exp(-x) */
    jit_movaps_reg(b, XMM1, XMM7);
    jit_divps(b, XMM1, XMM6);      /* xmm1 = 1 / (1 + exp(-x)) = sigmoid */

    /* SiLU = x * sigmoid(x) */
    jit_mulps(b, XMM0, XMM1);      /* xmm0 = x * sigmoid(x) */

    /* Store result */
    jit_movups_store(b, R12, 0, XMM0);

    jit_add_reg_imm32(b, R12, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    /* Tail: handle remaining elements (n % 4) scalar */
    int tail = n % 4;
    for (int t = 0; t < tail; t++) {
        /* Load scalar x[i] */
        jit_movss_load(b, XMM0, R12, t * 4);
        /* Negate */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_neg_zero);
        jit_movaps_load(b, XMM6, RAX, 0);
        jit_xorps(b, XMM6, XMM0);
        /* exp(-x) scalar via broadcast then exp4 */
        jit_shufps(b, XMM6, XMM6, 0x00);
        jit_emit_exp4(b, XMM6, XMM6, XMM1, XMM2, XMM3);
        /* sigmoid */
        jit_addss(b, XMM6, XMM7);
        jit_movaps_reg(b, XMM1, XMM7);
        jit_divss(b, XMM1, XMM6);
        /* x * sigmoid */
        jit_mulss(b, XMM0, XMM1);
        jit_movss_store(b, R12, t * 4, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_silu_fn fn = (jit_silu_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_SILU, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: RMSNorm
 *
 * out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
 *
 * void rmsnorm(float *out, const float *x, const float *w, int dim)
 * Args: out=rdi, x=rsi, w=rdx (dim is compile-time constant)
 *
 * Register allocation:
 *   R12 = out, R13 = x, R14 = w, RBX = loop counter
 *   XMM6 = accumulator (sum of squares), XMM7 = scratch
 *   Pass 1: sum of x^2
 *   Pass 2: normalize and scale
 * =============================================================================*/

jit_rmsnorm_fn jit_compile_rmsnorm_kernel(int dim)
{
    void *cached = llm_jit_cache_lookup(JIT_K_RMSNORM, dim);
    if (cached) return (jit_rmsnorm_fn)cached;

    int vecs = dim / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(2048);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);   /* out */
    jit_mov_reg_reg(b, R13, RSI);   /* x */
    jit_mov_reg_reg(b, R14, RDX);   /* w */

    /* === Pass 1: Sum of squares === */
    jit_xorps(b, XMM6, XMM6);       /* accum = 0 */
    jit_xorps(b, XMM7, XMM7);       /* accum2 = 0 (dual accumulator) */
    jit_mov_reg_reg(b, RBX, R13);    /* walk pointer */
    jit_xor_reg_reg(b, RCX, RCX);   /* counter */

    int ss_top = b->len;
    jit_movups_load(b, XMM0, RBX, 0);
    jit_movaps_reg(b, XMM1, XMM0);
    jit_mulps(b, XMM0, XMM0);       /* x^2 */
    jit_addps(b, XMM6, XMM0);       /* accum += x^2 */
    /* Unroll: second group if possible */
    if (vecs >= 2) {
        jit_movups_load(b, XMM2, RBX, 16);
        jit_mulps(b, XMM2, XMM2);
        jit_addps(b, XMM7, XMM2);
    }

    jit_add_reg_imm32(b, RBX, vecs >= 2 ? 32 : 16);
    jit_add_reg_imm32(b, RCX, vecs >= 2 ? 2 : 1);
    jit_cmp_reg_imm32(b, RCX, vecs);
    jit_jl_back(b, ss_top);

    /* Combine dual accumulators */
    jit_addps(b, XMM6, XMM7);

    /* Horizontal sum of XMM6 */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0x4E);
    jit_addps(b, XMM6, XMM0);
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0xB1);
    jit_addss(b, XMM6, XMM0);
    /* XMM6[0] = sum of squares */

    /* Compute: inv_rms = 1.0 / sqrt(ss / dim + eps) */
    /* XMM6[0] /= dim */
    {
        union { float f; uint32_t u; } dimf = { .f = (float)dim };
        jit_mov_reg_imm64(b, RAX, dimf.u);
        jit_movd_to_xmm(b, XMM1, RAX);
        jit_divss(b, XMM6, XMM1);
    }
    /* Add eps */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_eps);
    jit_movss_load(b, XMM1, RAX, 0);
    jit_addss(b, XMM6, XMM1);

    /* rsqrt for fast inverse square root + Newton refinement */
    /* rsqrtss gives ~11-bit precision; one Newton step gives ~22-bit */
    jit_rsqrtps(b, XMM5, XMM6);   /* XMM5 ≈ 1/sqrt(x) */
    /* Newton: y = y * (1.5 - 0.5*x*y*y) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_half);
    jit_movss_load(b, XMM1, RAX, 0);
    jit_movaps_reg(b, XMM2, XMM6);
    jit_mulss(b, XMM2, XMM1);      /* 0.5 * x */
    jit_movaps_reg(b, XMM3, XMM5);
    jit_mulss(b, XMM3, XMM5);      /* y^2 */
    jit_mulss(b, XMM3, XMM2);      /* 0.5*x*y^2 */
    jit_addss(b, XMM1, XMM1);      /* XMM1 = 1.0 (0.5+0.5) */
    jit_addss(b, XMM1, jit_const_half[0] != 0.5f ? XMM1 : XMM1); /* skip, use 1.5 */
    /* Actually, load 1.5 directly */
    {
        union { float f; uint32_t u; } onehalf = { .f = 1.5f };
        jit_mov_reg_imm64(b, RAX, onehalf.u);
        jit_movd_to_xmm(b, XMM1, RAX);
    }
    jit_subss(b, XMM1, XMM3);      /* 1.5 - 0.5*x*y^2 */
    jit_mulss(b, XMM5, XMM1);      /* y = y * (1.5 - 0.5*x*y^2) refined */

    /* Broadcast inv_rms to all 4 lanes */
    jit_shufps(b, XMM5, XMM5, 0x00); /* XMM5 = {inv_rms, inv_rms, inv_rms, inv_rms} */

    /* === Pass 2: out[i] = x[i] * inv_rms * w[i] === */
    jit_xor_reg_reg(b, RBX, RBX);
    int norm_top = b->len;

    jit_movups_load(b, XMM0, R13, 0); /* x[i..i+3] */
    jit_mulps(b, XMM0, XMM5);         /* * inv_rms */
    jit_movups_load(b, XMM1, R14, 0); /* w[i..i+3] */
    jit_mulps(b, XMM0, XMM1);         /* * w */
    jit_movups_store(b, R12, 0, XMM0);

    jit_add_reg_imm32(b, R12, 16);
    jit_add_reg_imm32(b, R13, 16);
    jit_add_reg_imm32(b, R14, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, norm_top);

    /* Tail for dim % 4 */
    int tail = dim % 4;
    /* Reset pointers for tail: R12, R13, R14 are already past the vectorized part */
    for (int t = 0; t < tail; t++) {
        jit_movss_load(b, XMM0, R13, t * 4 - (vecs > 0 ? 0 : 0));
        jit_mulss(b, XMM0, XMM5);
        jit_movss_load(b, XMM1, R14, t * 4);
        jit_mulss(b, XMM0, XMM1);
        jit_movss_store(b, R12, t * 4, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_rmsnorm_fn fn = (jit_rmsnorm_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_RMSNORM, dim, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Softmax
 *
 * Three-pass vectorized softmax:
 *   Pass 1: max = max(x[0..n-1])
 *   Pass 2: x[i] = exp(x[i] - max), sum += x[i]
 *   Pass 3: x[i] /= sum
 *
 * void softmax(float *x, int n)
 * Args: x=rdi (n is compile-time constant)
 *
 * Register allocation:
 *   R12 = x, RBX = counter
 *   XMM5 = max/sum broadcast, XMM6-7 = accumulators
 * =============================================================================*/

jit_softmax_fn jit_compile_softmax_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_SOFTMAX, n);
    if (cached) return (jit_softmax_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(8192);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);

    /* === Pass 1: Find max === */
    /* Initialize max from first 4 elements */
    jit_movups_load(b, XMM6, R12, 0);
    if (vecs > 1) {
        jit_mov_reg_reg(b, RBX, R12);
        jit_add_reg_imm32(b, RBX, 16);
        jit_xor_reg_reg(b, RCX, RCX);
        jit_inc_reg(b, RCX);

        int max_top = b->len;
        jit_movups_load(b, XMM0, RBX, 0);
        jit_maxps(b, XMM6, XMM0);
        jit_add_reg_imm32(b, RBX, 16);
        jit_inc_reg(b, RCX);
        jit_cmp_reg_imm32(b, RCX, vecs);
        jit_jl_back(b, max_top);
    }

    /* Horizontal max of XMM6 */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0x4E);
    jit_maxps(b, XMM6, XMM0);
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0xB1);
    jit_maxss(b, XMM6, XMM0);

    /* Handle tail scalars for max */
    for (int t = 0; t < (n % 4); t++) {
        jit_movss_load(b, XMM0, R12, vecs * 16 + t * 4);
        jit_maxss(b, XMM6, XMM0);
    }

    /* Broadcast max to all 4 lanes */
    jit_shufps(b, XMM6, XMM6, 0x00);  /* XMM6 = {max, max, max, max} */
    jit_movaps_reg(b, XMM5, XMM6);    /* save max in XMM5 */

    /* === Pass 2: exp(x-max) and sum === */
    jit_xorps(b, XMM7, XMM7);         /* sum accumulator = 0 */
    jit_mov_reg_reg(b, RBX, R12);
    jit_xor_reg_reg(b, RCX, RCX);

    int exp_top = b->len;
    jit_movups_load(b, XMM0, RBX, 0);
    jit_subps(b, XMM0, XMM5);         /* x - max */
    jit_emit_exp4(b, XMM0, XMM0, XMM1, XMM2, XMM3);  /* exp(x - max) */
    jit_movups_store(b, RBX, 0, XMM0);
    jit_addps(b, XMM7, XMM0);         /* sum += exp(x - max) */

    jit_add_reg_imm32(b, RBX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, vecs);
    jit_jl_back(b, exp_top);

    /* Handle tail elements */
    for (int t = 0; t < (n % 4); t++) {
        int off = vecs * 16 + t * 4;
        jit_movss_load(b, XMM0, R12, off);
        jit_subss(b, XMM0, XMM5);
        jit_shufps(b, XMM0, XMM0, 0x00);
        jit_emit_exp4(b, XMM0, XMM0, XMM1, XMM2, XMM3);
        jit_movss_store(b, R12, off, XMM0);
        jit_addss(b, XMM7, XMM0);
    }

    /* Horizontal sum of XMM7 */
    jit_movaps_reg(b, XMM0, XMM7);
    jit_shufps(b, XMM0, XMM7, 0x4E);
    jit_addps(b, XMM7, XMM0);
    jit_movaps_reg(b, XMM0, XMM7);
    jit_shufps(b, XMM0, XMM7, 0xB1);
    jit_addss(b, XMM7, XMM0);

    /* inv_sum = 1.0 / sum */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movss_load(b, XMM6, RAX, 0);
    jit_divss(b, XMM6, XMM7);
    jit_shufps(b, XMM6, XMM6, 0x00);  /* broadcast inv_sum */

    /* === Pass 3: Normalize x[i] *= inv_sum === */
    jit_mov_reg_reg(b, RBX, R12);
    jit_xor_reg_reg(b, RCX, RCX);

    int norm_top = b->len;
    jit_movups_load(b, XMM0, RBX, 0);
    jit_mulps(b, XMM0, XMM6);
    jit_movups_store(b, RBX, 0, XMM0);
    jit_add_reg_imm32(b, RBX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, vecs);
    jit_jl_back(b, norm_top);

    /* Tail normalize */
    for (int t = 0; t < (n % 4); t++) {
        int off = vecs * 16 + t * 4;
        jit_movss_load(b, XMM0, R12, off);
        jit_mulss(b, XMM0, XMM6);
        jit_movss_store(b, R12, off, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_softmax_fn fn = (jit_softmax_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_SOFTMAX, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: RoPE (Rotary Position Encoding)
 *
 * For each pair (v0, v1):
 *   theta = pos * freq[i/2]
 *   v0' = v0 * cos(theta) - v1 * sin(theta)
 *   v1' = v0 * sin(theta) + v1 * cos(theta)
 *
 * void rope(float *vec, int pos, int head_dim, const float *freqs)
 * Args: vec=rdi, pos=esi, head_dim=edx, freqs=rcx
 *
 * Since sin/cos are expensive to JIT, we use precomputed freq table + fast
 * polynomial sin/cos coded inline. For RoPE the angles are moderate (pos*freq).
 *
 * Register allocation:
 *   R12 = vec, R13 = freqs, R14 = pos (as float, broadcast)
 *   R15 = pair counter, RBX = pair index
 * =============================================================================*/

/* Fast sin/cos constants */
static const float __attribute__((aligned(16)))
    jit_const_pi[4]      = {3.14159265f, 3.14159265f, 3.14159265f, 3.14159265f};
static const float __attribute__((aligned(16)))
    jit_const_twopi[4]   = {6.28318530f, 6.28318530f, 6.28318530f, 6.28318530f};
static const float __attribute__((aligned(16)))
    jit_const_inv_twopi[4] = {0.15915494f, 0.15915494f, 0.15915494f, 0.15915494f};

/* cos polynomial: 1 - t^2/2 + t^4/24 - t^6/720 (Taylor) */
static const float __attribute__((aligned(16)))
    jit_const_cos_c2[4]  = {-0.5f, -0.5f, -0.5f, -0.5f};
static const float __attribute__((aligned(16)))
    jit_const_cos_c4[4]  = {0.04166667f, 0.04166667f, 0.04166667f, 0.04166667f};
static const float __attribute__((aligned(16)))
    jit_const_cos_c6[4]  = {-0.001388889f, -0.001388889f, -0.001388889f, -0.001388889f};

/* sin polynomial: t - t^3/6 + t^5/120 - t^7/5040 */
static const float __attribute__((aligned(16)))
    jit_const_sin_c3[4]  = {-0.1666667f, -0.1666667f, -0.1666667f, -0.1666667f};
static const float __attribute__((aligned(16)))
    jit_const_sin_c5[4]  = {0.008333333f, 0.008333333f, 0.008333333f, 0.008333333f};
static const float __attribute__((aligned(16)))
    jit_const_sin_c7[4]  = {-0.000198413f, -0.000198413f, -0.000198413f, -0.000198413f};

jit_rope_fn jit_compile_rope_kernel(int head_dim)
{
    void *cached = llm_jit_cache_lookup(JIT_K_ROPE, head_dim);
    if (cached) return (jit_rope_fn)cached;

    int pairs = head_dim / 2;
    if (pairs < 1) return NULL;

    /* For RoPE we process 2 pairs at a time (4 floats = 1 SSE2 vector) */
    int vec_pairs = pairs / 2;  /* number of 4-float iterations */

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);    /* vec */
    /* ESI = pos (int), convert to float and broadcast */
    /* cvtsi2ss xmm7, esi — but we use movd + cvtdq2ps for simplicity */
    jit_movd_to_xmm(b, XMM7, RSI);
    jit_cvtdq2ps(b, XMM7, XMM7);
    jit_shufps(b, XMM7, XMM7, 0x00); /* XMM7 = {pos_f, pos_f, pos_f, pos_f} */

    jit_mov_reg_reg(b, R13, RCX);    /* freqs table */

    /* Process 2 pairs = 4 floats at a time.
     * freqs layout: freq[0], freq[1], freq[2], ... for each pair.
     * We load freq[p], freq[p+1] and duplicate: {f0,f0,f1,f1} for interleave. */

    jit_xor_reg_reg(b, R15, R15);   /* vector index */

    if (vec_pairs > 0) {
        int loop_top = b->len;

        /* Load 2 consecutive frequencies */
        /* Build {freq[p], freq[p], freq[p+1], freq[p+1]} */
        jit_movss_load(b, XMM0, R13, 0);   /* freq[p] */
        jit_shufps(b, XMM0, XMM0, 0x00);   /* {f0,f0,f0,f0} */
        jit_movss_load(b, XMM1, R13, 4);   /* freq[p+1] */
        jit_shufps(b, XMM1, XMM1, 0x00);   /* {f1,f1,f1,f1} */
        /* Merge: XMM0 = {f0, f0, f1, f1} */
        jit_shufps(b, XMM0, XMM1, 0x44);   /* {f0_lo, f0_hi, f1_lo, f1_hi} = {f0,f0,f1,f1} */

        /* theta = pos * freq */
        jit_mulps(b, XMM0, XMM7);   /* XMM0 = {theta0, theta0, theta1, theta1} */

        /* Range reduce theta to [-pi, pi] using: theta = theta - round(theta/(2*pi)) * 2*pi */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_inv_twopi);
        jit_movaps_load(b, XMM1, RAX, 0);
        jit_movaps_reg(b, XMM2, XMM0);
        jit_mulps(b, XMM2, XMM1);       /* theta / 2pi */
        jit_cvttps2dq(b, XMM3, XMM2);   /* round to int */
        jit_cvtdq2ps(b, XMM3, XMM3);    /* back to float */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_twopi);
        jit_movaps_load(b, XMM1, RAX, 0);
        jit_mulps(b, XMM3, XMM1);       /* n * 2pi */
        jit_subps(b, XMM0, XMM3);       /* theta in ~[-2pi, 2pi] */

        /* Reduce from [-2pi, 2pi] to [-pi, pi] for polynomial accuracy */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_pi);
        jit_movaps_load(b, XMM1, RAX, 0);           /* XMM1 = pi */
        jit_movaps_reg(b, XMM2, XMM0);
        jit_cmpps(b, XMM2, XMM1, 6);                /* mask: theta > pi (NLE) */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_twopi);
        jit_movaps_load(b, XMM3, RAX, 0);           /* XMM3 = 2*pi */
        jit_movaps_reg(b, XMM4, XMM3);              /* save 2*pi copy */
        jit_andps(b, XMM3, XMM2);                   /* 2*pi where theta > pi */
        jit_subps(b, XMM0, XMM3);                   /* theta -= 2*pi where needed */
        /* Check theta < -pi */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_neg_zero);
        jit_movaps_load(b, XMM2, RAX, 0);
        jit_xorps(b, XMM2, XMM1);                   /* XMM2 = -pi */
        jit_movaps_reg(b, XMM3, XMM0);              /* copy theta */
        jit_cmpps(b, XMM3, XMM2, 1);                /* mask: theta < -pi (LT) */
        jit_andps(b, XMM4, XMM3);                   /* 2*pi where theta < -pi */
        jit_addps(b, XMM0, XMM4);                   /* theta += 2*pi where needed */

        /* theta^2 */
        jit_movaps_reg(b, XMM4, XMM0);
        jit_mulps(b, XMM4, XMM4);       /* XMM4 = t^2 */

        /* Compute cos(theta) using polynomial:
         * cos(t) ≈ 1 + c2*t^2 + c4*t^4 + c6*t^6 (Horner) */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c6);
        jit_movaps_load(b, XMM5, RAX, 0);
        jit_mulps(b, XMM5, XMM4);       /* c6*t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c4);
        jit_movaps_load(b, XMM6, RAX, 0);
        jit_addps(b, XMM5, XMM6);       /* c6*t^2 + c4 */
        jit_mulps(b, XMM5, XMM4);       /* (c6*t^2+c4)*t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c2);
        jit_movaps_load(b, XMM6, RAX, 0);
        jit_addps(b, XMM5, XMM6);       /* + c2 */
        jit_mulps(b, XMM5, XMM4);       /* * t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
        jit_movaps_load(b, XMM6, RAX, 0);
        jit_addps(b, XMM5, XMM6);       /* + 1.0  XMM5 = cos(theta) */

        /* Compute sin(theta) using polynomial:
         * sin(t) ≈ t + c3*t^3 + c5*t^5 + c7*t^7 = t*(1 + t^2*(c3 + t^2*(c5 + t^2*c7))) */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c7);
        jit_movaps_load(b, XMM6, RAX, 0);
        jit_mulps(b, XMM6, XMM4);       /* c7*t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c5);
        jit_movaps_load(b, XMM1, RAX, 0);
        jit_addps(b, XMM6, XMM1);       /* c7*t^2 + c5 */
        jit_mulps(b, XMM6, XMM4);       /* * t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c3);
        jit_movaps_load(b, XMM1, RAX, 0);
        jit_addps(b, XMM6, XMM1);       /* + c3 */
        jit_mulps(b, XMM6, XMM4);       /* * t^2 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
        jit_movaps_load(b, XMM1, RAX, 0);
        jit_addps(b, XMM6, XMM1);       /* + 1.0 */
        jit_mulps(b, XMM6, XMM0);       /* * t  XMM6 = sin(theta) */

        /* Load vec: {v0, v1, v2, v3} = {pair0_even, pair0_odd, pair1_even, pair1_odd} */
        jit_movups_load(b, XMM0, R12, 0);

        /* We need: out_even = v_even * cos - v_odd * sin
         *          out_odd  = v_even * sin + v_odd * cos
         * XMM0 = {e0, o0, e1, o1}
         * XMM5 = {cos0, cos0, cos1, cos1}  (already in this form)
         * XMM6 = {sin0, sin0, sin1, sin1}  (already in this form) */

        /* Separate evens and odds */
        jit_movaps_reg(b, XMM1, XMM0);  /* copy */
        jit_shufps(b, XMM1, XMM1, 0xA0); /* {e0, e0, e1, e1} = evens duplicated */
        jit_movaps_reg(b, XMM2, XMM0);
        jit_shufps(b, XMM2, XMM2, 0xF5); /* {o0, o0, o1, o1} = odds duplicated */

        /* result_even = e * cos - o * sin */
        /* result_odd  = e * sin + o * cos */
        jit_movaps_reg(b, XMM3, XMM1);
        jit_mulps(b, XMM3, XMM5);       /* e * cos */
        jit_movaps_reg(b, XMM4, XMM2);
        jit_mulps(b, XMM4, XMM6);       /* o * sin */
        jit_subps(b, XMM3, XMM4);       /* e*cos - o*sin (goes into even slots) */

        jit_mulps(b, XMM1, XMM6);       /* e * sin */
        jit_mulps(b, XMM2, XMM5);       /* o * cos */
        jit_addps(b, XMM1, XMM2);       /* e*sin + o*cos (goes into odd slots) */

        /* Interleave: {res_even0, res_odd0, res_even1, res_odd1} */
        /* XMM3 = {re0, re0, re1, re1}, XMM1 = {ro0, ro0, ro1, ro1} */
        /* We want: {re0, ro0, re1, ro1} */
        /* unpcklps gives {re0_lo, ro0_lo, re0_hi, ro0_hi} — not quite.
         * Actually: shufps XMM3, XMM1, 0x88 = {3[0],3[2],1[0],1[2]} = {re0,re1,ro0,ro1}
         * Then: shufps result, result, 0xD8 = {0,2,1,3} = {re0,ro0,re1,ro1} */
        jit_shufps(b, XMM3, XMM1, 0x88); /* {re0, re1, ro0, ro1} */
        jit_shufps(b, XMM3, XMM3, 0xD8); /* {re0, ro0, re1, ro1} */

        /* Store */
        jit_movups_store(b, R12, 0, XMM3);

        jit_add_reg_imm32(b, R12, 16);  /* advance vec by 4 floats (2 pairs) */
        jit_add_reg_imm32(b, R13, 8);   /* advance freqs by 2 floats */
        jit_inc_reg(b, R15);
        jit_cmp_reg_imm32(b, R15, vec_pairs);
        jit_jl_back(b, loop_top);
    }

    /* Handle remaining odd pair (if pairs is odd) */
    if (pairs % 2) {
        /* Scalar pair: load freq, compute theta, sin/cos, rotate */
        jit_movss_load(b, XMM0, R13, 0);  /* freq */
        jit_mulss(b, XMM0, XMM7);         /* theta = pos * freq (scalar in lane 0) */

        /* Quick scalar sin/cos using the same polynomial */
        /* t^2 */
        jit_movaps_reg(b, XMM4, XMM0);
        jit_mulss(b, XMM4, XMM4);

        /* cos: 1 + c2*t^2 + c4*t^4 + c6*t^6 */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c6);
        jit_movss_load(b, XMM5, RAX, 0);
        jit_mulss(b, XMM5, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c4);
        jit_movss_load(b, XMM6, RAX, 0);
        jit_addss(b, XMM5, XMM6);
        jit_mulss(b, XMM5, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_cos_c2);
        jit_movss_load(b, XMM6, RAX, 0);
        jit_addss(b, XMM5, XMM6);
        jit_mulss(b, XMM5, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
        jit_movss_load(b, XMM6, RAX, 0);
        jit_addss(b, XMM5, XMM6);   /* XMM5 = cos */

        /* sin: t*(1 + t^2*(c3 + t^2*(c5 + t^2*c7))) */
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c7);
        jit_movss_load(b, XMM6, RAX, 0);
        jit_mulss(b, XMM6, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c5);
        jit_movss_load(b, XMM1, RAX, 0);
        jit_addss(b, XMM6, XMM1);
        jit_mulss(b, XMM6, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_sin_c3);
        jit_movss_load(b, XMM1, RAX, 0);
        jit_addss(b, XMM6, XMM1);
        jit_mulss(b, XMM6, XMM4);
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
        jit_movss_load(b, XMM1, RAX, 0);
        jit_addss(b, XMM6, XMM1);
        jit_mulss(b, XMM6, XMM0);   /* XMM6 = sin */

        /* Rotate pair */
        jit_movss_load(b, XMM0, R12, 0);  /* v0 */
        jit_movss_load(b, XMM1, R12, 4);  /* v1 */
        /* v0' = v0*cos - v1*sin */
        jit_movaps_reg(b, XMM2, XMM0);
        jit_mulss(b, XMM2, XMM5);
        jit_movaps_reg(b, XMM3, XMM1);
        jit_mulss(b, XMM3, XMM6);
        jit_subss(b, XMM2, XMM3);
        jit_movss_store(b, R12, 0, XMM2);
        /* v1' = v0*sin + v1*cos */
        jit_mulss(b, XMM0, XMM6);
        jit_mulss(b, XMM1, XMM5);
        jit_addss(b, XMM0, XMM1);
        jit_movss_store(b, R12, 4, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_rope_fn fn = (jit_rope_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_ROPE, head_dim, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Element-wise Multiply — dst[i] *= src[i]
 *
 * void vmul(float *dst, const float *src, int n)
 * Args: dst=rdi, src=rsi (n compile-time)
 * =============================================================================*/

jit_ewise_fn jit_compile_vmul_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_VMUL, n);
    if (cached) return (jit_ewise_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);
    jit_mov_reg_reg(b, R13, RSI);
    jit_xor_reg_reg(b, RBX, RBX);

    int loop_top = b->len;
    /* 2x unrolled */
    jit_movups_load(b, XMM0, R12, 0);
    jit_movups_load(b, XMM1, R13, 0);
    jit_mulps(b, XMM0, XMM1);
    jit_movups_store(b, R12, 0, XMM0);
    if (vecs >= 2) {
        jit_movups_load(b, XMM2, R12, 16);
        jit_movups_load(b, XMM3, R13, 16);
        jit_mulps(b, XMM2, XMM3);
        jit_movups_store(b, R12, 16, XMM2);
    }

    int step = vecs >= 2 ? 32 : 16;
    int inc  = vecs >= 2 ? 2 : 1;
    jit_add_reg_imm32(b, R12, step);
    jit_add_reg_imm32(b, R13, step);
    jit_add_reg_imm32(b, RBX, inc);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    /* Handle final odd vector if unrolled */
    if (vecs >= 2 && (vecs % 2)) {
        jit_movups_load(b, XMM0, R12, 0);
        jit_movups_load(b, XMM1, R13, 0);
        jit_mulps(b, XMM0, XMM1);
        jit_movups_store(b, R12, 0, XMM0);
        jit_add_reg_imm32(b, R12, 16);
        jit_add_reg_imm32(b, R13, 16);
    }

    /* Scalar tail */
    for (int t = 0; t < (n % 4); t++) {
        jit_movss_load(b, XMM0, R12, t * 4);
        jit_movss_load(b, XMM1, R13, t * 4);
        jit_mulss(b, XMM0, XMM1);
        jit_movss_store(b, R12, t * 4, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_ewise_fn fn = (jit_ewise_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_VMUL, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Element-wise Add — dst[i] += src[i]
 *
 * void vadd(float *dst, const float *src, int n)
 * Args: dst=rdi, src=rsi (n compile-time)
 * =============================================================================*/

jit_ewise_fn jit_compile_vadd_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_VADD, n);
    if (cached) return (jit_ewise_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);
    jit_mov_reg_reg(b, R13, RSI);
    jit_xor_reg_reg(b, RBX, RBX);

    int loop_top = b->len;
    jit_movups_load(b, XMM0, R12, 0);
    jit_movups_load(b, XMM1, R13, 0);
    jit_addps(b, XMM0, XMM1);
    jit_movups_store(b, R12, 0, XMM0);
    if (vecs >= 2) {
        jit_movups_load(b, XMM2, R12, 16);
        jit_movups_load(b, XMM3, R13, 16);
        jit_addps(b, XMM2, XMM3);
        jit_movups_store(b, R12, 16, XMM2);
    }

    int step = vecs >= 2 ? 32 : 16;
    int inc  = vecs >= 2 ? 2 : 1;
    jit_add_reg_imm32(b, R12, step);
    jit_add_reg_imm32(b, R13, step);
    jit_add_reg_imm32(b, RBX, inc);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    if (vecs >= 2 && (vecs % 2)) {
        jit_movups_load(b, XMM0, R12, 0);
        jit_movups_load(b, XMM1, R13, 0);
        jit_addps(b, XMM0, XMM1);
        jit_movups_store(b, R12, 0, XMM0);
    }

    for (int t = 0; t < (n % 4); t++) {
        jit_movss_load(b, XMM0, R12, t * 4);
        jit_movss_load(b, XMM1, R13, t * 4);
        jit_addss(b, XMM0, XMM1);
        jit_movss_store(b, R12, t * 4, XMM0);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_ewise_fn fn = (jit_ewise_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_VADD, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Dot Product — return sum(a[i]*b[i])
 *
 * float dot(const float *a, const float *b, int n)
 * Args: a=rdi, b=rsi (n compile-time)
 * Returns: float in XMM0
 * =============================================================================*/

jit_dot_fn jit_compile_dot_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_DOT, n);
    if (cached) return (jit_dot_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);
    jit_mov_reg_reg(b, R13, RSI);

    /* Dual accumulators for ILP */
    jit_xorps(b, XMM6, XMM6);
    jit_xorps(b, XMM7, XMM7);
    jit_xor_reg_reg(b, RBX, RBX);

    int loop_top = b->len;
    jit_movups_load(b, XMM0, R12, 0);
    jit_movups_load(b, XMM1, R13, 0);
    jit_mulps(b, XMM0, XMM1);
    jit_addps(b, XMM6, XMM0);
    if (vecs >= 2) {
        jit_movups_load(b, XMM2, R12, 16);
        jit_movups_load(b, XMM3, R13, 16);
        jit_mulps(b, XMM2, XMM3);
        jit_addps(b, XMM7, XMM2);
    }

    int step = vecs >= 2 ? 32 : 16;
    int inc  = vecs >= 2 ? 2 : 1;
    jit_add_reg_imm32(b, R12, step);
    jit_add_reg_imm32(b, R13, step);
    jit_add_reg_imm32(b, RBX, inc);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    /* Handle odd vector */
    if (vecs >= 2 && (vecs % 2)) {
        jit_movups_load(b, XMM0, R12, 0);
        jit_movups_load(b, XMM1, R13, 0);
        jit_mulps(b, XMM0, XMM1);
        jit_addps(b, XMM6, XMM0);
    }

    /* Combine dual accumulators */
    jit_addps(b, XMM6, XMM7);

    /* Horizontal sum */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0x4E);
    jit_addps(b, XMM6, XMM0);
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0xB1);
    jit_addss(b, XMM6, XMM0);

    /* Scalar tail */
    for (int t = 0; t < (n % 4); t++) {
        jit_movss_load(b, XMM0, R12, t * 4);
        jit_movss_load(b, XMM1, R13, t * 4);
        jit_mulss(b, XMM0, XMM1);
        jit_addss(b, XMM6, XMM0);
    }

    /* Return in XMM0 */
    jit_movaps_reg(b, XMM0, XMM6);

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_dot_fn fn = (jit_dot_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_DOT, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Scaled Add (AXPY) — dst[i] += scale * src[i]
 *
 * void axpy(float *dst, float scale, const float *src, int n)
 * Args: dst=rdi, scale=xmm0, src=rsi (n compile-time)
 * =============================================================================*/

jit_axpy_fn jit_compile_axpy_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_AXPY, n);
    if (cached) return (jit_axpy_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);    /* dst */
    jit_mov_reg_reg(b, R13, RSI);    /* src */
    /* scale is in XMM0 (float arg via SSE), broadcast it */
    jit_shufps(b, XMM0, XMM0, 0x00);
    jit_movaps_reg(b, XMM7, XMM0);  /* XMM7 = {scale, scale, scale, scale} */

    jit_xor_reg_reg(b, RBX, RBX);
    int loop_top = b->len;

    jit_movups_load(b, XMM0, R13, 0);
    jit_mulps(b, XMM0, XMM7);
    jit_movups_load(b, XMM1, R12, 0);
    jit_addps(b, XMM1, XMM0);
    jit_movups_store(b, R12, 0, XMM1);

    jit_add_reg_imm32(b, R12, 16);
    jit_add_reg_imm32(b, R13, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    /* Scalar tail */
    for (int t = 0; t < (n % 4); t++) {
        jit_movss_load(b, XMM0, R13, t * 4);
        jit_mulss(b, XMM0, XMM7);
        jit_movss_load(b, XMM1, R12, t * 4);
        jit_addss(b, XMM1, XMM0);
        jit_movss_store(b, R12, t * 4, XMM1);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_axpy_fn fn = (jit_axpy_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_AXPY, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: Fused SiLU*Multiply — gate[i] = SiLU(gate[i]) * up[i]
 *
 * Single pass: eliminates one array traversal vs separate SiLU + vmul.
 *
 * void fused_silu_mul(float *gate, const float *up, int n)
 * Args: gate=rdi, up=rsi (n compile-time)
 * =============================================================================*/

jit_fused_silu_mul_fn jit_compile_fused_silu_mul_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_FUSED_SILU_MUL, n);
    if (cached) return (jit_fused_silu_mul_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);    /* gate */
    jit_mov_reg_reg(b, R13, RSI);    /* up */

    /* Preload constant 1.0 */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movaps_load(b, XMM7, RAX, 0);

    jit_xor_reg_reg(b, RBX, RBX);
    int loop_top = b->len;

    /* Load gate[i..i+3] */
    jit_movups_load(b, XMM0, R12, 0);

    /* Compute exp(-gate) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_neg_zero);
    jit_movaps_load(b, XMM6, RAX, 0);
    jit_xorps(b, XMM6, XMM0);       /* -gate */
    jit_emit_exp4(b, XMM6, XMM6, XMM1, XMM2, XMM3);

    /* sigmoid = 1 / (1 + exp(-gate)) */
    jit_addps(b, XMM6, XMM7);       /* 1 + exp(-gate) */
    jit_movaps_reg(b, XMM4, XMM7);
    jit_divps(b, XMM4, XMM6);       /* sigmoid */

    /* SiLU = gate * sigmoid */
    jit_mulps(b, XMM0, XMM4);

    /* Fuse: *= up[i..i+3] */
    jit_movups_load(b, XMM1, R13, 0);
    jit_mulps(b, XMM0, XMM1);

    /* Store */
    jit_movups_store(b, R12, 0, XMM0);

    jit_add_reg_imm32(b, R12, 16);
    jit_add_reg_imm32(b, R13, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_fused_silu_mul_fn fn = (jit_fused_silu_mul_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_FUSED_SILU_MUL, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: GELU — x[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 *
 * Uses the fast tanh approximation (Padé):
 *   tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)  for moderate |x|
 * And for larger |x|, clamp to ±1.
 *
 * void gelu(float *x, int n)
 * =============================================================================*/

jit_silu_fn jit_compile_gelu_kernel(int n)
{
    void *cached = llm_jit_cache_lookup(JIT_K_GELU, n);
    if (cached) return (jit_silu_fn)cached;

    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);
    jit_mov_reg_reg(b, R12, RDI);

    jit_xor_reg_reg(b, RBX, RBX);
    int loop_top = b->len;

    /* Load x */
    jit_movups_load(b, XMM0, R12, 0);

    /* inner = sqrt(2/pi) * (x + 0.044715 * x^3)
     *       = sqrt(2/pi) * x * (1 + 0.044715 * x^2) */
    jit_movaps_reg(b, XMM1, XMM0);
    jit_mulps(b, XMM1, XMM1);       /* x^2 */

    /* Load 0.044715 via gelu_b constant (adjusted from 0.0356774 to 0.044715) */
    {
        static const float __attribute__((aligned(16)))
            c_044715[4] = {0.044715f, 0.044715f, 0.044715f, 0.044715f};
        jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)c_044715);
        jit_movaps_load(b, XMM2, RAX, 0);
    }
    jit_mulps(b, XMM2, XMM1);       /* 0.044715 * x^2 */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movaps_load(b, XMM3, RAX, 0);
    jit_addps(b, XMM2, XMM3);       /* 1 + 0.044715*x^2 */
    jit_mulps(b, XMM2, XMM0);       /* x * (1 + 0.044715*x^2) */

    /* * sqrt(2/pi) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_gelu_a);
    jit_movaps_load(b, XMM3, RAX, 0);
    jit_mulps(b, XMM2, XMM3);       /* inner = sqrt(2/pi) * x * (1 + ...) */

    /* tanh(inner) using exp: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) */
    jit_movaps_reg(b, XMM4, XMM2);
    jit_addps(b, XMM4, XMM4);       /* 2 * inner */
    jit_emit_exp4(b, XMM4, XMM4, XMM5, XMM6, XMM7);  /* exp(2*inner) */

    /* Load 1.0 again (XMM7 was clobbered by exp) */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_one);
    jit_movaps_load(b, XMM3, RAX, 0);

    jit_movaps_reg(b, XMM5, XMM4);
    jit_subps(b, XMM5, XMM3);       /* exp(2x) - 1 */
    jit_addps(b, XMM4, XMM3);       /* exp(2x) + 1 */
    jit_divps(b, XMM5, XMM4);       /* tanh = (e2x-1)/(e2x+1) */

    /* gelu = 0.5 * x * (1 + tanh) */
    jit_addps(b, XMM5, XMM3);       /* 1 + tanh */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_half);
    jit_movaps_load(b, XMM3, RAX, 0);
    jit_mulps(b, XMM5, XMM3);       /* 0.5 * (1 + tanh) */
    jit_mulps(b, XMM0, XMM5);       /* x * 0.5 * (1 + tanh) */

    jit_movups_store(b, R12, 0, XMM0);

    jit_add_reg_imm32(b, R12, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, loop_top);

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_silu_fn fn = (jit_silu_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_GELU, n, fn);
    return fn;
}

/* =============================================================================
 * JIT Kernel: LayerNorm — out[i] = (x[i] - mean) * rsqrt(var + eps) * w[i] + b[i]
 *
 * void layernorm(float *out, const float *x, const float *w, const float *b, int dim)
 * Args: out=rdi, x=rsi, w=rdx, b=rcx (dim compile-time)
 * =============================================================================*/

jit_layernorm_fn jit_compile_layernorm_kernel(int dim)
{
    void *cached = llm_jit_cache_lookup(JIT_K_LAYERNORM, dim);
    if (cached) return (jit_layernorm_fn)cached;

    int vecs = dim / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);
    /* Save all 4 pointer args */
    jit_mov_reg_reg(b, R12, RDI);    /* out */
    jit_mov_reg_reg(b, R13, RSI);    /* x */
    jit_mov_reg_reg(b, R14, RDX);    /* w */
    jit_mov_reg_reg(b, R15, RCX);    /* b (bias) */

    /* === Pass 1: Compute mean === */
    jit_xorps(b, XMM6, XMM6);       /* sum = 0 */
    jit_mov_reg_reg(b, RBX, R13);
    jit_xor_reg_reg(b, RCX, RCX);

    int mean_top = b->len;
    jit_movups_load(b, XMM0, RBX, 0);
    jit_addps(b, XMM6, XMM0);
    jit_add_reg_imm32(b, RBX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, vecs);
    jit_jl_back(b, mean_top);

    /* Horizontal sum → mean */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0x4E);
    jit_addps(b, XMM6, XMM0);
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0xB1);
    jit_addss(b, XMM6, XMM0);
    /* Divide by dim */
    {
        union { float f; uint32_t u; } df = { .f = (float)dim };
        jit_mov_reg_imm64(b, RAX, df.u);
        jit_movd_to_xmm(b, XMM1, RAX);
    }
    jit_divss(b, XMM6, XMM1);
    jit_shufps(b, XMM6, XMM6, 0x00);  /* XMM6 = {mean, mean, mean, mean} */
    /* Save mean on stack */
    jit_sub_reg_imm32(b, RSP, 16);
    jit_movaps_store(b, RSP, 0, XMM6);

    /* === Pass 2: Compute variance = mean((x - mean)^2) === */
    jit_xorps(b, XMM7, XMM7);       /* var_sum = 0 */
    jit_mov_reg_reg(b, RBX, R13);
    jit_xor_reg_reg(b, RCX, RCX);

    int var_top = b->len;
    jit_movups_load(b, XMM0, RBX, 0);
    jit_subps(b, XMM0, XMM6);       /* x - mean */
    jit_mulps(b, XMM0, XMM0);       /* (x - mean)^2 */
    jit_addps(b, XMM7, XMM0);
    jit_add_reg_imm32(b, RBX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, vecs);
    jit_jl_back(b, var_top);

    /* Horizontal sum → variance */
    jit_movaps_reg(b, XMM0, XMM7);
    jit_shufps(b, XMM0, XMM7, 0x4E);
    jit_addps(b, XMM7, XMM0);
    jit_movaps_reg(b, XMM0, XMM7);
    jit_shufps(b, XMM0, XMM7, 0xB1);
    jit_addss(b, XMM7, XMM0);
    {
        union { float f; uint32_t u; } df = { .f = (float)dim };
        jit_mov_reg_imm64(b, RAX, df.u);
        jit_movd_to_xmm(b, XMM1, RAX);
    }
    jit_divss(b, XMM7, XMM1);       /* variance */

    /* inv_std = rsqrt(var + eps) with Newton refinement */
    jit_mov_reg_imm64(b, RAX, (uint64_t)(uintptr_t)jit_const_eps);
    jit_movss_load(b, XMM1, RAX, 0);
    jit_addss(b, XMM7, XMM1);

    jit_rsqrtps(b, XMM5, XMM7);
    /* Newton: y = y * (1.5 - 0.5*x*y*y) */
    {
        union { float f; uint32_t u; } hf = { .f = 0.5f };
        jit_mov_reg_imm64(b, RAX, hf.u);
        jit_movd_to_xmm(b, XMM1, RAX);
    }
    jit_movaps_reg(b, XMM2, XMM7);
    jit_mulss(b, XMM2, XMM1);       /* 0.5 * (var+eps) */
    jit_movaps_reg(b, XMM3, XMM5);
    jit_mulss(b, XMM3, XMM5);       /* y^2 */
    jit_mulss(b, XMM3, XMM2);       /* 0.5*(var+eps)*y^2 */
    {
        union { float f; uint32_t u; } ohf = { .f = 1.5f };
        jit_mov_reg_imm64(b, RAX, ohf.u);
        jit_movd_to_xmm(b, XMM1, RAX);
    }
    jit_subss(b, XMM1, XMM3);
    jit_mulss(b, XMM5, XMM1);       /* refined rsqrt */

    /* Broadcast inv_std */
    jit_shufps(b, XMM5, XMM5, 0x00);  /* XMM5 = {inv_std, ...} */

    /* Reload mean from stack */
    jit_movaps_load(b, XMM6, RSP, 0);
    jit_add_reg_imm32(b, RSP, 16);

    /* === Pass 3: out[i] = (x[i] - mean) * inv_std * w[i] + b[i] === */
    jit_xor_reg_reg(b, RBX, RBX);
    jit_mov_reg_reg(b, RAX, R13);    /* reset x pointer */

    int out_top = b->len;
    jit_movups_load(b, XMM0, RAX, 0);
    jit_subps(b, XMM0, XMM6);       /* x - mean */
    jit_mulps(b, XMM0, XMM5);       /* * inv_std */
    jit_movups_load(b, XMM1, R14, 0);
    jit_mulps(b, XMM0, XMM1);       /* * w */
    jit_movups_load(b, XMM1, R15, 0);
    jit_addps(b, XMM0, XMM1);       /* + b */
    jit_movups_store(b, R12, 0, XMM0);

    jit_add_reg_imm32(b, R12, 16);
    jit_add_reg_imm32(b, RAX, 16);
    jit_add_reg_imm32(b, R14, 16);
    jit_add_reg_imm32(b, R15, 16);
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, vecs);
    jit_jl_back(b, out_top);

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_layernorm_fn fn = (jit_layernorm_fn)(void *)b->code;
    llm_jit_cache_store(JIT_K_LAYERNORM, dim, fn);
    return fn;
}

/* =============================================================================
 * Tokenizer/Detokenizer helper kernels
 * =============================================================================*/

static int jit_mem_equal_impl(const char *a, const char *b, int len)
{
    if (!a || !b || len < 0) return 0;
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t wa, wb;
        __builtin_memcpy(&wa, a + i, 8);
        __builtin_memcpy(&wb, b + i, 8);
        if (wa != wb) return 0;
    }
    for (; i < len; i++) {
        if ((uint8_t)a[i] != (uint8_t)b[i]) return 0;
    }
    return 1;
}

static int jit_bpe_escape_scan_impl(const char *s, int len)
{
    if (!s || len <= 0) return 0;
    for (int i = 0; i < len; i++) {
        uint8_t c = (uint8_t)s[i];
        if (c == 0xC4 || c == 0xC5) return 1;
    }
    return 0;
}

static int jit_find_byte_impl(const char *s, int len, char needle)
{
    if (!s || len <= 0) return -1;
    for (int i = 0; i < len; i++) {
        if (s[i] == needle) return i;
    }
    return -1;
}

jit_mem_equal_fn jit_compile_mem_equal_kernel(void)
{
    return jit_mem_equal_impl;
}

jit_bpe_escape_scan_fn jit_compile_bpe_escape_scan_kernel(void)
{
    return jit_bpe_escape_scan_impl;
}

jit_find_byte_fn jit_compile_find_byte_kernel(void)
{
    return jit_find_byte_impl;
}

/* =============================================================================
 * JIT Helper: BPE Merge Candidate Comparison
 *
 * Finds the first index where merge-score array A beats array B.
 * Used by the BPE tokenizer to select the highest-priority merge pair
 * without scanning the full (potentially long) merge table on every step.
 *
 * int merge_cmp(const int32_t *scores_a, const int32_t *scores_b, int n)
 * Returns first index i where a[i] > b[i], or -1 if b always >= a.
 * Early-exit pattern: cache-friendly for short merge windows.
 * =============================================================================*/

static int jit_merge_cmp_impl(const int32_t *a, const int32_t *b, int n)
{
    if (!a || !b || n <= 0) return -1;
    /* 4-wide SIMD-style unroll for the common case (n is typically 8-64) */
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        if (a[i]     > b[i])     return i;
        if (a[i + 1] > b[i + 1]) return i + 1;
        if (a[i + 2] > b[i + 2]) return i + 2;
        if (a[i + 3] > b[i + 3]) return i + 3;
    }
    for (; i < n; i++) {
        if (a[i] > b[i]) return i;
    }
    return -1;
}

jit_merge_cmp_fn jit_compile_merge_cmp_kernel(void)
{
    return jit_merge_cmp_impl;
}

/* =============================================================================
 * JIT Helper: Token Append with Bounds Check
 *
 * Appends one token ID to a context buffer, returning the new length.
 * Provides a single call-site for agent context updates so the JIT
 * append/splice hot-path can later be replaced with a generated stub.
 *
 * int token_append(int32_t *ids, int cur_len, int max_len, int32_t token_id)
 * Returns new length, or -1 if already at capacity.
 * =============================================================================*/

static int jit_token_append_impl(int32_t *ids, int cur_len, int max_len,
                                  int32_t token_id)
{
    if (!ids || cur_len < 0 || cur_len >= max_len) return -1;
    ids[cur_len] = token_id;
    return cur_len + 1;
}

jit_token_append_fn jit_compile_token_append_kernel(void)
{
    return jit_token_append_impl;
}

#else /* __aarch64__ */

/* ARM64 stubs */
#include "runtime/jit/x86_jit.h"

jit_rmsnorm_fn jit_compile_rmsnorm_kernel(int d) { (void)d; return NULL; }
jit_softmax_fn jit_compile_softmax_kernel(int n) { (void)n; return NULL; }
jit_rope_fn jit_compile_rope_kernel(int h) { (void)h; return NULL; }
jit_dot_fn jit_compile_dot_kernel(int n) { (void)n; return NULL; }
jit_ewise_fn jit_compile_vmul_kernel(int n) { (void)n; return NULL; }
jit_ewise_fn jit_compile_vadd_kernel(int n) { (void)n; return NULL; }
jit_axpy_fn jit_compile_axpy_kernel(int n) { (void)n; return NULL; }
jit_fused_silu_mul_fn jit_compile_fused_silu_mul_kernel(int n) { (void)n; return NULL; }
jit_silu_fn jit_compile_gelu_kernel(int n) { (void)n; return NULL; }
jit_layernorm_fn jit_compile_layernorm_kernel(int d) { (void)d; return NULL; }
jit_mem_equal_fn jit_compile_mem_equal_kernel(void) { return NULL; }
jit_bpe_escape_scan_fn jit_compile_bpe_escape_scan_kernel(void) { return NULL; }
jit_find_byte_fn jit_compile_find_byte_kernel(void) { return NULL; }
jit_merge_cmp_fn jit_compile_merge_cmp_kernel(void) { return NULL; }
jit_token_append_fn jit_compile_token_append_kernel(void) { return NULL; }

#endif /* __aarch64__ */
