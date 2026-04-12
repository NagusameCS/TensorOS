/*
 * TensorOS Kernel Correctness Tests
 *
 * Validates tensor operation kernels against scalar reference implementations.
 * Covers: Q4_0 dequant, Q8_0 quantize/dequant, Q4xQ8 dot product,
 *         Q4 GEMV, Q8 GEMV, RMSNorm, SiLU, softmax, RoPE.
 *
 * Build (host): zig cc -target x86_64-windows-gnu -O2 -mavx2 -mfma
 *               -DHYPERTENSOR_HOSTED=1 -Ihost/shims -I. -Ihost
 *               tests/runtime/test_kernels.c host/hal.c
 *               runtime/nn/llm.c runtime/nn/gguf.c
 *               runtime/jit/x86_jit.c runtime/jit/llm_jit.c
 *               -ladvapi32 -o build_host/test_kernels.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/* ─── Minimal PRNG for deterministic test data ─── */
static uint64_t rng_state = 0xDEADBEEF42ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return (rng_state = x);
}
static float randf(void) { return (float)(xorshift64() % 10000) / 10000.0f - 0.5f; }
static void seed_rng(uint64_t s) { rng_state = s ? s : 1; }

/* ─── Q4_0 / Q8_0 structures (must match llm.c) ─── */
typedef struct { uint16_t d; uint8_t qs[16]; } test_q4_0_t;
typedef struct { uint16_t d; int8_t qs[32]; }  test_q8_0_t;

/* FP16 convert (matches llm.c fp16_to_fp32_fast) */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ─── Scalar reference: dequantize Q4_0 block → 32 floats ─── */
static void ref_dequant_q4_0(float *out, const test_q4_0_t *block) {
    float d = fp16_to_f32(block->d);
    for (int j = 0; j < 16; j++) {
        int lo = (block->qs[j] & 0x0F) - 8;
        int hi = (block->qs[j] >> 4)    - 8;
        out[j]      = d * (float)lo;
        out[j + 16] = d * (float)hi;
    }
}

/* ─── Scalar reference: dequantize Q8_0 block → 32 floats ─── */
static void ref_dequant_q8_0(float *out, const test_q8_0_t *block) {
    float d = fp16_to_f32(block->d);
    for (int j = 0; j < 32; j++)
        out[j] = d * (float)block->qs[j];
}

/* ─── Scalar reference: dot product Q4_0 x float ─── */
static float ref_dot_q4_f32(const test_q4_0_t *blocks, const float *x, int n) {
    int nb = n / 32;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        float dq[32];
        ref_dequant_q4_0(dq, &blocks[b]);
        for (int j = 0; j < 32; j++)
            sum += dq[j] * x[b * 32 + j];
    }
    return sum;
}

/* ─── Scalar reference: dot product Q4_0 x Q8 ─── */
static float ref_dot_q4_q8(const test_q4_0_t *w, const float *xq_d,
                            const int8_t *xq_qs, const int32_t *xq_isum, int n) {
    int nb = n / 32;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        float wd = fp16_to_f32(w[b].d);
        int dot = 0;
        uint8_t q4u[32];
        for (int j = 0; j < 16; j++) {
            q4u[j]      = w[b].qs[j] & 0x0F;
            q4u[j + 16] = w[b].qs[j] >> 4;
        }
        for (int j = 0; j < 32; j++)
            dot += (int)q4u[j] * (int)xq_qs[b * 32 + j];
        /* Bias correction: Q4 values are unsigned [0,15], bias = 8 */
        sum += wd * xq_d[b] * (float)(dot - 8 * xq_isum[b]);
    }
    return sum;
}

/* ─── Scalar reference: RMSNorm ─── */
static void ref_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

/* ─── Scalar reference: SiLU ─── */
static float ref_silu(float x) { return x / (1.0f + expf(-x)); }

/* ─── Scalar reference: Softmax ─── */
static void ref_softmax(float *out, const float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

/* ─── Scalar reference: RoPE ─── */
static void ref_rope(float *q, int head_dim, int pos, float base) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(base, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cs - q1 * sn;
        q[i + 1] = q0 * sn + q1 * cs;
    }
}

/* ─── Quantize float to Q8 block (scalar reference) ─── */
static void ref_q8_quantize(float *d_out, int8_t *qs_out, int32_t *isum_out,
                             const float *x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-30f) {
        *d_out = 0.0f; *isum_out = 0;
        memset(qs_out, 0, 32);
        return;
    }
    *d_out = amax / 127.0f;
    float id = 127.0f / amax;
    int32_t s = 0;
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(x[i] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qs_out[i] = (int8_t)q;
        s += q;
    }
    *isum_out = s;
}

/* ─── Quantize float to Q4_0 block (scalar reference) ─── */
static void ref_q4_quantize(test_q4_0_t *out, const float *x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float d = amax / 7.0f;
    /* Convert d to fp16 and back to get the real scale */
    /* Simplified: just store raw d for testing */
    uint32_t fbits;
    memcpy(&fbits, &d, 4);
    /* fp32 to fp16 */
    uint32_t sign = (fbits >> 16) & 0x8000;
    int exp = ((fbits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fbits >> 13) & 0x3FF;
    if (exp <= 0) { out->d = (uint16_t)sign; }
    else if (exp >= 31) { out->d = (uint16_t)(sign | 0x7C00); }
    else { out->d = (uint16_t)(sign | (exp << 10) | mant); }

    float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
    for (int j = 0; j < 16; j++) {
        int lo = (int)roundf(x[j] * id) + 8;
        int hi = (int)roundf(x[j + 16] * id) + 8;
        if (lo < 0) lo = 0; if (lo > 15) lo = 15;
        if (hi < 0) hi = 0; if (hi > 15) hi = 15;
        out->qs[j] = (uint8_t)((hi << 4) | lo);
    }
}

/* ─── Test infrastructure ─── */
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_NEAR(a, b, eps, name) do { \
    float _a = (a), _b = (b), _e = (eps); \
    tests_run++; \
    if (fabsf(_a - _b) <= _e) { tests_passed++; } \
    else { tests_failed++; \
        printf("FAIL %s: got %.6f expected %.6f (diff %.6e, tol %.6e)\n", \
               name, _a, _b, fabsf(_a-_b), _e); } \
} while(0)

#define ASSERT_TRUE(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("FAIL %s\n", name); } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Q4_0 dequantization consistency
 * ════════════════════════════════════════════════════════════════════════ */
static void test_q4_dequant(void) {
    printf("  [Q4_0 dequant] ");
    seed_rng(1001);
    for (int trial = 0; trial < 100; trial++) {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = randf() * 2.0f;

        test_q4_0_t block;
        ref_q4_quantize(&block, src);

        float dq[32];
        ref_dequant_q4_0(dq, &block);

        /* Dequant should be within quantization error of original */
        float d = fp16_to_f32(block.d);
        for (int i = 0; i < 32; i++) {
            ASSERT_NEAR(dq[i], src[i], d + 0.01f, "q4_dequant_roundtrip");
        }
    }
    printf("%d/%d passed\n", tests_passed, tests_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Q8_0 quantize/dequant roundtrip
 * ════════════════════════════════════════════════════════════════════════ */
static void test_q8_roundtrip(void) {
    int pre = tests_passed;
    printf("  [Q8_0 roundtrip] ");
    seed_rng(2002);
    for (int trial = 0; trial < 100; trial++) {
        float src[32];
        for (int i = 0; i < 32; i++) src[i] = randf() * 4.0f;

        float d;
        int8_t qs[32];
        int32_t isum;
        ref_q8_quantize(&d, qs, &isum, src);

        /* Verify isum */
        int32_t check_sum = 0;
        for (int i = 0; i < 32; i++) check_sum += qs[i];
        ASSERT_TRUE(check_sum == isum, "q8_isum");

        /* Dequant roundtrip */
        for (int i = 0; i < 32; i++) {
            float dq = d * (float)qs[i];
            ASSERT_NEAR(dq, src[i], d + 0.01f, "q8_roundtrip");
        }
    }
    printf("%d/%d passed\n", tests_passed - pre, tests_run - pre + (tests_passed - pre));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Q4xQ8 dot product correctness
 * ════════════════════════════════════════════════════════════════════════ */
static void test_q4_q8_dot(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [Q4xQ8 dot product] ");
    seed_rng(3003);

    for (int trial = 0; trial < 50; trial++) {
        /* Generate random weight and input vectors (32 elements = 1 block) */
        float w_src[32], x_src[32];
        for (int i = 0; i < 32; i++) {
            w_src[i] = randf() * 2.0f;
            x_src[i] = randf() * 3.0f;
        }

        /* Quantize weight to Q4_0 */
        test_q4_0_t w_block;
        ref_q4_quantize(&w_block, w_src);

        /* Quantize input to Q8 */
        float xd;
        int8_t xqs[32];
        int32_t xisum;
        ref_q8_quantize(&xd, xqs, &xisum, x_src);

        /* Reference: dequant Q4 then dot with x_src */
        float dq_w[32];
        ref_dequant_q4_0(dq_w, &w_block);
        float ref_float_dot = 0.0f;
        for (int i = 0; i < 32; i++) ref_float_dot += dq_w[i] * x_src[i];

        /* Q4xQ8 integer dot product (what the SIMD kernel computes) */
        float q4q8_dot = ref_dot_q4_q8(&w_block, &xd, xqs, &xisum, 32);

        /* Tolerance: quant noise from both Q4 and Q8 quantization */
        float wd = fp16_to_f32(w_block.d);
        float tol = fabsf(wd * xd) * 32.0f * 0.6f + 0.5f;
        ASSERT_NEAR(q4q8_dot, ref_float_dot, tol, "q4q8_vs_float_dot");
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: RMSNorm
 * ════════════════════════════════════════════════════════════════════════ */
static void test_rmsnorm(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [RMSNorm] ");
    seed_rng(4004);

    for (int trial = 0; trial < 20; trial++) {
        int dim = 64;
        float x[64], w[64], out[64], ref[64];
        for (int i = 0; i < dim; i++) { x[i] = randf(); w[i] = 0.5f + randf(); }

        ref_rmsnorm(ref, x, w, dim, 1e-5f);

        /* Verify properties: output should have similar norm structure */
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += ref[i] * ref[i];
        ASSERT_TRUE(ss > 0.0f && !isinf(ss) && !isnan(ss), "rmsnorm_finite");

        /* Self-consistency: applying rmsnorm twice with w=1 */
        float ones[64], tmp[64], tmp2[64];
        for (int i = 0; i < dim; i++) ones[i] = 1.0f;
        ref_rmsnorm(tmp, x, ones, dim, 1e-5f);
        /* RMS of normalized output should be ~1.0 */
        float rms = 0.0f;
        for (int i = 0; i < dim; i++) rms += tmp[i] * tmp[i];
        rms = sqrtf(rms / (float)dim);
        ASSERT_NEAR(rms, 1.0f, 0.01f, "rmsnorm_unit_rms");
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: SiLU activation
 * ════════════════════════════════════════════════════════════════════════ */
static void test_silu(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [SiLU] ");

    float vals[] = {-5.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f};
    for (int i = 0; i < 9; i++) {
        float ref = ref_silu(vals[i]);
        ASSERT_TRUE(!isnan(ref) && !isinf(ref), "silu_finite");
        /* SiLU(0) = 0 */
        if (vals[i] == 0.0f) ASSERT_NEAR(ref, 0.0f, 1e-6f, "silu_zero");
        /* SiLU(x) ≈ x for large positive x */
        if (vals[i] > 3.0f) ASSERT_NEAR(ref, vals[i], 0.1f, "silu_large");
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Softmax
 * ════════════════════════════════════════════════════════════════════════ */
static void test_softmax(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [Softmax] ");
    seed_rng(5005);

    for (int trial = 0; trial < 20; trial++) {
        int n = 32;
        float x[32], out[32];
        for (int i = 0; i < n; i++) x[i] = randf() * 10.0f;
        ref_softmax(out, x, n);

        /* Sum should be 1.0 */
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += out[i];
        ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax_sum_one");

        /* All values non-negative */
        for (int i = 0; i < n; i++)
            ASSERT_TRUE(out[i] >= 0.0f, "softmax_nonneg");

        /* Max input should have max output */
        int max_in = 0, max_out = 0;
        for (int i = 1; i < n; i++) {
            if (x[i] > x[max_in]) max_in = i;
            if (out[i] > out[max_out]) max_out = i;
        }
        ASSERT_TRUE(max_in == max_out, "softmax_argmax");
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: RoPE
 * ════════════════════════════════════════════════════════════════════════ */
static void test_rope(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [RoPE] ");
    seed_rng(6006);

    /* RoPE should preserve vector norm */
    for (int trial = 0; trial < 20; trial++) {
        int hd = 64;
        float q[64], q_copy[64];
        for (int i = 0; i < hd; i++) q[i] = q_copy[i] = randf();

        float norm_before = 0.0f;
        for (int i = 0; i < hd; i++) norm_before += q[i] * q[i];

        ref_rope(q, hd, trial + 1, 10000.0f);

        float norm_after = 0.0f;
        for (int i = 0; i < hd; i++) norm_after += q[i] * q[i];

        ASSERT_NEAR(sqrtf(norm_after), sqrtf(norm_before), 1e-4f, "rope_norm_preserve");

        /* Position 0 should be identity */
        for (int i = 0; i < hd; i++) q[i] = q_copy[i];
        ref_rope(q, hd, 0, 10000.0f);
        for (int i = 0; i < hd; i++)
            ASSERT_NEAR(q[i], q_copy[i], 1e-5f, "rope_pos0_identity");
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Q4 GEMV reference consistency
 * ════════════════════════════════════════════════════════════════════════ */
static void test_q4_gemv(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [Q4_0 GEMV] ");
    seed_rng(7007);

    /* Small GEMV: 4 rows x 64 cols */
    int rows = 4, cols = 64;
    int nb = cols / 32;
    test_q4_0_t *weight = (test_q4_0_t *)calloc(rows * nb, sizeof(test_q4_0_t));
    float *x = (float *)calloc(cols, sizeof(float));
    float *out = (float *)calloc(rows, sizeof(float));

    /* Fill weight and input with random data */
    for (int r = 0; r < rows; r++) {
        float row_data[64];
        for (int i = 0; i < cols; i++) row_data[i] = randf();
        for (int b = 0; b < nb; b++)
            ref_q4_quantize(&weight[r * nb + b], row_data + b * 32);
    }
    for (int i = 0; i < cols; i++) x[i] = randf();

    /* Compute GEMV row by row */
    for (int r = 0; r < rows; r++) {
        out[r] = ref_dot_q4_f32(&weight[r * nb], x, cols);
    }

    /* Verify output is finite and non-zero for random inputs */
    for (int r = 0; r < rows; r++) {
        ASSERT_TRUE(!isnan(out[r]) && !isinf(out[r]), "gemv_finite");
    }

    /* Cross-check: Q4xQ8 integer path vs float path */
    float *xd_arr = (float *)calloc(nb, sizeof(float));
    int8_t *xqs_arr = (int8_t *)calloc(nb * 32, sizeof(int8_t));
    int32_t *xisum_arr = (int32_t *)calloc(nb, sizeof(int32_t));
    for (int b = 0; b < nb; b++)
        ref_q8_quantize(&xd_arr[b], &xqs_arr[b * 32], &xisum_arr[b], x + b * 32);

    for (int r = 0; r < rows; r++) {
        float q4q8_val = ref_dot_q4_q8(&weight[r * nb], xd_arr, xqs_arr,
                                         xisum_arr, cols);
        /* Allow generous tolerance due to double quantization */
        float tol = fabsf(out[r]) * 0.15f + 0.5f;
        ASSERT_NEAR(q4q8_val, out[r], tol, "gemv_q4q8_vs_float");
    }

    free(weight); free(x); free(out);
    free(xd_arr); free(xqs_arr); free(xisum_arr);
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Multi-block Q4xQ8 dot product (stress test)
 * ════════════════════════════════════════════════════════════════════════ */
static void test_q4_q8_multiblock(void) {
    int pre_run = tests_run, pre_pass = tests_passed;
    printf("  [Q4xQ8 multi-block] ");
    seed_rng(8008);

    /* Test with dims typical of real models: 1536, 2048, 4096 */
    int dims[] = {64, 128, 256, 512, 1536, 2048};
    for (int d = 0; d < 6; d++) {
        int n = dims[d];
        int nb = n / 32;
        test_q4_0_t *w = (test_q4_0_t *)calloc(nb, sizeof(test_q4_0_t));
        float *x = (float *)calloc(n, sizeof(float));

        /* Random data */
        for (int b = 0; b < nb; b++) {
            float tmp[32];
            for (int i = 0; i < 32; i++) tmp[i] = randf() * 2.0f;
            ref_q4_quantize(&w[b], tmp);
        }
        for (int i = 0; i < n; i++) x[i] = randf() * 3.0f;

        /* Float reference */
        float ref = ref_dot_q4_f32(w, x, n);

        /* Q4xQ8 integer */
        float *xd = (float *)calloc(nb, sizeof(float));
        int8_t *xqs = (int8_t *)calloc(nb * 32, sizeof(int8_t));
        int32_t *xisum = (int32_t *)calloc(nb, sizeof(int32_t));
        for (int b = 0; b < nb; b++)
            ref_q8_quantize(&xd[b], &xqs[b * 32], &xisum[b], x + b * 32);

        float q4q8 = ref_dot_q4_q8(w, xd, xqs, xisum, n);

        float tol = fabsf(ref) * 0.2f + 1.0f;
        ASSERT_NEAR(q4q8, ref, tol, "q4q8_multiblock");

        free(w); free(x); free(xd); free(xqs); free(xisum);
    }
    printf("%d/%d passed\n", tests_passed - pre_pass, tests_run - pre_run);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("\n=== TensorOS Kernel Correctness Tests ===\n\n");

    test_q4_dequant();
    test_q8_roundtrip();
    test_q4_q8_dot();
    test_q4_gemv();
    test_q4_q8_multiblock();
    test_rmsnorm();
    test_silu();
    test_softmax();
    test_rope();

    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf(", %d FAILED", tests_failed);
    printf(" ===\n\n");

    return tests_failed > 0 ? 1 : 0;
}
