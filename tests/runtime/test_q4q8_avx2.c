/*
 * TensorOS Q4xQ8 AVX2 Kernel Validation
 *
 * Directly tests the q4_0_q8_dot_avx2 SIMD kernel against scalar reference.
 * This is the critical correctness gate for re-enabling the integer dot path.
 *
 * Build: zig cc -target x86_64-windows-gnu -O2 -mavx2 -mfma
 *        -DHYPERTENSOR_HOSTED=1 -Ihost/shims -I. -Ihost
 *        tests/runtime/test_q4q8_avx2.c host/hal.c
 *        runtime/nn/llm.c runtime/nn/gguf.c
 *        runtime/jit/x86_jit.c runtime/jit/llm_jit.c
 *        -ladvapi32 -o build_host/test_q4q8_avx2.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <intrin.h>
#endif

/* ─── Types matching llm.c ─── */
typedef struct { uint16_t d; uint8_t qs[16]; } ggml_q4_0_t;
typedef struct { float d; int32_t isum; int8_t qs[32]; } q8_input_t;

/* ─── FP16 → FP32 ─── */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 1; while (!(mant&0x400)){mant<<=1;exp--;} mant&=0x3FF;
               bits=sign|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant<<13);
    else bits = sign | ((exp+127-15)<<23) | (mant<<13);
    float f; memcpy(&f, &bits, 4); return f;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

/* ─── PRNG ─── */
static uint64_t rng = 0xFACEFEED;
static float randf(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)(rng % 20000) / 10000.0f - 1.0f;
}

/* ─── Scalar reference: Q4xQ8 dot product ─── */
static float ref_q4q8_dot(const ggml_q4_0_t *w, const q8_input_t *xq) {
    uint8_t q4u[32];
    for (int j = 0; j < 16; j++) {
        q4u[j]      = w->qs[j] & 0x0F;
        q4u[j + 16] = w->qs[j] >> 4;
    }
    int dot = 0;
    for (int j = 0; j < 32; j++)
        dot += (int)q4u[j] * (int)xq->qs[j];
    float wd = fp16_to_fp32(w->d);
    return wd * xq->d * (float)(dot - 8 * xq->isum);
}

/* ─── Scalar reference: dequant Q4 + float dot ─── */
static float ref_q4_float_dot(const ggml_q4_0_t *w, const float *x) {
    float d = fp16_to_fp32(w->d);
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
        int lo = (w->qs[j] & 0x0F) - 8;
        int hi = (w->qs[j] >> 4) - 8;
        sum += d * (float)lo * x[j];
        sum += d * (float)hi * x[j + 16];
    }
    return sum;
}

/* ─── Quantize float to Q8 (scalar, matches llm.c) ─── */
static void q8_quantize(q8_input_t *out, const float *x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-30f) { out->d = 0; out->isum = 0; memset(out->qs, 0, 32); return; }
    out->d = amax / 127.0f;
    float id = 127.0f / amax;
    int32_t s = 0;
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(x[i] * id);
        if (q > 127) q = 127; if (q < -127) q = -127;
        out->qs[i] = (int8_t)q;
        s += q;
    }
    out->isum = s;
}

/* ─── Quantize float to Q4_0 ─── */
static void q4_quantize(ggml_q4_0_t *out, const float *x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float d = amax / 7.0f;
    out->d = fp32_to_fp16(d);
    d = fp16_to_fp32(out->d);
    float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
    for (int j = 0; j < 16; j++) {
        int lo = (int)roundf(x[j] * id) + 8;
        int hi = (int)roundf(x[j + 16] * id) + 8;
        if (lo < 0) lo = 0; if (lo > 15) lo = 15;
        if (hi < 0) hi = 0; if (hi > 15) hi = 15;
        out->qs[j] = (uint8_t)((hi << 4) | lo);
    }
}

/* ─── AVX2 SIMD implementation (copy from llm.c for standalone testing) ─── */
typedef float v8f __attribute__((vector_size(32)));
typedef int   v8i  __attribute__((vector_size(32)));
typedef short v16s __attribute__((vector_size(32)));
typedef char  v32b __attribute__((vector_size(32)));

__attribute__((target("avx2,fma")))
static float simd_q4q8_dot(const ggml_q4_0_t *w, const q8_input_t *xq) {
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

    float wd = fp16_to_fp32(w->d);
    return wd * xq->d * (float)(dot - 8 * xq->isum);
}

int main(void) {
    printf("\n=== Q4xQ8 AVX2 Kernel Validation ===\n\n");

    int pass = 0, fail = 0;

    /* Test 1: Single block, many random vectors */
    printf("  [Single block × 1000 trials]\n");
    for (int trial = 0; trial < 1000; trial++) {
        float w_src[32], x_src[32];
        for (int i = 0; i < 32; i++) { w_src[i] = randf(); x_src[i] = randf(); }

        ggml_q4_0_t w_block;
        q4_quantize(&w_block, w_src);

        q8_input_t xq;
        q8_quantize(&xq, x_src);

        float ref  = ref_q4q8_dot(&w_block, &xq);
        float simd = simd_q4q8_dot(&w_block, &xq);

        if (fabsf(ref - simd) > 1e-5f) {
            if (fail < 5)
                printf("    FAIL trial %d: ref=%.6f simd=%.6f diff=%.6e\n",
                       trial, ref, simd, fabsf(ref-simd));
            fail++;
        } else {
            pass++;
        }
    }
    printf("    %d/1000 passed\n\n", pass);

    /* Test 2: Multi-block GEMV (simulate 4096-dim row) */
    printf("  [Multi-block GEMV: 4096-dim × 100 rows]\n");
    int gemv_pass = 0, gemv_fail = 0;
    int dim = 4096;
    int nb = dim / 32;

    for (int trial = 0; trial < 100; trial++) {
        /* Make weight row and input */
        ggml_q4_0_t *w_row = (ggml_q4_0_t *)calloc(nb, sizeof(ggml_q4_0_t));
        float *x = (float *)calloc(dim, sizeof(float));
        q8_input_t *xq = (q8_input_t *)calloc(nb, sizeof(q8_input_t));

        for (int b = 0; b < nb; b++) {
            float tmp[32];
            for (int i = 0; i < 32; i++) tmp[i] = randf();
            q4_quantize(&w_row[b], tmp);
        }
        for (int i = 0; i < dim; i++) x[i] = randf();
        for (int b = 0; b < nb; b++) q8_quantize(&xq[b], x + b * 32);

        /* Accumulate over blocks: both paths */
        float ref_sum = 0.0f, simd_sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            ref_sum  += ref_q4q8_dot(&w_row[b], &xq[b]);
            simd_sum += simd_q4q8_dot(&w_row[b], &xq[b]);
        }

        /* Also compare against float dot (ground truth) */
        float float_sum = 0.0f;
        for (int b = 0; b < nb; b++)
            float_sum += ref_q4_float_dot(&w_row[b], x + b * 32);

        float tol_simd_ref = 1e-4f;
        float tol_vs_float = fabsf(float_sum) * 0.15f + 1.0f;

        if (fabsf(ref_sum - simd_sum) > tol_simd_ref) {
            if (gemv_fail < 3)
                printf("    FAIL row %d: ref=%.4f simd=%.4f diff=%.4e\n",
                       trial, ref_sum, simd_sum, fabsf(ref_sum-simd_sum));
            gemv_fail++;
        } else {
            gemv_pass++;
        }

        /* Verify Q4xQ8 is within acceptable tolerance of float path */
        if (fabsf(simd_sum - float_sum) > tol_vs_float) {
            printf("    WARNING row %d: simd=%.4f float=%.4f diff=%.4f (tol=%.4f)\n",
                   trial, simd_sum, float_sum, fabsf(simd_sum-float_sum), tol_vs_float);
        }

        free(w_row); free(x); free(xq);
    }
    printf("    %d/100 passed (SIMD vs scalar ref)\n\n", gemv_pass);

    /* Test 3: Edge cases */
    printf("  [Edge cases]\n");
    int edge_pass = 0;
    {
        /* Zero weight */
        ggml_q4_0_t zw; memset(&zw, 0, sizeof(zw));
        float x_src[32]; for (int i = 0; i < 32; i++) x_src[i] = randf();
        q8_input_t xq; q8_quantize(&xq, x_src);
        float ref = ref_q4q8_dot(&zw, &xq);
        float simd = simd_q4q8_dot(&zw, &xq);
        if (fabsf(ref - simd) < 1e-6f && fabsf(ref) < 1e-6f) edge_pass++;
        else printf("    FAIL zero_weight: ref=%.6f simd=%.6f\n", ref, simd);

        /* Zero input */
        float zero[32] = {0};
        q8_input_t xq_zero; q8_quantize(&xq_zero, zero);
        float w_src[32]; for (int i = 0; i < 32; i++) w_src[i] = randf();
        ggml_q4_0_t wb; q4_quantize(&wb, w_src);
        ref = ref_q4q8_dot(&wb, &xq_zero);
        simd = simd_q4q8_dot(&wb, &xq_zero);
        if (fabsf(ref - simd) < 1e-6f && fabsf(ref) < 1e-6f) edge_pass++;
        else printf("    FAIL zero_input: ref=%.6f simd=%.6f\n", ref, simd);

        /* Max magnitude */
        float max_src[32];
        for (int i = 0; i < 32; i++) max_src[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        ggml_q4_0_t mw; q4_quantize(&mw, max_src);
        q8_input_t mxq; q8_quantize(&mxq, max_src);
        ref = ref_q4q8_dot(&mw, &mxq);
        simd = simd_q4q8_dot(&mw, &mxq);
        if (fabsf(ref - simd) < 1e-4f) edge_pass++;
        else printf("    FAIL max_mag: ref=%.6f simd=%.6f diff=%.6e\n",
                     ref, simd, fabsf(ref-simd));
    }
    printf("    %d/3 edge cases passed\n", edge_pass);

    printf("\n=== Summary: %d pass, %d fail ===\n\n",
           pass + gemv_pass + edge_pass, fail + gemv_fail + (3 - edge_pass));

    return (fail + gemv_fail + (3 - edge_pass)) > 0 ? 1 : 0;
}
