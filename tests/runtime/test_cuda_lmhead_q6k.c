/*
 * TensorOS CUDA lmhead Q6_K benchmark harness
 *
 * Measures:
 *  1) lmhead-like GEMV throughput on CUDA with Q6_K weights
 *  2) per-launch overhead on tiny GEMV shapes
 *
 * Build (host, Windows GNU toolchain):
 *   zig cc -O2 -target x86_64-windows-gnu -DHYPERTENSOR_HOSTED=1 -DENABLE_CUDA=1 \
 *     -Ihost/shims -I. -Ihost \
 *     tests/runtime/test_cuda_lmhead_q6k.c \
 *     runtime/nn/backend.c runtime/nn/backend_cuda.c host/hal.c \
 *     -ladvapi32 -o build_host/test_cuda_lmhead_q6k.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "runtime/nn/backend.h"

#ifdef ENABLE_CUDA

/* GGML Q6_K super-block: 256 elements in 210 bytes */
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} __attribute__((packed)) ggml_q6_k_t;

static uint64_t g_rng = 0xD1CEB00CULL;

static uint64_t xorshift64(void) {
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}

static float randf(float lo, float hi) {
    float t = (float)(xorshift64() & 0xFFFFFF) / (float)0x1000000;
    return lo + (hi - lo) * t;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7FFFFFu;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Encode one signed q6 value [-32,31] into packed ql/qh lanes. */
static void q6k_store_quant(ggml_q6_k_t *b, int half, int l, int slot, int q_signed) {
    int q = q_signed + 32; /* [0,63] */
    int lo = q & 0xF;
    int hi = (q >> 4) & 0x3;

    uint8_t *ql_h = b->ql + half * 64;
    uint8_t *qh_h = b->qh + half * 32;

    if (slot == 0) {
        ql_h[l] = (uint8_t)((ql_h[l] & 0xF0) | lo);
        qh_h[l] = (uint8_t)((qh_h[l] & ~0x03) | (hi << 0));
    } else if (slot == 1) {
        ql_h[l + 32] = (uint8_t)((ql_h[l + 32] & 0xF0) | lo);
        qh_h[l] = (uint8_t)((qh_h[l] & ~0x0C) | (hi << 2));
    } else if (slot == 2) {
        ql_h[l] = (uint8_t)((ql_h[l] & 0x0F) | (lo << 4));
        qh_h[l] = (uint8_t)((qh_h[l] & ~0x30) | (hi << 4));
    } else {
        ql_h[l + 32] = (uint8_t)((ql_h[l + 32] & 0x0F) | (lo << 4));
        qh_h[l] = (uint8_t)((qh_h[l] & ~0xC0) | (hi << 6));
    }
}

static void random_q6k_block(ggml_q6_k_t *b) {
    memset(b, 0, sizeof(*b));

    /* Small super-scale keeps output dynamic range stable. */
    b->d = fp32_to_fp16(randf(0.01f, 0.03f));

    for (int i = 0; i < 16; i++) {
        int sc = (int)(xorshift64() % 15) - 7; /* [-7,7] */
        if (sc == 0) sc = 1;
        b->scales[i] = (int8_t)sc;
    }

    for (int half = 0; half < 2; half++) {
        for (int l = 0; l < 32; l++) {
            int q0 = (int)(xorshift64() % 64) - 32;
            int q1 = (int)(xorshift64() % 64) - 32;
            int q2 = (int)(xorshift64() % 64) - 32;
            int q3 = (int)(xorshift64() % 64) - 32;
            q6k_store_quant(b, half, l, 0, q0);
            q6k_store_quant(b, half, l, 1, q1);
            q6k_store_quant(b, half, l, 2, q2);
            q6k_store_quant(b, half, l, 3, q3);
        }
    }
}

static float q6_k_dot256(const ggml_q6_k_t *block, const float *x) {
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;

    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_h = block->ql + half * 64;
        const uint8_t *qh_h = block->qh + half * 32;
        const int8_t *sc_h = block->scales + half * 8;
        const float *x_h = x + half * 128;

        for (int l = 0; l < 32; l++) {
            int q0 = (int)( ql_h[l]      & 0xF) | (int)(((qh_h[l] >> 0) & 3) << 4);
            int q1 = (int)( ql_h[l + 32] & 0xF) | (int)(((qh_h[l] >> 2) & 3) << 4);
            int q2 = (int)( ql_h[l]      >>  4) | (int)(((qh_h[l] >> 4) & 3) << 4);
            int q3 = (int)( ql_h[l + 32] >>  4) | (int)(((qh_h[l] >> 6) & 3) << 4);

            int si = l / 16;
            sum += (float)sc_h[0 + si] * (float)(q0 - 32) * x_h[l];
            sum += (float)sc_h[2 + si] * (float)(q1 - 32) * x_h[l + 32];
            sum += (float)sc_h[4 + si] * (float)(q2 - 32) * x_h[l + 64];
            sum += (float)sc_h[6 + si] * (float)(q3 - 32) * x_h[l + 96];
        }
    }

    return sum * d;
}

static void ref_gemv_q6k(float *out,
                         const ggml_q6_k_t *weights,
                         const float *x,
                         int out_dim,
                         int in_dim) {
    int nsb = in_dim / 256;
    for (int r = 0; r < out_dim; r++) {
        const ggml_q6_k_t *row = weights + (uint64_t)r * nsb;
        float s = 0.0f;
        for (int b = 0; b < nsb; b++) {
            s += q6_k_dot256(&row[b], x + b * 256);
        }
        out[r] = s;
    }
}

static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void run_lmhead_bench(const backend_t *cuda) {
    const int in_dim = 4096;
    const int out_dim = 4096;
    const int nsb = in_dim / 256;
    const int warmup = 5;
    const int iters = 40;

    ggml_q6_k_t *w = (ggml_q6_k_t *)malloc((size_t)out_dim * nsb * sizeof(ggml_q6_k_t));
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *out = (float *)malloc((size_t)out_dim * sizeof(float));
    float *ref = (float *)malloc((size_t)out_dim * sizeof(float));

    if (!w || !x || !out || !ref) {
        printf("[lmhead-q6k] allocation failed\n");
        free(w);
        free(x);
        free(out);
        free(ref);
        return;
    }

    for (int r = 0; r < out_dim * nsb; r++) random_q6k_block(&w[r]);
    for (int i = 0; i < in_dim; i++) x[i] = randf(-1.0f, 1.0f);

    ref_gemv_q6k(ref, w, x, 16, in_dim); /* spot-check only first 16 rows */

    for (int i = 0; i < warmup; i++) {
        cuda->compute.gemv(out, w, x, out_dim, in_dim, GGML_TYPE_Q6_K);
        cuda->mem.sync();
    }

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        cuda->compute.gemv(out, w, x, out_dim, in_dim, GGML_TYPE_Q6_K);
        cuda->mem.sync();
    }
    double t1 = now_sec();

    /* Numerical sanity against local CPU decoder. */
    double mae = 0.0;
    double maxe = 0.0;
    for (int i = 0; i < 16; i++) {
        double e = fabs((double)out[i] - (double)ref[i]);
        mae += e;
        if (e > maxe) maxe = e;
    }
    mae /= 16.0;

    double sec = t1 - t0;
    double calls_per_s = sec > 0.0 ? (double)iters / sec : 0.0;
    double us_per_call = sec > 0.0 ? (sec * 1e6) / (double)iters : 0.0;

    printf("[lmhead-q6k] shape=(out=%d,in=%d) iters=%d\\n", out_dim, in_dim, iters);
    printf("             time=%.3fs calls/s=%.2f us/call=%.1f\\n", sec, calls_per_s, us_per_call);
    printf("             spotcheck(rows=16): mae=%.6f max=%.6f\\n", mae, maxe);

    free(w);
    free(x);
    free(out);
    free(ref);
}

static void run_launch_overhead_bench(const backend_t *cuda) {
    const int in_dim = 256;
    const int out_dim = 1;
    const int nsb = 1;
    const int warmup = 64;
    const int iters = 5000;

    ggml_q6_k_t *w = (ggml_q6_k_t *)malloc((size_t)out_dim * nsb * sizeof(ggml_q6_k_t));
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    float out[1] = {0};

    if (!w || !x) {
        printf("[launch-overhead] allocation failed\\n");
        free(w);
        free(x);
        return;
    }

    random_q6k_block(&w[0]);
    for (int i = 0; i < in_dim; i++) x[i] = randf(-1.0f, 1.0f);

    for (int i = 0; i < warmup; i++) {
        cuda->compute.gemv(out, w, x, out_dim, in_dim, GGML_TYPE_Q6_K);
        cuda->mem.sync();
    }

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        cuda->compute.gemv(out, w, x, out_dim, in_dim, GGML_TYPE_Q6_K);
        cuda->mem.sync();
    }
    double t1 = now_sec();

    double sec = t1 - t0;
    double us_per_call = sec > 0.0 ? (sec * 1e6) / (double)iters : 0.0;

    printf("[launch-overhead] shape=(out=%d,in=%d) iters=%d\\n", out_dim, in_dim, iters);
    printf("                  total=%.3fs us/launch=%.2f\\n", sec, us_per_call);

    free(w);
    free(x);
}

int main(void) {
    printf("\\n=== TensorOS CUDA lmhead Q6_K Benchmark ===\\n\\n");

    backend_init_all();

    const backend_t *cuda = backend_get_by_id(BACKEND_CUDA);
    if (!cuda) {
        printf("CUDA backend not registered (build with -DENABLE_CUDA=1).\\n");
        return 0;
    }

    if (cuda->get_device_count && cuda->get_device_count() <= 0) {
        printf("CUDA backend loaded but no CUDA device available.\\n");
        return 0;
    }

    run_lmhead_bench(cuda);
    run_launch_overhead_bench(cuda);

    if (cuda->shutdown) cuda->shutdown();

    printf("\\n=== Done ===\\n");
    return 0;
}

#else

int main(void) {
    printf("Build with -DENABLE_CUDA=1 to run CUDA lmhead Q6_K benchmark.\\n");
    return 0;
}

#endif
