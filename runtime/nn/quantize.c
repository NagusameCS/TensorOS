/* =============================================================================
 * TensorOS - INT16 Quantized Neural Network Inference
 *
 * Quantizes FP32 weights to INT16 and uses SSE2 PMADDWD instruction for
 * 2× throughput on matrix-vector products. This is the same technique used
 * by production inference engines (TensorRT, ONNX Runtime) but running
 * directly on bare metal with zero OS overhead.
 *
 * PMADDWD: multiplies 8 packed int16 pairs and adds adjacent products
 * to produce 4 int32 results. Processes 8 MACs per SSE2 instruction
 * vs. 4 for FP32 mulps+addps.
 *
 * Quantization scheme: symmetric per-tensor
 *   scale = max(|W|) / 32767
 *   W_q = round(W / scale)
 *   W_float ≈ W_q * scale
 * =============================================================================*/

#include "runtime/nn/quantize.h"
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "runtime/tensor/tensor_cpu.h"

/* =============================================================================
 * SSE2 SIMD Types
 * =============================================================================*/

typedef float v4f  __attribute__((vector_size(16)));
typedef short v8i16 __attribute__((vector_size(16)));
typedef int   v4i32 __attribute__((vector_size(16)));

static inline v4i32 v4i32_zero(void) { return (v4i32){0, 0, 0, 0}; }

/* Multiply 8 int16 pairs, add adjacent to 4 int32 results.
 * result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]  for i=0..3
 * This is THE key instruction for quantized dot products. */
static inline v4i32 sse2_pmaddwd(v8i16 a, v8i16 b)
{
#if defined(__aarch64__)
    /* ARM64 NEON: SMULL (lower half) + SMLAL (upper half) */
    union { v8i16 v; int16_t s[8]; } ua = { .v = a };
    union { v8i16 v; int16_t s[8]; } ub = { .v = b };
    v4i32 result;
    union { v4i32 v; int32_t s[4]; } ur;
    for (int i = 0; i < 4; i++)
        ur.s[i] = (int32_t)ua.s[2*i] * ub.s[2*i] + (int32_t)ua.s[2*i+1] * ub.s[2*i+1];
    result = ur.v;
    return result;
#else
    v8i16 tmp = a;
    __asm__("pmaddwd %1, %0" : "+x"(tmp) : "x"(b));
    union { v8i16 i16; v4i32 i32; } u = { .i16 = tmp };
    return u.i32;
#endif
}

/* Horizontal sum of 4 int32 values */
static inline int32_t v4i32_hsum(v4i32 v)
{
    union { v4i32 vec; int32_t i[4]; } u = { .vec = v };
    return u.i[0] + u.i[1] + u.i[2] + u.i[3];
}

/* =============================================================================
 * Static Quantization Buffers
 *
 * Pre-allocated storage for quantized weights.
 * Max supported: 8 layers, 64×64 weights per layer = 32K int16 total.
 * =============================================================================*/

#define QWEIGHT_POOL_SIZE (256 * 1024)  /* 256K int16 values = 512KB */
static int16_t qweight_pool[QWEIGHT_POOL_SIZE] __attribute__((aligned(16)));
static int qweight_pool_offset = 0;

void nn_quant_reset_pool(void)
{
    qweight_pool_offset = 0;
}

static int16_t *qpool_alloc(int count)
{
    /* Align to 8 elements (16 bytes) */
    count = (count + 7) & ~7;
    if (qweight_pool_offset + count > QWEIGHT_POOL_SIZE) return NULL;
    int16_t *p = qweight_pool + qweight_pool_offset;
    qweight_pool_offset += count;
    return p;
}

/* =============================================================================
 * Quantization: FP32 → INT16
 * =============================================================================*/

/* INT16 quantization range.  Must be small enough that PMADDWD accumulation
 * doesn't overflow int32.  With Q=4096, max per PMADDWD slot = 2*4096^2 = 33M.
 * For in_dim=64 (8 iterations), max hsum = 4*8*33M = 1.07B < INT32_MAX. */
#define QUANT_RANGE 4096

static void quantize_tensor(int16_t *out, quant_params_t *qp,
                            const float *data, int n)
{
    /* Find max absolute value */
    float absmax = 0;
    for (int i = 0; i < n; i++) {
        float a = data[i] > 0 ? data[i] : -data[i];
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    qp->scale = absmax / (float)QUANT_RANGE;
    qp->inv_scale = (float)QUANT_RANGE / absmax;

    /* Quantize */
    for (int i = 0; i < n; i++) {
        float v = data[i] * qp->inv_scale;
        int32_t q = (int32_t)(v + (v > 0 ? 0.5f : -0.5f));
        if (q > QUANT_RANGE) q = QUANT_RANGE;
        if (q < -QUANT_RANGE) q = -QUANT_RANGE;
        out[i] = (int16_t)q;
    }

    /* Pad remainder to 8-aligned with zeros */
    int padded = (n + 7) & ~7;
    for (int i = n; i < padded; i++)
        out[i] = 0;
}

int nn_quantize_model(nn_qmodel_t *qm, const nn_model_t *fm)
{
    qm->num_layers = fm->num_layers;
    qm->max_dim = fm->max_dim;

    for (int l = 0; l < fm->num_layers; l++) {
        const nn_layer_t *fl = &fm->layers[l];
        nn_qlayer_t *ql = &qm->layers[l];

        ql->in_dim = fl->in_dim;
        ql->out_dim = fl->out_dim;
        ql->activation = fl->activation;

        int in_d = fl->in_dim;
        int out_d = fl->out_dim;
        int pad_in = (in_d + 7) & ~7;  /* Pad each row to 8-element boundary */

        /* Find global absmax for all weights in this layer */
        float absmax = 0;
        int wcount = out_d * in_d;
        for (int i = 0; i < wcount; i++) {
            float a = fl->weights[i] > 0 ? fl->weights[i] : -fl->weights[i];
            if (a > absmax) absmax = a;
        }
        if (absmax < 1e-10f) absmax = 1e-10f;
        ql->w_qp.scale = absmax / (float)QUANT_RANGE;
        ql->w_qp.inv_scale = (float)QUANT_RANGE / absmax;

        /* Allocate row-padded weight storage */
        int wcount_padded = out_d * pad_in;
        ql->weights_q = qpool_alloc(wcount_padded);
        if (!ql->weights_q) return -1;

        /* Quantize row by row with per-row zero-padding */
        float inv = ql->w_qp.inv_scale;
        for (int r = 0; r < out_d; r++) {
            for (int c = 0; c < in_d; c++) {
                float v = fl->weights[r * in_d + c] * inv;
                int32_t q = (int32_t)(v + (v > 0 ? 0.5f : -0.5f));
                if (q > QUANT_RANGE) q = QUANT_RANGE;
                if (q < -QUANT_RANGE) q = -QUANT_RANGE;
                ql->weights_q[r * pad_in + c] = (int16_t)q;
            }
            for (int c = in_d; c < pad_in; c++)
                ql->weights_q[r * pad_in + c] = 0;
        }

        /* Quantize bias */
        if (fl->bias) {
            ql->bias_q = qpool_alloc(fl->out_dim);
            if (!ql->bias_q) return -1;
            quantize_tensor(ql->bias_q, &ql->b_qp, fl->bias, fl->out_dim);
        } else {
            ql->bias_q = NULL;
        }
    }
    return 0;
}

/* =============================================================================
 * Quantized Forward Pass
 *
 * For each layer:
 *   1. Quantize input activations to INT16
 *   2. INT16 matrix-vector multiply using SSE2 PMADDWD
 *   3. De-quantize result to FP32
 *   4. Add bias (FP32) and apply activation
 *
 * The hot loop processes 8 int16 multiply-accumulates per cycle.
 * =============================================================================*/

void nn_qforward(nn_qmodel_t *model, float *output, const float *input)
{
    static float fbuf[2][1024] __attribute__((aligned(16)));
    static int16_t ibuf[1024] __attribute__((aligned(16)));
    const float *in = input;
    int cur = 0;

    for (int l = 0; l < model->num_layers; l++) {
        nn_qlayer_t *L = &model->layers[l];
        float *out = fbuf[cur];

        /* Step 1: Quantize input activations (vectorized absmax) */
        v4i32 sign_mask_i = (v4i32){0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        union { v4i32 i; v4f f; } umask = { .i = sign_mask_i };
        v4f vabs_mask = umask.f;  (void)vabs_mask;

        v4f vmax = (v4f){0, 0, 0, 0};
        int jj = 0;
        for (; jj + 4 <= L->in_dim; jj += 4) {
            v4f v = *(const v4f *)(in + jj);
            /* Portable absolute value + max using integer bitmask */
            v4i32 sign_clear = (v4i32){0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
            v4f va = (v4f)((v4i32)v & sign_clear);  /* abs(v) */
            typedef int v4i_cmp __attribute__((vector_size(16)));
            v4i_cmp gt = (v4i_cmp)(va > vmax);
            vmax = (v4f)(((v4i_cmp)va & gt) | ((v4i_cmp)vmax & ~gt));
        }
        union { v4f vec; float f[4]; } umax = { .vec = vmax };
        float in_absmax = umax.f[0];
        if (umax.f[1] > in_absmax) in_absmax = umax.f[1];
        if (umax.f[2] > in_absmax) in_absmax = umax.f[2];
        if (umax.f[3] > in_absmax) in_absmax = umax.f[3];
        for (; jj < L->in_dim; jj++) {
            float a = in[jj] > 0 ? in[jj] : -in[jj];
            if (a > in_absmax) in_absmax = a;
        }
        if (in_absmax < 1e-10f) in_absmax = 1e-10f;
        float in_scale = in_absmax / (float)QUANT_RANGE;
        float in_inv_scale = (float)QUANT_RANGE / in_absmax;

        for (int j = 0; j < L->in_dim; j++) {
            float v = in[j] * in_inv_scale;
            int32_t q = (int32_t)(v + (v > 0 ? 0.5f : -0.5f));
            if (q > QUANT_RANGE) q = QUANT_RANGE;
            if (q < -QUANT_RANGE) q = -QUANT_RANGE;
            ibuf[j] = (int16_t)q;
        }
        /* Pad to 8-aligned */
        int pad_dim = (L->in_dim + 7) & ~7;
        for (int j = L->in_dim; j < pad_dim; j++)
            ibuf[j] = 0;

        /* Combined scale: float_result = int32_accum * w_scale * in_scale */
        float combined_scale = L->w_qp.scale * in_scale;

        /* Step 2: INT16 matrix-vector multiply with PMADDWD (4x unrolled)
         * Four independent accumulator chains saturate the integer multiply pipeline.
         * Processes 32 int16 elements per iteration = 32 MACs. */
        for (int i = 0; i < L->out_dim; i++) {
            const int16_t *w_row = L->weights_q + i * pad_dim;
            v4i32 acc0 = v4i32_zero();
            v4i32 acc1 = v4i32_zero();
            v4i32 acc2 = v4i32_zero();
            v4i32 acc3 = v4i32_zero();

            /* Process 32 int16 elements per iteration via quad PMADDWD */
            int j;
            for (j = 0; j + 32 <= pad_dim; j += 32) {
                acc0 += sse2_pmaddwd(*(const v8i16 *)(w_row + j),      *(const v8i16 *)(ibuf + j));
                acc1 += sse2_pmaddwd(*(const v8i16 *)(w_row + j + 8),  *(const v8i16 *)(ibuf + j + 8));
                acc2 += sse2_pmaddwd(*(const v8i16 *)(w_row + j + 16), *(const v8i16 *)(ibuf + j + 16));
                acc3 += sse2_pmaddwd(*(const v8i16 *)(w_row + j + 24), *(const v8i16 *)(ibuf + j + 24));
            }
            acc0 = (acc0 + acc1) + (acc2 + acc3);
            /* 8-element remainder */
            for (; j + 8 <= pad_dim; j += 8)
                acc0 += sse2_pmaddwd(*(const v8i16 *)(w_row + j), *(const v8i16 *)(ibuf + j));

            int32_t isum = v4i32_hsum(acc0);

            /* Step 3: De-quantize to FP32 */
            out[i] = (float)isum * combined_scale;

            /* Add bias in FP32 domain */
            if (L->bias_q)
                out[i] += (float)L->bias_q[i] * L->b_qp.scale;
        }

        /* Step 4: Apply activation in FP32 */
        if (L->activation == NN_ACT_RELU)
            tensor_cpu_relu(out, out, L->out_dim);
        else if (L->activation == NN_ACT_SOFTMAX)
            tensor_cpu_softmax(out, out, L->out_dim);
        else if (L->activation == NN_ACT_SIGMOID)
            for (int i = 0; i < L->out_dim; i++)
                out[i] = 1.0f / (1.0f + fast_expf(-out[i]));

        in = out;
        cur ^= 1;
    }

    int final_dim = model->layers[model->num_layers - 1].out_dim;
    kmemcpy(output, in, (size_t)final_dim * sizeof(float));
}

/* =============================================================================
 * Demo: Quantization Accuracy & Performance
 * =============================================================================*/

/* Large benchmark weights (shared with inference.c) */
static float qbench_w1[64 * 32] __attribute__((aligned(16)));
static float qbench_b1[32] __attribute__((aligned(16)));
static float qbench_w2[32 * 16] __attribute__((aligned(16)));
static float qbench_b2[16] __attribute__((aligned(16)));

static void init_qbench_weights(void)
{
    for (int i = 0; i < 64 * 32; i++)
        qbench_w1[i] = ((float)((i * 7 + 13) % 97) - 48.0f) * 0.02f;
    for (int i = 0; i < 32; i++)
        qbench_b1[i] = ((float)(i % 7) - 3.0f) * 0.1f;
    for (int i = 0; i < 32 * 16; i++)
        qbench_w2[i] = ((float)((i * 11 + 7) % 89) - 44.0f) * 0.02f;
    for (int i = 0; i < 16; i++)
        qbench_b2[i] = ((float)(i % 5) - 2.0f) * 0.1f;
}

void nn_quant_demos(void)
{
    kprintf("\n[QUANT] INT16 Quantized Inference Engine\n");
    kprintf("  Method: symmetric per-tensor, SSE2 PMADDWD (8 MACs/cycle)\n\n");

    init_qbench_weights();

    /* Build FP32 model */
    nn_model_t fp_model;
    nn_model_init(&fp_model, 2);
    fp_model.max_dim = 64;
    fp_model.layers[0] = (nn_layer_t){ qbench_w1, qbench_b1, 64, 32, NN_ACT_RELU };
    fp_model.layers[1] = (nn_layer_t){ qbench_w2, qbench_b2, 32, 16, NN_ACT_NONE };

    /* Quantize model */
    nn_qmodel_t q_model;
    nn_quant_reset_pool();
    int rc = nn_quantize_model(&q_model, &fp_model);
    if (rc != 0) {
        kprintf("  Quantization FAILED\n");
        return;
    }
    kprintf("  Model quantized: FP32 -> INT16 (2x memory reduction)\n");

    /* Test input */
    static float input[64] __attribute__((aligned(16)));
    for (int i = 0; i < 64; i++)
        input[i] = ((float)(i % 11) - 5.0f) * 0.1f;

    /* Verify accuracy */
    float fp_out[16] __attribute__((aligned(16)));
    float q_out[16] __attribute__((aligned(16)));
    nn_forward(&fp_model, fp_out, input);
    nn_qforward(&q_model, q_out, input);

    float max_err = 0;
    for (int i = 0; i < 16; i++) {
        float diff = fp_out[i] - q_out[i];
        if (diff < 0) diff = -diff;
        if (diff > max_err) max_err = diff;
    }
    int err_pct = (int)(max_err * 10000.0f);
    kprintf("  Accuracy: max error %d.%d%d%d%d\n",
            err_pct / 10000, (err_pct / 1000) % 10, (err_pct / 100) % 10,
            (err_pct / 10) % 10, err_pct % 10);

    /* Argmax match check */
    int fp_argmax = tensor_cpu_argmax(fp_out, 16);
    int q_argmax = tensor_cpu_argmax(q_out, 16);
    kprintf("  Argmax: FP32=%d INT16=%d %s\n",
            fp_argmax, q_argmax, fp_argmax == q_argmax ? "MATCH" : "MISMATCH");

    /* Benchmark: FP32 vs INT16 (10000 inferences) */
    int iters = 10000;

    uint64_t t0 = rdtsc_fenced();
    for (int r = 0; r < iters; r++)
        nn_forward(&fp_model, fp_out, input);
    uint64_t t1 = rdtsc_fenced();

    uint64_t t2 = rdtsc_fenced();
    for (int r = 0; r < iters; r++)
        nn_qforward(&q_model, q_out, input);
    uint64_t t3 = rdtsc_fenced();

    uint64_t fp_ns = perf_cycles_to_ns(t1 - t0) / iters;
    uint64_t q_ns  = perf_cycles_to_ns(t3 - t2) / iters;

    /* FLOPS: each inference = 2*64*32 + 2*32*16 = 5120 FLOPs equivalent */
    uint64_t flops_per = 2ULL * 64 * 32 + 2ULL * 32 * 16;
    uint64_t fp_us = perf_cycles_to_us(t1 - t0);
    uint64_t q_us  = perf_cycles_to_us(t3 - t2);
    uint64_t fp_mflops = (fp_us > 0) ? (iters * flops_per / fp_us) : 0;
    uint64_t q_mflops  = (q_us > 0) ? (iters * flops_per / q_us) : 0;

    kprintf("  FP32:  %lu ns/inference (%lu MFLOPS)\n", fp_ns, fp_mflops);
    kprintf("  INT16: %lu ns/inference (%lu MFLOPS)\n", q_ns, q_mflops);

    if (q_ns > 0 && fp_ns > 0) {
        uint32_t speedup = (uint32_t)((fp_ns * 100ULL) / q_ns);
        kprintf("  Speedup: %u.%u%ux\n",
                speedup / 100, (speedup / 10) % 10, speedup % 10);
    }

    kprintf("[QUANT] Complete\n");
}
