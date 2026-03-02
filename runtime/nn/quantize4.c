/* =============================================================================
 * TensorOS - INT4 Block Quantization Engine (Q4_0 Format)
 *
 * This is the same quantization format used by GGML/llama.cpp — the engine
 * that made LLM inference possible on consumer hardware. TensorOS runs it
 * on bare metal with SSE2 SIMD, achieving maximum throughput with zero
 * OS overhead.
 *
 * Q4_0 format:
 *   Block of 32 values → 20 bytes (1 float scale + 16 packed nibble bytes)
 *   Compression: 6.4x vs FP32, 3.2x vs INT16
 *   Quantization: symmetric per-block, range [-8, 7]
 *
 * Key innovation: the dequantize-and-dot kernel unpacks nibbles to float
 * and computes the dot product in a single pass, using SSE2 4-wide SIMD
 * to process 4 elements at a time within each 32-element block.
 * =============================================================================*/

#include "runtime/nn/quantize4.h"
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "runtime/tensor/tensor_cpu.h"
#include "runtime/nn/quantize.h"   /* for INT16 comparison */

/* =============================================================================
 * SSE2 SIMD Types (local definitions for freestanding)
 * =============================================================================*/

typedef float q4_v4f __attribute__((vector_size(16)));

/* =============================================================================
 * Block Pool: Static allocation for Q4 blocks (no heap needed)
 * =============================================================================*/

/* Pool for quantized weight blocks. Each block is 20 bytes.
 * 16384 blocks × 20 bytes = 320 KB — enough for large models. */
#define Q4_POOL_BLOCKS 16384

static q4_block_t q4_pool[Q4_POOL_BLOCKS] __attribute__((aligned(16)));
static int q4_pool_used = 0;

void q4_reset_pool(void)
{
    q4_pool_used = 0;
}

static q4_block_t *q4_pool_alloc(int count)
{
    if (q4_pool_used + count > Q4_POOL_BLOCKS) return 0;
    q4_block_t *p = &q4_pool[q4_pool_used];
    q4_pool_used += count;
    return p;
}

/* =============================================================================
 * Fast Math Helpers
 * =============================================================================*/

static inline float q4_fabsf(float x)
{
    return x < 0.0f ? -x : x;
}

static inline int q4_roundf(float x)
{
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}

static inline int q4_clamp(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* =============================================================================
 * Q4_0 Block Operations
 * =============================================================================*/

void q4_quantize_block(q4_block_t *block, const float *values)
{
    /* Find absmax for scale computation */
    float absmax = 0.0f;
    for (int i = 0; i < Q4_BLOCK_SIZE; i++) {
        float a = q4_fabsf(values[i]);
        if (a > absmax) absmax = a;
    }

    /* Scale: maps [-absmax, absmax] → [-8, 7] */
    float scale = absmax / 7.0f;
    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
    block->scale = scale;

    /* Quantize and pack: two 4-bit values per byte.
     * q[i] = round(value[i] / scale) clamped to [-8, 7]
     * stored as (q[i] + 8) in range [0, 15] */
    for (int i = 0; i < 16; i++) {
        int lo = q4_clamp(q4_roundf(values[2 * i]     * inv_scale), -8, 7) + 8;
        int hi = q4_clamp(q4_roundf(values[2 * i + 1] * inv_scale), -8, 7) + 8;
        block->data[i] = (uint8_t)((hi << 4) | lo);
    }
}

void q4_dequantize_block(float *out, const q4_block_t *block)
{
    float scale = block->scale;
    for (int i = 0; i < 16; i++) {
        uint8_t packed = block->data[i];
        out[2 * i]     = (float)((int)(packed & 0x0F) - 8) * scale;
        out[2 * i + 1] = (float)((int)(packed >> 4)   - 8) * scale;
    }
}

/* =============================================================================
 * SSE2-Optimized Q4 Dot Product
 *
 * Computes dot(Q4_block, FP32_vector) for 32 elements.
 * Strategy: unpack 4 nibbles at a time → float, multiply with input,
 * accumulate in a SIMD register. Final horizontal sum + scale.
 * =============================================================================*/

float q4_dot_block(const q4_block_t *block, const float *x)
{
    q4_v4f acc0 = {0, 0, 0, 0};
    q4_v4f acc1 = {0, 0, 0, 0};

    /* Process 8 values per iteration (4 bytes = 8 nibbles) */
    for (int i = 0; i < 16; i += 4) {
        /* Unpack 4 bytes → 8 nibbles → 8 floats, processed as 2× v4f */
        uint8_t b0 = block->data[i];
        uint8_t b1 = block->data[i + 1];
        uint8_t b2 = block->data[i + 2];
        uint8_t b3 = block->data[i + 3];

        q4_v4f q0 = {
            (float)((int)(b0 & 0x0F) - 8),
            (float)((int)(b0 >> 4)   - 8),
            (float)((int)(b1 & 0x0F) - 8),
            (float)((int)(b1 >> 4)   - 8)
        };
        q4_v4f q1 = {
            (float)((int)(b2 & 0x0F) - 8),
            (float)((int)(b2 >> 4)   - 8),
            (float)((int)(b3 & 0x0F) - 8),
            (float)((int)(b3 >> 4)   - 8)
        };

        int base = i * 2;   /* element index = byte_index * 2 */
        acc0 += q0 * *(const q4_v4f *)(x + base);
        acc1 += q1 * *(const q4_v4f *)(x + base + 4);
    }

    q4_v4f total = acc0 + acc1;
    union { q4_v4f vec; float f[4]; } u = { .vec = total };
    return block->scale * (u.f[0] + u.f[1] + u.f[2] + u.f[3]);
}

/* =============================================================================
 * Q4 GEMV: Matrix-Vector Multiply with Q4 Weights
 *
 * Computes out[i] = sum_j(W_q4[i][j] * x[j]) for each output neuron.
 * Weights are stored as Q4 blocks, row-major. Each row has ceil(in_dim/32)
 * blocks. We process one output neuron at a time.
 *
 * 2-row unrolled: processes 2 output neurons simultaneously to improve
 * instruction-level parallelism and reduce loop overhead.
 * =============================================================================*/

void q4_gemv(float *out, const q4_block_t *weights, const float *input,
             int out_dim, int in_dim)
{
    int blocks_per_row = (in_dim + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    int i = 0;

    /* 2-row unrolled for ILP */
    for (; i + 2 <= out_dim; i += 2) {
        const q4_block_t *row0 = weights + (i)     * blocks_per_row;
        const q4_block_t *row1 = weights + (i + 1) * blocks_per_row;
        float sum0 = 0.0f, sum1 = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const float *xp = input + b * Q4_BLOCK_SIZE;
            sum0 += q4_dot_block(&row0[b], xp);
            sum1 += q4_dot_block(&row1[b], xp);
        }
        out[i]     = sum0;
        out[i + 1] = sum1;
    }

    /* Remainder */
    for (; i < out_dim; i++) {
        const q4_block_t *row = weights + i * blocks_per_row;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++)
            sum += q4_dot_block(&row[b], input + b * Q4_BLOCK_SIZE);
        out[i] = sum;
    }
}

/* =============================================================================
 * Model Quantization: FP32 → Q4_0
 * =============================================================================*/

int q4_quantize_model(q4_model_t *qm, const nn_model_t *fm)
{
    if (fm->num_layers > Q4_MAX_LAYERS) return -1;
    qm->num_layers = fm->num_layers;
    qm->max_dim = fm->max_dim;

    for (int l = 0; l < fm->num_layers; l++) {
        const nn_layer_t *fl = &fm->layers[l];
        q4_layer_t *ql = &qm->layers[l];

        ql->in_dim = fl->in_dim;
        ql->out_dim = fl->out_dim;
        ql->activation = fl->activation;
        ql->bias = fl->bias;   /* Share FP32 bias */

        /* Calculate blocks needed: each row of the weight matrix
         * (one output neuron) is divided into blocks of 32 */
        int blocks_per_row = (fl->in_dim + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
        ql->num_blocks = fl->out_dim * blocks_per_row;

        ql->w_blocks = q4_pool_alloc(ql->num_blocks);
        if (!ql->w_blocks) return -1;

        /* Quantize each row of weights. We need to pad the last block
         * with zeros if in_dim is not a multiple of 32. */
        float pad_buf[Q4_BLOCK_SIZE] __attribute__((aligned(16)));
        for (int r = 0; r < fl->out_dim; r++) {
            const float *row = fl->weights + r * fl->in_dim;
            q4_block_t *dst = ql->w_blocks + r * blocks_per_row;

            int j = 0;
            /* Full blocks */
            for (; j + Q4_BLOCK_SIZE <= fl->in_dim; j += Q4_BLOCK_SIZE)
                q4_quantize_block(&dst[j / Q4_BLOCK_SIZE], row + j);

            /* Partial last block (zero-padded) */
            if (j < fl->in_dim) {
                kmemset(pad_buf, 0, sizeof(pad_buf));
                for (int k = 0; j + k < fl->in_dim; k++)
                    pad_buf[k] = row[j + k];
                q4_quantize_block(&dst[j / Q4_BLOCK_SIZE], pad_buf);
            }
        }
    }
    return 0;
}

/* =============================================================================
 * Q4 Forward Pass
 * =============================================================================*/

void q4_forward(q4_model_t *model, float *output, const float *input)
{
    static float buf[2][1024] __attribute__((aligned(16)));
    const float *in = input;
    int cur = 0;

    for (int l = 0; l < model->num_layers; l++) {
        q4_layer_t *L = &model->layers[l];
        float *out = buf[cur];

        /* Q4 GEMV: W_q4 × input */
        q4_gemv(out, L->w_blocks, in, L->out_dim, L->in_dim);

        /* Add FP32 bias */
        if (L->bias) {
            for (int i = 0; i < L->out_dim; i++)
                out[i] += L->bias[i];
        }

        /* Activation */
        if (L->activation == NN_ACT_RELU)
            tensor_cpu_relu(out, out, L->out_dim);
        else if (L->activation == NN_ACT_SOFTMAX)
            tensor_cpu_softmax(out, out, L->out_dim);

        in = out;
        cur ^= 1;
    }

    /* Copy final output */
    int last_dim = model->layers[model->num_layers - 1].out_dim;
    kmemcpy(output, in, (uint64_t)last_dim * sizeof(float));
}

/* =============================================================================
 * Quantization Error Analysis
 * =============================================================================*/

__attribute__((unused))
static float q4_max_error(const nn_model_t *fm, const q4_model_t *qm,
                          const float *input)
{
    float fp_out[64] __attribute__((aligned(16)));
    float q4_out[64] __attribute__((aligned(16)));

    nn_forward((nn_model_t *)fm, fp_out, input);
    q4_forward((q4_model_t *)qm, q4_out, input);

    int out_dim = fm->layers[fm->num_layers - 1].out_dim;
    float max_err = 0.0f;
    for (int i = 0; i < out_dim; i++) {
        float err = q4_fabsf(fp_out[i] - q4_out[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* =============================================================================
 * Demo: INT4 Block Quantization Showcase
 *
 * Demonstrates:
 *   1. FP32 → Q4_0 quantization with error analysis
 *   2. Three-way speed comparison: FP32 vs INT16 vs Q4
 *   3. Memory footprint comparison
 *   4. Accuracy verification (argmax preservation)
 * =============================================================================*/

void q4_run_demos(void)
{
    kprintf("\n============================================================\n");
    kprintf("  INT4 BLOCK QUANTIZATION ENGINE (Q4_0 Format)\n");
    kprintf("  GGML/llama.cpp-class Quantization on Bare Metal\n");
    kprintf("============================================================\n");
    kprintf("  Format: Q4_0 (32 values -> 20 bytes, 6.4x compression)\n");
    kprintf("  Method: symmetric per-block, range [-8,7], SSE2 SIMD\n");
    kprintf("  This is the technology behind llama.cpp and GGML.\n\n");

    /* Build test model: 64->32->16->8 (same as INT16 demo for comparison) */
    static float w1[64 * 32] __attribute__((aligned(16)));
    static float b1[32] __attribute__((aligned(16)));
    static float w2[32 * 16] __attribute__((aligned(16)));
    static float b2[16] __attribute__((aligned(16)));
    static float w3[16 * 8]  __attribute__((aligned(16)));
    static float b3[8]  __attribute__((aligned(16)));

    for (int i = 0; i < 64 * 32; i++)
        w1[i] = ((float)((i * 7 + 3) % 97) - 48.0f) * 0.02f;
    for (int i = 0; i < 32; i++)
        b1[i] = ((float)(i % 11) - 5.0f) * 0.05f;
    for (int i = 0; i < 32 * 16; i++)
        w2[i] = ((float)((i * 11 + 5) % 83) - 41.0f) * 0.025f;
    for (int i = 0; i < 16; i++)
        b2[i] = ((float)(i % 7) - 3.0f) * 0.04f;
    for (int i = 0; i < 16 * 8; i++)
        w3[i] = ((float)((i * 3 + 1) % 67) - 33.0f) * 0.03f;
    for (int i = 0; i < 8; i++)
        b3[i] = ((float)(i % 5) - 2.0f) * 0.06f;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 64;
    model.layers[0] = (nn_layer_t){ w1, b1, 64, 32, NN_ACT_RELU, NN_LAYER_DENSE,
                                    0,0,0,0,0,0,0,0 };
    model.layers[1] = (nn_layer_t){ w2, b2, 32, 16, NN_ACT_RELU, NN_LAYER_DENSE,
                                    0,0,0,0,0,0,0,0 };
    model.layers[2] = (nn_layer_t){ w3, b3, 16, 8,  NN_ACT_NONE, NN_LAYER_DENSE,
                                    0,0,0,0,0,0,0,0 };

    /* Quantize to INT16 */
    nn_qmodel_t qmodel16;
    nn_quant_reset_pool();
    nn_quantize_model(&qmodel16, &model);

    /* Quantize to Q4_0 */
    q4_model_t qmodel4;
    q4_reset_pool();
    q4_quantize_model(&qmodel4, &model);

    /* --- Accuracy Comparison --- */
    kprintf("  --- Accuracy: FP32 vs INT16 vs INT4 ---\n");

    float input[64] __attribute__((aligned(16)));
    for (int i = 0; i < 64; i++)
        input[i] = ((float)(i * 7 % 50) - 25.0f) * 0.1f;

    float fp_out[8]   __attribute__((aligned(16)));
    float q16_out[8]  __attribute__((aligned(16)));
    float q4_out[8]   __attribute__((aligned(16)));

    nn_forward(&model, fp_out, input);
    nn_qforward(&qmodel16, q16_out, input);
    q4_forward(&qmodel4, q4_out, input);

    /* Find argmax for each */
    int fp_argmax = 0, q16_argmax = 0, q4_argmax = 0;
    for (int i = 1; i < 8; i++) {
        if (fp_out[i]  > fp_out[fp_argmax])   fp_argmax  = i;
        if (q16_out[i] > q16_out[q16_argmax]) q16_argmax = i;
        if (q4_out[i]  > q4_out[q4_argmax])   q4_argmax  = i;
    }

    /* Compute max errors */
    float err16 = 0.0f, err4 = 0.0f;
    for (int i = 0; i < 8; i++) {
        float e16 = q4_fabsf(fp_out[i] - q16_out[i]);
        float e4  = q4_fabsf(fp_out[i] - q4_out[i]);
        if (e16 > err16) err16 = e16;
        if (e4  > err4)  err4  = e4;
    }

    int e16 = (int)(err16 * 1000.0f);
    int e4  = (int)(err4  * 1000.0f);

    kprintf("  FP32  argmax: %d\n", fp_argmax);
    kprintf("  INT16 argmax: %d  (max error: %d.%d%d%d)\n", q16_argmax,
            e16/1000, (e16/100)%10, (e16/10)%10, e16%10);
    kprintf("  INT4  argmax: %d  (max error: %d.%d%d%d)\n", q4_argmax,
            e4/1000, (e4/100)%10, (e4/10)%10, e4%10);
    kprintf("  Argmax match: %s\n",
            (fp_argmax == q16_argmax && fp_argmax == q4_argmax) ? "ALL MATCH" : "MISMATCH");

    /* --- Memory Footprint Comparison --- */
    kprintf("\n  --- Memory Footprint ---\n");
    int total_params = 64*32 + 32 + 32*16 + 16 + 16*8 + 8;
    int fp32_bytes = total_params * 4;

    /* INT16: 2 bytes per weight */
    int int16_weight_params = 64*32 + 32*16 + 16*8;
    int int16_bytes = int16_weight_params * 2 + (32 + 16 + 8) * 2; /* weights+bias */

    /* Q4: 20 bytes per block of 32, bias stays FP32 */
    int q4_weight_blocks = (64 + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE * 32
                         + (32 + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE * 16
                         + (16 + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE * 8;
    int q4_bytes = q4_weight_blocks * 20 + (32 + 16 + 8) * 4;

    kprintf("  Model: 64->32->16->8 (%d params)\n", total_params);
    kprintf("  FP32:  %d bytes (baseline)\n", fp32_bytes);
    kprintf("  INT16: %d bytes (%d.%dx compression)\n",
            int16_bytes,
            fp32_bytes * 10 / int16_bytes / 10,
            (fp32_bytes * 10 / int16_bytes) % 10);
    kprintf("  Q4_0:  %d bytes (%d.%dx compression)\n",
            q4_bytes,
            fp32_bytes * 10 / q4_bytes / 10,
            (fp32_bytes * 10 / q4_bytes) % 10);

    /* --- Speed Benchmark: FP32 vs INT16 vs Q4 --- */
    kprintf("\n  --- Speed Benchmark (5000 inferences) ---\n");
    int iters = 5000;

    /* FP32 benchmark */
    uint64_t t0 = rdtsc_fenced();
    for (int r = 0; r < iters; r++)
        nn_forward(&model, fp_out, input);
    uint64_t t1 = rdtsc_fenced();

    /* INT16 benchmark */
    uint64_t t2 = rdtsc_fenced();
    for (int r = 0; r < iters; r++)
        nn_qforward(&qmodel16, q16_out, input);
    uint64_t t3 = rdtsc_fenced();

    /* Q4 benchmark */
    uint64_t t4 = rdtsc_fenced();
    for (int r = 0; r < iters; r++)
        q4_forward(&qmodel4, q4_out, input);
    uint64_t t5 = rdtsc_fenced();

    uint64_t fp_ns  = perf_cycles_to_ns(t1 - t0) / iters;
    uint64_t q16_ns = perf_cycles_to_ns(t3 - t2) / iters;
    uint64_t q4_ns  = perf_cycles_to_ns(t5 - t4) / iters;

    /* FLOPS: 2*(64*32 + 32*16 + 16*8) = 5376 */
    uint64_t flops = 2ULL * (64*32 + 32*16 + 16*8);
    uint64_t fp_us  = perf_cycles_to_us(t1 - t0);
    uint64_t q16_us = perf_cycles_to_us(t3 - t2);
    uint64_t q4_us  = perf_cycles_to_us(t5 - t4);
    uint64_t fp_mf  = (fp_us > 0)  ? (iters * flops / fp_us)  : 0;
    uint64_t q16_mf = (q16_us > 0) ? (iters * flops / q16_us) : 0;
    uint64_t q4_mf  = (q4_us > 0)  ? (iters * flops / q4_us)  : 0;

    kprintf("  FP32:  %lu ns/inf (%lu MFLOPS)\n",  fp_ns,  fp_mf);
    kprintf("  INT16: %lu ns/inf (%lu MFLOPS)\n",  q16_ns, q16_mf);
    kprintf("  Q4_0:  %lu ns/inf (%lu MFLOPS)\n",  q4_ns,  q4_mf);

    /* Speedups */
    if (q16_ns > 0 && fp_ns > 0) {
        uint32_t sp16 = (uint32_t)((fp_ns * 10ULL) / q16_ns);
        kprintf("  INT16 speedup: %u.%ux vs FP32\n", sp16 / 10, sp16 % 10);
    }
    if (q4_ns > 0 && fp_ns > 0) {
        uint32_t sp4 = (uint32_t)((fp_ns * 10ULL) / q4_ns);
        kprintf("  Q4_0 speedup:  %u.%ux vs FP32\n", sp4 / 10, sp4 % 10);
    }
    kprintf("  Note: Q4 trades compute for memory -- enables 5.5x larger models\n");

    /* --- Block-level analysis --- */
    kprintf("\n  --- Q4_0 Block Analysis ---\n");
    {
        float test_vals[32] __attribute__((aligned(16)));
        for (int i = 0; i < 32; i++)
            test_vals[i] = ((float)(i * 7 % 50) - 25.0f) * 0.04f;

        q4_block_t blk;
        q4_quantize_block(&blk, test_vals);

        float dequant[32] __attribute__((aligned(16)));
        q4_dequantize_block(dequant, &blk);

        float block_err = 0.0f, block_l2 = 0.0f;  (void)block_l2;
        for (int i = 0; i < 32; i++) {
            float e = q4_fabsf(test_vals[i] - dequant[i]);
            if (e > block_err) block_err = e;
            block_l2 += e * e;
        }
        /* Scale factor tells us the precision per step */
        int scale_1000 = (int)(blk.scale * 1000.0f);
        int err_1000 = (int)(block_err * 1000.0f);

        kprintf("  Block of 32 values: scale=0.%d%d%d, max_err=0.%d%d%d\n",
                scale_1000/100, (scale_1000/10)%10, scale_1000%10,
                err_1000/100, (err_1000/10)%10, err_1000%10);
        kprintf("  20 bytes packed (vs 128 bytes FP32)\n");
    }

    /* --- Dot product verification --- */
    kprintf("\n  --- Q4 Dot Product Verification ---\n");
    {
        float a[32] __attribute__((aligned(16)));
        float b[32] __attribute__((aligned(16)));
        for (int i = 0; i < 32; i++) {
            a[i] = ((float)(i * 3 % 20) - 10.0f) * 0.1f;
            b[i] = ((float)(i * 7 % 30) - 15.0f) * 0.1f;
        }

        /* Exact FP32 dot product */
        float exact = 0.0f;
        for (int i = 0; i < 32; i++) exact += a[i] * b[i];

        /* Q4 dot product */
        q4_block_t blk;
        q4_quantize_block(&blk, a);
        float q4_dot = q4_dot_block(&blk, b);

        int ex10 = (int)(exact * 10.0f);
        int q410 = (int)(q4_dot * 10.0f);
        int derr10 = (int)(q4_fabsf(exact - q4_dot) * 10.0f);
        /* Use 1-decimal format to avoid sign/padding issues */
        kprintf("  FP32 dot: %s%d.%d | Q4 dot: %s%d.%d | error: %d.%d\n",
                (exact < 0 ? "-" : ""),
                (ex10 < 0 ? -ex10 : ex10) / 10, (ex10 < 0 ? -ex10 : ex10) % 10,
                (q4_dot < 0 ? "-" : ""),
                (q410 < 0 ? -q410 : q410) / 10, (q410 < 0 ? -q410 : q410) % 10,
                derr10 / 10, derr10 % 10);
    }

    kprintf("\n============================================================\n");
    kprintf("  Q4_0: 6.4x compression, llama.cpp-class quantization\n");
    kprintf("  Running on bare metal. No OS. No frameworks.\n");
    kprintf("============================================================\n");
}
