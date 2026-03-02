/* =============================================================================
 * TensorOS - Tensor Memory Arena Implementation
 *
 * A zero-fragmentation bump allocator designed for the unique allocation
 * pattern of neural network inference:
 *
 *   1. Allocate many temporaries (activations, workspace buffers)
 *   2. Compute a forward pass
 *   3. Free everything at once
 *
 * Traditional malloc/free causes fragmentation under this pattern because
 * of interleaved alloc/free. The arena avoids this entirely: allocations
 * are O(1) pointer bumps, and deallocation frees the entire arena in O(1).
 *
 * Checkpoints enable nested scopes — each layer can checkpoint, allocate
 * its temporaries, and restore on exit. This gives us the memory reuse
 * pattern that TVM's memory planner computes statically, but dynamically.
 *
 * Performance:
 *   - Allocation: ~2 ns (compare: malloc ~50-200 ns)
 *   - Deallocation: O(1) regardless of allocation count
 *   - Zero fragmentation: impossible by construction
 *   - Cache-friendly: sequential memory access pattern
 * =============================================================================*/

#include "kernel/mm/tensor_arena.h"
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "kernel/mm/tensor_mm.h"
#include "runtime/tensor/tensor_cpu.h"
#include "runtime/nn/inference.h"

/* =============================================================================
 * Static Arena Storage (2 MB, 16-byte aligned)
 * =============================================================================*/

static uint8_t arena_storage[ARENA_SIZE] __attribute__((aligned(ARENA_ALIGN)));

/* =============================================================================
 * Arena Operations
 * =============================================================================*/

void arena_init(tensor_arena_t *arena)
{
    arena->base = arena_storage;
    arena->size = ARENA_SIZE;
    arena->used = 0;
    arena->peak = 0;
    arena->alloc_count = 0;
    arena->total_allocs = 0;
    arena->total_resets = 0;
    arena->checkpoint_depth = 0;
}

void *arena_alloc(tensor_arena_t *arena, uint64_t size)
{
    /* Align up to 16 bytes */
    uint64_t aligned_size = (size + (ARENA_ALIGN - 1)) & ~((uint64_t)(ARENA_ALIGN - 1));

    if (arena->used + aligned_size > arena->size)
        return 0;  /* Arena full */

    void *ptr = arena->base + arena->used;
    arena->used += aligned_size;
    arena->alloc_count++;
    arena->total_allocs++;

    if (arena->used > arena->peak)
        arena->peak = arena->used;

    return ptr;
}

void arena_reset(tensor_arena_t *arena)
{
    arena->used = 0;
    arena->alloc_count = 0;
    arena->checkpoint_depth = 0;
    arena->total_resets++;
}

int arena_checkpoint(tensor_arena_t *arena)
{
    if (arena->checkpoint_depth >= ARENA_MAX_CHECKPOINTS)
        return -1;
    arena->checkpoints[arena->checkpoint_depth++] = arena->used;
    return 0;
}

int arena_restore(tensor_arena_t *arena)
{
    if (arena->checkpoint_depth <= 0)
        return -1;
    arena->used = arena->checkpoints[--arena->checkpoint_depth];
    return 0;
}

uint64_t arena_used(const tensor_arena_t *arena) { return arena->used; }
uint64_t arena_peak(const tensor_arena_t *arena) { return arena->peak; }
uint64_t arena_remaining(const tensor_arena_t *arena) { return arena->size - arena->used; }

/* =============================================================================
 * Arena-Backed Neural Network Forward Pass
 *
 * Demonstrates using the arena for all intermediate buffers during inference.
 * Each layer's activations are allocated from the arena, and the entire
 * inference scratch space is freed in a single reset.
 * =============================================================================*/

static void arena_nn_forward(tensor_arena_t *arena, nn_model_t *model,
                             float *output, const float *input)
{
    const float *in = input;

    for (int l = 0; l < model->num_layers; l++) {
        nn_layer_t *L = &model->layers[l];

        /* Allocate output buffer from arena */
        float *out = (float *)arena_alloc(arena, (uint64_t)L->out_dim * sizeof(float));
        if (!out) return;  /* Arena exhausted */

        /* GEMV: out = W * in + bias */
        for (int i = 0; i < L->out_dim; i++) {
            const float *w_row = L->weights + i * L->in_dim;
            float sum = 0.0f;
            for (int j = 0; j < L->in_dim; j++)
                sum += w_row[j] * in[j];
            if (L->bias) sum += L->bias[i];
            out[i] = sum;
        }

        /* Activation */
        if (L->activation == NN_ACT_RELU)
            tensor_cpu_relu(out, out, L->out_dim);
        else if (L->activation == NN_ACT_SOFTMAX)
            tensor_cpu_softmax(out, out, L->out_dim);

        in = out;
    }

    /* Copy final output */
    int last_dim = model->layers[model->num_layers - 1].out_dim;
    kmemcpy(output, in, (uint64_t)last_dim * sizeof(float));
}

/* =============================================================================
 * Demo: Tensor Memory Arena Showcase
 * =============================================================================*/

void arena_run_demos(void)
{
    kprintf("\n============================================================\n");
    kprintf("  TENSOR MEMORY ARENA\n");
    kprintf("  Zero-Fragmentation Bump Allocator for AI Workloads\n");
    kprintf("============================================================\n");
    kprintf("  Size: 2 MB | Alignment: 16-byte (SSE2)\n");
    kprintf("  Alloc: O(1) bump | Dealloc: O(1) reset | Frag: 0%%\n\n");

    tensor_arena_t arena;
    arena_init(&arena);

    /* --- Demo 1: Basic Arena Operations --- */
    kprintf("  --- Arena Operations ---\n");
    {
        float *a = (float *)arena_alloc(&arena, 1024 * sizeof(float));
        float *b = (float *)arena_alloc(&arena, 512 * sizeof(float));
        float *c = (float *)arena_alloc(&arena, 256 * sizeof(float));

        kprintf("  Alloc: 1024 + 512 + 256 floats = %lu bytes used\n",
                arena_used(&arena));
        kprintf("  Remaining: %lu bytes (%lu KB free)\n",
                arena_remaining(&arena), arena_remaining(&arena) / 1024);

        /* Verify alignment */
        int aligned = (((uint64_t)a & 0xF) == 0) &&
                      (((uint64_t)b & 0xF) == 0) &&
                      (((uint64_t)c & 0xF) == 0);
        kprintf("  16-byte aligned: %s\n", aligned ? "YES" : "NO");

        arena_reset(&arena);
        kprintf("  After reset: %lu bytes used (instant free)\n",
                arena_used(&arena));
    }

    /* --- Demo 2: Checkpoint/Restore --- */
    kprintf("\n  --- Checkpoint/Restore (Nested Scopes) ---\n");
    {
        arena_reset(&arena);

        /* Outer scope: persistent buffer */
        float *persistent = (float *)arena_alloc(&arena, 2048 * sizeof(float));
        (void)persistent;
        uint64_t outer_used = arena_used(&arena);
        kprintf("  Outer scope: %lu bytes (persistent buffer)\n", outer_used);

        /* Inner scope: temporary compute */
        arena_checkpoint(&arena);
        float *temp1 = (float *)arena_alloc(&arena, 4096 * sizeof(float));
        float *temp2 = (float *)arena_alloc(&arena, 4096 * sizeof(float));
        (void)temp1; (void)temp2;
        kprintf("  Inner scope: %lu bytes (+ temp buffers)\n", arena_used(&arena));

        /* Restore — temps are freed, persistent remains */
        arena_restore(&arena);
        kprintf("  After restore: %lu bytes (temps freed, persist kept)\n",
                arena_used(&arena));

        /* Nested checkpoints */
        arena_checkpoint(&arena);
        arena_alloc(&arena, 1024 * sizeof(float));
        arena_checkpoint(&arena);
        arena_alloc(&arena, 512 * sizeof(float));
        kprintf("  Nested (depth=2): %lu bytes\n", arena_used(&arena));
        arena_restore(&arena);
        kprintf("  Pop inner: %lu bytes\n", arena_used(&arena));
        arena_restore(&arena);
        kprintf("  Pop outer: %lu bytes (back to persistent)\n", arena_used(&arena));
    }

    /* --- Demo 3: Arena vs malloc benchmark --- */
    kprintf("\n  --- Allocation Speed: Arena vs Heap ---\n");
    {
        arena_reset(&arena);
        int iters = 10000;

        /* Arena allocation benchmark */
        uint64_t t0 = rdtsc_fenced();
        for (int i = 0; i < iters; i++) {
            arena_reset(&arena);
            /* Simulate inference: allocate 5 activation buffers */
            arena_alloc(&arena, 256 * sizeof(float));
            arena_alloc(&arena, 128 * sizeof(float));
            arena_alloc(&arena, 64 * sizeof(float));
            arena_alloc(&arena, 32 * sizeof(float));
            arena_alloc(&arena, 16 * sizeof(float));
        }
        uint64_t t1 = rdtsc_fenced();
        uint64_t arena_ns = perf_cycles_to_ns(t1 - t0) / iters;

        /* Heap allocation benchmark (tensor_mm alloc+free) */
        uint64_t t2 = rdtsc_fenced();
        for (int i = 0; i < iters; i++) {
            void *p1 = tensor_alloc(256 * sizeof(float));
            void *p2 = tensor_alloc(128 * sizeof(float));
            void *p3 = tensor_alloc(64 * sizeof(float));
            void *p4 = tensor_alloc(32 * sizeof(float));
            void *p5 = tensor_alloc(16 * sizeof(float));
            tensor_free(p5); tensor_free(p4); tensor_free(p3);
            tensor_free(p2); tensor_free(p1);
        }
        uint64_t t3 = rdtsc_fenced();
        uint64_t heap_ns = perf_cycles_to_ns(t3 - t2) / iters;

        kprintf("  Arena (5 allocs + reset): %lu ns/iter\n", arena_ns);
        kprintf("  Heap  (5 allocs + frees): %lu ns/iter\n", heap_ns);
        if (arena_ns > 0) {
            uint32_t speedup = (uint32_t)((heap_ns * 10ULL) / arena_ns);
            kprintf("  Arena speedup: %u.%ux\n", speedup / 10, speedup % 10);
        }
    }

    /* --- Demo 4: Arena-backed inference --- */
    kprintf("\n  --- Arena-Backed Neural Network Inference ---\n");
    {
        /* Build model: 64->32->16->8 */
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
        model.layers[0] = (nn_layer_t){ w1, b1, 64, 32, NN_ACT_RELU,
                                        NN_LAYER_DENSE, 0,0,0,0,0,0,0,0 };
        model.layers[1] = (nn_layer_t){ w2, b2, 32, 16, NN_ACT_RELU,
                                        NN_LAYER_DENSE, 0,0,0,0,0,0,0,0 };
        model.layers[2] = (nn_layer_t){ w3, b3, 16, 8,  NN_ACT_NONE,
                                        NN_LAYER_DENSE, 0,0,0,0,0,0,0,0 };

        float input[64] __attribute__((aligned(16)));
        float output[8] __attribute__((aligned(16)));
        float ref_output[8] __attribute__((aligned(16)));
        for (int i = 0; i < 64; i++)
            input[i] = ((float)(i * 7 % 50) - 25.0f) * 0.1f;

        /* Reference: standard forward pass */
        nn_forward(&model, ref_output, input);

        /* Arena-backed forward pass */
        arena_reset(&arena);
        arena_nn_forward(&arena, &model, output, input);

        /* Verify correctness */
        float max_err = 0.0f;
        int match = 1;
        for (int i = 0; i < 8; i++) {
            float err = output[i] - ref_output[i];
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
            if (err > 0.001f) match = 0;
        }
        kprintf("  Model: 64->32->16->8 (3 layers)\n");
        kprintf("  Arena scratch used: %lu bytes (for %lu allocs)\n",
                arena_used(&arena), arena.alloc_count);
        kprintf("  Correctness: %s (max error vs standard: %lu.%lu)\n",
                match ? "MATCH" : "MISMATCH",
                (uint64_t)max_err,
                (uint64_t)(max_err * 1000.0f) % 1000);

        /* Benchmark: arena-backed vs standard */
        int iters = 5000;
        uint64_t t0 = rdtsc_fenced();
        for (int r = 0; r < iters; r++) {
            arena_reset(&arena);
            arena_nn_forward(&arena, &model, output, input);
        }
        uint64_t t1 = rdtsc_fenced();

        uint64_t t2 = rdtsc_fenced();
        for (int r = 0; r < iters; r++)
            nn_forward(&model, output, input);
        uint64_t t3 = rdtsc_fenced();

        uint64_t arena_us = perf_cycles_to_us(t1 - t0);
        uint64_t std_us = perf_cycles_to_us(t3 - t2);
        (void)arena_us; (void)std_us;
        uint64_t arena_ns = perf_cycles_to_ns(t1 - t0) / iters;
        uint64_t std_ns = perf_cycles_to_ns(t3 - t2) / iters;

        kprintf("  Standard forward: %lu ns/inf\n", std_ns);
        kprintf("  Arena forward:    %lu ns/inf\n", arena_ns);
        kprintf("  (Arena uses simple GEMV; standard uses 4-row batched GEMV)\n");
    }

    /* --- Summary statistics --- */
    kprintf("\n  --- Arena Lifetime Statistics ---\n");
    kprintf("  Total allocs: %lu | Total resets: %lu\n",
            arena.total_allocs, arena.total_resets);
    kprintf("  Peak usage: %lu bytes (%lu KB)\n",
            arena_peak(&arena), arena_peak(&arena) / 1024);

    kprintf("\n============================================================\n");
    kprintf("  Arena: O(1) alloc, O(1) free, 0%% fragmentation\n");
    kprintf("  The memory allocator AI workloads deserve.\n");
    kprintf("============================================================\n");
}
