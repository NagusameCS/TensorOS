/* =============================================================================
 * TensorOS - Neural Architecture Search via Neuroevolution
 *
 * Evolves neural network architectures AND weights simultaneously using a
 * population-based evolutionary strategy. Each candidate genome encodes:
 *   1. Architecture: layer dimensions and activation functions
 *   2. Weights: the actual learned parameters
 *
 * Evolution loop (runs during boot):
 *   1. Initialize random population of 16 candidate networks
 *   2. Evaluate each: build model → forward pass on XOR → measure fitness
 *   3. Select top 4 elites (survive unchanged)
 *   4. Create 12 offspring via mutation of elites
 *   5. Repeat for N generations until XOR is solved
 *
 * Fitness = accuracy_score - 0.5 * MSE + speed_bonus
 * where speed_bonus rewards faster (smaller) networks.
 *
 * The OS literally discovers how to solve XOR from random initialization,
 * with no human-designed weights and no gradient descent.
 * =============================================================================*/

#include "runtime/nn/evolution.h"
#include "runtime/nn/inference.h"
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "runtime/tensor/tensor_cpu.h"

/* =============================================================================
 * Pseudo-Random Number Generator (LCG)
 * Fast, deterministic, seeded from TSC for variety across boots.
 * =============================================================================*/

static uint32_t evo_seed;

static uint32_t evo_rand(void)
{
    evo_seed = evo_seed * 1103515245u + 12345u;
    return (evo_seed >> 16) & 0x7FFF;
}

/* Random float in [-1, 1] */
static float evo_randf(void)
{
    return ((float)evo_rand() / 16384.0f) - 1.0f;
}

/* Random float in [0, 1] */
__attribute__((unused))
static float evo_rand01(void)
{
    return (float)evo_rand() / 32768.0f;
}

/* =============================================================================
 * Genome Operations
 * =============================================================================*/

/* Count total weights (weight matrices + biases) for a genome */
static int genome_count_weights(evo_genome_t *g)
{
    int total = 0;
    for (int l = 0; l < g->num_layers; l++) {
        int in_d = g->dims[l];
        int out_d = g->dims[l + 1];
        total += out_d * in_d;  /* weight matrix */
        total += out_d;         /* bias vector */
    }
    return total;
}

/* Initialize a genome with random architecture and weights */
static void genome_init_random(evo_genome_t *g)
{
    /* Architecture: always 2 layers for XOR (input→hidden→output) */
    g->num_layers = 2;
    g->dims[0] = 4;                            /* Input (padded to 4) */
    g->dims[1] = 4 + (evo_rand() % 4) * 4;    /* Hidden: 4, 8, 12, or 16 */
    g->dims[2] = 4;                            /* Output (padded to 4) */
    g->activations[0] = NN_ACT_RELU;
    g->activations[1] = NN_ACT_NONE;

    /* Xavier-ish random weights */
    g->num_weights = genome_count_weights(g);
    if (g->num_weights > EVO_MAX_WEIGHTS) g->num_weights = EVO_MAX_WEIGHTS;

    int idx = 0;
    for (int l = 0; l < g->num_layers; l++) {
        int fan_in = g->dims[l];
        int fan_out = g->dims[l + 1];
        float range = fast_sqrtf(6.0f / (float)(fan_in + fan_out));
        /* Weights */
        for (int i = 0; i < fan_out * fan_in; i++)
            g->weights[idx++] = evo_randf() * range;
        /* Biases (zero-init) */
        for (int i = 0; i < fan_out; i++)
            g->weights[idx++] = 0.0f;
    }
    g->fitness = 0;
    g->accuracy = 0;
    g->mse = 999.0f;
    g->latency_ns = 0;
}

/* Build an nn_model_t from a genome (points into genome's weight storage) */
static void genome_to_model(evo_genome_t *g, nn_model_t *model)
{
    nn_model_init(model, g->num_layers);
    model->max_dim = 0;

    int woff = 0;
    for (int l = 0; l < g->num_layers; l++) {
        int in_d = g->dims[l];
        int out_d = g->dims[l + 1];
        model->layers[l].weights = &g->weights[woff];
        woff += out_d * in_d;
        model->layers[l].bias = &g->weights[woff];
        woff += out_d;
        model->layers[l].in_dim = in_d;
        model->layers[l].out_dim = out_d;
        model->layers[l].activation = g->activations[l];
        if (in_d > model->max_dim) model->max_dim = in_d;
        if (out_d > model->max_dim) model->max_dim = out_d;
    }
}

/* =============================================================================
 * Fitness Evaluation
 *
 * Tests genome on XOR problem. Fitness rewards:
 *   - Correct binary classification (high accuracy)
 *   - Low MSE (output close to target)
 *   - Fast inference (smaller networks preferred)
 * =============================================================================*/

static float xor_inputs[4][4] __attribute__((aligned(16))) = {
    {0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}
};
static float xor_targets[4] = { 0.0f, 1.0f, 1.0f, 0.0f };

static void genome_evaluate(evo_genome_t *g)
{
    nn_model_t model;
    nn_model_init(&model, 0);
    genome_to_model(g, &model);

    float total_mse = 0;
    int correct = 0;
    float output[4] __attribute__((aligned(16)));

    for (int t = 0; t < 4; t++) {
        nn_forward(&model, output, xor_inputs[t]);
        float diff = output[0] - xor_targets[t];
        total_mse += diff * diff;
        int predicted = output[0] > 0.5f ? 1 : 0;
        int expected = xor_targets[t] > 0.5f ? 1 : 0;
        if (predicted == expected) correct++;
    }

    g->mse = total_mse / 4.0f;
    g->accuracy = (float)correct / 4.0f;

    /* Measure latency (average of 50 inferences) */
    uint64_t t0 = rdtsc_fenced();
    for (int r = 0; r < 50; r++)
        nn_forward(&model, output, xor_inputs[r & 3]);
    uint64_t t1 = rdtsc_fenced();
    g->latency_ns = perf_cycles_to_ns(t1 - t0) / 50;

    /* Fitness: higher is better
     * - Accuracy bonus: 10 points per correct answer
     * - MSE penalty: lower is better
     * - Speed bonus: faster networks get small bonus */
    float speed_bonus = 100.0f / (float)(g->latency_ns + 1);
    g->fitness = (float)correct * 10.0f - g->mse * 5.0f + speed_bonus;
}

/* =============================================================================
 * Evolutionary Operators
 * =============================================================================*/

/* Copy genome */
static void genome_copy(evo_genome_t *dst, const evo_genome_t *src)
{
    kmemcpy(dst, src, sizeof(evo_genome_t));
}

/* Mutate a genome: perturb weights with Gaussian-ish noise */
static void genome_mutate(evo_genome_t *g, float sigma)
{
    for (int i = 0; i < g->num_weights; i++) {
        /* Each weight has 30% chance of mutation */
        if (evo_rand() % 100 < 30) {
            /* Box-Muller-ish: sum of 3 uniform ≈ Gaussian (CLT) */
            float noise = (evo_randf() + evo_randf() + evo_randf()) * 0.577f;
            g->weights[i] += noise * sigma;
        }
    }

    /* 5% chance of activation function mutation */
    if (evo_rand() % 100 < 5) {
        int layer = evo_rand() % g->num_layers;
        if (layer < g->num_layers - 1)  /* Don't mutate output activation */
            g->activations[layer] = (evo_rand() % 2 == 0) ? NN_ACT_RELU : NN_ACT_NONE;
    }
}

/* Crossover: blend weights from two parents */
static void genome_crossover(evo_genome_t *child,
                             const evo_genome_t *p1, const evo_genome_t *p2)
{
    /* Use parent1's architecture */
    genome_copy(child, p1);

    /* Blend weights: 50% from each parent at random cutpoint */
    int cut = evo_rand() % (child->num_weights + 1);
    for (int i = cut; i < child->num_weights && i < p2->num_weights; i++)
        child->weights[i] = p2->weights[i];
}

/* Sort population by fitness (descending) — simple insertion sort */
static void population_sort(evo_genome_t *pop, int n)
{
    for (int i = 1; i < n; i++) {
        evo_genome_t key;
        genome_copy(&key, &pop[i]);
        int j = i - 1;
        while (j >= 0 && pop[j].fitness < key.fitness) {
            genome_copy(&pop[j + 1], &pop[j]);
            j--;
        }
        genome_copy(&pop[j + 1], &key);
    }
}

/* =============================================================================
 * Main Evolution Loop: Discover XOR Solver
 *
 * Strategy: (μ+λ) evolution strategy
 *   - μ = 4 elites survive each generation
 *   - λ = 12 offspring created via mutation/crossover
 *   - Adaptive mutation: sigma decreases as fitness improves
 *   - Early termination when 4/4 XOR correct with low MSE
 * =============================================================================*/

void nn_evolve_demos(void)
{
    kprintf("\n[EVO] Neural Architecture Search via Neuroevolution\n");
    kprintf("  Strategy: (4+12)-ES, population=%d, XOR task\n\n", EVO_POP_SIZE);

    /* Seed PRNG from TSC for variety across boots */
    evo_seed = (uint32_t)(rdtsc() ^ 0xDEADBEEF);

    /* Allocate population on stack (each genome ~1.4KB, 16 total = 22KB) */
    static evo_genome_t pop[EVO_POP_SIZE];

    kprintf("  Initializing %d random genomes...\n", EVO_POP_SIZE);

    /* Initialize random population */
    for (int i = 0; i < EVO_POP_SIZE; i++)
        genome_init_random(&pop[i]);

    /* Evaluate initial population */
    kprintf("  Evaluating fitness (4 XOR tests each)...\n");
    for (int i = 0; i < EVO_POP_SIZE; i++)
        genome_evaluate(&pop[i]);

    population_sort(pop, EVO_POP_SIZE);

    uint64_t evo_start = rdtsc_fenced();
    int max_gens = 100;
    int solved_gen = -1;
    float sigma = 0.5f;  /* Initial mutation strength */

    for (int gen = 0; gen < max_gens; gen++) {
        /* Print progress every 25 generations, or first 3 */
        if (gen % 25 == 0 || gen < 3) {
            int fit_10 = (int)(pop[0].fitness * 10.0f);
            if (fit_10 < 0) fit_10 = 0;
            int acc_n = (int)(pop[0].accuracy * 4.0f + 0.1f);
            int mse_100 = (int)(pop[0].mse * 100.0f);
            if (mse_100 < 0) mse_100 = 0;
            kprintf("  Gen %d: fit=%d.%d acc=%d/4 mse=%d.%d%d ",
                    gen,
                    fit_10 / 10, fit_10 % 10,
                    acc_n,
                    mse_100 / 100, (mse_100 / 10) % 10, mse_100 % 10);
            kprintf("arch=%d", pop[0].dims[0]);
            for (int l = 1; l <= pop[0].num_layers; l++)
                kprintf("->%d", pop[0].dims[l]);
            kprintf("\n");
        }

        /* Check if solved: 4/4 correct AND MSE < 0.1 */
        if (pop[0].accuracy >= 1.0f && pop[0].mse < 0.1f) {
            solved_gen = gen;
            break;
        }

        /* Create offspring: elites (top 4) survive, rest are offspring */
        for (int i = EVO_ELITES; i < EVO_POP_SIZE; i++) {
            int p1_idx = evo_rand() % EVO_ELITES;
            if (evo_rand() % 100 < 30 && EVO_ELITES >= 2) {
                /* 30% chance: crossover of two elites */
                int p2_idx = (p1_idx + 1 + evo_rand() % (EVO_ELITES - 1)) % EVO_ELITES;
                genome_crossover(&pop[i], &pop[p1_idx], &pop[p2_idx]);
            } else {
                /* 70% chance: mutated copy of an elite */
                genome_copy(&pop[i], &pop[p1_idx]);
            }
            genome_mutate(&pop[i], sigma);
        }

        /* Evaluate all offspring */
        for (int i = EVO_ELITES; i < EVO_POP_SIZE; i++)
            genome_evaluate(&pop[i]);

        /* Also re-evaluate elites (in case mutation changed them) */
        /* Actually elites are unchanged, but let's re-sort */
        population_sort(pop, EVO_POP_SIZE);

        /* Adaptive sigma: reduce as fitness improves */
        if (pop[0].accuracy >= 0.75f)
            sigma = 0.2f;
        if (pop[0].accuracy >= 1.0f)
            sigma = 0.1f;
    }

    uint64_t evo_end = rdtsc_fenced();
    uint64_t evo_us = perf_cycles_to_us(evo_end - evo_start);

    kprintf("\n  --- Evolution Results ---\n");
    if (solved_gen >= 0) {
        kprintf("  *** XOR SOLVED in generation %d ***\n", solved_gen);
    } else {
        int best_acc = (int)(pop[0].accuracy * 100.0f);
        kprintf("  Best accuracy: %d%% (not fully solved in %d gens)\n",
                best_acc, max_gens);
    }

    /* Print best genome details */
    evo_genome_t *best = &pop[0];
    kprintf("  Best architecture: ");
    for (int i = 0; i <= best->num_layers; i++) {
        if (i > 0) kprintf("->");
        kprintf("%d", best->dims[i]);
    }
    kprintf(" (%d weights)\n", best->num_weights);

    /* Show XOR predictions of best genome */
    nn_model_t model;
    nn_model_init(&model, 0);
    genome_to_model(best, &model);
    float output[4] __attribute__((aligned(16)));

    kprintf("  Predictions: ");
    for (int t = 0; t < 4; t++) {
        nn_forward(&model, output, xor_inputs[t]);
        int val = (int)(output[0] * 100.0f + 0.5f);
        if (val < 0) val = 0;
        if (val > 199) val = 199;
        kprintf("[%d,%d]=%d.%d%d ",
                (int)xor_inputs[t][0], (int)xor_inputs[t][1],
                val / 100, (val / 10) % 10, val % 10);
    }
    kprintf("\n");

    /* JIT compile the best evolved network */
    nn_jit_fn jit_fn = nn_jit_compile_model(&model);
    if (jit_fn) {
        /* Benchmark evolved JIT vs eager */
        uint64_t t0 = rdtsc_fenced();
        for (int r = 0; r < 1000; r++)
            nn_forward(&model, output, xor_inputs[r & 3]);
        uint64_t t1 = rdtsc_fenced();

        uint64_t t2 = rdtsc_fenced();
        for (int r = 0; r < 1000; r++)
            jit_fn(output, xor_inputs[r & 3]);
        uint64_t t3 = rdtsc_fenced();

        uint64_t eager_ns = perf_cycles_to_ns(t1 - t0) / 1000;
        uint64_t jit_ns = perf_cycles_to_ns(t3 - t2) / 1000;
        kprintf("  JIT compiled evolved network: eager %lu vs JIT %lu ns\n",
                eager_ns, jit_ns);
    }

    kprintf("  Evolution time: %lu us (%lu ms)\n", evo_us, evo_us / 1000);
    kprintf("[EVO] Architecture search complete\n");
}
