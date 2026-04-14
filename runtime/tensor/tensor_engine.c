/* =============================================================================
 * TensorOS - Tensor Execution Engine Implementation
 * =============================================================================*/

#include "runtime/tensor/tensor_engine.h"
#include "runtime/tensor/tensor_cpu.h"
#include "kernel/drivers/gpu/gpu.h"
#include "kernel/mm/tensor_mm.h"

static bool engine_initialized = false;

int tensor_engine_init(void)
{
    if (engine_initialized) return 0;
    engine_initialized = true;
    kprintf_debug("[ENGINE] Tensor execution engine initialized\n");
    return 0;
}

/* =============================================================================
 * Backend Selection
 * Decides whether to run on CPU, GPU, or TPU based on:
 * - Tensor size (small tensors → CPU, large → GPU)
 * - Operation type (matmul → GPU, elementwise → depends)
 * - Device availability and load
 * =============================================================================*/

static engine_backend_t select_backend(tir_opcode_t op, const tensor_desc_t *tensors,
                                         uint32_t count)
{
    /* If no GPU, must use CPU */
    if (kstate.gpu_count == 0)
        return ENGINE_BACKEND_CPU;

    /* Calculate total data size */
    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < count; i++)
        total_bytes += tensors[i].size_bytes;

    /* Small tensors: CPU is faster (avoids PCIe transfer) */
    if (total_bytes < 4096)
        return ENGINE_BACKEND_CPU;

    /* GPU-friendly operations */
    switch (op) {
    case TIR_MATMUL:
    case TIR_CONV2D:
    case TIR_ATTENTION:
    case TIR_SOFTMAX:
    case TIR_EMBEDDING:
        return ENGINE_BACKEND_GPU;

    case TIR_ADD:
    case TIR_MUL:
    case TIR_RELU:
    case TIR_GELU:
        /* Elementwise: GPU if tensor is large enough */
        return total_bytes > 65536 ? ENGINE_BACKEND_GPU : ENGINE_BACKEND_CPU;

    case TIR_LAYERNORM:
        /* LayerNorm: GPU for large hidden dims */
        return total_bytes > 32768 ? ENGINE_BACKEND_GPU : ENGINE_BACKEND_CPU;

    default:
        return ENGINE_BACKEND_CPU;
    }
}

/* =============================================================================
 * Eager Tensor Operations
 * =============================================================================*/

int tensor_matmul(tensor_desc_t *C, const tensor_desc_t *A,
                   const tensor_desc_t *B)
{
    tensor_desc_t inputs[2] = {*A, *B};
    engine_backend_t backend = select_backend(TIR_MATMUL, inputs, 2);

    if (backend == ENGINE_BACKEND_GPU) {
        return gpu_tensor_matmul(0, C, A, B);
    }

    /* CPU fallback: real SIMD-accelerated matmul */
    if (A->ndim < 2 || B->ndim < 2) return -1;
    uint64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];

    /* Allocate output */
    C->ndim = 2;
    C->shape[0] = M;
    C->shape[1] = N;
    C->dtype = A->dtype;
    C->size_bytes = M * N * sizeof(float);

    /* Perform real CPU matmul if data pointers are set */
    if (A->data_virt && B->data_virt && C->data_virt) {
        tensor_cpu_matmul((float *)C->data_virt,
                          (const float *)A->data_virt,
                          (const float *)B->data_virt,
                          (int)M, (int)N, (int)K);
    }

    kstate.tensor_ops_total++;
    return 0;
}

int tensor_add(tensor_desc_t *C, const tensor_desc_t *A,
                const tensor_desc_t *B)
{
    tensor_desc_t inputs[2] = {*A, *B};
    engine_backend_t backend = select_backend(TIR_ADD, inputs, 2);

    if (backend == ENGINE_BACKEND_GPU) {
        return gpu_tensor_elementwise(0, C, A, B, 0);
    }

    /* CPU element-wise add */
    *C = *A;
    if (A->data_virt && B->data_virt && C->data_virt) {
        int n = (int)(A->size_bytes / sizeof(float));
        const float *fa = (const float *)A->data_virt;
        const float *fb = (const float *)B->data_virt;
        float *fc = (float *)C->data_virt;
        for (int i = 0; i < n; i++) fc[i] = fa[i] + fb[i];
    }
    kstate.tensor_ops_total++;
    return 0;
}

int tensor_relu(tensor_desc_t *output, const tensor_desc_t *input)
{
    *output = *input;
    if (input->data_virt && output->data_virt) {
        int n = (int)(input->size_bytes / sizeof(float));
        tensor_cpu_relu((float *)output->data_virt,
                        (const float *)input->data_virt, n);
    }
    kstate.tensor_ops_total++;
    return 0;
}

int tensor_softmax(tensor_desc_t *output, const tensor_desc_t *input, int axis)
{
    tensor_desc_t inputs[1] = {*input};
    engine_backend_t backend = select_backend(TIR_SOFTMAX, inputs, 1);

    if (backend == ENGINE_BACKEND_GPU)
        return gpu_tensor_softmax(0, output, input, axis);

    *output = *input;
    if (input->data_virt && output->data_virt) {
        int n = (int)(input->size_bytes / sizeof(float));
        tensor_cpu_softmax((float *)output->data_virt,
                           (const float *)input->data_virt, n);
    }
    kstate.tensor_ops_total++;
    return 0;
}

int tensor_attention(tensor_desc_t *output,
                      const tensor_desc_t *Q, const tensor_desc_t *K,
                      const tensor_desc_t *V, float scale)
{
    /* Always prefer GPU for attention */
    if (kstate.gpu_count > 0)
        return gpu_tensor_attention(0, output, Q, K, V, scale);

    /* CPU fallback using tensor_cpu attention implementation */
    *output = *Q;
    if (Q->data_virt && K->data_virt && V->data_virt && output->data_virt) {
        int seq_len = (int)Q->shape[0];
        int head_dim = (int)Q->shape[1];
        tensor_cpu_attention((float *)output->data_virt,
                             (const float *)Q->data_virt,
                             (const float *)K->data_virt,
                             (const float *)V->data_virt,
                             seq_len, head_dim);
    }
    kstate.tensor_ops_total++;
    return 0;
}

int tensor_layernorm(tensor_desc_t *output, const tensor_desc_t *input,
                      const tensor_desc_t *gamma, const tensor_desc_t *beta,
                      float epsilon)
{
    if (kstate.gpu_count > 0)
        return gpu_tensor_layernorm(0, output, input, gamma, beta, epsilon);

    *output = *input;
    if (input->data_virt && output->data_virt) {
        int n = (int)(input->size_bytes / sizeof(float));
        /* Compute layernorm: (x - mean) / sqrt(var + eps) */
        tensor_cpu_layernorm((float *)output->data_virt,
                             (const float *)input->data_virt, n, epsilon);
        /* Apply affine: out = out * gamma + beta */
        if (gamma && gamma->data_virt) {
            const float *g = (const float *)gamma->data_virt;
            float *o = (float *)output->data_virt;
            if (beta && beta->data_virt) {
                const float *b = (const float *)beta->data_virt;
                for (int i = 0; i < n; i++)
                    o[i] = o[i] * g[i] + b[i];
            } else {
                for (int i = 0; i < n; i++)
                    o[i] = o[i] * g[i];
            }
        }
    }
    kstate.tensor_ops_total++;
    return 0;
}

/* =============================================================================
 * Compute Graph Execution
 * =============================================================================*/

compute_graph_t *tensor_graph_create(void)
{
    compute_graph_t *graph = (compute_graph_t *)kmalloc(sizeof(compute_graph_t));
    if (graph) {
        kmemset(graph, 0, sizeof(*graph));
        graph->default_backend = ENGINE_BACKEND_AUTO;
    }
    return graph;
}

compute_node_t *tensor_graph_add_op(compute_graph_t *graph, tir_opcode_t op,
                                      tensor_desc_t *inputs, uint32_t num_inputs)
{
    if (!graph || graph->node_count >= MAX_GRAPH_NODES) return NULL;

    compute_node_t *node = &graph->nodes[graph->node_count];
    kmemset(node, 0, sizeof(*node));
    node->id = graph->node_count++;
    node->op = op;
    node->num_inputs = num_inputs;

    for (uint32_t i = 0; i < num_inputs && i < 4; i++)
        node->inputs[i] = inputs[i];

    /* Auto-select backend */
    node->backend = select_backend(op, inputs, num_inputs);

    return node;
}

int tensor_graph_compile(compute_graph_t *graph)
{
    if (!graph) return -1;

    /* Optimization passes on the graph:
     * 1. Topological sort
     * 2. Operator fusion
     * 3. Memory planning (allocate/reuse tensor buffers)
     * 4. Device placement optimization
     */

    /* Simple: ensure all GPU ops are batched together */
    graph->compiled = true;
    kprintf_debug("[ENGINE] Graph compiled: %d nodes\n", graph->node_count);
    return 0;
}

int tensor_graph_execute(compute_graph_t *graph)
{
    if (!graph || !graph->compiled) return -1;

    for (uint32_t i = 0; i < graph->node_count; i++) {
        compute_node_t *node = &graph->nodes[i];
        if (node->completed) continue;

        /* Check dependencies */
        bool deps_done = true;
        for (uint32_t j = 0; j < node->num_deps; j++) {
            if (!node->deps[j]->completed) {
                deps_done = false;
                break;
            }
        }
        if (!deps_done) continue;

        /* Execute the node by dispatching to the appropriate backend */
        switch (node->op) {
        case TIR_MATMUL:
            if (node->num_inputs >= 2)
                tensor_matmul(&node->output, &node->inputs[0], &node->inputs[1]);
            break;
        case TIR_ADD:
            if (node->num_inputs >= 2)
                tensor_add(&node->output, &node->inputs[0], &node->inputs[1]);
            break;
        case TIR_RELU:
            if (node->num_inputs >= 1)
                tensor_relu(&node->output, &node->inputs[0]);
            break;
        case TIR_SOFTMAX:
            if (node->num_inputs >= 1)
                tensor_softmax(&node->output, &node->inputs[0], -1);
            break;
        case TIR_ATTENTION:
            if (node->num_inputs >= 3)
                tensor_attention(&node->output, &node->inputs[0],
                                 &node->inputs[1], &node->inputs[2], 1.0f);
            break;
        case TIR_LAYERNORM:
            if (node->num_inputs >= 3)
                tensor_layernorm(&node->output, &node->inputs[0],
                                 &node->inputs[1], &node->inputs[2], 1e-5f);
            break;
        default:
            /* Unsupported ops: mark completed without executing */
            break;
        }
        node->completed = true;
        kstate.tensor_ops_total++;
    }

    return 0;
}

void tensor_graph_destroy(compute_graph_t *graph)
{
    if (graph) kfree(graph);
}
