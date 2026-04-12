/*
 * TensorOS Tensor Injection / Hidden-State Bridge
 *
 * Implementation of the hidden-state bridge for daisy-chaining LLMs.
 */

#include "runtime/nn/tensor_bridge.h"

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

void tensor_bridge_init(tensor_bridge_t *bridge) {
    kmemset(bridge, 0, sizeof(*bridge));
    bridge->mode = BRIDGE_MODE_NONE;
    bridge->capture_layer = -1;
    bridge->inject_layer = -1;
    bridge->proj_type = BRIDGE_PROJ_NONE;
}

int tensor_bridge_set_capture(tensor_bridge_t *bridge, int layer, int dim) {
    if (dim <= 0) return -1;

    /* Allocate capture buffer */
    if (bridge->capture_buf.data) tensor_free(bridge->capture_buf.data);
    bridge->capture_buf.data = (float *)tensor_alloc(dim * sizeof(float));
    if (!bridge->capture_buf.data) return -1;

    bridge->capture_buf.dim = dim;
    bridge->capture_layer = layer;
    bridge->capture_buf.valid = 0;

    if (bridge->mode == BRIDGE_MODE_INJECT)
        bridge->mode = BRIDGE_MODE_BOTH;
    else
        bridge->mode = BRIDGE_MODE_CAPTURE;
    return 0;
}

int tensor_bridge_set_inject(tensor_bridge_t *bridge, int layer, int dim) {
    if (dim <= 0) return -1;

    if (bridge->inject_buf.data) tensor_free(bridge->inject_buf.data);
    bridge->inject_buf.data = (float *)tensor_alloc(dim * sizeof(float));
    if (!bridge->inject_buf.data) return -1;

    bridge->inject_buf.dim = dim;
    bridge->inject_layer = layer;
    bridge->inject_buf.valid = 0;

    if (bridge->mode == BRIDGE_MODE_CAPTURE)
        bridge->mode = BRIDGE_MODE_BOTH;
    else
        bridge->mode = BRIDGE_MODE_INJECT;
    return 0;
}

int tensor_bridge_set_projection(tensor_bridge_t *bridge,
                                  const float *weight, int src_dim, int dst_dim) {
    if (!weight || src_dim <= 0 || dst_dim <= 0) return -1;

    if (bridge->proj_weight) tensor_free(bridge->proj_weight);
    uint64_t size = (uint64_t)dst_dim * src_dim * sizeof(float);
    bridge->proj_weight = (float *)tensor_alloc(size);
    if (!bridge->proj_weight) return -1;

    kmemcpy(bridge->proj_weight, weight, size);
    bridge->proj_src_dim = src_dim;
    bridge->proj_dst_dim = dst_dim;
    bridge->proj_type = BRIDGE_PROJ_LINEAR;
    return 0;
}

void tensor_bridge_capture(tensor_bridge_t *bridge,
                           const float *hidden, int dim, int pos) {
    if (!bridge || !hidden) return;
    if (!(bridge->mode & BRIDGE_MODE_CAPTURE)) return;
    if (!bridge->capture_buf.data) return;

    int copy_dim = dim < bridge->capture_buf.dim ? dim : bridge->capture_buf.dim;
    kmemcpy(bridge->capture_buf.data, hidden, copy_dim * sizeof(float));

    /* Zero-pad if capture buffer is larger */
    if (copy_dim < bridge->capture_buf.dim)
        kmemset(bridge->capture_buf.data + copy_dim, 0,
                (bridge->capture_buf.dim - copy_dim) * sizeof(float));

    bridge->capture_buf.layer = bridge->capture_layer;
    bridge->capture_buf.pos = pos;
    bridge->capture_buf.valid = 1;
    bridge->capture_buf.seq_id++;
    bridge->captures++;
}

int tensor_bridge_inject(tensor_bridge_t *bridge,
                         float *hidden, int dim, int pos) {
    if (!bridge || !hidden) return 0;
    if (!(bridge->mode & BRIDGE_MODE_INJECT)) return 0;
    if (!bridge->inject_buf.data || !bridge->inject_buf.valid) return 0;

    (void)pos; /* pos available for future sequence-aware injection */

    int src_dim = bridge->inject_buf.dim;

    if (bridge->proj_type == BRIDGE_PROJ_LINEAR && bridge->proj_weight) {
        /* Apply linear projection: hidden[dim] = W[dim × src_dim] · inject[src_dim] */
        const float *w = bridge->proj_weight;
        const float *x = bridge->inject_buf.data;
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < src_dim; j++)
                sum += w[i * src_dim + j] * x[j];
            hidden[i] = sum;
        }
    } else if (bridge->proj_type == BRIDGE_PROJ_TRUNCATE) {
        int copy_dim = dim < src_dim ? dim : src_dim;
        kmemcpy(hidden, bridge->inject_buf.data, copy_dim * sizeof(float));
    } else if (bridge->proj_type == BRIDGE_PROJ_PAD) {
        int copy_dim = dim < src_dim ? dim : src_dim;
        kmemcpy(hidden, bridge->inject_buf.data, copy_dim * sizeof(float));
        if (copy_dim < dim)
            kmemset(hidden + copy_dim, 0, (dim - copy_dim) * sizeof(float));
    } else {
        /* No projection: dims must match */
        if (src_dim != dim) return 0;
        kmemcpy(hidden, bridge->inject_buf.data, dim * sizeof(float));
    }

    bridge->injections++;
    return 1;
}

int tensor_bridge_has_capture(const tensor_bridge_t *bridge) {
    return bridge && bridge->capture_buf.valid;
}

const float *tensor_bridge_get_capture(const tensor_bridge_t *bridge,
                                        int *out_dim, int *out_layer) {
    if (!bridge || !bridge->capture_buf.valid) return (const float *)0;
    if (out_dim) *out_dim = bridge->capture_buf.dim;
    if (out_layer) *out_layer = bridge->capture_buf.layer;
    return bridge->capture_buf.data;
}

int tensor_bridge_transfer(tensor_bridge_t *dst, const tensor_bridge_t *src) {
    if (!dst || !src) return -1;
    if (!src->capture_buf.valid) return -1;
    if (!dst->inject_buf.data) return -1;

    int src_dim = src->capture_buf.dim;
    int dst_dim = dst->inject_buf.dim;

    if (dst->proj_type == BRIDGE_PROJ_LINEAR && dst->proj_weight) {
        /* Project: inject[dst_dim] = W[dst_dim × src_dim] · capture[src_dim] */
        const float *w = dst->proj_weight;
        const float *x = src->capture_buf.data;
        for (int i = 0; i < dst_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < src_dim; j++)
                sum += w[i * src_dim + j] * x[j];
            dst->inject_buf.data[i] = sum;
        }
    } else {
        int copy_dim = dst_dim < src_dim ? dst_dim : src_dim;
        kmemcpy(dst->inject_buf.data, src->capture_buf.data,
                copy_dim * sizeof(float));
        if (copy_dim < dst_dim)
            kmemset(dst->inject_buf.data + copy_dim, 0,
                    (dst_dim - copy_dim) * sizeof(float));
    }

    dst->inject_buf.layer = src->capture_buf.layer;
    dst->inject_buf.pos = src->capture_buf.pos;
    dst->inject_buf.valid = 1;
    dst->inject_buf.seq_id = src->capture_buf.seq_id;
    return 0;
}

void tensor_bridge_free(tensor_bridge_t *bridge) {
    if (!bridge) return;
    if (bridge->capture_buf.data) tensor_free(bridge->capture_buf.data);
    if (bridge->inject_buf.data) tensor_free(bridge->inject_buf.data);
    if (bridge->proj_weight) tensor_free(bridge->proj_weight);
    kmemset(bridge, 0, sizeof(*bridge));
}
