/*
 * TensorOS Tensor Injection / Hidden-State Bridge
 *
 * Enables daisy-chaining LLMs by providing a mechanism to:
 * 1. Capture hidden states (activations) at any transformer layer
 * 2. Inject hidden states into the forward pass at any layer
 * 3. Bridge between models with different dimensions via learned projections
 *
 * Use cases:
 * - Speculative decoding: draft model → verify with larger model
 * - Mixture of experts: route tokens through specialized sub-models
 * - Model composition: chain encoder → decoder across separate GGUFs
 * - Distillation: capture teacher hidden states for student training
 */

#ifndef TENSOROS_TENSOR_BRIDGE_H
#define TENSOROS_TENSOR_BRIDGE_H

#include <stdint.h>

/* ─── Bridge configuration ─── */
typedef enum {
    BRIDGE_MODE_NONE    = 0, /* No bridging active */
    BRIDGE_MODE_CAPTURE = 1, /* Capture hidden state at specified layer */
    BRIDGE_MODE_INJECT  = 2, /* Inject hidden state at specified layer */
    BRIDGE_MODE_BOTH    = 3, /* Capture + inject at different layers */
} bridge_mode_t;

/* ─── Projection type for dimension mismatch ─── */
typedef enum {
    BRIDGE_PROJ_NONE     = 0, /* No projection (dims must match) */
    BRIDGE_PROJ_LINEAR   = 1, /* Linear projection W[dst_dim × src_dim] */
    BRIDGE_PROJ_TRUNCATE = 2, /* Truncate to min(src, dst) dimensions */
    BRIDGE_PROJ_PAD      = 3, /* Zero-pad to dst dimensions */
} bridge_proj_t;

/* ─── Hidden state buffer ─── */
typedef struct {
    float   *data;      /* Hidden state vector [dim] */
    int      dim;       /* Dimension of hidden state */
    int      layer;     /* Layer this was captured from (or injected to) */
    int      pos;       /* Position in sequence */
    int      valid;     /* 1 if buffer contains valid data */
    uint64_t seq_id;    /* Sequence ID for tracking */
} hidden_state_t;

/* ─── Bridge context ─── */
typedef struct {
    bridge_mode_t mode;

    /* Capture settings */
    int           capture_layer;    /* Layer index to capture after (-1 = last) */
    hidden_state_t capture_buf;     /* Captured hidden state */

    /* Injection settings */
    int           inject_layer;     /* Layer index to inject before (-1 = first) */
    hidden_state_t inject_buf;      /* Hidden state to inject */

    /* Projection (for dimension mismatch between models) */
    bridge_proj_t proj_type;
    float        *proj_weight;      /* [dst_dim × src_dim] projection matrix */
    int           proj_src_dim;
    int           proj_dst_dim;

    /* Statistics */
    uint64_t      captures;
    uint64_t      injections;
} tensor_bridge_t;

/* ─── API ─── */

/* Initialize bridge context. Call once before use. */
void tensor_bridge_init(tensor_bridge_t *bridge);

/* Configure capture: save hidden state after layer `layer` completes.
 * -1 for last layer. dim is the model's hidden dimension. */
int tensor_bridge_set_capture(tensor_bridge_t *bridge, int layer, int dim);

/* Configure injection: inject hidden state before layer `layer` starts.
 * -1 for first layer (replaces embedding). */
int tensor_bridge_set_inject(tensor_bridge_t *bridge, int layer, int dim);

/* Set linear projection for dimension mismatch.
 * weight is [dst_dim × src_dim] row-major float matrix. */
int tensor_bridge_set_projection(tensor_bridge_t *bridge,
                                  const float *weight, int src_dim, int dst_dim);

/* Capture: copy hidden state into bridge buffer.
 * Called from llm_forward_token after the specified layer. */
void tensor_bridge_capture(tensor_bridge_t *bridge,
                           const float *hidden, int dim, int pos);

/* Inject: overwrite hidden state from bridge buffer.
 * Called from llm_forward_token before the specified layer.
 * Returns 1 if injection occurred, 0 otherwise. */
int tensor_bridge_inject(tensor_bridge_t *bridge,
                         float *hidden, int dim, int pos);

/* Check if capture buffer has valid data. */
int tensor_bridge_has_capture(const tensor_bridge_t *bridge);

/* Get captured hidden state (read-only). Returns NULL if no capture. */
const float *tensor_bridge_get_capture(const tensor_bridge_t *bridge,
                                        int *out_dim, int *out_layer);

/* Transfer: copy capture from one bridge to another's injection buffer.
 * Applies projection if dimensions differ. */
int tensor_bridge_transfer(tensor_bridge_t *dst, const tensor_bridge_t *src);

/* Free bridge resources. */
void tensor_bridge_free(tensor_bridge_t *bridge);

#endif /* TENSOROS_TENSOR_BRIDGE_H */
