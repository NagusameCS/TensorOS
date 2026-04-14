/* =============================================================================
 * TensorOS - Neural Network Training Engine (Backpropagation)
 *
 * The most revolutionary feature in any OS: full gradient-based training
 * running in the kernel. No Python, no frameworks, no userspace — just
 * raw SSE2-accelerated backpropagation computing gradients and updating
 * weights during boot.
 *
 * This implements:
 *   - Forward pass with activation caching (for backward)
 *   - Cross-entropy and MSE loss functions
 *   - Full backward pass with chain rule gradient propagation
 *   - SGD with momentum optimizer
 *   - Mini-batch training loop
 *
 * The OS literally learns from data. During boot. On bare metal.
 * =============================================================================*/

#ifndef TENSOROS_NN_TRAIN_H
#define TENSOROS_NN_TRAIN_H

#include <stdint.h>
#include "runtime/nn/inference.h"

/* Training hyperparameters */
#define OPTIM_SGD   0
#define OPTIM_ADAM  1

typedef struct {
    float learning_rate;    /* Step size (typical: 0.01-0.1) */
    float momentum;         /* SGD momentum (0 = vanilla SGD, 0.9 = typical) */
    float weight_decay;     /* L2 regularization (0 = none) */
    float beta1;            /* Adam first moment decay (0.9) */
    float beta2;            /* Adam second moment decay (0.999) */
    float epsilon;          /* Adam numerical stability (1e-8) */
    int   optimizer;        /* OPTIM_SGD or OPTIM_ADAM */
    int   epochs;           /* Number of training passes */
    int   batch_size;       /* Samples per gradient update */
} nn_train_config_t;

/* Training state — holds gradients and optimizer state */
#define TRAIN_MAX_WEIGHTS  65536 /* Max total weights across all layers */
#define TRAIN_MAX_DIM      256   /* Max neurons per layer */

typedef struct {
    /* Gradient storage */
    float dW[NN_MAX_LAYERS][TRAIN_MAX_DIM * TRAIN_MAX_DIM]; /* Weight gradients */
    float db[NN_MAX_LAYERS][TRAIN_MAX_DIM];                  /* Bias gradients */

    /* SGD momentum / Adam first moment (m) */
    float vW[NN_MAX_LAYERS][TRAIN_MAX_DIM * TRAIN_MAX_DIM]; /* Weight velocity / m_W */
    float vb[NN_MAX_LAYERS][TRAIN_MAX_DIM];                  /* Bias velocity / m_b */

    /* Adam second moment (v) */
    float sW[NN_MAX_LAYERS][TRAIN_MAX_DIM * TRAIN_MAX_DIM]; /* Second moment weights */
    float sb[NN_MAX_LAYERS][TRAIN_MAX_DIM];                  /* Second moment biases */

    /* Cached activations from forward pass (needed for backward) */
    float activations[NN_MAX_LAYERS + 1][TRAIN_MAX_DIM];    /* Pre/post activation */
    float pre_act[NN_MAX_LAYERS][TRAIN_MAX_DIM];             /* Before activation */

    /* Backprop workspace */
    float delta[TRAIN_MAX_DIM];          /* Error signal for current layer */
    float delta_next[TRAIN_MAX_DIM];     /* Error signal for next layer back */

    /* Adam step counter */
    int t;
} nn_train_state_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Train a model on a dataset using backpropagation + SGD.
 * model: the neural network (weights will be modified in-place)
 * X: input data [num_samples × input_dim], row-major
 * Y: target data [num_samples × output_dim], row-major
 * Returns final loss value. */
float nn_train(nn_model_t *model, const float *X, const float *Y,
               int num_samples, int input_dim, int output_dim,
               const nn_train_config_t *config);

/* Run training demos during boot */
void nn_train_demos(void);

/* -------------------------------------------------------------------
 * Policy Gradient Step (REINFORCE with Baseline)
 *
 * model    — shallow policy network mapping step features → advantage
 * steps    — token-native rollout steps (llm_rollout_get_steps output)
 * returns  — discounted returns (llm_rollout_compute_returns output)
 * n_steps  — number of steps
 * lr       — policy learning rate
 *
 * Returns mean |advantage| as a diagnostic. Weights updated in-place.
 * ----------------------------------------------------------------- */

/* Forward-declare rollout step type to avoid circular include of llm.h */
#ifndef LLM_ROLLOUT_STEP_FWDDECL
#define LLM_ROLLOUT_STEP_FWDDECL
typedef struct {
    int   token_id;
    float logprob;
    float value;
    float reward;
    int   done;
} llm_rollout_step_t;
#endif

float nn_policy_gradient_step(nn_model_t *model,
                               const llm_rollout_step_t *steps,
                               const float *returns,
                               int n_steps,
                               float lr);

#endif /* TENSOROS_NN_TRAIN_H */
