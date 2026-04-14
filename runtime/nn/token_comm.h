/*
 * HyperTensor Token-Space Communication
 *
 * Enables two or more LLMs to communicate directly in token space —
 * exchanging raw logit distributions, token sequences, and attention
 * patterns without converting to/from text.
 *
 * This implements "distributional structure" communication where:
 *   - Model A outputs a probability distribution over the vocabulary
 *   - Model B consumes that distribution as soft input (not argmax)
 *   - Both models share a token channel for bidirectional exchange
 *
 * Communication modes:
 *   HARD: Exchange discrete token IDs (standard pipeline)
 *   SOFT: Exchange full logit distributions (distributional)
 *   MIXED: Exchange top-K logits + hidden states
 *   HYBRID: Combine distributional + hidden-state bridge
 *
 * Architecture:
 *   ┌──────────┐  logits/tokens  ┌──────────┐
 *   │  Model A  │ ──────────────→ │  Model B  │
 *   │  (sender) │ ←────────────── │ (receiver)│
 *   └──────────┘  feedback/ctrl  └──────────┘
 *       ↕                             ↕
 *   [token_channel_t] ←──shared──→ [token_channel_t]
 */

#ifndef HYPERTENSOR_TOKEN_COMM_H
#define HYPERTENSOR_TOKEN_COMM_H

#include <stdint.h>

/* ── Constants ─────────────────────────────────────────────────────────── */
#define TOKEN_COMM_MAX_VOCAB   262144
#define TOKEN_COMM_MAX_TOPK    64
#define TOKEN_COMM_RING_SIZE   4096    /* Ring buffer for token history */

/* ── Communication Modes ───────────────────────────────────────────────── */
typedef enum {
    TOKEN_COMM_HARD   = 0,  /* Exchange discrete token IDs only */
    TOKEN_COMM_SOFT   = 1,  /* Exchange full logit distributions */
    TOKEN_COMM_MIXED  = 2,  /* Top-K logits + associated hidden state */
    TOKEN_COMM_HYBRID = 3,  /* Distributional + tensor bridge */
} token_comm_mode_t;

/* ── Top-K Entry ───────────────────────────────────────────────────────── */
typedef struct {
    int32_t token_id;
    float   logit;
    float   prob;      /* After softmax */
} topk_entry_t;

/* ── Token Message (single exchange unit) ──────────────────────────────── */
typedef struct {
    /* Hard tokens (always available) */
    int32_t token;             /* Selected token (argmax or sampled) */
    int     position;          /* Sequence position */

    /* Soft distribution (if mode != HARD) */
    float  *logits;            /* Full vocab logits [vocab_size] or NULL */
    int     vocab_size;

    /* Top-K summary (if mode == MIXED) */
    topk_entry_t topk[TOKEN_COMM_MAX_TOPK];
    int          n_topk;

    /* Entropy / uncertainty metrics */
    float entropy;             /* Shannon entropy of distribution */
    float max_prob;            /* Probability of selected token */
    float perplexity;          /* exp(cross-entropy) */

    /* Metadata */
    uint64_t seq_id;           /* Sequence identifier */
    int      model_id;         /* Source model */
    uint64_t timestamp_us;     /* Microsecond timestamp */
} token_message_t;

/* ── Token Channel (shared communication pipe) ─────────────────────────── */
typedef struct {
    token_comm_mode_t mode;

    /* Ring buffer of recent token messages */
    token_message_t ring[TOKEN_COMM_RING_SIZE];
    int             ring_head;
    int             ring_tail;
    int             ring_count;

    /* Vocab mapping (for models with different tokenizers) */
    int32_t *vocab_map_a_to_b;   /* Map model_a token → model_b token */
    int32_t *vocab_map_b_to_a;   /* Map model_b token → model_a token */
    int      vocab_a_size;
    int      vocab_b_size;
    int      has_vocab_map;

    /* Logit buffer for soft communication */
    float   *shared_logits;      /* [max(vocab_a, vocab_b)] scratch buffer */

    /* Attention feedback channel */
    float   *attention_feedback; /* [max_seq] attention weights from receiver */
    int      feedback_len;

    /* Statistics */
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t soft_exchanges;
    uint64_t hard_exchanges;
    float    avg_entropy;
    float    agreement_rate;     /* How often both models agree on top token */
} token_channel_t;

/* ── API ───────────────────────────────────────────────────────────────── */

/* Initialize a token communication channel.
 * vocab_a/vocab_b are the vocab sizes of the two models. */
int token_comm_init(token_channel_t *ch, token_comm_mode_t mode,
                    int vocab_a, int vocab_b);

/* Build a vocabulary mapping between two models' tokenizers.
 * vocab_a/vocab_b are arrays of (string, length) pairs.
 * This enables soft communication even when tokenizers differ. */
int token_comm_build_vocab_map(token_channel_t *ch,
                               const char **vocab_a_strs, const int *vocab_a_lens,
                               int n_vocab_a,
                               const char **vocab_b_strs, const int *vocab_b_lens,
                               int n_vocab_b);

/* Send a hard token (discrete ID). */
int token_comm_send_token(token_channel_t *ch, int32_t token,
                          int position, int model_id);

/* Send a full logit distribution (soft communication).
 * logits is [vocab_size] raw logits before softmax. */
int token_comm_send_logits(token_channel_t *ch, const float *logits,
                           int vocab_size, int position, int model_id);

/* Send top-K logits (mixed mode). */
int token_comm_send_topk(token_channel_t *ch,
                         const topk_entry_t *topk, int k,
                         int32_t selected_token, int position, int model_id);

/* Receive the most recent token message from the channel.
 * Returns 1 if a message was available, 0 if empty. */
int token_comm_receive(token_channel_t *ch, token_message_t *out);

/* Peek at the channel without consuming (non-destructive read). */
int token_comm_peek(const token_channel_t *ch, token_message_t *out);

/* Remap logits from model A's vocabulary to model B's vocabulary.
 * out_logits[vocab_b] = mapped from in_logits[vocab_a]. */
int token_comm_remap_logits(const token_channel_t *ch,
                            float *out_logits, int out_vocab,
                            const float *in_logits, int in_vocab,
                            int direction);  /* 0=A→B, 1=B→A */

/* Send attention feedback (receiver → sender).
 * Allows the receiver to influence which tokens the sender focuses on. */
int token_comm_send_feedback(token_channel_t *ch,
                             const float *attention_weights, int seq_len);

/* Get attention feedback from the channel. */
int token_comm_get_feedback(const token_channel_t *ch,
                            float *out_weights, int max_len);

/* Compute agreement between two models over the channel history.
 * Returns fraction of positions where both models selected same top token. */
float token_comm_agreement(const token_channel_t *ch);

/* Get channel statistics. */
void token_comm_stats(const token_channel_t *ch,
                      uint64_t *sent, uint64_t *received,
                      float *avg_entropy, float *agreement);

/* Free channel resources. */
void token_comm_free(token_channel_t *ch);

#endif /* HYPERTENSOR_TOKEN_COMM_H */
