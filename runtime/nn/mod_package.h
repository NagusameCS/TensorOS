/*
 * HyperTensor Modification Packaging
 *
 * Packages model outputs (token sequences) together with edit operations,
 * allowing modifications to be applied directly in token space without
 * the lossy detokenize → re-tokenize round-trip.
 *
 * Use cases:
 *   - Code completion + inline edit suggestions
 *   - Multi-model pipelines: model A generates, model B refines in-place
 *   - Speculative decoding: draft tokens + corrections merged efficiently
 *   - Distillation: teacher output + student corrections as single package
 *
 * Package format:
 *   [header][original_tokens][edit_ops][metadata]
 *
 * Edit operations:
 *   INSERT(pos, token_ids[])   — insert tokens at position
 *   DELETE(pos, count)         — remove tokens starting at position
 *   REPLACE(pos, count, new[]) — replace span with new tokens
 *   REWRITE(start, end, new[]) — rewrite range (insert + delete)
 */

#ifndef HYPERTENSOR_MOD_PACKAGE_H
#define HYPERTENSOR_MOD_PACKAGE_H

#include <stdint.h>

/* ── Constants ─────────────────────────────────────────────────────────── */
#define MOD_PKG_MAGIC       0x4D4F4450  /* "MODP" */
#define MOD_PKG_VERSION     1
#define MOD_MAX_EDITS       256
#define MOD_MAX_TOKENS      8192
#define MOD_MAX_EDIT_TOKENS 512

/* ── Edit Operation Types ──────────────────────────────────────────────── */
typedef enum {
    MOD_EDIT_INSERT  = 1,   /* Insert tokens at position */
    MOD_EDIT_DELETE  = 2,   /* Delete N tokens at position */
    MOD_EDIT_REPLACE = 3,   /* Replace N tokens with new sequence */
} mod_edit_type_t;

/* ── Single Edit Operation ─────────────────────────────────────────────── */
typedef struct {
    mod_edit_type_t type;
    int             pos;            /* Position in original token stream */
    int             count;          /* Number of original tokens affected */
    int32_t         new_tokens[MOD_MAX_EDIT_TOKENS]; /* Replacement tokens */
    int             n_new;          /* Number of new tokens */
    float           confidence;     /* Model confidence in this edit (0-1) */
    int             source_model;   /* Which model proposed this edit */
} mod_edit_t;

/* ── Modification Package ──────────────────────────────────────────────── */
typedef struct {
    uint32_t magic;
    uint32_t version;

    /* Original token sequence (from first model) */
    int32_t  original_tokens[MOD_MAX_TOKENS];
    int      n_original;

    /* Edit operations (from second model or refinement pass) */
    mod_edit_t edits[MOD_MAX_EDITS];
    int        n_edits;

    /* Hidden state snapshot at point of divergence (optional) */
    float   *hidden_state;          /* [hidden_dim] or NULL */
    int      hidden_dim;
    int      hidden_layer;          /* Layer the state was captured from */

    /* Metadata */
    float    original_perplexity;   /* PPL of original sequence */
    float    modified_perplexity;   /* PPL after edits applied */
    uint64_t timestamp;
    int      source_model_id;       /* Model that generated original */
    int      editor_model_id;       /* Model that proposed edits */
} mod_package_t;

/* ── API ───────────────────────────────────────────────────────────────── */

/* Initialize an empty modification package. */
void mod_package_init(mod_package_t *pkg);

/* Set the original token sequence. */
int mod_package_set_original(mod_package_t *pkg,
                             const int32_t *tokens, int n_tokens);

/* Add an edit operation to the package. Returns edit index or -1 on error. */
int mod_package_add_insert(mod_package_t *pkg, int pos,
                           const int32_t *tokens, int n_tokens,
                           float confidence);

int mod_package_add_delete(mod_package_t *pkg, int pos, int count,
                           float confidence);

int mod_package_add_replace(mod_package_t *pkg, int pos, int count,
                            const int32_t *new_tokens, int n_new,
                            float confidence);

/* Apply all edits to produce the final token sequence.
 * Returns number of tokens in output, or -1 on error. */
int mod_package_apply(const mod_package_t *pkg,
                      int32_t *out_tokens, int max_out);

/* Attach a hidden state snapshot to the package. */
int mod_package_attach_hidden(mod_package_t *pkg,
                              const float *hidden, int dim, int layer);

/* Serialize package to binary buffer. Returns bytes written or -1 on error. */
int mod_package_serialize(const mod_package_t *pkg,
                          void *buf, int buf_size);

/* Deserialize package from binary buffer. Returns 0 on success. */
int mod_package_deserialize(mod_package_t *pkg,
                            const void *buf, int buf_size);

/* Merge edits from a second package into the first.
 * Resolves conflicts by confidence score. */
int mod_package_merge(mod_package_t *dst, const mod_package_t *src);

/* Get statistics about the edit operations. */
void mod_package_stats(const mod_package_t *pkg,
                       int *n_inserts, int *n_deletes, int *n_replaces,
                       int *tokens_added, int *tokens_removed);

/* Free any dynamically allocated resources in the package. */
void mod_package_free(mod_package_t *pkg);

#endif /* HYPERTENSOR_MOD_PACKAGE_H */
