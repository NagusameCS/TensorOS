/*
 * TensorOS BLT (Bytes, Letters, Tokens) Tokenizer
 *
 * Hybrid tokenizer supporting multiple granularity levels:
 * - Byte-level: raw UTF-8 bytes (256 tokens), no UNK possible
 * - Letter-level: Unicode codepoints (for char-level models)
 * - Token-level: BPE/SentencePiece subword tokens (standard)
 *
 * Key features:
 * - Fallback chain: try token → letter → byte (never fails)
 * - Dynamic patching: encode unknown substrings at byte level
 * - Mixed-granularity sequences for robust handling of:
 *   * Code with unusual identifiers
 *   * Multilingual text with rare scripts
 *   * Binary data / hex dumps
 *   * Adversarial / corrupted inputs
 *
 * Based on Meta's BLT paper (2024): "Better Language Models via
 * Bytes, Letters, and Tokens" — but adapted for inference-only use.
 */

#ifndef TENSOROS_BLT_H
#define TENSOROS_BLT_H

#include <stdint.h>

/* ─── Token range allocation ─── */
/* Standard vocab: tokens 0..vocab_size-1 (from GGUF)
 * Byte tokens:    256000..256255 (raw bytes 0x00..0xFF)
 * Letter tokens:  257000..367000 (Unicode BMP codepoints)
 * Control tokens: 999000+ (BOS, EOS, PAD, etc.) */
#define BLT_BYTE_BASE    256000
#define BLT_LETTER_BASE  257000
#define BLT_CONTROL_BASE 999000

/* ─── Granularity levels ─── */
typedef enum {
    BLT_LEVEL_TOKEN  = 0, /* Standard BPE/SPM subword */
    BLT_LEVEL_LETTER = 1, /* Unicode codepoint */
    BLT_LEVEL_BYTE   = 2, /* Raw byte */
} blt_level_t;

/* ─── BLT token ─── */
typedef struct {
    int        id;    /* Token ID (may be in byte/letter/token range) */
    blt_level_t level; /* What granularity this token is at */
    uint8_t    byte;  /* For byte-level: the actual byte value */
} blt_token_t;

/* ─── BLT context ─── */
typedef struct {
    /* Underlying vocabulary (from GGUF/SPM) */
    const char **vocab;     /* Token strings */
    const float *scores;    /* Token scores (for BPE merge priority) */
    int          vocab_size;

    /* Settings */
    int          enable_byte_fallback; /* 1 = fall back to bytes for unknown */
    int          enable_letter_level;  /* 1 = try letter level before bytes */

    /* Statistics */
    uint64_t     tokens_encoded;
    uint64_t     byte_fallbacks;
    uint64_t     letter_fallbacks;
} blt_ctx_t;

/* ─── API ─── */

/* Initialize BLT context with vocabulary from loaded model.
 * vocab/scores arrays must remain valid for lifetime of ctx. */
void blt_init(blt_ctx_t *ctx, const char **vocab, const float *scores,
              int vocab_size);

/* Encode text to BLT token sequence.
 * Returns number of tokens written to out_tokens.
 * Uses fallback chain: token → letter → byte for any unrecognized input. */
int blt_encode(const blt_ctx_t *ctx, const char *text, int text_len,
               blt_token_t *out_tokens, int max_tokens);

/* Decode BLT token sequence back to text.
 * Returns number of bytes written to out_text. */
int blt_decode(const blt_ctx_t *ctx, const blt_token_t *tokens,
               int n_tokens, char *out_text, int max_len);

/* Check if a token ID is in the byte range */
static inline int blt_is_byte_token(int id) {
    return id >= BLT_BYTE_BASE && id < BLT_BYTE_BASE + 256;
}

/* Check if a token ID is in the letter range */
static inline int blt_is_letter_token(int id) {
    return id >= BLT_LETTER_BASE && id < BLT_LETTER_BASE + 110000;
}

/* Get byte value from a byte token */
static inline uint8_t blt_byte_value(int id) {
    return (uint8_t)(id - BLT_BYTE_BASE);
}

/* Get unicode codepoint from a letter token */
static inline int blt_letter_codepoint(int id) {
    return id - BLT_LETTER_BASE;
}

/* Create a byte token */
static inline blt_token_t blt_make_byte(uint8_t b) {
    blt_token_t t;
    t.id = BLT_BYTE_BASE + b;
    t.level = BLT_LEVEL_BYTE;
    t.byte = b;
    return t;
}

/* Create a token-level token */
static inline blt_token_t blt_make_token(int id) {
    blt_token_t t;
    t.id = id;
    t.level = BLT_LEVEL_TOKEN;
    t.byte = 0;
    return t;
}

#endif /* TENSOROS_BLT_H */
