/*
 * TensorOS BLT (Bytes, Letters, Tokens) Tokenizer
 *
 * Hybrid tokenizer with byte-level fallback for robust text handling.
 */

#include "runtime/nn/blt.h"

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * String helpers
 * ════════════════════════════════════════════════════════════════════════ */

static int blt_strlen(const char *s) {
    int n = 0; while (s[n]) n++;
    return n;
}

static int blt_strncmp(const char *a, const char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return (unsigned char)a[i] - (unsigned char)b[i];
        if (a[i] == 0) return 0;
    }
    return 0;
}

/* Decode UTF-8 codepoint from bytes. Returns bytes consumed (1-4). */
static int blt_utf8_decode(const uint8_t *p, int remaining, int *out_cp) {
    if (remaining <= 0) { *out_cp = 0; return 0; }
    uint8_t b = p[0];
    if (b < 0x80) { *out_cp = b; return 1; }
    if ((b & 0xE0) == 0xC0 && remaining >= 2) {
        *out_cp = ((b & 0x1F) << 6) | (p[1] & 0x3F);
        return 2;
    }
    if ((b & 0xF0) == 0xE0 && remaining >= 3) {
        *out_cp = ((b & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        return 3;
    }
    if ((b & 0xF8) == 0xF0 && remaining >= 4) {
        *out_cp = ((b & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
                  ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        return 4;
    }
    /* Invalid UTF-8: treat as raw byte */
    *out_cp = b;
    return 1;
}

/* Encode UTF-8 codepoint to bytes. Returns bytes written (1-4). */
static int blt_utf8_encode(int cp, char *out, int max) {
    if (cp < 0x80 && max >= 1) {
        out[0] = (char)cp; return 1;
    }
    if (cp < 0x800 && max >= 2) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000 && max >= 3) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp < 0x110000 && max >= 4) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════════════ */

void blt_init(blt_ctx_t *ctx, const char **vocab, const float *scores,
              int vocab_size) {
    kmemset(ctx, 0, sizeof(*ctx));
    ctx->vocab = vocab;
    ctx->scores = scores;
    ctx->vocab_size = vocab_size;
    ctx->enable_byte_fallback = 1;
    ctx->enable_letter_level = 0; /* disabled by default */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Greedy longest-match tokenizer with byte fallback
 * ════════════════════════════════════════════════════════════════════════ */

/* Find the longest vocabulary token matching at position p in text.
 * Returns token ID, or -1 if no match. Sets *match_len to bytes consumed. */
static int blt_find_longest_token(const blt_ctx_t *ctx,
                                   const char *text, int remaining,
                                   int *match_len) {
    int best_id = -1;
    int best_len = 0;

    for (int i = 0; i < ctx->vocab_size; i++) {
        const char *tok = ctx->vocab[i];
        if (!tok) continue;
        int tlen = blt_strlen(tok);
        if (tlen > remaining || tlen <= best_len) continue;
        if (blt_strncmp(text, tok, tlen) == 0) {
            best_id = i;
            best_len = tlen;
        }
    }

    *match_len = best_len;
    return best_id;
}

int blt_encode(const blt_ctx_t *ctx, const char *text, int text_len,
               blt_token_t *out_tokens, int max_tokens) {
    if (!ctx || !text || !out_tokens) return 0;
    if (text_len <= 0) text_len = blt_strlen(text);

    int pos = 0;
    int n_out = 0;
    blt_ctx_t *mctx = (blt_ctx_t *)ctx; /* for stats */

    while (pos < text_len && n_out < max_tokens) {
        int match_len = 0;
        int tok_id = blt_find_longest_token(ctx, text + pos, text_len - pos,
                                             &match_len);

        if (tok_id >= 0 && match_len > 0) {
            /* Standard token match */
            out_tokens[n_out++] = blt_make_token(tok_id);
            pos += match_len;
            mctx->tokens_encoded++;
        } else if (ctx->enable_byte_fallback) {
            /* Byte-level fallback */
            uint8_t b = (uint8_t)text[pos];
            out_tokens[n_out++] = blt_make_byte(b);
            pos++;
            mctx->byte_fallbacks++;
        } else {
            /* Skip byte if no fallback */
            pos++;
        }
    }

    return n_out;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Decoding
 * ════════════════════════════════════════════════════════════════════════ */

int blt_decode(const blt_ctx_t *ctx, const blt_token_t *tokens,
               int n_tokens, char *out_text, int max_len) {
    if (!ctx || !tokens || !out_text || max_len <= 0) return 0;

    int pos = 0;
    for (int i = 0; i < n_tokens && pos < max_len - 1; i++) {
        if (tokens[i].level == BLT_LEVEL_BYTE) {
            out_text[pos++] = (char)tokens[i].byte;
        } else if (tokens[i].level == BLT_LEVEL_LETTER) {
            int cp = blt_letter_codepoint(tokens[i].id);
            int wrote = blt_utf8_encode(cp, out_text + pos, max_len - pos - 1);
            pos += wrote;
        } else {
            /* Standard token: look up in vocab */
            int id = tokens[i].id;
            if (id >= 0 && id < ctx->vocab_size && ctx->vocab[id]) {
                const char *tok = ctx->vocab[id];
                int tlen = blt_strlen(tok);
                int copy = tlen < (max_len - pos - 1) ? tlen : (max_len - pos - 1);
                for (int j = 0; j < copy; j++)
                    out_text[pos++] = tok[j];
            }
        }
    }
    out_text[pos] = '\0';
    return pos;
}
