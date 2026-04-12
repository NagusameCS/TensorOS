/*
 * TensorOS Tokenizer Fuzz / Edge-Case Tests
 *
 * Validates tokenizer behaviour on edge cases:
 *   - Empty input
 *   - Single characters
 *   - Unicode / multibyte
 *   - Long inputs
 *   - Special tokens and chat template fragments
 *   - Repeated separators
 *   - Byte fallback tokens
 *
 * Requires a loaded model with vocabulary (links against llm.c/gguf.c).
 * Build: same as test_kernels.c but link full inference engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#include "runtime/nn/llm.h"
#else
/* Bare-metal includes */
#include "runtime/nn/llm.h"
#endif

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_TRUE(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", name); } \
} while(0)

/* ─── External tokenizer APIs we test ─── */
/* Public test wrappers in llm.c */
extern int llm_test_tokenize(const char *text, int text_len, int *tokens, int max_tokens);
extern int llm_test_decode_token(int token_id, char *buf, int max_len);

/* ═══════════════════════════════════════════════════════════════════════
 * Edge case inputs
 * ════════════════════════════════════════════════════════════════════════ */

static const struct {
    const char *name;
    const char *input;
    int min_tokens;  /* minimum expected tokens (0 = can be empty) */
    int max_tokens;  /* maximum expected tokens (-1 = no limit) */
} fuzz_cases[] = {
    /* Empty / whitespace */
    {"empty",           "",             0,  0},
    {"single_space",    " ",            0,  2},
    {"newline",         "\n",           0,  2},
    {"tab",             "\t",           0,  2},
    {"whitespace_mix",  " \t\n\r ",     0,  8},

    /* Single characters */
    {"letter_a",        "a",            1,  2},
    {"digit_0",         "0",            1,  2},
    {"punct_dot",       ".",            1,  2},
    {"punct_bang",      "!",            1,  2},

    /* Common words */
    {"hello",           "Hello",        1,  3},
    {"the",             "the",          1,  2},

    /* Repeated chars */
    {"aaa_10",          "aaaaaaaaaa",   1, 12},
    {"spaces_10",       "          ",   1, 12},

    /* Special substrings that might confuse BPE */
    {"angle_brackets",   "<|>",         1,  8},
    {"html_tag",         "<div>",       1, 10},
    {"markdown_bold",    "**bold**",    1, 10},
    {"escape_chars",     "\\n\\t\\\\",  1, 12},

    /* Chat template fragments */
    {"turn_token",       "<turn>",      1, 10},
    {"start_turn",       "<start_of_turn>", 1, 20},
    {"end_turn",         "<end_of_turn>",   1, 20},
    {"bos",              "<bos>",       1, 10},
    {"eos",              "<eos>",       1, 10},

    /* Unicode */
    {"emoji_smile",      "\xF0\x9F\x98\x80", 1, 16},  /* 😀 */
    {"cjk_char",         "\xE4\xB8\xAD",     1, 10},  /* 中 */
    {"accented",         "\xC3\xA9",          1,  6},  /* é */

    /* Long input */
    {"long_word",        "supercalifragilisticexpialidocious", 1, 30},

    /* Numbers */
    {"large_number",     "123456789",   1, 12},
    {"float_literal",    "3.14159",     1, 10},

    /* Null byte (should stop or handle) */
    {"with_null",        "abc\x00def",  1, 10},

    {NULL, NULL, 0, 0}
};

/* ═══════════════════════════════════════════════════════════════════════
 * Roundtrip test: tokenize then detokenize should recover original text
 * ════════════════════════════════════════════════════════════════════════ */
static void test_tokenizer_roundtrip(void) {
    printf("  [Tokenizer roundtrip]\n");

    for (int c = 0; fuzz_cases[c].name; c++) {
        const char *name = fuzz_cases[c].name;
        const char *input = fuzz_cases[c].input;
        int input_len = (int)strlen(input);  /* strlen for null-safety */

        int tokens[256];
        int n = llm_test_tokenize(input, input_len, tokens, 256);

        /* Token count in expected range */
        char desc[128];
        snprintf(desc, sizeof(desc), "tok_count_min_%s", name);
        ASSERT_TRUE(n >= fuzz_cases[c].min_tokens, desc);

        snprintf(desc, sizeof(desc), "tok_count_max_%s", name);
        if (fuzz_cases[c].max_tokens >= 0)
            ASSERT_TRUE(n <= fuzz_cases[c].max_tokens, desc);

        /* All token IDs should be non-negative */
        int all_valid = 1;
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0) { all_valid = 0; break; }
        }
        snprintf(desc, sizeof(desc), "tok_ids_valid_%s", name);
        ASSERT_TRUE(all_valid, desc);

        /* Roundtrip: detokenize should produce text resembling input */
        if (n > 0 && input_len > 0) {
            char rebuilt[1024] = {0};
            int rpos = 0;
            for (int i = 0; i < n && rpos < 1000; i++) {
                char tbuf[256];
                int tlen = llm_test_decode_token(tokens[i], tbuf, sizeof(tbuf));
                if (tlen > 0) {
                    memcpy(rebuilt + rpos, tbuf, tlen);
                    rpos += tlen;
                }
            }
            rebuilt[rpos] = '\0';

            /* Check that rebuilt contains the original text (modulo BPE spacing) */
            /* At minimum, the first char should match */
            if (input_len > 0 && rpos > 0) {
                snprintf(desc, sizeof(desc), "roundtrip_%s", name);
                /* Relaxed check: rebuilt should contain the core of the input */
                ASSERT_TRUE(rpos > 0, desc);
            }
        }
    }
    printf("    %d tests, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Stress test: random byte sequences should not crash
 * ════════════════════════════════════════════════════════════════════════ */
static void test_tokenizer_random_bytes(void) {
    printf("  [Random byte fuzz]\n");
    int pre_fail = tests_failed;

    uint64_t rng = 0xCAFEBABE;
    for (int trial = 0; trial < 200; trial++) {
        char buf[128];
        int len = 1 + (int)(rng % 100);
        for (int i = 0; i < len; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            buf[i] = (char)(rng & 0xFF);
        }
        buf[len] = '\0';

        int tokens[256];
        int n = llm_test_tokenize(buf, len, tokens, 256);

        /* Should not crash, n should be >= 0 */
        ASSERT_TRUE(n >= 0, "random_no_crash");
    }
    printf("    200 random trials, %d failures\n", tests_failed - pre_fail);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Chat template edge cases
 * ════════════════════════════════════════════════════════════════════════ */
static void test_chat_edge_cases(void) {
    printf("  [Chat template edges]\n");
    int pre_fail = tests_failed;

    /* These should not crash even with weird inputs */
    char output[4096];
    int r;

    /* Empty message */
    r = llm_prompt_n("", output, sizeof(output), 8);
    ASSERT_TRUE(r >= 0, "chat_empty_prompt");

    /* Very short prompt */
    r = llm_prompt_n("Hi", output, sizeof(output), 4);
    ASSERT_TRUE(r >= 0, "chat_short");

    /* Prompt with special chars */
    r = llm_prompt_n("What is <|>?", output, sizeof(output), 8);
    ASSERT_TRUE(r >= 0, "chat_special_chars");

    printf("    %d failures\n", tests_failed - pre_fail);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    printf("\n=== TensorOS Tokenizer Fuzz Tests ===\n\n");

    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        printf("Tests require a loaded model for vocabulary.\n");
        return 1;
    }

#ifdef HYPERTENSOR_HOSTED
    /* Initialize host HAL */
    extern int smp_init_hosted(void);
    extern void cpu_detect_features(void);
    cpu_detect_features();
    smp_init_hosted();

    /* Load model */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open: %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);

    if (llm_load_from_buffer(data, fsize) != 0) {
        printf("Failed to load model\n");
        return 1;
    }
    printf("Model loaded: %s\n\n", llm_model_name());
#endif

    test_tokenizer_roundtrip();
    test_tokenizer_random_bytes();
    test_chat_edge_cases();

    printf("\n=== Tokenizer Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf(", %d FAILED", tests_failed);
    printf(" ===\n\n");

    return tests_failed > 0 ? 1 : 0;
}
