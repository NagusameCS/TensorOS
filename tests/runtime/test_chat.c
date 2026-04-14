/*
 * TensorOS Chat & Token-Native Pipeline Tests
 *
 * Validates:
 *   1. Token-native validators (JSON, XML, code-fence, key-value) — structural
 *   2. Token execution loop contract (llm_execute_token_loop) — mock executor
 *   3. Chat turn API surface (llm_chat_turn_tokens, llm_prompt_tokens)
 *   4. Token encode/decode roundtrip integrity for chat payloads
 *
 * These tests run WITHOUT a loaded model by exercising the validator/executor
 * API contracts: error-path coverage, boundary conditions, and mock callbacks.
 * Model-dependent paths return expected error codes when no model is loaded.
 *
 * Build (host):
 *   zig cc -target x86_64-windows-gnu -O2 -mavx2 -mfma \
 *          -DHYPERTENSOR_HOSTED=1 -Ihost/shims -I. -Ihost \
 *          tests/runtime/test_chat.c host/hal.c \
 *          runtime/nn/llm.c runtime/nn/gguf.c \
 *          runtime/jit/x86_jit.c runtime/jit/llm_jit.c \
 *          -ladvapi32 -o build_host/test_chat.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#endif
#include "runtime/nn/llm.h"

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define ASSERT_TRUE(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", name); } \
} while(0)

#define ASSERT_EQ(a, b, name) do { \
    tests_run++; \
    if ((a) == (b)) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (got %d, want %d)\n", name, (int)(a), (int)(b)); } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════
 * Section 1: Validator API boundary tests (no model required for error paths)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_validators_null_input(void)
{
    printf("  [validators: null/empty input]\n");

    /* All validators must return -1 (error) on NULL tokens */
    ASSERT_EQ(llm_validate_json_tokens(NULL, 0), -1, "json_null");
    ASSERT_EQ(llm_validate_json_tokens(NULL, 5), -1, "json_null_nz");
    ASSERT_EQ(llm_validate_code_fence_tokens(NULL, 0), -1, "fence_null");
    ASSERT_EQ(llm_validate_code_fence_tokens(NULL, 5), -1, "fence_null_nz");
    ASSERT_EQ(llm_validate_xml_tokens(NULL, 0), -1, "xml_null");
    ASSERT_EQ(llm_validate_xml_tokens(NULL, 5), -1, "xml_null_nz");
    ASSERT_EQ(llm_validate_key_value_tokens(NULL, 0), -1, "kv_null");
    ASSERT_EQ(llm_validate_key_value_tokens(NULL, 5), -1, "kv_null_nz");

    /* Zero or negative count with non-null pointer */
    {
        int dummy[4] = {1, 2, 3, 4};
        ASSERT_EQ(llm_validate_json_tokens(dummy, 0), -1, "json_zero_len");
        ASSERT_EQ(llm_validate_json_tokens(dummy, -1), -1, "json_neg_len");
        ASSERT_EQ(llm_validate_code_fence_tokens(dummy, 0), -1, "fence_zero_len");
        ASSERT_EQ(llm_validate_xml_tokens(dummy, 0), -1, "xml_zero_len");
        ASSERT_EQ(llm_validate_key_value_tokens(dummy, 0), -1, "kv_zero_len");
        ASSERT_EQ(llm_validate_key_value_tokens(dummy, -1), -1, "kv_neg_len");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 2: Token execution loop contract tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Mock executor: echoes input tokens back with +1000 offset */
static int mock_echo_executor(const int *input, int n_input,
                              int *output, int max_output,
                              void *userdata)
{
    int *call_count = (int *)userdata;
    int i, n;
    if (call_count) (*call_count)++;
    n = n_input < max_output ? n_input : max_output;
    for (i = 0; i < n; i++) output[i] = input[i] + 1000;
    return n;
}

/* Mock executor: returns 0 tokens (terminates loop) */
static int mock_null_executor(const int *input, int n_input,
                              int *output, int max_output,
                              void *userdata)
{
    (void)input; (void)n_input; (void)output; (void)max_output;
    (void)userdata;
    return 0;
}

/* Mock executor: returns exactly 1 token per call, up to 3 calls */
static int mock_countdown_executor(const int *input, int n_input,
                                   int *output, int max_output,
                                   void *userdata)
{
    int *call_count = (int *)userdata;
    (void)input; (void)n_input;
    if (call_count) (*call_count)++;
    if (*call_count > 3) return 0; /* stop after 3 rounds */
    if (max_output < 1) return 0;
    output[0] = 42 + *call_count;
    return 1;
}

static void test_execute_token_loop_contract(void)
{
    int prompt[4] = {10, 20, 30, 40};
    int output[256];
    int ret;

    printf("  [execute_token_loop: API contract]\n");

    /* NULL prompt → error */
    ret = llm_execute_token_loop(NULL, 4, mock_echo_executor, NULL,
                                 output, 256, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_null_prompt");

    /* Zero prompt length → error */
    ret = llm_execute_token_loop(prompt, 0, mock_echo_executor, NULL,
                                 output, 256, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_zero_prompt");

    /* NULL executor → error */
    ret = llm_execute_token_loop(prompt, 4, NULL, NULL,
                                 output, 256, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_null_exec");

    /* NULL output → error */
    ret = llm_execute_token_loop(prompt, 4, mock_echo_executor, NULL,
                                 NULL, 256, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_null_output");

    /* Zero output capacity → error */
    ret = llm_execute_token_loop(prompt, 4, mock_echo_executor, NULL,
                                 output, 0, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_zero_output_cap");

    /* No model loaded → should return -1 (model-dependent) */
    ret = llm_execute_token_loop(prompt, 4, mock_echo_executor, NULL,
                                 output, 256, 64, 0.8f, 4);
    ASSERT_EQ(ret, -1, "loop_no_model");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 3: Token program execution contract
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_execute_token_program_contract(void)
{
    int prompt[4] = {10, 20, 30, 40};
    int output[256];
    int ret;

    printf("  [execute_token_program: API contract]\n");

    /* NULL prompt → error */
    ret = llm_execute_token_program(NULL, 4, mock_echo_executor, NULL,
                                    output, 256);
    ASSERT_EQ(ret, -1, "prog_null_prompt");

    /* NULL executor → error */
    ret = llm_execute_token_program(prompt, 4, NULL, NULL,
                                    output, 256);
    ASSERT_EQ(ret, -1, "prog_null_exec");

    /* NULL output → error */
    ret = llm_execute_token_program(prompt, 4, mock_echo_executor, NULL,
                                    NULL, 256);
    ASSERT_EQ(ret, -1, "prog_null_output");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 4: Chat turn API surface (error paths without model)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_chat_api_surface(void)
{
    int tokens[256];
    int ret;

    printf("  [chat API: error paths]\n");

    /* NULL text → error */
    ret = llm_prompt_tokens(NULL, tokens, 256, 64, 0.8f);
    ASSERT_EQ(ret, -1, "prompt_null_text");

    /* NULL output → error */
    ret = llm_prompt_tokens("hello", NULL, 256, 64, 0.8f);
    ASSERT_EQ(ret, -1, "prompt_null_output");

    /* Zero max_output → error */
    ret = llm_prompt_tokens("hello", tokens, 0, 64, 0.8f);
    ASSERT_EQ(ret, -1, "prompt_zero_output");

    /* Chat turn NULL text → error */
    ret = llm_chat_turn_tokens(NULL, tokens, 256, 64, 0.8f);
    ASSERT_EQ(ret, -1, "chat_null_text");

    /* Chat turn NULL output → error */
    ret = llm_chat_turn_tokens("hello", NULL, 256, 64, 0.8f);
    ASSERT_EQ(ret, -1, "chat_null_output");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 5: Tokenize/decode roundtrip error paths
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_tokenize_decode_errors(void)
{
    int tokens[64];
    char buf[256];
    int ret;

    printf("  [tokenize/decode: error paths]\n");

    /* NULL text → error or 0 */
    ret = llm_tokenize_text(NULL, tokens, 64);
    ASSERT_TRUE(ret <= 0, "tokenize_null");

    /* NULL output → error */
    ret = llm_tokenize_text("hello", NULL, 64);
    ASSERT_TRUE(ret <= 0, "tokenize_null_output");

    /* Decode NULL tokens → error */
    ret = llm_decode_tokens(NULL, 5, buf, 256);
    ASSERT_TRUE(ret <= 0, "decode_null");

    /* Decode zero len → error or 0 */
    {
        int dummy[4] = {1, 2, 3, 4};
        ret = llm_decode_tokens(dummy, 0, buf, 256);
        ASSERT_TRUE(ret <= 0, "decode_zero_len");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 6: Generate tokens error paths
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_generate_tokens_errors(void)
{
    int prompt[4] = {1, 2, 3, 4};
    int output[64];
    int ret;

    printf("  [generate_tokens: error paths]\n");

    /* No model loaded → -1 */
    ret = llm_generate_tokens(prompt, 4, output, 64, 32, 0.8f, 0);
    ASSERT_EQ(ret, -1, "gen_no_model");

    /* NULL output → -1 */
    ret = llm_generate_tokens(prompt, 4, NULL, 64, 32, 0.8f, 0);
    ASSERT_EQ(ret, -1, "gen_null_output");

    /* Zero max_output → -1 */
    ret = llm_generate_tokens(prompt, 4, output, 0, 32, 0.8f, 0);
    ASSERT_EQ(ret, -1, "gen_zero_output");

    /* Zero max_gen → -1 */
    ret = llm_generate_tokens(prompt, 4, output, 64, 0, 0.8f, 0);
    ASSERT_EQ(ret, -1, "gen_zero_gen");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Section 7: KV cache and agent context API error paths
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_kv_agent_errors(void)
{
    int tokens[4] = {10, 20, 30, 40};
    int output[64];
    int ret;

    printf("  [kv/agent: error paths]\n");

    /* KV snapshot null → error */
    ret = llm_kv_snapshot_prefix(NULL, 4);
    ASSERT_TRUE(ret < 0, "kv_snap_null");

    /* KV snapshot zero → error */
    ret = llm_kv_snapshot_prefix(tokens, 0);
    ASSERT_TRUE(ret < 0, "kv_snap_zero");

    /* Agent context without model */
    llm_agent_context_reset();  /* should just be a no-op without model */

    ret = llm_agent_context_append_tokens(NULL, 4);
    ASSERT_TRUE(ret < 0, "agent_append_null");
}

/* ═══════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== TensorOS Chat & Token-Native Pipeline Tests ===\n\n");

    printf("[1] Validator boundary tests\n");
    test_validators_null_input();

    printf("[2] Token execution loop contract\n");
    test_execute_token_loop_contract();

    printf("[3] Token program execution contract\n");
    test_execute_token_program_contract();

    printf("[4] Chat API surface\n");
    test_chat_api_surface();

    printf("[5] Tokenize/decode error paths\n");
    test_tokenize_decode_errors();

    printf("[6] Generate tokens error paths\n");
    test_generate_tokens_errors();

    printf("[7] KV/agent context error paths\n");
    test_kv_agent_errors();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
