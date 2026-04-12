/*
 * Test: Model metadata normalization
 * Build: zig cc -O2 -target x86_64-windows-gnu -DHYPERTENSOR_HOSTED=1 -Ihost/shims -I. -Ihost tests/runtime/test_model_meta.c runtime/nn/model_meta.c host/hal.c -o build_host/test_model_meta.exe
 */

#include <stdio.h>
#include <string.h>
#include "runtime/nn/model_meta.h"

static int tests_passed = 0, tests_failed = 0;

static void check_norm(const char *hf, const char *expected) {
    char out[128];
    int rc = model_normalize_tensor_name(hf, out, sizeof(out));
    if (rc != 0) {
        printf("FAIL: '%s' => error %d (expected '%s')\n", hf, rc, expected);
        tests_failed++;
        return;
    }
    if (strcmp(out, expected) != 0) {
        printf("FAIL: '%s' => '%s' (expected '%s')\n", hf, out, expected);
        tests_failed++;
        return;
    }
    tests_passed++;
}

static void check_format(const char *label, const unsigned char *data, int size,
                          model_format_t expected) {
    model_format_t got = model_detect_format(data, size);
    if (got != expected) {
        printf("FAIL format: %s => %d (expected %d)\n", label, got, expected);
        tests_failed++;
    } else {
        tests_passed++;
    }
}

int main(void) {
    printf("=== Model Metadata Normalization Tests ===\n\n");

    /* Global tensor names */
    check_norm("model.embed_tokens.weight", "token_embd.weight");
    check_norm("model.norm.weight", "output_norm.weight");
    check_norm("lm_head.weight", "output.weight");
    check_norm("transformer.wte.weight", "token_embd.weight");
    check_norm("transformer.ln_f.weight", "output_norm.weight");

    /* Already canonical */
    check_norm("token_embd.weight", "token_embd.weight");
    check_norm("output_norm.weight", "output_norm.weight");
    check_norm("blk.0.attn_q.weight", "blk.0.attn_q.weight");

    /* Per-layer: LLaMA style */
    check_norm("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight");
    check_norm("model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight");
    check_norm("model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight");
    check_norm("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight");
    check_norm("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight");
    check_norm("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight");
    check_norm("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight");
    check_norm("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight");
    check_norm("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight");

    /* Multi-digit layer numbers */
    check_norm("model.layers.15.self_attn.q_proj.weight", "blk.15.attn_q.weight");
    check_norm("model.layers.127.mlp.down_proj.weight", "blk.127.ffn_down.weight");

    /* GPT-2 style */
    check_norm("transformer.h.0.attn.c_attn.weight", "blk.0.attn_qkv.weight");
    check_norm("transformer.h.3.mlp.c_fc.weight", "blk.3.ffn_up.weight");
    check_norm("transformer.h.3.ln_1.weight", "blk.3.attn_norm.weight");

    /* Phi-3 fused QKV */
    check_norm("model.layers.0.self_attn.qkv_proj.weight", "blk.0.attn_qkv.weight");

    /* Bias variants */
    check_norm("model.layers.5.self_attn.q_proj.bias", "blk.5.attn_q.bias");
    check_norm("model.norm.bias", "output_norm.bias");

    /* Format detection */
    unsigned char gguf_magic[] = {'G','G','U','F', 0,0,0,0};
    check_format("gguf", gguf_magic, 8, MODEL_FMT_GGUF);

    unsigned char st_magic[32] = {0x10,0,0,0, 0,0,0,0, '{'}; /* header_size=16, then '{' */
    check_format("safetensors", st_magic, 32, MODEL_FMT_SAFETENSORS);

    unsigned char pk_magic[] = {'P','K',0x03,0x04, 0,0,0,0};
    check_format("pytorch", pk_magic, 8, MODEL_FMT_PYTORCH);

    unsigned char unknown[] = {0xDE,0xAD,0xBE,0xEF, 0,0,0,0};
    check_format("unknown", unknown, 8, MODEL_FMT_UNKNOWN);

    printf("\n%d/%d tests passed\n", tests_passed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
