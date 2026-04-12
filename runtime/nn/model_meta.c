/*
 * TensorOS Model Metadata Normalization
 *
 * Maps tensor names from HuggingFace/PyTorch to GGUF canonical names.
 * Supports: LLaMA, Gemma, Qwen2, Phi-2/3, Mistral, GPT-2 naming conventions.
 */

#include "runtime/nn/model_meta.h"

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * String helpers (bare-metal safe, no libc dependency)
 * ════════════════════════════════════════════════════════════════════════ */

static int mm_strlen(const char *s) {
    int n = 0; while (s[n]) n++;
    return n;
}

static int mm_strncmp(const char *a, const char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return (unsigned char)a[i] - (unsigned char)b[i];
        if (a[i] == 0) return 0;
    }
    return 0;
}

static int mm_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}

static void mm_strcpy(char *d, const char *s) {
    while ((*d++ = *s++));
}

static int mm_starts_with(const char *str, const char *prefix) {
    while (*prefix) {
        if (*str++ != *prefix++) return 0;
    }
    return 1;
}

static int mm_contains(const char *str, const char *sub) {
    int slen = mm_strlen(sub);
    int len = mm_strlen(str);
    for (int i = 0; i <= len - slen; i++) {
        if (mm_strncmp(str + i, sub, slen) == 0) return 1;
    }
    return 0;
}

/* Append string to buffer, respecting max length. Returns new position. */
static int mm_append(char *buf, int pos, int max, const char *s) {
    while (*s && pos < max - 1) buf[pos++] = *s++;
    buf[pos] = '\0';
    return pos;
}

/* Extract layer number from "model.layers.N." or "transformer.h.N." pattern.
 * Returns layer number, or -1 if not found. Sets *after to position after "N." */
static int mm_extract_layer(const char *name, int *after) {
    /* Try "model.layers.N." */
    const char *p = (const char *)0;
    if (mm_starts_with(name, "model.layers.")) p = name + 13;
    else if (mm_starts_with(name, "transformer.h.")) p = name + 14;
    else if (mm_starts_with(name, "transformer.layers.")) p = name + 19;
    else if (mm_starts_with(name, "encoder.layer.")) p = name + 14;
    else if (mm_starts_with(name, "gpt_neox.layers.")) p = name + 16;

    if (!p) return -1;

    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        p++;
    }
    if (*p != '.') return -1;
    *after = (int)(p + 1 - name);
    return layer;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Global tensor name mapping table
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *hf_name;     /* HuggingFace suffix (after model prefix) */
    const char *gguf_name;   /* GGUF canonical name */
} name_map_t;

/* Global (non-layer) tensor name mappings */
static const name_map_t global_map[] = {
    /* LLaMA / Gemma / Qwen2 / Mistral */
    { "model.embed_tokens.weight",         "token_embd.weight" },
    { "model.norm.weight",                 "output_norm.weight" },
    { "model.norm.bias",                   "output_norm.bias" },
    { "lm_head.weight",                    "output.weight" },
    { "lm_head.bias",                      "output.bias" },

    /* GPT-2 / GPT-NeoX */
    { "transformer.wte.weight",            "token_embd.weight" },
    { "transformer.wpe.weight",            "position_embd.weight" },
    { "transformer.ln_f.weight",           "output_norm.weight" },
    { "transformer.ln_f.bias",             "output_norm.bias" },

    /* Phi-2 */
    { "model.final_layernorm.weight",      "output_norm.weight" },
    { "model.final_layernorm.bias",        "output_norm.bias" },

    /* GPT-NeoX */
    { "gpt_neox.embed_in.weight",          "token_embd.weight" },
    { "gpt_neox.final_layer_norm.weight",  "output_norm.weight" },
    { "gpt_neox.final_layer_norm.bias",    "output_norm.bias" },
    { "embed_out.weight",                  "output.weight" },

    { (const char *)0, (const char *)0 }
};

/* Per-layer suffix mappings (after "model.layers.N." or similar prefix) */
typedef struct {
    const char *hf_suffix;
    const char *gguf_suffix;
} layer_map_t;

static const layer_map_t layer_map[] = {
    /* Attention */
    { "self_attn.q_proj.weight",       "attn_q.weight" },
    { "self_attn.k_proj.weight",       "attn_k.weight" },
    { "self_attn.v_proj.weight",       "attn_v.weight" },
    { "self_attn.o_proj.weight",       "attn_output.weight" },
    { "self_attn.q_proj.bias",         "attn_q.bias" },
    { "self_attn.k_proj.bias",         "attn_k.bias" },
    { "self_attn.v_proj.bias",         "attn_v.bias" },
    { "self_attn.o_proj.bias",         "attn_output.bias" },

    /* Fused QKV (Phi-3, some custom models) */
    { "self_attn.qkv_proj.weight",     "attn_qkv.weight" },
    { "self_attn.qkv_proj.bias",       "attn_qkv.bias" },

    /* Norms */
    { "input_layernorm.weight",        "attn_norm.weight" },
    { "input_layernorm.bias",          "attn_norm.bias" },
    { "post_attention_layernorm.weight","ffn_norm.weight" },
    { "post_attention_layernorm.bias",  "ffn_norm.bias" },

    /* Gemma4 norms */
    { "self_attn.q_norm.weight",       "attn_q_norm.weight" },
    { "self_attn.k_norm.weight",       "attn_k_norm.weight" },
    { "post_attention_norm.weight",    "post_attention_norm.weight" },
    { "post_feedforward_norm.weight",  "post_ffw_norm.weight" },

    /* FFN - SwiGLU / GeGLU */
    { "mlp.gate_proj.weight",          "ffn_gate.weight" },
    { "mlp.up_proj.weight",            "ffn_up.weight" },
    { "mlp.down_proj.weight",          "ffn_down.weight" },
    { "mlp.gate_proj.bias",            "ffn_gate.bias" },
    { "mlp.up_proj.bias",              "ffn_up.bias" },
    { "mlp.down_proj.bias",            "ffn_down.bias" },

    /* FFN - Phi-2 style */
    { "mlp.fc1.weight",                "ffn_up.weight" },
    { "mlp.fc1.bias",                  "ffn_up.bias" },
    { "mlp.fc2.weight",               "ffn_down.weight" },
    { "mlp.fc2.bias",                  "ffn_down.bias" },

    /* GPT-2 style */
    { "attn.c_attn.weight",           "attn_qkv.weight" },
    { "attn.c_attn.bias",             "attn_qkv.bias" },
    { "attn.c_proj.weight",           "attn_output.weight" },
    { "attn.c_proj.bias",             "attn_output.bias" },
    { "ln_1.weight",                   "attn_norm.weight" },
    { "ln_1.bias",                     "attn_norm.bias" },
    { "ln_2.weight",                   "ffn_norm.weight" },
    { "ln_2.bias",                     "ffn_norm.bias" },
    { "mlp.c_fc.weight",              "ffn_up.weight" },
    { "mlp.c_fc.bias",                "ffn_up.bias" },
    { "mlp.c_proj.weight",            "ffn_down.weight" },
    { "mlp.c_proj.bias",              "ffn_down.bias" },

    /* Gemma4 ISWA */
    { "iswa.inp_gate.weight",          "inp_gate.weight" },
    { "iswa.proj.weight",              "proj.weight" },
    { "iswa.post_norm.weight",         "post_norm.weight" },
    { "layer_output_scale",            "layer_output_scale.weight" },

    { (const char *)0, (const char *)0 }
};

/* ═══════════════════════════════════════════════════════════════════════
 * Format Detection
 * ════════════════════════════════════════════════════════════════════════ */

model_format_t model_detect_format(const void *data, uint64_t size) {
    if (size < 8) return MODEL_FMT_UNKNOWN;

    const uint8_t *b = (const uint8_t *)data;

    /* GGUF: magic "GGUF" at offset 0 (little-endian uint32 0x46554747) */
    if (b[0] == 'G' && b[1] == 'G' && b[2] == 'U' && b[3] == 'F')
        return MODEL_FMT_GGUF;

    /* Safetensors: starts with 8-byte little-endian header size,
     * followed by '{' (JSON header). Header size is typically < 10MB. */
    {
        uint64_t hdr_sz = 0;
        for (int i = 0; i < 8; i++)
            hdr_sz |= (uint64_t)b[i] << (i * 8);
        if (hdr_sz > 0 && hdr_sz < 100 * 1024 * 1024 && hdr_sz + 8 <= size) {
            if (b[8] == '{')
                return MODEL_FMT_SAFETENSORS;
        }
    }

    /* ONNX: protobuf format, first byte is field tag 0x08 or 0x0A */
    if ((b[0] == 0x08 || b[0] == 0x0A) && size > 16) {
        /* Heuristic: check for common ONNX strings */
        int found_onnx = 0;
        int check_len = size < 256 ? (int)size : 256;
        for (int i = 0; i < check_len - 3; i++) {
            if (b[i] == 'o' && b[i+1] == 'n' && b[i+2] == 'n' && b[i+3] == 'x') {
                found_onnx = 1;
                break;
            }
        }
        if (found_onnx) return MODEL_FMT_ONNX;
    }

    /* PyTorch: ZIP format (PK magic) */
    if (b[0] == 'P' && b[1] == 'K' && b[2] == 0x03 && b[3] == 0x04)
        return MODEL_FMT_PYTORCH;

    return MODEL_FMT_UNKNOWN;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tensor Name Normalization
 * ════════════════════════════════════════════════════════════════════════ */

/* Small int-to-string for layer numbers */
static int mm_itoa(int n, char *buf) {
    if (n == 0) { buf[0] = '0'; buf[1] = '\0'; return 1; }
    char tmp[12];
    int k = 0;
    while (n > 0) { tmp[k++] = '0' + (n % 10); n /= 10; }
    int len = k;
    for (int i = 0; i < k; i++) buf[i] = tmp[k - 1 - i];
    buf[len] = '\0';
    return len;
}

int model_normalize_tensor_name(const char *hf_name, char *out_name, int out_len) {
    if (!hf_name || !out_name || out_len < 2) return -1;

    /* 1. Check if already in GGUF canonical form */
    if (mm_starts_with(hf_name, "token_embd.") ||
        mm_starts_with(hf_name, "output_norm.") ||
        mm_starts_with(hf_name, "output.") ||
        mm_starts_with(hf_name, "blk.") ||
        mm_starts_with(hf_name, "rope_factors") ||
        mm_starts_with(hf_name, "per_layer_")) {
        int len = mm_strlen(hf_name);
        if (len >= out_len) return -1;
        mm_strcpy(out_name, hf_name);
        return 0;
    }

    /* 2. Try global (non-layer) mappings */
    for (int i = 0; global_map[i].hf_name; i++) {
        if (mm_strcmp(hf_name, global_map[i].hf_name) == 0) {
            int len = mm_strlen(global_map[i].gguf_name);
            if (len >= out_len) return -1;
            mm_strcpy(out_name, global_map[i].gguf_name);
            return 0;
        }
    }

    /* 3. Try per-layer mappings */
    int after = 0;
    int layer = mm_extract_layer(hf_name, &after);
    if (layer >= 0) {
        const char *suffix = hf_name + after;
        for (int i = 0; layer_map[i].hf_suffix; i++) {
            if (mm_strcmp(suffix, layer_map[i].hf_suffix) == 0) {
                /* Build "blk.{layer}.{gguf_suffix}" */
                char num[12];
                mm_itoa(layer, num);
                int pos = 0;
                pos = mm_append(out_name, pos, out_len, "blk.");
                pos = mm_append(out_name, pos, out_len, num);
                pos = mm_append(out_name, pos, out_len, ".");
                pos = mm_append(out_name, pos, out_len, layer_map[i].gguf_suffix);
                return 0;
            }
        }
    }

    return -1; /* Unknown name */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Architecture Inference
 * ════════════════════════════════════════════════════════════════════════ */

model_arch_t model_infer_arch(const char **tensor_names, int n_tensors) {
    int has_gate_proj = 0;
    int has_qkv_fused = 0;
    int has_q_norm = 0;
    int has_iswa = 0;
    int has_gpt2_attn = 0;
    int has_c_fc = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char *name = tensor_names[i];
        if (mm_contains(name, "gate_proj")) has_gate_proj = 1;
        if (mm_contains(name, "qkv_proj")) has_qkv_fused = 1;
        if (mm_contains(name, "q_norm")) has_q_norm = 1;
        if (mm_contains(name, "iswa") || mm_contains(name, "inp_gate")) has_iswa = 1;
        if (mm_contains(name, "c_attn")) has_gpt2_attn = 1;
        if (mm_contains(name, "c_fc")) has_c_fc = 1;
    }

    /* Gemma4: has ISWA injection tensors */
    if (has_iswa && has_q_norm) return MODEL_ARCH_GEMMA4;

    /* Gemma2: has Q/K norms but no ISWA */
    if (has_q_norm && has_gate_proj) return MODEL_ARCH_GEMMA2;

    /* GPT-2: has c_attn and c_fc */
    if (has_gpt2_attn && has_c_fc) return MODEL_ARCH_GPT2;

    /* Phi-3: fused QKV */
    if (has_qkv_fused) return MODEL_ARCH_PHI3;

    /* LLaMA family (Qwen2, Mistral, etc.): has gate_proj */
    if (has_gate_proj) return MODEL_ARCH_LLAMA;

    return MODEL_ARCH_UNKNOWN;
}
