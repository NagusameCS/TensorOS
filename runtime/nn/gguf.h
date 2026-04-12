/* =============================================================================
 * TensorOS - GGUF Model Format Parser
 * Compatible with llama.cpp / GGML GGUF v3 format
 * Enables loading real quantized LLM models
 * =============================================================================*/

#ifndef TENSOROS_GGUF_H
#define TENSOROS_GGUF_H

#include <stdint.h>
#include <stddef.h>

/* GGUF magic number: "GGUF" in little-endian */
#define GGUF_MAGIC  0x46554747u

/* GGUF versions we support */
#define GGUF_VERSION_MIN 2
#define GGUF_VERSION_MAX 3

/* GGUF metadata value types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

/* GGML tensor types (quantization formats) */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT,
} ggml_type_t;

/* Size info for each quantization type */
typedef struct {
    const char *name;
    uint32_t    block_size;   /* Elements per block */
    uint32_t    type_size;    /* Bytes per block */
} ggml_type_info_t;

/* GGUF string (length-prefixed) */
typedef struct {
    uint64_t len;
    const char *data;   /* NOT null-terminated in file; points into mapped data */
} gguf_string_t;

/* GGUF key-value pair (parsed) */
#define GGUF_MAX_KV       512
#define GGUF_MAX_TENSORS  4096
#define GGUF_MAX_DIMS     4

typedef struct {
    gguf_string_t key;
    gguf_type_t   type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        uint8_t   bool_val;
        gguf_string_t str;
        struct {
            gguf_type_t elem_type;
            uint64_t    count;
            const void *data;
        } array;
    } value;
} gguf_kv_t;

/* GGUF tensor info (parsed) */
typedef struct {
    gguf_string_t name;
    uint32_t      n_dims;
    uint64_t      dims[GGUF_MAX_DIMS];
    ggml_type_t   type;
    uint64_t      offset;       /* Offset from start of data section */
    /* Computed fields */
    uint64_t      n_elements;   /* Total element count */
    uint64_t      size_bytes;   /* Size in bytes */
    const void   *data;         /* Pointer to actual tensor data in memory */
} gguf_tensor_info_t;

/* Parsed GGUF file context */
typedef struct {
    /* Header */
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    /* Metadata KV pairs */
    gguf_kv_t kv[GGUF_MAX_KV];
    uint32_t  kv_count;

    /* Tensor info */
    gguf_tensor_info_t tensors[GGUF_MAX_TENSORS];
    uint32_t           tensor_count;

    /* Data section pointer */
    const void *data_start;
    uint64_t    data_size;

    /* Model architecture info (extracted from metadata) */
    char     arch[64];          /* e.g. "llama", "gpt2", "phi" */
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_ctx;             /* Context length */
    uint32_t n_ff;              /* Feed-forward hidden size */
    float    rope_freq_base;
    float    rope_freq_scale;
    uint32_t n_embd_head_k;     /* attention.key_length (0 = use n_embd/n_heads) */
    uint32_t n_embd_head_v;     /* attention.value_length (0 = same as key_length) */
    float    final_logit_softcap; /* final_logit_softcapping (0 = disabled) */
    uint32_t n_embd_head_k_swa; /* attention.key_length_swa (0 = same as key_length) */
    float    rope_freq_base_swa;/* rope.freq_base_swa (0 = same as rope_freq_base) */
    uint32_t shared_kv_layers;  /* attention.shared_kv_layers (0 = none) */
    uint32_t n_embd_per_layer;  /* embedding_length_per_layer (0 = no ISWA) */

    /* Total model size */
    uint64_t total_weight_bytes;
    uint64_t total_param_count;
} gguf_ctx_t;

/* =============================================================================
 * API
 * =============================================================================*/

/**
 * Parse a GGUF file from a memory buffer.
 * Returns 0 on success, negative error code on failure.
 * The ctx structure is filled with parsed metadata and tensor pointers.
 * The data pointer must remain valid for the lifetime of ctx.
 */
int gguf_parse(gguf_ctx_t *ctx, const void *data, uint64_t size);

/**
 * Find a tensor by name in a parsed GGUF context.
 * Returns pointer to tensor info, or NULL if not found.
 */
const gguf_tensor_info_t *gguf_find_tensor(const gguf_ctx_t *ctx, const char *name);

/**
 * Find a metadata key-value pair by key name.
 * Returns pointer to KV entry, or NULL if not found.
 */
const gguf_kv_t *gguf_find_kv(const gguf_ctx_t *ctx, const char *key);

/**
 * Get a uint32 metadata value by key. Returns default_val if not found.
 */
uint32_t gguf_get_u32(const gguf_ctx_t *ctx, const char *key, uint32_t default_val);

/**
 * Get a float32 metadata value by key. Returns default_val if not found.
 */
float gguf_get_f32(const gguf_ctx_t *ctx, const char *key, float default_val);

/**
 * Get a string metadata value by key. Returns NULL if not found.
 */
const char *gguf_get_str(const gguf_ctx_t *ctx, const char *key);

/**
 * Print GGUF model summary to serial/VGA.
 */
void gguf_print_info(const gguf_ctx_t *ctx);

/**
 * Compute the byte size of a GGML tensor given type and element count.
 */
uint64_t ggml_tensor_size(ggml_type_t type, uint64_t n_elements);

/**
 * Get type info for a GGML type.
 */
const ggml_type_info_t *ggml_get_type_info(ggml_type_t type);

/**
 * Run GGUF demos: parse a synthetic test model in memory.
 */
void gguf_run_demos(void);

#endif /* TENSOROS_GGUF_H */
