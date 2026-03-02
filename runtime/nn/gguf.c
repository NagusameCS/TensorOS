/* =============================================================================
 * TensorOS - GGUF Model Format Parser Implementation
 * Parses GGUF v2/v3 files, extracts metadata and tensor pointers
 * Compatible with llama.cpp / GGML quantized models
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "runtime/nn/gguf.h"

/* =============================================================================
 * GGML type info table
 * =============================================================================*/

static const ggml_type_info_t type_info[] = {
    [GGML_TYPE_F32]     = { "F32",     1,  4 },
    [GGML_TYPE_F16]     = { "F16",     1,  2 },
    [GGML_TYPE_Q4_0]    = { "Q4_0",   32, 18 },  /* 32 elements: 2 bytes scale + 16 bytes quants */
    [GGML_TYPE_Q4_1]    = { "Q4_1",   32, 20 },  /* 32 elements: 2+2 bytes scale/min + 16 bytes quants */
    [GGML_TYPE_Q5_0]    = { "Q5_0",   32, 22 },
    [GGML_TYPE_Q5_1]    = { "Q5_1",   32, 24 },
    [GGML_TYPE_Q8_0]    = { "Q8_0",   32, 34 },  /* 32 elements: 2 bytes scale + 32 bytes quants */
    [GGML_TYPE_Q8_1]    = { "Q8_1",   32, 36 },
    [GGML_TYPE_Q2_K]    = { "Q2_K",  256, 84 },
    [GGML_TYPE_Q3_K]    = { "Q3_K",  256, 110 },
    [GGML_TYPE_Q4_K]    = { "Q4_K",  256, 144 },
    [GGML_TYPE_Q5_K]    = { "Q5_K",  256, 176 },
    [GGML_TYPE_Q6_K]    = { "Q6_K",  256, 210 },
    [GGML_TYPE_IQ2_XXS] = { "IQ2_XXS",256, 66 },
    [GGML_TYPE_IQ2_XS]  = { "IQ2_XS", 256, 74 },
    [GGML_TYPE_IQ3_XXS] = { "IQ3_XXS",256, 98 },
    [GGML_TYPE_IQ1_S]   = { "IQ1_S",  256, 50 },
    [GGML_TYPE_IQ4_NL]  = { "IQ4_NL", 32,  18 },
};

const ggml_type_info_t *ggml_get_type_info(ggml_type_t type)
{
    if ((unsigned)type >= GGML_TYPE_COUNT) return &type_info[0];
    if (type_info[type].block_size == 0) return &type_info[0]; /* unknown type */
    return &type_info[type];
}

uint64_t ggml_tensor_size(ggml_type_t type, uint64_t n_elements)
{
    const ggml_type_info_t *ti = ggml_get_type_info(type);
    if (ti->block_size <= 1) {
        return n_elements * ti->type_size;
    }
    /* Quantized: size = (n_elements / block_size) * type_size */
    uint64_t n_blocks = (n_elements + ti->block_size - 1) / ti->block_size;
    return n_blocks * ti->type_size;
}

/* =============================================================================
 * Binary reader helpers (little-endian, unaligned)
 * =============================================================================*/

typedef struct {
    const uint8_t *data;
    uint64_t pos;
    uint64_t size;
} reader_t;

static int reader_init(reader_t *r, const void *data, uint64_t size)
{
    r->data = (const uint8_t *)data;
    r->pos = 0;
    r->size = size;
    return 0;
}

static int reader_has(const reader_t *r, uint64_t n)
{
    return (r->pos + n) <= r->size;
}

static uint8_t read_u8(reader_t *r)
{
    if (!reader_has(r, 1)) return 0;
    return r->data[r->pos++];
}

static uint16_t read_u16(reader_t *r)
{
    if (!reader_has(r, 2)) return 0;
    uint16_t v = (uint16_t)r->data[r->pos] |
                 ((uint16_t)r->data[r->pos + 1] << 8);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r)
{
    if (!reader_has(r, 4)) return 0;
    uint32_t v = (uint32_t)r->data[r->pos] |
                 ((uint32_t)r->data[r->pos + 1] << 8) |
                 ((uint32_t)r->data[r->pos + 2] << 16) |
                 ((uint32_t)r->data[r->pos + 3] << 24);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r)
{
    if (!reader_has(r, 8)) return 0;
    uint64_t lo = read_u32(r);
    uint64_t hi = read_u32(r);
    return lo | (hi << 32);
}

static float read_f32(reader_t *r)
{
    uint32_t bits = read_u32(r);
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

static double read_f64(reader_t *r)
{
    uint64_t bits = read_u64(r);
    double d;
    kmemcpy(&d, &bits, 8);
    return d;
}

static gguf_string_t read_string(reader_t *r)
{
    gguf_string_t s = {0, 0};
    s.len = read_u64(r);
    if (s.len > 0 && reader_has(r, s.len)) {
        s.data = (const char *)(r->data + r->pos);
        r->pos += s.len;
    }
    return s;
}

/* =============================================================================
 * String comparison for GGUF (not null-terminated)
 * =============================================================================*/

static int gguf_str_eq(const gguf_string_t *s, const char *cstr)
{
    size_t clen = kstrlen(cstr);
    if (s->len != clen) return 0;
    for (uint64_t i = 0; i < s->len; i++) {
        if (s->data[i] != cstr[i]) return 0;
    }
    return 1;
}

static void gguf_str_copy(char *dst, const gguf_string_t *s, size_t max)
{
    size_t n = s->len < (max - 1) ? s->len : (max - 1);
    kmemcpy(dst, s->data, n);
    dst[n] = '\0';
}

/* =============================================================================
 * Parse a single KV value
 * =============================================================================*/

static int parse_kv_value(reader_t *r, gguf_kv_t *kv, gguf_type_t type)
{
    kv->type = type;
    switch (type) {
    case GGUF_TYPE_UINT8:   kv->value.u8 = read_u8(r); break;
    case GGUF_TYPE_INT8:    kv->value.i8 = (int8_t)read_u8(r); break;
    case GGUF_TYPE_UINT16:  kv->value.u16 = read_u16(r); break;
    case GGUF_TYPE_INT16:   kv->value.i16 = (int16_t)read_u16(r); break;
    case GGUF_TYPE_UINT32:  kv->value.u32 = read_u32(r); break;
    case GGUF_TYPE_INT32:   kv->value.i32 = (int32_t)read_u32(r); break;
    case GGUF_TYPE_FLOAT32: kv->value.f32 = read_f32(r); break;
    case GGUF_TYPE_BOOL:    kv->value.bool_val = read_u8(r); break;
    case GGUF_TYPE_STRING:  kv->value.str = read_string(r); break;
    case GGUF_TYPE_UINT64:  kv->value.u64 = read_u64(r); break;
    case GGUF_TYPE_INT64:   kv->value.i64 = (int64_t)read_u64(r); break;
    case GGUF_TYPE_FLOAT64: kv->value.f64 = read_f64(r); break;
    case GGUF_TYPE_ARRAY: {
        kv->value.array.elem_type = (gguf_type_t)read_u32(r);
        kv->value.array.count = read_u64(r);
        kv->value.array.data = r->data + r->pos;
        /* Skip array data — we just record the pointer */
        for (uint64_t i = 0; i < kv->value.array.count; i++) {
            switch (kv->value.array.elem_type) {
            case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: r->pos += 1; break;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: r->pos += 2; break;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: r->pos += 4; break;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: r->pos += 8; break;
            case GGUF_TYPE_STRING: { gguf_string_t s = read_string(r); (void)s; } break;
            default: return -4; /* unsupported array element type */
            }
        }
        break;
    }
    default: return -3; /* unknown type */
    }
    return 0;
}

/* =============================================================================
 * Main GGUF parser
 * =============================================================================*/

int gguf_parse(gguf_ctx_t *ctx, const void *data, uint64_t size)
{
    kmemset(ctx, 0, sizeof(*ctx));

    if (size < 24) return -1; /* too small for header */

    reader_t r;
    reader_init(&r, data, size);

    /* Read header */
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        kprintf("[GGUF] Bad magic: %x (expected %x)\n", magic, GGUF_MAGIC);
        return -1;
    }

    ctx->version = read_u32(&r);
    if (ctx->version < GGUF_VERSION_MIN || ctx->version > GGUF_VERSION_MAX) {
        kprintf("[GGUF] Unsupported version: %u\n", ctx->version);
        return -2;
    }

    ctx->n_tensors = read_u64(&r);
    ctx->n_kv = read_u64(&r);

    kprintf("[GGUF] Version %u, %lu tensors, %lu KV pairs\n",
            ctx->version, ctx->n_tensors, ctx->n_kv);

    /* Parse KV metadata */
    uint32_t kv_parsed = 0;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (!reader_has(&r, 8)) break;
        gguf_string_t key = read_string(&r);
        uint32_t type = read_u32(&r);

        if (kv_parsed < GGUF_MAX_KV) {
            gguf_kv_t *kv = &ctx->kv[kv_parsed];
            kv->key = key;
            int rc = parse_kv_value(&r, kv, (gguf_type_t)type);
            if (rc < 0) {
                kprintf("[GGUF] KV parse error at index %lu: %d\n", i, rc);
                return rc;
            }
            kv_parsed++;
        } else {
            /* Skip this KV — too many */
            gguf_kv_t tmp;
            parse_kv_value(&r, &tmp, (gguf_type_t)type);
        }
    }
    ctx->kv_count = kv_parsed;

    /* Parse tensor infos */
    uint32_t tensor_parsed = 0;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (!reader_has(&r, 8)) break;

        gguf_string_t name = read_string(&r);
        uint32_t n_dims = read_u32(&r);

        uint64_t dims[GGUF_MAX_DIMS] = {0};
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < n_dims && d < GGUF_MAX_DIMS; d++) {
            dims[d] = read_u64(&r);
            n_elements *= dims[d];
        }
        /* Skip extra dims if any */
        for (uint32_t d = GGUF_MAX_DIMS; d < n_dims; d++) {
            read_u64(&r);
        }

        uint32_t type = read_u32(&r);
        uint64_t offset = read_u64(&r);

        if (tensor_parsed < GGUF_MAX_TENSORS) {
            gguf_tensor_info_t *ti = &ctx->tensors[tensor_parsed];
            ti->name = name;
            ti->n_dims = n_dims;
            for (uint32_t d = 0; d < GGUF_MAX_DIMS; d++) ti->dims[d] = dims[d];
            ti->type = (ggml_type_t)type;
            ti->offset = offset;
            ti->n_elements = n_elements;
            ti->size_bytes = ggml_tensor_size((ggml_type_t)type, n_elements);
            ti->data = NULL; /* Will be set after alignment */
            tensor_parsed++;
        }
    }
    ctx->tensor_count = tensor_parsed;

    /* Compute data section start (aligned to 32 bytes for GGUF v3) */
    uint64_t alignment = 32;
    /* Check for custom alignment in metadata */
    const gguf_kv_t *align_kv = gguf_find_kv(ctx, "general.alignment");
    if (align_kv && align_kv->type == GGUF_TYPE_UINT32) {
        alignment = align_kv->value.u32;
    }

    uint64_t data_offset = (r.pos + alignment - 1) & ~(alignment - 1);
    ctx->data_start = (const uint8_t *)data + data_offset;
    ctx->data_size = (data_offset < size) ? (size - data_offset) : 0;

    /* Set tensor data pointers */
    uint64_t total_bytes = 0;
    uint64_t total_params = 0;
    for (uint32_t i = 0; i < ctx->tensor_count; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[i];
        ti->data = (const uint8_t *)ctx->data_start + ti->offset;
        total_bytes += ti->size_bytes;
        total_params += ti->n_elements;
    }
    ctx->total_weight_bytes = total_bytes;
    ctx->total_param_count = total_params;

    /* Extract architecture info from metadata */
    const gguf_kv_t *arch_kv = gguf_find_kv(ctx, "general.architecture");
    if (arch_kv && arch_kv->type == GGUF_TYPE_STRING) {
        gguf_str_copy(ctx->arch, &arch_kv->value.str, sizeof(ctx->arch));
    } else {
        kstrcpy(ctx->arch, "unknown");
    }

    /* Build architecture-specific key prefix */
    char prefix[80];
    kstrcpy(prefix, ctx->arch);
    /* Append dot for key lookup */
    size_t plen = kstrlen(prefix);
    prefix[plen] = '.';
    prefix[plen + 1] = '\0';

    /* Helper macro for arch-prefixed keys */
    #define ARCH_KEY(suffix, buf) do { \
        kstrcpy(buf, prefix); \
        kstrcpy(buf + kstrlen(buf), suffix); \
    } while(0)

    char keybuf[128];

    ARCH_KEY("embedding_length", keybuf);
    ctx->n_embd = gguf_get_u32(ctx, keybuf, 0);

    ARCH_KEY("block_count", keybuf);
    ctx->n_layers = gguf_get_u32(ctx, keybuf, 0);

    ARCH_KEY("attention.head_count", keybuf);
    ctx->n_heads = gguf_get_u32(ctx, keybuf, 0);

    ARCH_KEY("attention.head_count_kv", keybuf);
    ctx->n_kv_heads = gguf_get_u32(ctx, keybuf, ctx->n_heads);

    ARCH_KEY("context_length", keybuf);
    ctx->n_ctx = gguf_get_u32(ctx, keybuf, 2048);

    ARCH_KEY("feed_forward_length", keybuf);
    ctx->n_ff = gguf_get_u32(ctx, keybuf, 0);

    ARCH_KEY("vocab_size", keybuf);
    ctx->n_vocab = gguf_get_u32(ctx, keybuf, 0);

    ARCH_KEY("rope.freq_base", keybuf);
    ctx->rope_freq_base = gguf_get_f32(ctx, keybuf, 10000.0f);

    #undef ARCH_KEY

    return 0;
}

/* =============================================================================
 * Lookup functions
 * =============================================================================*/

const gguf_tensor_info_t *gguf_find_tensor(const gguf_ctx_t *ctx, const char *name)
{
    for (uint32_t i = 0; i < ctx->tensor_count; i++) {
        if (gguf_str_eq(&ctx->tensors[i].name, name))
            return &ctx->tensors[i];
    }
    return NULL;
}

const gguf_kv_t *gguf_find_kv(const gguf_ctx_t *ctx, const char *key)
{
    for (uint32_t i = 0; i < ctx->kv_count; i++) {
        if (gguf_str_eq(&ctx->kv[i].key, key))
            return &ctx->kv[i];
    }
    return NULL;
}

uint32_t gguf_get_u32(const gguf_ctx_t *ctx, const char *key, uint32_t default_val)
{
    const gguf_kv_t *kv = gguf_find_kv(ctx, key);
    if (!kv) return default_val;
    switch (kv->type) {
    case GGUF_TYPE_UINT32: return kv->value.u32;
    case GGUF_TYPE_INT32:  return (uint32_t)kv->value.i32;
    case GGUF_TYPE_UINT16: return kv->value.u16;
    case GGUF_TYPE_UINT8:  return kv->value.u8;
    default: return default_val;
    }
}

float gguf_get_f32(const gguf_ctx_t *ctx, const char *key, float default_val)
{
    const gguf_kv_t *kv = gguf_find_kv(ctx, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_FLOAT32) return kv->value.f32;
    return default_val;
}

const char *gguf_get_str(const gguf_ctx_t *ctx, const char *key)
{
    const gguf_kv_t *kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return NULL;
    /* Note: not null-terminated — caller must handle length */
    return kv->value.str.data;
}

/* =============================================================================
 * Print model info
 * =============================================================================*/

void gguf_print_info(const gguf_ctx_t *ctx)
{
    kprintf("[GGUF] Model Architecture: %s\n", ctx->arch);
    kprintf("[GGUF] Layers: %u, Heads: %u, KV-Heads: %u\n",
            ctx->n_layers, ctx->n_heads, ctx->n_kv_heads);
    kprintf("[GGUF] Embedding: %u, FFN: %u, Vocab: %u\n",
            ctx->n_embd, ctx->n_ff, ctx->n_vocab);
    kprintf("[GGUF] Context: %u tokens\n", ctx->n_ctx);
    kprintf("[GGUF] Tensors: %u, Weights: %lu KB (%lu params)\n",
            ctx->tensor_count,
            ctx->total_weight_bytes / 1024,
            ctx->total_param_count);

    /* Print first few tensors */
    uint32_t show = ctx->tensor_count < 8 ? ctx->tensor_count : 8;
    for (uint32_t i = 0; i < show; i++) {
        const gguf_tensor_info_t *t = &ctx->tensors[i];
        const ggml_type_info_t *ti = ggml_get_type_info(t->type);
        char name_buf[64];
        gguf_str_copy(name_buf, &t->name, sizeof(name_buf));
        kprintf("  [%u] %s %s [", i, name_buf, ti->name);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d > 0) kprintf("x");
            kprintf("%lu", t->dims[d]);
        }
        kprintf("] %lu KB\n", t->size_bytes / 1024);
    }
    if (ctx->tensor_count > show) {
        kprintf("  ... and %u more tensors\n", ctx->tensor_count - show);
    }
}

/* =============================================================================
 * GGUF Demo: Synthesize a test GGUF file in memory and parse it
 * =============================================================================*/

/* Helper to write bytes into a buffer */
typedef struct {
    uint8_t *buf;
    uint64_t pos;
    uint64_t capacity;
} writer_t;

static void write_u8(writer_t *w, uint8_t v) { if (w->pos < w->capacity) w->buf[w->pos++] = v; }
static void write_u16(writer_t *w, uint16_t v) { write_u8(w, v & 0xFF); write_u8(w, (v >> 8) & 0xFF); }
static void write_u32(writer_t *w, uint32_t v) { write_u16(w, v & 0xFFFF); write_u16(w, (v >> 16) & 0xFFFF); }
static void write_u64(writer_t *w, uint64_t v) { write_u32(w, (uint32_t)v); write_u32(w, (uint32_t)(v >> 32)); }
static void write_f32(writer_t *w, float v) { uint32_t bits; kmemcpy(&bits, &v, 4); write_u32(w, bits); }
static void write_bytes(writer_t *w, const void *data, uint64_t len) {
    for (uint64_t i = 0; i < len && w->pos < w->capacity; i++)
        w->buf[w->pos++] = ((const uint8_t*)data)[i];
}
static void write_string(writer_t *w, const char *s) {
    uint64_t len = kstrlen(s);
    write_u64(w, len);
    write_bytes(w, s, len);
}
static void write_kv_str(writer_t *w, const char *key, const char *val) {
    write_string(w, key);
    write_u32(w, GGUF_TYPE_STRING);
    write_string(w, val);
}
static void write_kv_u32(writer_t *w, const char *key, uint32_t val) {
    write_string(w, key);
    write_u32(w, GGUF_TYPE_UINT32);
    write_u32(w, val);
}
static void write_kv_f32(writer_t *w, const char *key, float val) {
    write_string(w, key);
    write_u32(w, GGUF_TYPE_FLOAT32);
    write_f32(w, val);
}

void gguf_run_demos(void)
{
    kprintf("\n=== GGUF Model Format Parser Demo ===\n");

    /* Create a synthetic GGUF file simulating a tiny LLaMA model */
    static uint8_t gguf_buf[8192] __attribute__((aligned(64)));
    writer_t w = { gguf_buf, 0, sizeof(gguf_buf) };

    /* Header */
    write_u32(&w, GGUF_MAGIC);
    write_u32(&w, 3);  /* version */

    /* We'll create 4 tensors for a tiny 1-layer model */
    uint32_t n_tensors = 4;
    uint32_t n_kv = 8;
    write_u64(&w, n_tensors);
    write_u64(&w, n_kv);

    /* KV metadata */
    write_kv_str(&w, "general.architecture", "llama");
    write_kv_str(&w, "general.name", "TinyLlama-TensorOS-Demo");
    write_kv_u32(&w, "llama.embedding_length", 64);
    write_kv_u32(&w, "llama.block_count", 1);
    write_kv_u32(&w, "llama.attention.head_count", 4);
    write_kv_u32(&w, "llama.attention.head_count_kv", 4);
    write_kv_u32(&w, "llama.context_length", 512);
    write_kv_u32(&w, "llama.feed_forward_length", 128);

    /* Tensor infos: token_embd.weight [32000 x 64] Q4_0 */
    uint64_t offset = 0;

    /* Tensor 0: token_embd.weight [32000, 64] Q4_0 */
    write_string(&w, "token_embd.weight");
    write_u32(&w, 2); /* n_dims */
    write_u64(&w, 32000);
    write_u64(&w, 64);
    write_u32(&w, GGML_TYPE_Q4_0);
    write_u64(&w, offset);
    uint64_t t0_size = ggml_tensor_size(GGML_TYPE_Q4_0, 32000 * 64);
    offset += t0_size;

    /* Tensor 1: blk.0.attn_q.weight [64, 64] F32 */
    write_string(&w, "blk.0.attn_q.weight");
    write_u32(&w, 2);
    write_u64(&w, 64);
    write_u64(&w, 64);
    write_u32(&w, GGML_TYPE_F32);
    write_u64(&w, offset);
    uint64_t t1_size = ggml_tensor_size(GGML_TYPE_F32, 64 * 64);
    offset += t1_size;

    /* Tensor 2: blk.0.ffn_gate.weight [128, 64] Q8_0 */
    write_string(&w, "blk.0.ffn_gate.weight");
    write_u32(&w, 2);
    write_u64(&w, 128);
    write_u64(&w, 64);
    write_u32(&w, GGML_TYPE_Q8_0);
    write_u64(&w, offset);
    uint64_t t2_size = ggml_tensor_size(GGML_TYPE_Q8_0, 128 * 64);
    offset += t2_size;

    /* Tensor 3: output.weight [32000, 64] Q4_0 */
    write_string(&w, "output.weight");
    write_u32(&w, 2);
    write_u64(&w, 32000);
    write_u64(&w, 64);
    write_u32(&w, GGML_TYPE_Q4_0);
    write_u64(&w, offset);
    uint64_t t3_size = ggml_tensor_size(GGML_TYPE_Q4_0, 32000 * 64);
    offset += t3_size;

    /* Pad to alignment */
    while (w.pos % 32 != 0) write_u8(&w, 0);

    /* Write some fake tensor data (just fill with pattern) */
    uint64_t data_start = w.pos;  (void)data_start;
    for (uint64_t i = 0; i < offset && w.pos < w.capacity; i++) {
        write_u8(&w, (uint8_t)(i & 0xFF));
    }

    kprintf("[GGUF] Synthetic test file: %lu bytes, %u tensors\n", w.pos, n_tensors);

    /* Parse it */
    static gguf_ctx_t ctx;
    int rc = gguf_parse(&ctx, gguf_buf, w.pos);
    if (rc != 0) {
        kprintf("[GGUF] PARSE FAILED: %d\n", rc);
        return;
    }

    kprintf("[GGUF] Parse successful!\n");
    gguf_print_info(&ctx);

    /* Verify tensor lookups */
    const gguf_tensor_info_t *embd = gguf_find_tensor(&ctx, "token_embd.weight");
    if (embd) {
        kprintf("[GGUF] Lookup 'token_embd.weight': OK (%lu elements, %lu KB)\n",
                embd->n_elements, embd->size_bytes / 1024);
    }

    const gguf_tensor_info_t *q_weight = gguf_find_tensor(&ctx, "blk.0.attn_q.weight");
    if (q_weight) {
        kprintf("[GGUF] Lookup 'blk.0.attn_q.weight': OK (F32, %lu elements)\n",
                q_weight->n_elements);
    }

    /* Verify metadata */
    kprintf("[GGUF] Architecture: %s\n", ctx.arch);
    kprintf("[GGUF] Parsed: embd=%u layers=%u heads=%u ctx=%u\n",
            ctx.n_embd, ctx.n_layers, ctx.n_heads, ctx.n_ctx);

    kprintf("[GGUF] Model loader ready for real GGUF files\n");
}
