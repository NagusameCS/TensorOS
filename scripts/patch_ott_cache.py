"""
patch_ott_cache.py — Add LRU cache + RMSNorm key normalization to ott_get_hidden_state.

Todo 21: LRU cache keyed by (token_id, layer) — skips llm_generate_tokens on hit.
Todo 24: RMSNorm applied to captured hidden state before storing (prevents magnitude bias).
"""

FILEPATH = r"C:\Users\legom\HyperTensor\runtime\nn\axiom_beta.c"

with open(FILEPATH, "r", encoding="utf-8") as f:
    src = f.read()

# ── Replacement: swap out ott_get_hidden_state with cached + rms-normed version ──

OLD_HS = """\
/* \u2500\u2500 OTT helper: last-layer hidden state capture \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
 * Runs one prefill+decode forward pass and captures the last-layer hidden
 * state via tensor_bridge. BRIDGE_MODE_CAP_ONCE prevents the decode step
 * (pos=1) from overwriting the prefill capture.
 *
 * layer == -1  =>  last layer (llm_model_layers() - 1)
 * Returns 0 on success, -1 on failure (caller should fall back to embedding).
 * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
static int ott_get_hidden_state(int token_id, int layer, float *out, int dim)
{
    tensor_bridge_t *bridge = llm_get_bridge();
    if (!bridge || dim <= 0 || !out) return -1;
    tensor_bridge_init(bridge);
    if (layer < 0) {
        layer = llm_model_layers() - 1;
        if (layer < 0) return -1;
    }
    if (tensor_bridge_set_capture(bridge, layer, dim) != 0) {
        bridge->mode = BRIDGE_MODE_NONE;
        return -1;
    }
    /* CAP_ONCE: lock the buffer after the prefill capture so the decode step
     * cannot overwrite the input token's hidden state. */
    bridge->mode = (bridge_mode_t)(bridge->mode | BRIDGE_MODE_CAP_ONCE);
    int prompt[1] = { token_id };
    int out_tok[2];
    static int ott_fail_count = 0;
    int gen_rc = llm_generate_tokens(prompt, 1, out_tok, 2, 1, 0.0f, 0);
    int ok = (gen_rc >= 0) && bridge->capture_buf.valid &&
             bridge->capture_buf.data && bridge->capture_buf.dim >= dim;
    if (!ok && ++ott_fail_count <= 3)
        kprintf("[OTT-HS] FAIL call #%d: gen_rc=%d valid=%d dim=%d need=%d\\n",
                ott_fail_count, gen_rc, (int)bridge->capture_buf.valid,
                bridge->capture_buf.dim, dim);
    if (ok) memcpy(out, bridge->capture_buf.data, (size_t)dim * sizeof(float));
    bridge->mode = BRIDGE_MODE_NONE;
    llm_reset_cache();
    return ok ? 0 : -1;
}"""

NEW_HS = """\
/* \u2500\u2500 OTT hidden-state LRU cache (todo 21) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
 * Keyed by (token_id, layer).  On a hit the forward pass is skipped entirely,
 * reducing Phase-3 sampling from ~1900 LLM calls to a small fraction.
 * Entry count: OTT_HS_CACHE_CAP (2048 by default).
 * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
#define OTT_HS_CACHE_CAP 2048

typedef struct {
    int    token_id;
    int    layer;
    int    dim;
    float *data;        /* heap-allocated float[dim]; NULL = empty slot */
    int    lru_stamp;
} ott_hs_entry_t;

static ott_hs_entry_t ott_hs_cache[OTT_HS_CACHE_CAP];
static int            ott_hs_lru_clock = 0;
static int            ott_hs_hits  = 0;
static int            ott_hs_misses = 0;

static float *ott_hs_cache_lookup(int token_id, int layer, int dim)
{
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        ott_hs_entry_t *e = &ott_hs_cache[i];
        if (e->data && e->token_id == token_id && e->layer == layer &&
            e->dim == dim) {
            e->lru_stamp = ++ott_hs_lru_clock;
            return e->data;
        }
    }
    return NULL;
}

static void ott_hs_cache_insert(int token_id, int layer, int dim,
                                 const float *data)
{
    /* Find empty slot or LRU victim */
    int victim = 0;
    int min_stamp = ott_hs_cache[0].lru_stamp;
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        if (!ott_hs_cache[i].data) { victim = i; break; }   /* empty */
        if (ott_hs_cache[i].lru_stamp < min_stamp) {
            min_stamp = ott_hs_cache[i].lru_stamp;
            victim = i;
        }
    }
    ott_hs_entry_t *e = &ott_hs_cache[victim];
    /* Reuse allocation if dimension matches, otherwise reallocate */
    if (!e->data || e->dim != dim) {
        if (e->data) tensor_free(e->data);
        e->data = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
        if (!e->data) return;
    }
    e->token_id  = token_id;
    e->layer     = layer;
    e->dim       = dim;
    e->lru_stamp = ++ott_hs_lru_clock;
    memcpy(e->data, data, (size_t)dim * sizeof(float));
}

/* Flush the cache (call when model changes or at axiom run start) */
static void ott_hs_cache_flush(void)
{
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        if (ott_hs_cache[i].data) {
            tensor_free(ott_hs_cache[i].data);
            ott_hs_cache[i].data = NULL;
        }
        ott_hs_cache[i].lru_stamp = 0;
    }
    ott_hs_lru_clock = 0;
    ott_hs_hits = ott_hs_misses = 0;
}

/* \u2500\u2500 OTT helper: last-layer hidden state capture (with LRU cache + RMSNorm key) \u2500
 * Runs one prefill+decode forward pass and captures the last-layer hidden
 * state via tensor_bridge. BRIDGE_MODE_CAP_ONCE prevents the decode step
 * (pos=1) from overwriting the prefill capture.
 *
 * Todo 21: On cache hit the forward pass is skipped entirely.
 * Todo 24: The captured vector is RMSNorm-normalised before storage so that
 *          large-magnitude late-layer activations don't bias Phase-3 covariance.
 *
 * layer == -1  =>  last layer (llm_model_layers() - 1)
 * Returns 0 on success, -1 on failure (caller should fall back to embedding).
 * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
static int ott_get_hidden_state(int token_id, int layer, float *out, int dim)
{
    if (dim <= 0 || !out) return -1;

    /* Resolve -1 to the concrete last layer index */
    int resolved_layer = layer;
    if (resolved_layer < 0) {
        resolved_layer = llm_model_layers() - 1;
        if (resolved_layer < 0) return -1;
    }

    /* \u2500\u2500 Cache lookup (todo 21) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
    const float *cached = ott_hs_cache_lookup(token_id, resolved_layer, dim);
    if (cached) {
        memcpy(out, cached, (size_t)dim * sizeof(float));
        ott_hs_hits++;
        return 0;
    }
    ott_hs_misses++;

    /* \u2500\u2500 Forward pass to capture hidden state \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
    tensor_bridge_t *bridge = llm_get_bridge();
    if (!bridge) return -1;
    tensor_bridge_init(bridge);
    if (tensor_bridge_set_capture(bridge, resolved_layer, dim) != 0) {
        bridge->mode = BRIDGE_MODE_NONE;
        return -1;
    }
    bridge->mode = (bridge_mode_t)(bridge->mode | BRIDGE_MODE_CAP_ONCE);
    int prompt[1] = { token_id };
    int out_tok[2];
    static int ott_fail_count = 0;
    int gen_rc = llm_generate_tokens(prompt, 1, out_tok, 2, 1, 0.0f, 0);
    int ok = (gen_rc >= 0) && bridge->capture_buf.valid &&
             bridge->capture_buf.data && bridge->capture_buf.dim >= dim;
    if (!ok && ++ott_fail_count <= 3)
        kprintf("[OTT-HS] FAIL call #%d: gen_rc=%d valid=%d dim=%d need=%d\\n",
                ott_fail_count, gen_rc, (int)bridge->capture_buf.valid,
                bridge->capture_buf.dim, dim);
    if (!ok) { bridge->mode = BRIDGE_MODE_NONE; llm_reset_cache(); return -1; }

    /* \u2500\u2500 RMSNorm key normalisation (todo 24) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
     * Apply RMSNorm to the captured hidden state before storing so that
     * large-magnitude late-layer activations don't bias the Phase-3 metric
     * field covariance.  phi(q,k) = exp(q^T * RMSNorm(k)) (AttnRes eq. 2).
     * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
    const float *raw = bridge->capture_buf.data;
    double ms2 = 0.0;
    for (int j = 0; j < dim; j++) ms2 += (double)raw[j] * (double)raw[j];
    double rms_inv = 1.0 / sqrt(ms2 / (double)dim + 1e-6);
    for (int j = 0; j < dim; j++) out[j] = (float)((double)raw[j] * rms_inv);

    bridge->mode = BRIDGE_MODE_NONE;
    llm_reset_cache();

    /* Store RMSNorm-normalised vector in cache */
    ott_hs_cache_insert(token_id, resolved_layer, dim, out);
    return 0;
}"""

count = src.count(OLD_HS)
if count == 0:
    print("PATTERN NOT FOUND")
else:
    if count > 1:
        print(f"WARNING: {count} occurrences, replacing first")
    out = src.replace(OLD_HS, NEW_HS, 1)
    with open(FILEPATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Done. Size delta: +{len(out)-len(src)} bytes")
