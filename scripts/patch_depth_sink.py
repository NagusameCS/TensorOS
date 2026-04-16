#!/usr/bin/env python3
"""Todo 25: Depth-sink layer detection.
   Probes 8 uniformly-spaced vocab tokens at N candidate layers (every 5th + last).
   Picks the layer with highest mean pairwise cosine similarity —
   depth-sink layers have consistent representations across diverse inputs,
   giving a more stable manifold basis for Phase 3 and Phase 5.
   Overhead: 8 tokens x 7 layers = 56 forward passes (~4s).
"""
import sys

path = r"C:\Users\legom\HyperTensor\runtime\nn\axiom_beta.c"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

orig = len(src)

# ── 1. Add static global + detection function right after ott_hs_cache_flush ──
AFTER_FLUSH_ANCHOR = (
    "/* reset hidden-state cache each run */\n"
    "    report->beta_version = 3;"
)
if AFTER_FLUSH_ANCHOR not in src:
    print("ERROR: flush anchor not found"); sys.exit(1)

# ── 2. Add static global before ott_get_hidden_state header comment ──
BEFORE_OTT_HELPER = (
    "/* ── OTT helper: last-layer hidden state capture (with LRU cache + RMSNorm key) ─\n"
    " * Runs one prefill+decode forward pass and captures the last-layer hidden\n"
    " * state via tensor_bridge. BRIDGE_MODE_CAP_ONCE prevents the decode step\n"
    " * (pos=1) from overwriting the prefill capture.\n"
    " *\n"
    " * Todo 21: On cache hit the forward pass is skipped entirely.\n"
    " * Todo 24: The captured vector is RMSNorm-normalised before storage so that\n"
    " *          large-magnitude late-layer activations don't bias Phase-3 covariance.\n"
    " *\n"
    " * layer == -1  =>  last layer (llm_model_layers() - 1)\n"
    " * Returns 0 on success, -1 on failure (caller should fall back to embedding)."
)
if BEFORE_OTT_HELPER not in src:
    print("ERROR: ott_helper anchor not found"); sys.exit(1)

DEPTH_SINK_GLOBAL = (
    "/* Todo 25: Depth-sink layer global.\n"
    " * -1 = not yet detected, use last layer as fallback.\n"
    " * Detected at axiom_beta_run() start; all ott_get_hidden_state(-1) calls\n"
    " * resolve to this layer once set. */\n"
    "static int ott_depth_sink_layer = -1;\n"
    "\n"
)

src = src.replace(BEFORE_OTT_HELPER,
                  DEPTH_SINK_GLOBAL + BEFORE_OTT_HELPER, 1)
print("  + Added ott_depth_sink_layer global")

# ── 3. In ott_get_hidden_state, change layer=-1 resolution to use depth-sink ──
# Current code (added by earlier patch):
#     int resolved_layer = layer;
#     if (resolved_layer < 0) {
#         resolved_layer = llm_model_layers() - 1;
#         if (resolved_layer < 0) return -1;
#     }
OLD_LAYER_RESOLVE = (
    "    int resolved_layer = layer;\n"
    "    if (resolved_layer < 0) {\n"
    "        resolved_layer = llm_model_layers() - 1;\n"
    "        if (resolved_layer < 0) return -1;\n"
    "    }"
)
NEW_LAYER_RESOLVE = (
    "    int resolved_layer = layer;\n"
    "    if (resolved_layer < 0) {\n"
    "        /* Todo 25: prefer detected depth-sink layer over raw last layer */\n"
    "        if (ott_depth_sink_layer >= 0)\n"
    "            resolved_layer = ott_depth_sink_layer;\n"
    "        else\n"
    "            resolved_layer = llm_model_layers() - 1;\n"
    "        if (resolved_layer < 0) return -1;\n"
    "    }"
)
if OLD_LAYER_RESOLVE not in src:
    print("ERROR: layer resolve anchor not found"); sys.exit(1)
src = src.replace(OLD_LAYER_RESOLVE, NEW_LAYER_RESOLVE, 1)
print("  + Patched ott_get_hidden_state layer=-1 resolution to use depth-sink")

# ── 4. Add ott_detect_depth_sink() before axiom_beta_default_config ──
BEFORE_DEFAULT_CONFIG = "void axiom_beta_default_config(axiom_beta_config_t *cfg)\n{"
if BEFORE_DEFAULT_CONFIG not in src:
    print("ERROR: default_config anchor not found"); sys.exit(1)

DETECT_FUNC = """\
/* ── Todo 25: Depth-sink layer detection ──────────────────────────────────
 * Probes n_probe uniformly-spaced tokens at each candidate layer.
 * Computes mean pairwise cosine similarity; layer with max = depth sink.
 * Writes result to ott_depth_sink_layer. */
static void ott_detect_depth_sink(uint64_t *seed)
{
    int n_layers = llm_model_layers();
    int dim      = llm_model_dim();
    int vocab    = llm_model_vocab();
    if (n_layers <= 0 || dim <= 0 || vocab <= 0) return;

    /* Candidate layers: every 5 layers + last */
    int cands[16];
    int n_cands = 0;
    for (int l = 4; l < n_layers && n_cands < 15; l += 5)
        cands[n_cands++] = l;
    if (n_cands == 0 || cands[n_cands - 1] != n_layers - 1)
        cands[n_cands++] = n_layers - 1;

    int n_probe = 8;  /* diverse tokens to probe */
    float *hs   = (float *)tensor_alloc((uint64_t)n_probe * dim * sizeof(float));
    if (!hs) return;

    int    best_layer = n_layers - 1;
    double best_score = -1.0;

    for (int ci = 0; ci < n_cands; ci++) {
        int lyr = cands[ci];
        /* Sample n_probe uniformly-spaced vocab tokens */
        int got = 0;
        for (int p = 0; p < n_probe; p++) {
            int blk = p * (vocab / n_probe);
            int jit = (vocab / n_probe > 1) ? (int)(ax_rng_range(seed, 0, vocab / n_probe)) : 0;
            int tok = blk + jit;
            if (tok >= vocab) tok = vocab - 1;
            int rc = ott_get_hidden_state(tok, lyr, hs + (uint64_t)got * dim, dim);
            if (rc == 0) got++;
        }
        if (got < 2) continue;

        /* Mean pairwise cosine similarity */
        double sum_cos = 0.0;
        int    n_pairs = 0;
        for (int a = 0; a < got; a++) {
            double na = 0.0;
            for (int j = 0; j < dim; j++) na += (double)hs[a * dim + j] * hs[a * dim + j];
            na = sqrt(na) + 1e-12;
            for (int b = a + 1; b < got; b++) {
                double nb = 0.0, dot = 0.0;
                for (int j = 0; j < dim; j++) {
                    nb  += (double)hs[b * dim + j] * hs[b * dim + j];
                    dot += (double)hs[a * dim + j] * hs[b * dim + j];
                }
                nb = sqrt(nb) + 1e-12;
                sum_cos += dot / (na * nb);
                n_pairs++;
            }
        }
        double mean_cos = n_pairs > 0 ? sum_cos / (double)n_pairs : 0.0;
        kprintf("[OTT-SINK] layer=%d mean_pairwise_cos=%.4f\\n", lyr, mean_cos);
        if (mean_cos > best_score) {
            best_score = mean_cos;
            best_layer = lyr;
        }
    }
    tensor_free(hs);

    ott_depth_sink_layer = best_layer;
    kprintf("[OTT-SINK] depth-sink layer=%d (score=%.4f, %d candidates tested)\\n",
            best_layer, best_score, n_cands);
}

"""

src = src.replace(BEFORE_DEFAULT_CONFIG,
                  DETECT_FUNC + BEFORE_DEFAULT_CONFIG, 1)
print("  + Added ott_detect_depth_sink() function")

# ── 5. Call ott_detect_depth_sink() after ott_hs_cache_flush() ──
OLD_FLUSH_CALL = (
    "    ott_hs_cache_flush(); /* reset hidden-state cache each run */\n"
    "    report->beta_version = 3;"
)
NEW_FLUSH_CALL = (
    "    ott_hs_cache_flush(); /* reset hidden-state cache each run */\n"
    "    ott_depth_sink_layer = -1; /* reset so detection re-runs for this model */\n"
    "    ott_detect_depth_sink(&seed); /* Todo 25: find most informative layer */\n"
    "    ott_hs_cache_flush(); /* flush cache populated during depth-sink probe */\n"
    "    report->beta_version = 3;"
)
if OLD_FLUSH_CALL not in src:
    print("ERROR: flush call anchor not found"); sys.exit(1)
src = src.replace(OLD_FLUSH_CALL, NEW_FLUSH_CALL, 1)
print("  + Wired ott_detect_depth_sink() into axiom_beta_run()")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"Done. Size delta: {len(src) - orig:+d} bytes")
print("  - 8 probe tokens x up to 7 candidate layers = ~56 forward passes (~4-6s overhead)")
print("  - depth-sink layer replaces constant layer=-1 in all ott_get_hidden_state calls")
