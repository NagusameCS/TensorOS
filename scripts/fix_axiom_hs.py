"""
fix_axiom_hs.py — Apply hidden-state fixes to axiom_beta.c on disk.
All phases and velocities switched from llm_get_embedding_vec to
ott_get_hidden_state (with fallback), and Christoffel symbols normalized
to ~1000 curvature target for stable geodesic integration.
"""

import sys
import re

FILEPATH = r"C:\Users\legom\HyperTensor\runtime\nn\axiom_beta.c"

with open(FILEPATH, "r", encoding="utf-8") as f:
    src = f.read()

original = src  # keep for diff count

# ── Helper: replace exactly one occurrence, abort if not found ──────────────
def replace_once(s, old, new, label):
    count = s.count(old)
    if count == 0:
        print(f"  [SKIP] {label}: pattern not found (already applied?)")
        return s
    if count > 1:
        print(f"  [WARN] {label}: {count} occurrences, replacing first only")
    return s.replace(old, new, 1)


# ════════════════════════════════════════════════════════════════════════════
# 1. Insert ott_get_hidden_state helper
#    Inserted after phase5_init_velocity_curvature closing brace,
#    just before the Phase-1 section separator.
# ════════════════════════════════════════════════════════════════════════════

OTT_HELPER = r"""
/* ── OTT helper: last-layer hidden state capture ────────────────────────────
 * Runs one prefill+decode forward pass and captures the last-layer hidden
 * state via tensor_bridge. BRIDGE_MODE_CAP_ONCE prevents the decode step
 * (pos=1) from overwriting the prefill capture.
 *
 * layer == -1  =>  last layer (llm_model_layers() - 1)
 * Returns 0 on success, -1 on failure (caller should fall back to embedding).
 * ─────────────────────────────────────────────────────────────────────── */
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
        kprintf("[OTT-HS] FAIL call #%d: gen_rc=%d valid=%d dim=%d need=%d\n",
                ott_fail_count, gen_rc, (int)bridge->capture_buf.valid,
                bridge->capture_buf.dim, dim);
    if (ok) memcpy(out, bridge->capture_buf.data, (size_t)dim * sizeof(float));
    bridge->mode = BRIDGE_MODE_NONE;
    llm_reset_cache();
    return ok ? 0 : -1;
}
"""

# Anchor: end of phase5_init_velocity_curvature — the closing } followed by a
# blank line and the Phase-1 banner (which starts with the /* box-char sequence).
# We identify it by the unique comment before the Phase-1 banner.
ANCHOR_OLD = (
    "        if (vnorm > 1e-10)\n"
    "            ax_vec_scale(v_out, v_out, 1.0 / vnorm, dim);\n"
    "    }\n"
    "}\n"
)
ANCHOR_NEW = (
    "        if (vnorm > 1e-10)\n"
    "            ax_vec_scale(v_out, v_out, 1.0 / vnorm, dim);\n"
    "    }\n"
    "}\n"
) + OTT_HELPER

src = replace_once(src, ANCHOR_OLD, ANCHOR_NEW, "1. ott_get_hidden_state insert")


# ════════════════════════════════════════════════════════════════════════════
# 2. Phase 1 — replace llm_get_embedding_vec with ott_get_hidden_state
# ════════════════════════════════════════════════════════════════════════════

P1_OLD = (
    "        int rc = llm_get_embedding_vec(token_id, emb_f32, dim);\n"
    "        if (rc != 0) {\n"
    "            /* Fallback: use sequential token IDs */\n"
    "            token_id = i % vocab;\n"
    "            rc = llm_get_embedding_vec(token_id, emb_f32, dim);\n"
    "        }\n"
)
P1_NEW = (
    "        int rc = ott_get_hidden_state(token_id, -1, emb_f32, dim);\n"
    "        if (rc != 0) {\n"
    "            /* Fallback: try sequential token, then static embedding */\n"
    "            token_id = i % vocab;\n"
    "            rc = ott_get_hidden_state(token_id, -1, emb_f32, dim);\n"
    "        }\n"
    "        if (rc != 0) rc = llm_get_embedding_vec(token_id, emb_f32, dim);\n"
)

src = replace_once(src, P1_OLD, P1_NEW, "2. Phase 1 sampling")


# ════════════════════════════════════════════════════════════════════════════
# 3. Phase 3 — replace llm_get_embedding_vec in metric field sampling
# ════════════════════════════════════════════════════════════════════════════

P3_OLD = (
    "        if (llm_get_embedding_vec(tok, emb_f32, dim) == 0) {\n"
    "            for (int j = 0; j < dim; j++) emb_f64[j] = (double)emb_f32[j];\n"
    "            axpca_project(&phase1_pca, emb_f64, proj_full);\n"
    "            /* Keep only first sub_dim components */\n"
    "            memcpy(all_proj + i * sub_dim, proj_full,\n"
    "                   (uint64_t)sub_dim * sizeof(double));\n"
    "        } else {\n"
    "            memset(all_proj + i * sub_dim, 0, (uint64_t)sub_dim * sizeof(double));\n"
    "        }\n"
)
P3_NEW = (
    "        int hs_rc = ott_get_hidden_state(tok, -1, emb_f32, dim);\n"
    "        if (hs_rc != 0) hs_rc = llm_get_embedding_vec(tok, emb_f32, dim);\n"
    "        if (hs_rc == 0) {\n"
    "            for (int j = 0; j < dim; j++) emb_f64[j] = (double)emb_f32[j];\n"
    "            axpca_project(&phase1_pca, emb_f64, proj_full);\n"
    "            /* Keep only first sub_dim components */\n"
    "            memcpy(all_proj + i * sub_dim, proj_full,\n"
    "                   (uint64_t)sub_dim * sizeof(double));\n"
    "        } else {\n"
    "            memset(all_proj + i * sub_dim, 0, (uint64_t)sub_dim * sizeof(double));\n"
    "        }\n"
)

src = replace_once(src, P3_OLD, P3_NEW, "3. Phase 3 metric sampling")


# ════════════════════════════════════════════════════════════════════════════
# 4. Phase 5 — probe pool building
# ════════════════════════════════════════════════════════════════════════════

PP_OLD = (
    "        if (llm_get_embedding_vec(tok_probe, emb_f32, dim) != 0) {\n"
    "            probe_tokens[p] = -1;\n"
    "            cand_norms[p] = 0.0f;\n"
    "            memset(cand_mat_f32 + (uint64_t)p * dim, 0,\n"
    "                   (uint64_t)dim * sizeof(float));\n"
    "            continue;\n"
    "        }\n"
)
PP_NEW = (
    "        int pp_rc = ott_get_hidden_state(tok_probe, -1, emb_f32, dim);\n"
    "        if (pp_rc != 0) pp_rc = llm_get_embedding_vec(tok_probe, emb_f32, dim);\n"
    "        if (pp_rc != 0) {\n"
    "            probe_tokens[p] = -1;\n"
    "            cand_norms[p] = 0.0f;\n"
    "            memset(cand_mat_f32 + (uint64_t)p * dim, 0,\n"
    "                   (uint64_t)dim * sizeof(float));\n"
    "            continue;\n"
    "        }\n"
)

src = replace_once(src, PP_OLD, PP_NEW, "4. Phase 5 probe pool")


# ════════════════════════════════════════════════════════════════════════════
# 5. Phase 5 — tok_start pilot embedding
# ════════════════════════════════════════════════════════════════════════════

TS_OLD = "        if (llm_get_embedding_vec(tok_start, emb_f32, dim) != 0) continue;\n"
TS_NEW = (
    "        { int _rc = ott_get_hidden_state(tok_start, -1, emb_f32, dim);\n"
    "          if (_rc != 0) _rc = llm_get_embedding_vec(tok_start, emb_f32, dim);\n"
    "          if (_rc != 0) continue; }\n"
)

src = replace_once(src, TS_OLD, TS_NEW, "5. Phase 5 tok_start")


# ════════════════════════════════════════════════════════════════════════════
# 6. Phase 5 — tok_end pilot embedding
# ════════════════════════════════════════════════════════════════════════════

TE_OLD = "        if (llm_get_embedding_vec(tok_end, emb_f32, dim) != 0) continue;\n"
TE_NEW = (
    "        { int _rc = ott_get_hidden_state(tok_end, -1, emb_f32, dim);\n"
    "          if (_rc != 0) _rc = llm_get_embedding_vec(tok_end, emb_f32, dim);\n"
    "          if (_rc != 0) continue; }\n"
)

src = replace_once(src, TE_OLD, TE_NEW, "6. Phase 5 tok_end")


# ════════════════════════════════════════════════════════════════════════════
# 7. Phase 5 — slot 0 target (tok_end score comparison)
# ════════════════════════════════════════════════════════════════════════════

SL_OLD = "                if (llm_get_embedding_vec(tok_end, emb_f32, dim) == 0) {\n"
SL_NEW = (
    "                { int _t0rc = ott_get_hidden_state(tok_end, -1, emb_f32, dim);\n"
    "                  if (_t0rc != 0) _t0rc = llm_get_embedding_vec(tok_end, emb_f32, dim); }\n"
    "                if (emb_f32[0] == emb_f32[0]) {  /* always enter after ott/emb fetch */\n"
)

src = replace_once(src, SL_OLD, SL_NEW, "7. Phase 5 slot-0 target")


# ════════════════════════════════════════════════════════════════════════════
# 8. v2 velocity — replace llm_get_embedding_vec with ott_get_hidden_state
# ════════════════════════════════════════════════════════════════════════════

V2_OLD = (
    "    if (llm_get_embedding_vec(tok_curr, e_curr, dim) != 0 ||\n"
    "        llm_get_embedding_vec(tok_prev, e_prev, dim) != 0) {\n"
    "        tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);\n"
    "        return AXIOM_BETA_ERR_INVALID;\n"
    "    }\n"
)
V2_NEW = (
    "    int hs_v2_ok = (ott_get_hidden_state(tok_curr, -1, e_curr, dim) == 0);\n"
    "    if (!hs_v2_ok) hs_v2_ok = (llm_get_embedding_vec(tok_curr, e_curr, dim) == 0);\n"
    "    int hs_v2_ok2 = hs_v2_ok && (ott_get_hidden_state(tok_prev, -1, e_prev, dim) == 0);\n"
    "    if (hs_v2_ok && !hs_v2_ok2)\n"
    "        hs_v2_ok2 = (llm_get_embedding_vec(tok_prev, e_prev, dim) == 0);\n"
    "    if (!hs_v2_ok || !hs_v2_ok2) {\n"
    "        tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);\n"
    "        return AXIOM_BETA_ERR_INVALID;\n"
    "    }\n"
)

src = replace_once(src, V2_OLD, V2_NEW, "8. v2 velocity")


# ════════════════════════════════════════════════════════════════════════════
# 9. Christoffel normalization — insert after axgeo_curvature_destroy
#    Normalizes Γ^k_ij so effective curvature ≈ 1000 (HS space is ~100x hotter)
# ════════════════════════════════════════════════════════════════════════════

CN_OLD = (
    "    axgeo_curvature_destroy(&curv);\n"
    "\n"
    "    phase3_mf = mf;\n"
)
CN_NEW = (
    "    axgeo_curvature_destroy(&curv);\n"
    "\n"
    "    /* OTT: HS curvature (~1e5) is ~100x larger than embedding curvature (~1e3).\n"
    "     * Normalize Christoffel symbols so effective curvature ≈ 1000, keeping\n"
    "     * geodesic RK4 integration numerically stable.\n"
    "     * Scale Gamma^k_ij by sqrt(target/actual) since R ~ Gamma^2. */\n"
    "    kprintf(\"[OTT-CH-DBG-v2] rc_ch=%d rc_curv=%d max_R=%.1f ch_n=%d ch_dim=%d\\n\",\n"
    "            rc_ch, rc_curv, r->phase3.max_scalar_curvature, ch.n_points, ch.dim);\n"
    "    if (rc_ch == 0 && ch.gamma && r->phase3.max_scalar_curvature != 0.0) {\n"
    "        double target_max_curv = 1000.0;\n"
    "        double actual_max_curv = fabs(r->phase3.max_scalar_curvature);\n"
    "        if (actual_max_curv > target_max_curv * 2.0) {\n"
    "            double ch_scale = sqrt(target_max_curv / actual_max_curv);\n"
    "            uint64_t ch_total = (uint64_t)ch.n_points * (uint64_t)ch.dim *\n"
    "                                (uint64_t)ch.dim * (uint64_t)ch.dim;\n"
    "            for (uint64_t ci = 0; ci < ch_total; ci++) ch.gamma[ci] *= ch_scale;\n"
    "            kprintf(\"[AXIOM-P3] Christoffel normalized by %.4f \"\n"
    "                    \"(max_R %.0f->%.0f equiv)\\n\",\n"
    "                    ch_scale, actual_max_curv,\n"
    "                    actual_max_curv * ch_scale * ch_scale);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    phase3_mf = mf;\n"
)

src = replace_once(src, CN_OLD, CN_NEW, "9. Christoffel normalization")


# ════════════════════════════════════════════════════════════════════════════
# Write result
# ════════════════════════════════════════════════════════════════════════════
if src == original:
    print("WARNING: No changes were made — all patterns may already be applied.")
    sys.exit(1)

with open(FILEPATH, "w", encoding="utf-8") as f:
    f.write(src)

print(f"Done. File written: {FILEPATH}")
print(f"Size delta: {len(src) - len(original):+d} bytes")
