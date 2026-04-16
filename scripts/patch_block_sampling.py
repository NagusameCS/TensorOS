#!/usr/bin/env python3
"""Todo 22: Block-partitioned manifold sampling in Phase 3.
   Reduces k_local from 32 to 4, adds block-partitioned token selection
   (vocab divided into n_total_samples windows, one token sampled per window).
   Expected speedup: 4096 HS calls -> 512 HS calls, ~36s vs ~289s.
"""
import sys

path = r"C:\Users\legom\HyperTensor\runtime\nn\axiom_beta.c"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

orig = len(src)

OLD = (
    "    int k_local = 32;  /* neighbors for local covariance */\n"
    "    int n_total_samples = n_mp * k_local;\n"
    "    if (n_total_samples > vocab) n_total_samples = vocab;\n"
    "\n"
    "    /* Sample embeddings and project to PCA subspace */\n"
    "    int pca_full = phase1_pca.n_components;  /* full PCA output dim */\n"
    "    float *emb_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));\n"
    "    double *emb_f64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));\n"
    "    double *proj_full = (double *)tensor_alloc((uint64_t)pca_full * sizeof(double));\n"
    "\n"
    "    /* Store all projected samples for local covariance computation */\n"
    "    double *all_proj = (double *)tensor_alloc((uint64_t)n_total_samples * sub_dim * sizeof(double));\n"
    "\n"
    "    if (!emb_f32 || !emb_f64 || !proj_full || !all_proj) {\n"
    "        if (emb_f32)   tensor_free(emb_f32);\n"
    "        if (emb_f64)   tensor_free(emb_f64);\n"
    "        if (proj_full) tensor_free(proj_full);\n"
    "        if (all_proj)  tensor_free(all_proj);\n"
    "        r->phase3_us = hal_timer_us() - t0;\n"
    "        return -1;\n"
    "    }\n"
    "\n"
    "    /* Collect projected embedding samples (first sub_dim components) */\n"
    "    for (int i = 0; i < n_total_samples; i++) {\n"
    "        int tok = ax_rng_range(seed, 0, vocab);\n"
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
    "        }"
)

NEW = (
    "    /* Todo 22: block-partitioned sampling -- use k_local=4 neighbors per\n"
    "     * metric point (down from 32) with deterministic block partition so\n"
    "     * each sample window covers an equal slice of the vocabulary.\n"
    "     * Total HS calls: n_mp*4 vs n_mp*32 (8x fewer forward passes). */\n"
    "    int k_local = 4;  /* neighbors for local covariance (block-partitioned) */\n"
    "    int n_total_samples = n_mp * k_local;\n"
    "    if (n_total_samples > vocab) n_total_samples = vocab;\n"
    "\n"
    "    /* Sample embeddings and project to PCA subspace */\n"
    "    int pca_full = phase1_pca.n_components;  /* full PCA output dim */\n"
    "    float *emb_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));\n"
    "    double *emb_f64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));\n"
    "    double *proj_full = (double *)tensor_alloc((uint64_t)pca_full * sizeof(double));\n"
    "\n"
    "    /* Store all projected samples for local covariance computation */\n"
    "    double *all_proj = (double *)tensor_alloc((uint64_t)n_total_samples * sub_dim * sizeof(double));\n"
    "\n"
    "    if (!emb_f32 || !emb_f64 || !proj_full || !all_proj) {\n"
    "        if (emb_f32)   tensor_free(emb_f32);\n"
    "        if (emb_f64)   tensor_free(emb_f64);\n"
    "        if (proj_full) tensor_free(proj_full);\n"
    "        if (all_proj)  tensor_free(all_proj);\n"
    "        r->phase3_us = hal_timer_us() - t0;\n"
    "        return -1;\n"
    "    }\n"
    "\n"
    "    /* Block-partitioned sampling: divide vocab into n_total_samples equal\n"
    "     * windows; sample one token per window with a jitter from the seed.\n"
    "     * This ensures uniform vocab coverage and deterministic reuse across runs. */\n"
    "    int blk_w = (vocab > n_total_samples) ? (vocab / n_total_samples) : 1;\n"
    "    kprintf(\"[AXIOM-P3] Block-sampled %d tokens (k_local=%d, blk_w=%d)\\n\",\n"
    "            n_total_samples, k_local, blk_w);\n"
    "    for (int i = 0; i < n_total_samples; i++) {\n"
    "        int blk_base = i * blk_w;\n"
    "        int jitter   = (blk_w > 1) ? (int)(ax_rng_range(seed, 0, blk_w)) : 0;\n"
    "        int tok      = blk_base + jitter;\n"
    "        if (tok >= vocab) tok = vocab - 1;\n"
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
    "        }"
)

if OLD not in src:
    print("ERROR: anchor text not found — check file version")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
# Also fix the covariance divisor guard: k_local=4 means (k_local-1)=3, still valid.
# Nothing else to change — the rest of the metric field computation loop
# already uses the updated k_local variable.

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"Done. Size delta: {len(src) - orig:+d} bytes")
print("  - k_local 32 -> 4 (8x fewer HS forward passes in Phase 3)")
print("  - Block-partitioned token sampling (uniform vocab coverage)")
print("  - Expected Phase 3: ~36s (from ~289s)")
