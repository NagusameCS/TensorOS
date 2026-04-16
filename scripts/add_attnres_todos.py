"""Add Beta-4++ Memory Efficiency section to GEODESSICAL_PLAN.md."""

FILEPATH = r"C:\Users\legom\HyperTensor\docs\GEODESSICAL_PLAN.md"

NEW_SECTION = (
    "\n"
    "### Beta-4++ Memory Efficiency (from arXiv:2603.15031 \u00a74 & Appendix)\n"
    "**Goal**: Eliminate redundant ott_get_hidden_state forward passes and reduce probe pool peak memory,\n"
    "using concrete infrastructure techniques from the AttnRes paper.\n"
    "\n"
    "21. **ott_get_hidden_state Result Cache (LRU token_id \u2192 hidden state)**\n"
    "   - Problem: Phase 1 = 31.5s (256 calls ~130ms each), Phase 3 = 248s (~1900 calls), Phase 5 = 65.6s (~1032 calls). Same token IDs are resampled across phases.\n"
    "   - Fix: In-process LRU cache keyed by (token_id, layer): `float[dim]`. On hit, skip llm_generate_tokens entirely.\n"
    "   - Size: 2048 entries x 576 x 4B = 4.7MB (negligible relative to model).\n"
    "   - Expected: 70-90% cache hit rate on Phase 3's ~1900 random vocab samples; Phase 3: 248s -> ~25-75s.\n"
    "   - Paper basis (SS4): Block AttnRes caches N block representations and reuses them across all L layers. Cross-stage cache eliminates V* redundant inter-stage transfers.\n"
    "\n"
    "22. **Block-Level Manifold Sampling -- Reduce ott_get_hidden_state Call Count**\n"
    "   - Problem: Phase 1 samples 256 tokens individually; Phase 3 samples ~1900 individually.\n"
    "   - Fix: Partition vocab into N=8 blocks by token_id range. Sample S tokens per block; aggregate b_n = mean(h_i for i in block_n). Use N block vectors for PCA / metric field fitting.\n"
    "   - Reduction: Phase 1: 256 -> N*S = 8*4 = 32 calls (8x). Phase 3: ~1900 -> N*S = 8*32 = 256 calls (7x). Phase 3: 248s -> ~33s at 130ms/call.\n"
    "   - Memory: O(N*d) = 8*576*4B = 18KB vs O(n_samples*d) = 590KB.\n"
    "   - Paper basis (SS3): 'N=~8 recovers most of the benefit across model scales.' Block AttnRes reduces memory from O(Ld) to O(Nd).\n"
    "\n"
    "23. **Online Softmax Merge for Phase 5 Probe Pool Scoring**\n"
    "   - Problem: cand_mat_f32[n_probe * dim] materializes all 1024 probe hidden states: 1024*576*4B = 2.3MB. GPU scores entire matrix in one shot.\n"
    "   - Fix: Process probe tokens in chunks of S=32. Maintain running (m_max, l_logsumexp, o_weighted_sum). Merge chunks via Milakov 2018 online softmax:\n"
    "     m = max(m1, m2); l = exp(m1-m)*l1 + exp(m2-m)*l2; o = (exp(m1-m)*o1 + exp(m2-m)*o2) / l\n"
    "   - Peak active memory: 32*576*4B = 72KB (32x reduction). GPU allocation drops proportionally.\n"
    "   - Paper basis (SS4 Algorithm 1 Phase 2): online softmax merge naturally admits kernel fusion with RMSNorm; used for intra-block sequential + inter-block parallel merge.\n"
    "\n"
    "24. **RMSNorm Normalization of Captured Hidden State Keys**\n"
    "   - Problem: Last-layer hidden states have widely variable magnitudes across token types; high-magnitude tokens bias Phase 3 covariance structure.\n"
    "   - Fix: After ott_get_hidden_state captures h, compute k = h / sqrt(mean(h^2) + 1e-6). Use normalized k as the metric field sample point.\n"
    "   - Expected: Phase 3 max_R reduced organically; Christoffel normalization scale factor approaches 1.0 naturally.\n"
    "   - Paper basis (SS3 eq. phi(q,k) = exp(q^T * RMSNorm(k))): 'The RMSNorm inside phi prevents layers with large-magnitude outputs from dominating the attention weights.'\n"
    "\n"
    "25. **Depth-Sink Layer Detection for Optimal Hidden State Capture Layer**\n"
    "   - Problem: ott_get_hidden_state uses layer=-1 uniformly. Certain layers are depth sinks -- they attract consistently high attention weight regardless of input, giving more stable representations.\n"
    "   - Fix: On axiom_beta_run init, probe 16 diverse tokens with tensor_bridge across all L layers. For each layer, compute variance of activation L2-norm across the 16 tokens. Layer with minimum variance = depth-sink candidate. Cache as ott_sink_layer; use for all subsequent captures.\n"
    "   - Paper basis (SS6.1 Discussions): 'Input-dependent M of AttnRes reveals depth-wise attention sinks, where certain layers consistently attract high weight regardless of input -- mirroring sequence-wise attention sinks (Xiao 2023).'\n"
    "\n"
    "26. **Two-Phase Batch I/O for Hidden State Collection (10x I/O reduction)**\n"
    "   - Problem: Each ott_get_hidden_state runs a full sequential forward pass. For Phase 3 metric sampling, all target tokens are known upfront -- ideal for batching.\n"
    "   - Fix (Appendix formula): Partition n_total_samples tokens into N=8 blocks of S tokens each.\n"
    "     Phase 1 (parallel): batch all S queries against N cached block-KV pairs via single matmul.\n"
    "     Phase 2 (sequential): walk intra-block sums; merge with Phase 1 via online softmax.\n"
    "   - I/O per token: (S+N)*d = (32+8)*576 = 23,040 floats vs naive full-scan per call.\n"
    "   - Paper formula (Appendix A): Read per layer = (S+N-2)*d, Write = 2*d. Total I/O = (S+N)*d.\n"
    "     Typical (L=54, N=9, S=6): 15*d = 8,640 floats vs naive 55*d = 31,680 floats = 3.7x I/O reduction.\n"
    "   - Requires: new llm_batch_hidden_states(token_ids[], n, layer, out[][dim]) API in llm.c.\n"
    "\n"
)

with open(FILEPATH, "r", encoding="utf-8") as f:
    src = f.read()

ANCHOR_OLD = "- Quality target: measurable gain in Phase-5 token metrics (top1/MRR) at equal compute budget.\n\n### Axiom Beta Benchmark Snapshot"
ANCHOR_NEW = "- Quality target: measurable gain in Phase-5 token metrics (top1/MRR) at equal compute budget.\n" + NEW_SECTION + "### Axiom Beta Benchmark Snapshot"

if ANCHOR_OLD in src:
    out = src.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    with open(FILEPATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Done. Size delta: +{len(out)-len(src)} bytes")
else:
    # Try to find what's near
    idx = src.find("Quality target")
    print("ANCHOR NOT FOUND. Context:", repr(src[idx:idx+200]))
