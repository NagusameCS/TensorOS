# OTT k⁴ Full Realization Plan

This is the engineering plan for turning the paper's k^4 claim into a concrete implementation path.

## Mathematical Foundation

The paper's core claim: for a k-dimensional Riemannian submanifold embedded in the
d-dimensional transformer activation space, precompute Christoffel symbols Γ^μ_νρ
**once** at O(k⁴) cost, then run O(k²) geodesic integration per token instead of
the full O(d²) transformer forward pass.

### What "k⁴" Means

For k sample points in k-dimensional coordinate space:
- k³ Christoffel entries per sample point (Γ^μ_νρ, μ,ν,ρ ∈ [0,k))
- Finite-difference metric derivatives require k neighbors
- Total work: k (points) × k³ (symbol entries) × k (FD neighbors) = **k⁴**
- For Gemma-4 with estimated intrinsic dim k≈24: k⁴ = 331,776 operations

This is a **one-time precomputation** run once at model load.

### The Missing Piece: Pullback Metric

**Current (approximation):** metric = token embedding covariance in PCA subspace  
**Required (exact):** metric = pullback of transformer Jacobian through PCA basis

For a linear layer y = Wx, the induced (pullback) metric on input space:
```
g_ij = Σ_k W_ki · W_kj     (Fisher metric = W^T W)
```

For an RMSNorm(x) layer, the induced metric is the **Fisher metric on the
(d-1)-sphere**: g_ij = δ_ij/||x||² - x_i x_j/||x||⁴

**The diffeomorphism ϕ** (paper §11): the map that absorbs LayerNorm into
Christoffel curvature. For RMSNorm: ϕ maps activations to their normalized
counterparts on the sphere, and the connection picks up a correction term:

```
Γ^k_ij += (norm correction from RMSNorm sphere geometry)
         = -δ^k_j x_i/||x||² - δ^k_i x_j/||x||² + 2 x^k x_i x_j/||x||⁴
```

This correction is computable from the input activation alone.

---

## Implementation Roadmap

### Step 1 — Weight-Derived Pullback Metric (IMPLEMENT NOW)

**File:** `runtime/nn/axiom_geo.c` + `axiom_beta.c`

Replace Phase 3 metric (embedding covariance) with:
1. Take U ∈ R^{d×k} from Phase 1 PCA basis (already computed)
2. For each sample position (embedding projection p ∈ R^k):
   - Find nearest real token embedding
   - Dequantize the QKV weight matrices for that token's layer
   - Pullback: G_p = (W · U)^T (W · U)  ∈ R^{k×k}
   - This is the true local geometry at position p
3. Blend with Fisher Info as before, but now G_p is weight-grounded

**New function:** `axgeo_build_metric_from_weights(mf, U, k, d, layers, n_layers)`

**Expected improvement:** Christoffel symbols now reflect actual model curvature, not
just embedding distribution statistics. Geodesic endpoint closer to true output.

### Step 2 — Layer-Stratified Christoffel (precompute per-layer)

Current: one global metric field for all layers  
Required: per-layer Christoffel symbols Γ^μ_νρ[L] for L ∈ [0, n_layers)

**Why:** The transformer residual stream x evolves through layers. At layer L,
the relevant geometry is the pullback of layers 0..L of the weight stack.

**Storage:** n_layers × k³ doubles
- k=24, n_layers=35: 35 × 13,824 × 8 bytes = ~3.8 MB ← completely feasible

**New struct:** `axgeo_layer_christoffel_t` (array of Christoffel per layer)

### Step 3 — Geodesic Hot Path Integration

**File:** `runtime/nn/llm.c` (decode loop)

After each layer's residual update `x += attn_out`:
1. Project x → p ∈ R^k via U^T x  (k matrix-vector: O(k²))  
2. Geodesic step: p' = p + v·dt - ½Γ^k_ij v^i v^j dt²  (O(k²))
3. Logit correction: map geodesic residual back to d-space and add as bias

**Gating condition:** only activate when `||proj - prev_proj|| > curvature_threshold`
i.e., only when the activation has moved significantly in the manifold.

**No performance cost when inactive.** When active, adds ~k² = 576 FLOPs per layer —
negligible vs. the 1536×1536 GEMV per layer.

### Step 4 — RMSNorm Sphere Absorption (ϕ construction)

**File:** `runtime/nn/axiom_geo.c`

Implement the connection correction for RMSNorm:

```c
// At each layer's pre-attn RMSNorm, add Christoffel correction:
// ΔΓ^k_ij = -(δ^k_j p_i + δ^k_i p_j)/||p||² + 2 p^k p_i p_j/||p||⁴
void axgeo_apply_rmsnorm_connection(double *gamma, const double *p, int k);
```

This makes the geodesic path account for the normalization operation,
which is the key mathematical step from the paper.

### Step 5 — GRC Library (Jacobi Reuse Cache)

When two tokens produce similar p_0 under Phase 1 PCA:
- Cache the geodesic correction vector δ ∈ R^k
- Reuse with Jacobi field correction: δ' = δ + J(δ) × Δp  where J = ∂²geodesic/∂p_0²
- This is the "GRC O(k²) per hit" claim: Jacobi field along stored geodesic

**Cache:** LRU of ~512 geodesic trajectories, keyed by (token_id, layer_bucket)

### Step 6 — Sampling Logit Shortcut

When geodesic error < threshold ε:
- Skip the final LM head GEMV (d×vocab = huge)
- Use geodesic endpoint to bias the existing logit distribution
- Only needed when model is operating in expected regime

This is the 4800× speedup claim — most tokens are "expected" and geodesic suffices.

---

## Current Gaps vs. Paper

| Paper Requirement | Current State | Gap |
|---|---|---|
| Metric from weight Jacobians | Embedding covariance | **Step 1** |
| Per-layer Christoffel | Single global field | **Step 2** |
| Hot-path geodesic integration | Phase 5 only (offline) | **Step 3** |
| RMSNorm ϕ construction | No absorption | **Step 4** |
| GRC reuse cache | None | **Step 5** |
| Sampling shortcut | None | **Step 6** |

---

## Implementation Order (Priority)

1. **Step 1** → most impactful: better geometry = better geodesics
2. **Step 3** → wire to decode: makes it real-time instead of offline analysis
3. **Step 4** → ϕ construction: mathematical correctness
4. **Step 2** → per-layer: better accuracy  
5. **Step 5** → GRC cache: speedup path
6. **Step 6** → sampling shortcut: the 4800× speedup

---

## Key Numbers for Gemma-4 E2B

- d = 1536 (model dim)
- k ≈ 24 (estimated TwoNN intrinsic dimension from Phase 1 runs)
- n_layers = 35
- k^4 = 331,776 operations (precomputation, ~1ms on CPU)
- k^3 × n_layers = 483,840 doubles = 3.9MB (Christoffel storage per-layer)
- Per-token geodesic cost: k² × n_layers × 2 = 40,320 FLOPs ← ~0.01% of GEMV cost

The geodesic hot-path correction is essentially **free** computationally.
The question is accuracy — does it help token quality?
