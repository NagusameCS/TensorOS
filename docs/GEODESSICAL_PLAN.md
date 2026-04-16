# Geodesic Axiomatic Subsystem — Development Plan

Based on Organic Training Theory (OTT): model weights form a Riemannian manifold
whose geodesics approximate the transformer forward pass.

## Architecture

```
axiom_linalg.h/c  — Dense linear algebra: PCA, TwoNN, dequantization, vector ops
axiom_geo.h/c     — Differential geometry: metric field, Christoffel, Riemann, geodesic, Fisher
axiom_beta.h/c    — 5-phase discovery pipeline: manifold ID → symmetry → curvature → axioms → geodesic
```

## Milestone Status

### Beta-3: Real Geometry ✅ (current)
- [x] Fisher Information Matrix via embedding perturbation (random probe estimator)
- [x] Full Riemann curvature tensor (∂Γ derivative terms + Γ·Γ algebraic terms)
- [x] Real symmetry mining (head-subspace distributional cosine similarity)
- [x] Geodesic forward pass using real Phase 3 metric (not synthetic)
- [x] Phase 3→5 metric sharing (persistent metric field)
- [x] Geometry-derived axiom candidates (dimension, symmetry, curvature, metric structure)
- [x] Build integration (build.ps1 sources + llm.h declarations)

### Beta-4: Geodesic Inference Prototype
- [ ] Full geodesic → logit projection pipeline
- [ ] Side-by-side comparison with transformer forward pass
- [ ] O(n·k²) complexity validation
- [ ] Multi-layer geodesic (layer-by-layer metric fields)

### Beta-5: Knowledge Injection
- [ ] Axiom-guided weight perturbation
- [ ] Geodesic-based fine-tuning
- [ ] Active axiom learning with formal grammar

### v1.0: Production
- [ ] Validated O(n·k²) geodesic inference
- [ ] Full OTT theorem verification
- [ ] Performance benchmarks vs transformer baseline
