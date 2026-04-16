/* =============================================================================
 * TensorOS Axiomatic Subsystem — Differential Geometry Engine
 *
 * Implements the Riemannian geometry primitives from Organic Training Theory:
 *   - Sampled metric tensor field with IDW interpolation
 *   - Christoffel symbols Γ^k_ij via finite differences of g_ij
 *   - Full Riemann curvature tensor R^l_{ijk} with derivative + algebraic terms
 *   - Ricci tensor R_ij and scalar curvature R
 *   - RK4 geodesic solver with trajectory recording
 *   - Fisher Information Matrix computation via forward-pass perturbation
 * =============================================================================*/

#ifndef TENSOROS_AXIOM_GEO_H
#define TENSOROS_AXIOM_GEO_H

#include <stdint.h>

/* Maximum manifold dimension for stack-sized temporaries */
#define AXGEO_MAX_DIM 256

/* ─── Sampled Metric Tensor Field ─── */
typedef struct {
    float *points;    /* [N * dim]  sample point coordinates */
    float *metrics;   /* [N * dim * dim]  metric tensor at each sample */
    int    N;         /* number of sample points */
    int    dim;       /* manifold dimension */
} axgeo_metric_field_t;

axgeo_metric_field_t axgeo_metric_field_create(int N, int dim);
void axgeo_metric_field_destroy(axgeo_metric_field_t *mf);

/* Interpolate metric at arbitrary point x[dim] using IDW (k-nearest, Shepard) */
void axgeo_metric_at(const axgeo_metric_field_t *mf,
                     const float *x, float *g_out);

/* ─── Christoffel Symbols ─── */
typedef struct {
    float *gamma;     /* [N * dim * dim * dim]  Γ^k_ij at each sample point */
    int    N;
    int    dim;
} axgeo_christoffel_t;

axgeo_christoffel_t axgeo_compute_christoffel(const axgeo_metric_field_t *mf);
void axgeo_christoffel_destroy(axgeo_christoffel_t *ch);

/* ─── Curvature (Ricci + scalar) ─── */
typedef struct {
    float *ricci;     /* [N * dim * dim]  Ricci tensor R_ij at each sample */
    float *scalar;    /* [N]  scalar curvature R at each sample */
    int    N;
    int    dim;
} axgeo_curvature_t;

/* Compute full Riemann curvature tensor including derivative terms */
axgeo_curvature_t axgeo_compute_curvature(const axgeo_metric_field_t *mf,
                                           const axgeo_christoffel_t *ch);
void axgeo_curvature_destroy(axgeo_curvature_t *cv);

/* ─── Geodesic Solver (RK4) ─── */
typedef struct {
    float *trajectory;   /* [(max_steps+1) * dim]  recorded positions */
    float *velocity;     /* [dim]  current velocity */
    int    n_steps;      /* steps actually taken */
    int    max_steps;
    int    dim;
    int    diverged;     /* 1 if solver detected divergence */
} axgeo_geodesic_t;

/* Solve geodesic from x0[dim] with initial velocity v0[dim].
 * Uses metric field for Christoffel interpolation at each step. */
axgeo_geodesic_t axgeo_solve_geodesic(const axgeo_metric_field_t *mf,
                                       const float *x0, const float *v0,
                                       float dt, int max_steps);
void axgeo_geodesic_destroy(axgeo_geodesic_t *g);

/* Compute arc length of a geodesic trajectory */
float axgeo_geodesic_length(const axgeo_geodesic_t *g,
                            const axgeo_metric_field_t *mf);

/* ─── Fisher Information Matrix ─── */
typedef struct {
    float *matrix;    /* [dim * dim]  Fisher Information Matrix */
    int    dim;       /* dimension (= model hidden dim or PCA dim) */
    int    n_probes;  /* number of perturbation probes used */
} axgeo_fisher_t;

axgeo_fisher_t axgeo_fisher_create(int dim);
void axgeo_fisher_destroy(axgeo_fisher_t *f);

/* Compute FIM via embedding perturbation:
 * For n_probes random directions, perturb embedding, measure output
 * distribution change (KL-divergence proxy), build outer products.
 *
 * embed_fn: given perturbed embedding[dim], runs forward pass and writes
 *           output logits[vocab_size] (softmax'd).
 * base_embed: the unperturbed embedding vector [dim].
 * epsilon: perturbation magnitude.
 */
typedef void (*axgeo_embed_fn)(const float *embed, int dim,
                               float *logits, int vocab_size, void *ctx);

void axgeo_compute_fisher(axgeo_fisher_t *fim,
                          axgeo_embed_fn embed_fn, void *ctx,
                          const float *base_embed, int dim,
                          int vocab_size, int n_probes, float epsilon);

/* Build metric field from Fisher matrices at multiple sample points */
axgeo_metric_field_t axgeo_fisher_to_metric_field(
    const float *sample_points, const float *fisher_matrices,
    int n_samples, int dim);

#endif /* TENSOROS_AXIOM_GEO_H */
