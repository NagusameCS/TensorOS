/* =============================================================================
 * TensorOS Axiomatic Subsystem — Dense Linear Algebra
 *
 * Provides core linear algebra routines for the geodesic axiomatic pipeline:
 *   - Dense matrix type (row-major, heap-allocated)
 *   - Jacobi eigendecomposition for real symmetric matrices
 *   - Economy PCA (keep only top-k components by variance)
 *   - TwoNN intrinsic dimension estimator (Facco et al. 2017)
 *   - GGUF weight dequantization (Q4_0, Q8_0, Q4_K, Q6_K, F16, F32)
 *   - Vector utilities (dot, norm, add, scale)
 * =============================================================================*/

#ifndef TENSOROS_AXIOM_LINALG_H
#define TENSOROS_AXIOM_LINALG_H

#include <stdint.h>

/* ─── Dense matrix (row-major) ─── */
typedef struct {
    float *data;   /* rows * cols floats */
    int    rows;
    int    cols;
} axmat_t;

axmat_t  axmat_create(int rows, int cols);
void     axmat_destroy(axmat_t *m);
void     axmat_zero(axmat_t *m);
float    axmat_get(const axmat_t *m, int r, int c);
void     axmat_set(axmat_t *m, int r, int c, float v);

/* ─── Eigendecomposition (Jacobi, symmetric only) ─── */
/* Decomposes A = V * diag(eigenvalues) * V^T in-place.
 * eigenvalues[dim], eigenvectors is dim x dim column-major. */
void ax_jacobi_eigen(const float *A, int dim, float *eigenvalues, float *eigenvectors);

/* ─── Economy PCA ─── */
typedef struct {
    float  *mean;        /* [dim] column means */
    float  *components;  /* [n_components * dim] row-major PC vectors */
    float  *variances;   /* [n_components] eigenvalues (variances) */
    float   total_var;   /* sum of all eigenvalues */
    int     dim;         /* original dimensionality */
    int     n_components;/* number of retained components */
} axpca_t;

/* Compute PCA from data[n_samples * dim]. Retains components explaining
 * >= variance_ratio of total variance, capped at max_components. */
axpca_t ax_pca_fit(const float *data, int n_samples, int dim,
                   float variance_ratio, int max_components);
void ax_pca_destroy(axpca_t *pca);

/* Project x[dim] → out[n_components] using fitted PCA. */
void ax_pca_project(const axpca_t *pca, const float *x, float *out);

/* ─── TwoNN intrinsic dimension estimator ─── */
/* Estimates intrinsic dimensionality from data[n_samples * dim].
 * Returns estimated dimension (>= 1). */
float ax_twonn_estimate(const float *data, int n_samples, int dim);

/* ─── GGUF weight dequantization ─── */
/* Dequantize a row of GGUF weights into out[cols].
 * Supports Q4_0, Q8_0, Q4_K, Q6_K, F16, F32, BF16. */
void ax_dequant_row(float *out, const void *data, int cols, int ggml_type);

/* ─── Vector utilities ─── */
float  ax_vec_dot(const float *a, const float *b, int n);
float  ax_vec_norm(const float *a, int n);
void   ax_vec_add(float *dst, const float *a, const float *b, int n);
void   ax_vec_sub(float *dst, const float *a, const float *b, int n);
void   ax_vec_scale(float *dst, const float *a, float s, int n);
float  ax_vec_cosine(const float *a, const float *b, int n);

/* ─── Simple PRNG (xoshiro128+) ─── */
typedef struct { uint32_t s[4]; } ax_rng_t;
void     ax_rng_seed(ax_rng_t *rng, uint32_t seed);
uint32_t ax_rng_next(ax_rng_t *rng);
float    ax_rng_uniform(ax_rng_t *rng);           /* [0, 1) */
float    ax_rng_normal(ax_rng_t *rng);             /* N(0,1) via Box-Muller */
int      ax_rng_range(ax_rng_t *rng, int lo, int hi); /* [lo, hi) */

#endif /* TENSOROS_AXIOM_LINALG_H */
