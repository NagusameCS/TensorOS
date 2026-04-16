/* =============================================================================
 * TensorOS Axiomatic Subsystem — Differential Geometry Engine (Implementation)
 *
 * Full Riemannian geometry with proper derivative-based Riemann curvature
 * and Fisher Information Matrix for real model-derived metrics.
 * =============================================================================*/

#include "runtime/nn/axiom_geo.h"
#include "runtime/nn/axiom_linalg.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/kernel.h"

#include <stddef.h>

/* ── math helpers ── */
static float geo_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    float r = x;
    for (int i = 0; i < 20; i++) r = 0.5f * (r + x / r);
    return r;
}
static float geo_fabsf(float x) { return x < 0 ? -x : x; }
static float geo_logf(float x) {
    if (x <= 0.0f) return -1e30f;
    float y = (x - 1.0f) / (x + 1.0f);
    float y2 = y * y, s = y, term = y;
    for (int k = 3; k <= 21; k += 2) { term *= y2; s += term / (float)k; }
    return 2.0f * s;
}
static float geo_expf(float x) {
    if (x > 80.0f) return 1e30f;
    if (x < -80.0f) return 0.0f;
    float sum = 1.0f, term = 1.0f;
    for (int i = 1; i <= 20; i++) { term *= x / (float)i; sum += term; }
    return sum;
}

/* ─── Metric Field ─── */

axgeo_metric_field_t axgeo_metric_field_create(int N, int dim) {
    axgeo_metric_field_t mf;
    mf.N = N;
    mf.dim = dim;
    mf.points = (float *)tensor_alloc((uint64_t)N * dim * sizeof(float));
    mf.metrics = (float *)tensor_alloc((uint64_t)N * dim * dim * sizeof(float));
    if (mf.points) kmemset(mf.points, 0, (uint64_t)N * dim * sizeof(float));
    if (mf.metrics) kmemset(mf.metrics, 0, (uint64_t)N * dim * dim * sizeof(float));
    return mf;
}

void axgeo_metric_field_destroy(axgeo_metric_field_t *mf) {
    if (mf->points) { tensor_free(mf->points); mf->points = NULL; }
    if (mf->metrics) { tensor_free(mf->metrics); mf->metrics = NULL; }
    mf->N = mf->dim = 0;
}

/* IDW interpolation: Shepard weighting with k nearest neighbors */
void axgeo_metric_at(const axgeo_metric_field_t *mf,
                     const float *x, float *g_out) {
    int N = mf->N, dim = mf->dim;
    int k = 8; if (k > N) k = N;
    int d2 = dim * dim;

    /* Find k nearest neighbors */
    float best_dist[8];  int best_idx[8];
    for (int i = 0; i < k; i++) { best_dist[i] = 1e30f; best_idx[i] = 0; }

    for (int n = 0; n < N; n++) {
        float dist2 = 0.0f;
        const float *p = mf->points + n * dim;
        for (int j = 0; j < dim; j++) { float d = x[j] - p[j]; dist2 += d * d; }
        /* Insert into sorted best list */
        for (int i = 0; i < k; i++) {
            if (dist2 < best_dist[i]) {
                for (int j = k - 1; j > i; j--) {
                    best_dist[j] = best_dist[j-1];
                    best_idx[j] = best_idx[j-1];
                }
                best_dist[i] = dist2; best_idx[i] = n;
                break;
            }
        }
    }

    /* Shepard IDW: w_i = 1/d_i^2, normalize */
    kmemset(g_out, 0, (uint64_t)d2 * sizeof(float));
    float w_sum = 0.0f;
    for (int i = 0; i < k; i++) {
        float d = geo_sqrtf(best_dist[i]);
        if (d < 1e-12f) {
            /* Exact match — copy metric directly */
            kmemcpy(g_out, mf->metrics + (uint64_t)best_idx[i] * d2, (uint64_t)d2 * sizeof(float));
            return;
        }
        float w = 1.0f / (d * d);
        w_sum += w;
        const float *g = mf->metrics + (uint64_t)best_idx[i] * d2;
        for (int j = 0; j < d2; j++) g_out[j] += w * g[j];
    }
    if (w_sum > 0.0f) {
        float inv = 1.0f / w_sum;
        for (int j = 0; j < d2; j++) g_out[j] *= inv;
    }
}

/* ─── Invert symmetric matrix (Gauss-Jordan with regularization) ─── */
static void invert_symmetric(const float *A, float *Ainv, int n) {
    float *work = (float *)tensor_alloc((uint64_t)n * 2 * n * sizeof(float));
    if (!work) {
        /* Fallback: return identity */
        kmemset(Ainv, 0, (uint64_t)n * n * sizeof(float));
        for (int i = 0; i < n; i++) Ainv[i * n + i] = 1.0f;
        return;
    }

    /* [A | I] → work */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) work[i * 2 * n + j] = A[i * n + j];
        for (int j = 0; j < n; j++) work[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
    }

    /* Regularize diagonal */
    float trace = 0.0f;
    for (int i = 0; i < n; i++) trace += geo_fabsf(A[i * n + i]);
    float reg = 1e-6f * (trace / n + 1e-10f);
    for (int i = 0; i < n; i++) work[i * 2 * n + i] += reg;

    /* Forward elimination with partial pivoting */
    for (int col = 0; col < n; col++) {
        int pivot = col; float pval = geo_fabsf(work[col * 2 * n + col]);
        for (int row = col + 1; row < n; row++) {
            float v = geo_fabsf(work[row * 2 * n + col]);
            if (v > pval) { pval = v; pivot = row; }
        }
        if (pivot != col) {
            for (int j = 0; j < 2 * n; j++) {
                float tmp = work[col * 2 * n + j];
                work[col * 2 * n + j] = work[pivot * 2 * n + j];
                work[pivot * 2 * n + j] = tmp;
            }
        }
        float diag = work[col * 2 * n + col];
        if (geo_fabsf(diag) < 1e-15f) diag = 1e-15f;
        float inv = 1.0f / diag;
        for (int j = 0; j < 2 * n; j++) work[col * 2 * n + j] *= inv;
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            float f = work[row * 2 * n + col];
            for (int j = 0; j < 2 * n; j++)
                work[row * 2 * n + j] -= f * work[col * 2 * n + j];
        }
    }

    /* Extract inverse */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ainv[i * n + j] = work[i * 2 * n + n + j];

    tensor_free(work);
}

/* ─── Christoffel Symbols ─── */

axgeo_christoffel_t axgeo_compute_christoffel(const axgeo_metric_field_t *mf) {
    axgeo_christoffel_t ch;
    int N = mf->N, dim = mf->dim;
    ch.N = N; ch.dim = dim;
    ch.gamma = (float *)tensor_alloc((uint64_t)N * dim * dim * dim * sizeof(float));
    if (!ch.gamma) return ch;
    kmemset(ch.gamma, 0, (uint64_t)N * dim * dim * dim * sizeof(float));

    int d2 = dim * dim;

    /* Compute finite-difference step from point spacing */
    float avg_spacing = 0.0f;
    int count = 0;
    for (int n = 0; n < N && n < 20; n++) {
        float min_dist = 1e30f;
        for (int m = 0; m < N; m++) {
            if (m == n) continue;
            float d = 0;
            for (int j = 0; j < dim; j++) {
                float dd = mf->points[n * dim + j] - mf->points[m * dim + j];
                d += dd * dd;
            }
            if (d < min_dist) min_dist = d;
        }
        if (min_dist < 1e20f) { avg_spacing += geo_sqrtf(min_dist); count++; }
    }
    float h = (count > 0) ? avg_spacing / count * 0.1f : 0.01f;
    if (h < 1e-6f) h = 1e-6f;

    /* Temporary buffers */
    float *g_plus = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));
    float *g_minus = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));
    float *dg = (float *)tensor_alloc((uint64_t)dim * d2 * sizeof(float));
    float *g_inv = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));
    float *g_here = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));
    float *x_pert = (float *)tensor_alloc((uint64_t)dim * sizeof(float));

    if (!g_plus || !g_minus || !dg || !g_inv || !g_here || !x_pert) goto cleanup;

    for (int n = 0; n < N; n++) {
        const float *pt = mf->points + n * dim;

        /* Get metric at this point */
        axgeo_metric_at(mf, pt, g_here);
        invert_symmetric(g_here, g_inv, dim);

        /* Compute ∂_m g_{ij} via central differences */
        for (int m = 0; m < dim; m++) {
            kmemcpy(x_pert, pt, (uint64_t)dim * sizeof(float));
            x_pert[m] = pt[m] + h;
            axgeo_metric_at(mf, x_pert, g_plus);
            x_pert[m] = pt[m] - h;
            axgeo_metric_at(mf, x_pert, g_minus);
            for (int ij = 0; ij < d2; ij++)
                dg[m * d2 + ij] = (g_plus[ij] - g_minus[ij]) / (2.0f * h);
        }

        /* Γ^k_ij = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}) */
        float *gamma_n = ch.gamma + (uint64_t)n * dim * dim * dim;
        for (int k = 0; k < dim; k++)
            for (int i = 0; i < dim; i++)
                for (int j = 0; j <= i; j++) { /* symmetric in i,j */
                    float val = 0.0f;
                    for (int l = 0; l < dim; l++) {
                        float bracket = dg[i * d2 + j * dim + l]  /* ∂_i g_{jl} */
                                      + dg[j * d2 + i * dim + l]  /* ∂_j g_{il} */
                                      - dg[l * d2 + i * dim + j]; /* ∂_l g_{ij} */
                        val += g_inv[k * dim + l] * bracket;
                    }
                    val *= 0.5f;
                    gamma_n[k * d2 + i * dim + j] = val;
                    gamma_n[k * d2 + j * dim + i] = val; /* symmetry */
                }
    }

cleanup:
    if (g_plus) tensor_free(g_plus);
    if (g_minus) tensor_free(g_minus);
    if (dg) tensor_free(dg);
    if (g_inv) tensor_free(g_inv);
    if (g_here) tensor_free(g_here);
    if (x_pert) tensor_free(x_pert);
    return ch;
}

void axgeo_christoffel_destroy(axgeo_christoffel_t *ch) {
    if (ch->gamma) { tensor_free(ch->gamma); ch->gamma = NULL; }
    ch->N = ch->dim = 0;
}

/* ─── Full Riemann Curvature Tensor ─── */
/* R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij} + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}
 *
 * The derivative terms ∂_j Γ^l_{ik} are estimated by finite differences
 * using nearby sample points where Christoffel symbols are already computed.
 */

/* Find nearest sample point index to x[dim] */
static int nearest_sample(const axgeo_metric_field_t *mf, const float *x) {
    int best = 0;
    float best_d = 1e30f;
    for (int n = 0; n < mf->N; n++) {
        float d = 0;
        for (int j = 0; j < mf->dim; j++) {
            float dd = x[j] - mf->points[n * mf->dim + j];
            d += dd * dd;
        }
        if (d < best_d) { best_d = d; best = n; }
    }
    return best;
}

/* Find k nearest neighbors to point index n, storing indices in nn_idx */
static int find_knn(const axgeo_metric_field_t *mf, int n, int *nn_idx, int k) {
    float nn_dist[16];
    int dim = mf->dim;
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) { nn_dist[i] = 1e30f; nn_idx[i] = -1; }

    const float *pt = mf->points + n * dim;
    for (int m = 0; m < mf->N; m++) {
        if (m == n) continue;
        float d = 0;
        for (int j = 0; j < dim; j++) {
            float dd = pt[j] - mf->points[m * dim + j];
            d += dd * dd;
        }
        for (int i = 0; i < k; i++) {
            if (d < nn_dist[i]) {
                for (int j = k - 1; j > i; j--) {
                    nn_dist[j] = nn_dist[j-1]; nn_idx[j] = nn_idx[j-1];
                }
                nn_dist[i] = d; nn_idx[i] = m;
                break;
            }
        }
    }

    int actual = 0;
    for (int i = 0; i < k; i++) if (nn_idx[i] >= 0) actual++;
    return actual;
}

axgeo_curvature_t axgeo_compute_curvature(const axgeo_metric_field_t *mf,
                                           const axgeo_christoffel_t *ch) {
    axgeo_curvature_t cv;
    int N = mf->N, dim = mf->dim;
    cv.N = N; cv.dim = dim;
    int d2 = dim * dim;
    int d3 = dim * dim * dim;

    cv.ricci = (float *)tensor_alloc((uint64_t)N * d2 * sizeof(float));
    cv.scalar = (float *)tensor_alloc((uint64_t)N * sizeof(float));
    if (!cv.ricci || !cv.scalar) return cv;
    kmemset(cv.ricci, 0, (uint64_t)N * d2 * sizeof(float));
    kmemset(cv.scalar, 0, (uint64_t)N * sizeof(float));

    /* For Christoffel derivatives, we use least-squares fit of Γ values
     * at neighboring points to estimate ∂_j Γ^l_{ik} */

    /* Temporary: Christoffel derivative ∂_j Γ^l_{ik} at point n */
    /* Index: dGamma[j * d3 + l * d2 + i * dim + k] = ∂_j Γ^l_{ik} */
    float *dGamma = (float *)tensor_alloc((uint64_t)dim * d3 * sizeof(float));
    float *g_inv = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));
    float *g_here = (float *)tensor_alloc((uint64_t)d2 * sizeof(float));

    if (!dGamma || !g_inv || !g_here) {
        if (dGamma) tensor_free(dGamma);
        if (g_inv) tensor_free(g_inv);
        if (g_here) tensor_free(g_here);
        return cv;
    }

    for (int n = 0; n < N; n++) {
        const float *pt = mf->points + n * dim;
        const float *gamma_n = ch->gamma + (uint64_t)n * d3;

        /* Estimate ∂_j Γ^l_{ik} via finite differences from nearest neighbors */
        kmemset(dGamma, 0, (uint64_t)dim * d3 * sizeof(float));

        int nn_idx[16];
        int n_nn = find_knn(mf, n, nn_idx, (dim + 1 < 8) ? dim + 1 : 8);

        if (n_nn >= 2) {
            /* For each coordinate direction j, find best directional derivative
             * by pairing with neighbor that has largest displacement in direction j */
            for (int j = 0; j < dim; j++) {
                /* Find neighbor with largest |delta_j| */
                int best_m = -1;
                float best_dj = 0.0f;
                for (int nn = 0; nn < n_nn; nn++) {
                    int m = nn_idx[nn];
                    float dj = mf->points[m * dim + j] - pt[j];
                    if (geo_fabsf(dj) > geo_fabsf(best_dj)) {
                        best_dj = dj; best_m = m;
                    }
                }
                if (best_m >= 0 && geo_fabsf(best_dj) > 1e-12f) {
                    const float *gamma_m = ch->gamma + (uint64_t)best_m * d3;
                    float inv_dj = 1.0f / best_dj;
                    for (int lik = 0; lik < d3; lik++)
                        dGamma[j * d3 + lik] = (gamma_m[lik] - gamma_n[lik]) * inv_dj;
                }
            }
        }

        /* Full Riemann tensor → Ricci contraction:
         * R_{ik} = R^l_{ilk} = ∂_l Γ^l_{ik} - ∂_k Γ^l_{il}
         *                    + Γ^l_{lm} Γ^m_{ik} - Γ^l_{km} Γ^m_{il}
         */
        float *ricci_n = cv.ricci + (uint64_t)n * d2;

        for (int i = 0; i < dim; i++) {
            for (int k = 0; k < dim; k++) {
                float R_ik = 0.0f;

                for (int l = 0; l < dim; l++) {
                    /* Derivative terms (∂Γ) */
                    /* ∂_l Γ^l_{ik} */
                    R_ik += dGamma[l * d3 + l * d2 + i * dim + k];
                    /* -∂_k Γ^l_{il} */
                    R_ik -= dGamma[k * d3 + l * d2 + i * dim + l];

                    /* Algebraic terms (Γ·Γ) */
                    for (int m = 0; m < dim; m++) {
                        /* + Γ^l_{lm} Γ^m_{ik} */
                        R_ik += gamma_n[l * d2 + l * dim + m] *
                                gamma_n[m * d2 + i * dim + k];
                        /* - Γ^l_{km} Γ^m_{il} */
                        R_ik -= gamma_n[l * d2 + k * dim + m] *
                                gamma_n[m * d2 + i * dim + l];
                    }
                }

                ricci_n[i * dim + k] = R_ik;
            }
        }

        /* Scalar curvature: R = g^{ij} R_{ij} */
        axgeo_metric_at(mf, pt, g_here);
        invert_symmetric(g_here, g_inv, dim);

        float R = 0.0f;
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                R += g_inv[i * dim + j] * ricci_n[i * dim + j];
        cv.scalar[n] = R;
    }

    tensor_free(dGamma);
    tensor_free(g_inv);
    tensor_free(g_here);
    return cv;
}

void axgeo_curvature_destroy(axgeo_curvature_t *cv) {
    if (cv->ricci) { tensor_free(cv->ricci); cv->ricci = NULL; }
    if (cv->scalar) { tensor_free(cv->scalar); cv->scalar = NULL; }
    cv->N = cv->dim = 0;
}

/* ─── Geodesic Solver (RK4) ─── */

/* Interpolate Christoffel at arbitrary point via nearest-sample lookup */
static void christoffel_at(const axgeo_metric_field_t *mf,
                           const axgeo_christoffel_t *ch,
                           const float *x, float *gamma_out) {
    int idx = nearest_sample(mf, x);
    int d3 = mf->dim * mf->dim * mf->dim;
    kmemcpy(gamma_out, ch->gamma + (uint64_t)idx * d3, (uint64_t)d3 * sizeof(float));
}

/* Geodesic equation: d²x^k/dt² = -Γ^k_ij dx^i/dt dx^j/dt
 * As first-order system: dx/dt = v, dv^k/dt = -Γ^k_ij v^i v^j */
static void geodesic_accel(const float *gamma, const float *v, int dim, float *a) {
    for (int k = 0; k < dim; k++) {
        float ak = 0.0f;
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                ak -= gamma[k * dim * dim + i * dim + j] * v[i] * v[j];
        a[k] = ak;
    }
}

axgeo_geodesic_t axgeo_solve_geodesic(const axgeo_metric_field_t *mf,
                                       const float *x0, const float *v0,
                                       float dt, int max_steps) {
    axgeo_geodesic_t g;
    int dim = mf->dim;
    g.dim = dim;
    g.max_steps = max_steps;
    g.n_steps = 0;
    g.diverged = 0;

    g.trajectory = (float *)tensor_alloc((uint64_t)(max_steps + 1) * dim * sizeof(float));
    g.velocity = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!g.trajectory || !g.velocity) return g;

    /* Compute Christoffel for the metric field */
    axgeo_christoffel_t ch = axgeo_compute_christoffel(mf);
    if (!ch.gamma) return g;

    /* RK4 temporaries */
    float *x = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *v = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *gamma_buf = (float *)tensor_alloc((uint64_t)dim * dim * dim * sizeof(float));
    float *kx1 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kv1 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kx2 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kv2 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kx3 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kv3 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kx4 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *kv4 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *xt = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *vt = (float *)tensor_alloc((uint64_t)dim * sizeof(float));

    if (!x || !v || !gamma_buf || !kx1 || !kv1 || !kx2 || !kv2 ||
        !kx3 || !kv3 || !kx4 || !kv4 || !xt || !vt) {
        axgeo_christoffel_destroy(&ch);
        goto rk4_cleanup;
    }

    /* Initial conditions */
    kmemcpy(x, x0, (uint64_t)dim * sizeof(float));
    kmemcpy(v, v0, (uint64_t)dim * sizeof(float));
    kmemcpy(g.trajectory, x, (uint64_t)dim * sizeof(float));

    for (int step = 0; step < max_steps; step++) {
        /* k1 */
        for (int i = 0; i < dim; i++) kx1[i] = v[i];
        christoffel_at(mf, &ch, x, gamma_buf);
        geodesic_accel(gamma_buf, v, dim, kv1);

        /* k2 */
        for (int i = 0; i < dim; i++) {
            xt[i] = x[i] + 0.5f * dt * kx1[i];
            vt[i] = v[i] + 0.5f * dt * kv1[i];
        }
        for (int i = 0; i < dim; i++) kx2[i] = vt[i];
        christoffel_at(mf, &ch, xt, gamma_buf);
        geodesic_accel(gamma_buf, vt, dim, kv2);

        /* k3 */
        for (int i = 0; i < dim; i++) {
            xt[i] = x[i] + 0.5f * dt * kx2[i];
            vt[i] = v[i] + 0.5f * dt * kv2[i];
        }
        for (int i = 0; i < dim; i++) kx3[i] = vt[i];
        christoffel_at(mf, &ch, xt, gamma_buf);
        geodesic_accel(gamma_buf, vt, dim, kv3);

        /* k4 */
        for (int i = 0; i < dim; i++) {
            xt[i] = x[i] + dt * kx3[i];
            vt[i] = v[i] + dt * kv3[i];
        }
        for (int i = 0; i < dim; i++) kx4[i] = vt[i];
        christoffel_at(mf, &ch, xt, gamma_buf);
        geodesic_accel(gamma_buf, vt, dim, kv4);

        /* Update */
        for (int i = 0; i < dim; i++) {
            x[i] += dt / 6.0f * (kx1[i] + 2.0f * kx2[i] + 2.0f * kx3[i] + kx4[i]);
            v[i] += dt / 6.0f * (kv1[i] + 2.0f * kv2[i] + 2.0f * kv3[i] + kv4[i]);
        }

        /* Check for divergence */
        float vnorm = 0.0f;
        for (int i = 0; i < dim; i++) vnorm += v[i] * v[i];
        if (vnorm > 1e12f || vnorm != vnorm) { /* NaN check */
            g.diverged = 1;
            break;
        }

        g.n_steps = step + 1;
        kmemcpy(g.trajectory + (uint64_t)(step + 1) * dim, x, (uint64_t)dim * sizeof(float));
    }

    kmemcpy(g.velocity, v, (uint64_t)dim * sizeof(float));

    axgeo_christoffel_destroy(&ch);

rk4_cleanup:
    if (x) tensor_free(x);
    if (v) tensor_free(v);
    if (gamma_buf) tensor_free(gamma_buf);
    if (kx1) tensor_free(kx1); if (kv1) tensor_free(kv1);
    if (kx2) tensor_free(kx2); if (kv2) tensor_free(kv2);
    if (kx3) tensor_free(kx3); if (kv3) tensor_free(kv3);
    if (kx4) tensor_free(kx4); if (kv4) tensor_free(kv4);
    if (xt) tensor_free(xt); if (vt) tensor_free(vt);
    return g;
}

void axgeo_geodesic_destroy(axgeo_geodesic_t *g) {
    if (g->trajectory) { tensor_free(g->trajectory); g->trajectory = NULL; }
    if (g->velocity) { tensor_free(g->velocity); g->velocity = NULL; }
    g->n_steps = g->max_steps = g->dim = 0;
}

float axgeo_geodesic_length(const axgeo_geodesic_t *g,
                            const axgeo_metric_field_t *mf) {
    if (!g->trajectory || g->n_steps < 1) return 0.0f;
    int dim = g->dim;
    float length = 0.0f;

    float *g_buf = (float *)tensor_alloc((uint64_t)dim * dim * sizeof(float));
    float *diff = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!g_buf || !diff) {
        if (g_buf) tensor_free(g_buf);
        if (diff) tensor_free(diff);
        return 0.0f;
    }

    for (int s = 0; s < g->n_steps; s++) {
        const float *p0 = g->trajectory + (uint64_t)s * dim;
        const float *p1 = g->trajectory + (uint64_t)(s + 1) * dim;

        /* Midpoint metric */
        float *mid = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
        if (!mid) break;
        for (int i = 0; i < dim; i++) {
            mid[i] = 0.5f * (p0[i] + p1[i]);
            diff[i] = p1[i] - p0[i];
        }
        axgeo_metric_at(mf, mid, g_buf);
        tensor_free(mid);

        /* ds² = g_ij dx^i dx^j */
        float ds2 = 0.0f;
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                ds2 += g_buf[i * dim + j] * diff[i] * diff[j];
        if (ds2 > 0.0f) length += geo_sqrtf(ds2);
    }

    tensor_free(g_buf);
    tensor_free(diff);
    return length;
}

/* ─── Fisher Information Matrix ─── */

axgeo_fisher_t axgeo_fisher_create(int dim) {
    axgeo_fisher_t f;
    f.dim = dim;
    f.n_probes = 0;
    f.matrix = (float *)tensor_alloc((uint64_t)dim * dim * sizeof(float));
    if (f.matrix) kmemset(f.matrix, 0, (uint64_t)dim * dim * sizeof(float));
    return f;
}

void axgeo_fisher_destroy(axgeo_fisher_t *f) {
    if (f->matrix) { tensor_free(f->matrix); f->matrix = NULL; }
    f->dim = f->n_probes = 0;
}

/* Compute FIM via finite-difference perturbation of embeddings.
 *
 * F_{ij} ≈ E_k[ ∂log p(y|θ)/∂θ_i · ∂log p(y|θ)/∂θ_j ]
 *
 * Estimated by: perturb θ in direction e_i by ±ε, measure KL-divergence
 * of output distributions, accumulate outer products of score vectors.
 *
 * For efficiency, we use random probe directions instead of axis-aligned,
 * then recover the matrix via E[v v^T · s²] where s = directional score.
 */
void axgeo_compute_fisher(axgeo_fisher_t *fim,
                          axgeo_embed_fn embed_fn, void *ctx,
                          const float *base_embed, int dim,
                          int vocab_size, int n_probes, float epsilon) {
    if (!fim->matrix || !embed_fn || dim <= 0) return;

    int logit_cap = vocab_size;
    if (logit_cap > 4096) logit_cap = 4096; /* cap for memory */

    float *base_logits = (float *)tensor_alloc((uint64_t)logit_cap * sizeof(float));
    float *pert_logits = (float *)tensor_alloc((uint64_t)logit_cap * sizeof(float));
    float *pert_embed = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *direction = (float *)tensor_alloc((uint64_t)dim * sizeof(float));

    if (!base_logits || !pert_logits || !pert_embed || !direction) goto fisher_cleanup;

    /* Get base output distribution */
    embed_fn(base_embed, dim, base_logits, logit_cap, ctx);

    /* Softmax the base logits for KL computation */
    {
        float max_l = base_logits[0];
        for (int i = 1; i < logit_cap; i++)
            if (base_logits[i] > max_l) max_l = base_logits[i];
        float sum = 0.0f;
        for (int i = 0; i < logit_cap; i++) {
            base_logits[i] = geo_expf(base_logits[i] - max_l);
            sum += base_logits[i];
        }
        float inv_sum = 1.0f / (sum + 1e-30f);
        for (int i = 0; i < logit_cap; i++) base_logits[i] *= inv_sum;
    }

    ax_rng_t rng;
    ax_rng_seed(&rng, 42 + (uint32_t)(uint64_t)base_embed);

    kmemset(fim->matrix, 0, (uint64_t)dim * dim * sizeof(float));

    for (int probe = 0; probe < n_probes; probe++) {
        /* Random unit direction */
        float norm = 0.0f;
        for (int i = 0; i < dim; i++) {
            direction[i] = ax_rng_normal(&rng);
            norm += direction[i] * direction[i];
        }
        norm = geo_sqrtf(norm);
        if (norm < 1e-12f) continue;
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < dim; i++) direction[i] *= inv_norm;

        /* Perturb embedding: θ + ε·v */
        for (int i = 0; i < dim; i++)
            pert_embed[i] = base_embed[i] + epsilon * direction[i];

        /* Forward pass with perturbed embedding */
        embed_fn(pert_embed, dim, pert_logits, logit_cap, ctx);

        /* Softmax perturbed logits */
        {
            float max_l = pert_logits[0];
            for (int i = 1; i < logit_cap; i++)
                if (pert_logits[i] > max_l) max_l = pert_logits[i];
            float sum = 0.0f;
            for (int i = 0; i < logit_cap; i++) {
                pert_logits[i] = geo_expf(pert_logits[i] - max_l);
                sum += pert_logits[i];
            }
            float inv_sum = 1.0f / (sum + 1e-30f);
            for (int i = 0; i < logit_cap; i++) pert_logits[i] *= inv_sum;
        }

        /* Compute directional score:
         * s = (1/ε) * Σ_y p(y|θ) * [log p(y|θ+εv) - log p(y|θ)]²
         * This is the directional Fisher info in direction v */
        float score = 0.0f;
        for (int y = 0; y < logit_cap; y++) {
            if (base_logits[y] > 1e-15f && pert_logits[y] > 1e-15f) {
                float log_ratio = geo_logf(pert_logits[y]) - geo_logf(base_logits[y]);
                score += base_logits[y] * log_ratio * log_ratio;
            }
        }
        score /= (epsilon * epsilon + 1e-30f);

        /* Accumulate: F += score * v v^T */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j <= i; j++) {
                float val = score * direction[i] * direction[j];
                fim->matrix[i * dim + j] += val;
                if (i != j) fim->matrix[j * dim + i] += val;
            }
    }

    /* Average over probes */
    if (n_probes > 0) {
        float inv_n = 1.0f / (float)n_probes;
        for (int i = 0; i < dim * dim; i++) fim->matrix[i] *= inv_n;
    }

    /* Scale correction: random projection estimator needs factor dim
     * because E[v_i v_j] = δ_{ij}/dim for unit random vectors */
    for (int i = 0; i < dim * dim; i++) fim->matrix[i] *= (float)dim;

    fim->n_probes = n_probes;

fisher_cleanup:
    if (base_logits) tensor_free(base_logits);
    if (pert_logits) tensor_free(pert_logits);
    if (pert_embed) tensor_free(pert_embed);
    if (direction) tensor_free(direction);
}

/* ─── Fisher to Metric Field ─── */
axgeo_metric_field_t axgeo_fisher_to_metric_field(
    const float *sample_points, const float *fisher_matrices,
    int n_samples, int dim) {
    axgeo_metric_field_t mf = axgeo_metric_field_create(n_samples, dim);
    if (!mf.points || !mf.metrics) return mf;

    kmemcpy(mf.points, sample_points, (uint64_t)n_samples * dim * sizeof(float));
    kmemcpy(mf.metrics, fisher_matrices, (uint64_t)n_samples * dim * dim * sizeof(float));
    return mf;
}
