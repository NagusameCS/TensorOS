/* =============================================================================
 * TensorOS Axiomatic Subsystem — Dense Linear Algebra (Implementation)
 * =============================================================================*/

#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/gguf.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/kernel.h"

#include <stddef.h>

/* ── helpers ── */
static float ax_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    float r = x;
    for (int i = 0; i < 20; i++) r = 0.5f * (r + x / r);
    return r;
}
static float ax_fabsf(float x) { return x < 0 ? -x : x; }
static float ax_logf(float x) {
    if (x <= 0.0f) return -1e30f;
    /* ln via Padé: ln((1+t)/(1-t)) = 2*(t + t^3/3 + t^5/5 + ...) */
    float y = (x - 1.0f) / (x + 1.0f);
    float y2 = y * y, s = y;
    for (int k = 3; k <= 21; k += 2) { y *= y2; s += y / (float)k; }
    return 2.0f * s;
}

/* ── PRNG (xoshiro128+) ── */
static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

void ax_rng_seed(ax_rng_t *rng, uint32_t seed) {
    for (int i = 0; i < 4; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        rng->s[i] = seed;
    }
}

uint32_t ax_rng_next(ax_rng_t *rng) {
    uint32_t result = rng->s[0] + rng->s[3];
    uint32_t t = rng->s[1] << 9;
    rng->s[2] ^= rng->s[0]; rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2]; rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t; rng->s[3] = rotl(rng->s[3], 11);
    return result;
}

float ax_rng_uniform(ax_rng_t *rng) {
    return (float)(ax_rng_next(rng) >> 8) / 16777216.0f;
}

float ax_rng_normal(ax_rng_t *rng) {
    float u1 = ax_rng_uniform(rng) + 1e-10f;
    float u2 = ax_rng_uniform(rng);
    float r = ax_sqrtf(-2.0f * ax_logf(u1));
    float theta = 6.283185307f * u2;
    /* sin via Taylor */
    float x = theta;
    while (x > 3.14159265f) x -= 6.28318530f;
    while (x < -3.14159265f) x += 6.28318530f;
    float x2 = x * x;
    float s = x * (1.0f - x2 / 6.0f * (1.0f - x2 / 20.0f * (1.0f - x2 / 42.0f)));
    return r * s;
}

int ax_rng_range(ax_rng_t *rng, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(ax_rng_uniform(rng) * (float)(hi - lo));
}

/* ── Dense matrix ── */
axmat_t axmat_create(int rows, int cols) {
    axmat_t m;
    m.rows = rows; m.cols = cols;
    m.data = (float *)tensor_alloc((uint64_t)rows * cols * sizeof(float));
    if (m.data) kmemset(m.data, 0, (uint64_t)rows * cols * sizeof(float));
    return m;
}

void axmat_destroy(axmat_t *m) {
    if (m->data) { tensor_free(m->data); m->data = NULL; }
    m->rows = m->cols = 0;
}

void axmat_zero(axmat_t *m) {
    if (m->data) kmemset(m->data, 0, (uint64_t)m->rows * m->cols * sizeof(float));
}

float axmat_get(const axmat_t *m, int r, int c) {
    return m->data[r * m->cols + c];
}

void axmat_set(axmat_t *m, int r, int c, float v) {
    m->data[r * m->cols + c] = v;
}

/* ── Vector utilities ── */
float ax_vec_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

float ax_vec_norm(const float *a, int n) {
    return ax_sqrtf(ax_vec_dot(a, a, n));
}

void ax_vec_add(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

void ax_vec_sub(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] - b[i];
}

void ax_vec_scale(float *dst, const float *a, float s, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * s;
}

float ax_vec_cosine(const float *a, const float *b, int n) {
    float d = ax_vec_dot(a, b, n);
    float na = ax_vec_norm(a, n);
    float nb = ax_vec_norm(b, n);
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return d / (na * nb);
}

/* ── Jacobi eigendecomposition (real symmetric) ── */
void ax_jacobi_eigen(const float *A, int dim, float *eigenvalues, float *eigenvectors) {
    int n = dim;
    /* Copy A into working matrix (row-major) */
    float *S = (float *)tensor_alloc((uint64_t)n * n * sizeof(float));
    if (!S) return;
    kmemcpy(S, A, (uint64_t)n * n * sizeof(float));

    /* Init eigenvectors to identity */
    kmemset(eigenvectors, 0, (uint64_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++) eigenvectors[i * n + i] = 1.0f;

    int max_iter = 100 * n * n;
    if (max_iter > 50000) max_iter = 50000;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Find largest off-diagonal |S[p][q]| */
        int p = 0, q = 1;
        float maxval = 0.0f;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (ax_fabsf(S[i * n + j]) > maxval) {
                    maxval = ax_fabsf(S[i * n + j]);
                    p = i; q = j;
                }
        if (maxval < 1e-10f) break;

        /* Compute rotation angle */
        float Spq = S[p * n + q];
        float diff = S[q * n + q] - S[p * n + p];
        float t;
        if (ax_fabsf(diff) < 1e-20f) {
            t = 1.0f;
        } else {
            float tau = diff / (2.0f * Spq);
            t = 1.0f / (ax_fabsf(tau) + ax_sqrtf(1.0f + tau * tau));
            if (tau < 0) t = -t;
        }
        float c = 1.0f / ax_sqrtf(1.0f + t * t);
        float s = t * c;

        /* Apply Givens rotation to S */
        float Spp = S[p * n + p], Sqq = S[q * n + q];
        S[p * n + p] = Spp - t * Spq;
        S[q * n + q] = Sqq + t * Spq;
        S[p * n + q] = 0.0f;
        S[q * n + p] = 0.0f;

        for (int r = 0; r < n; r++) {
            if (r == p || r == q) continue;
            float Srp = S[r * n + p], Srq = S[r * n + q];
            S[r * n + p] = S[p * n + r] = c * Srp - s * Srq;
            S[r * n + q] = S[q * n + r] = s * Srp + c * Srq;
        }

        /* Accumulate eigenvectors */
        for (int r = 0; r < n; r++) {
            float Vp = eigenvectors[r * n + p], Vq = eigenvectors[r * n + q];
            eigenvectors[r * n + p] = c * Vp - s * Vq;
            eigenvectors[r * n + q] = s * Vp + c * Vq;
        }
    }

    /* Extract eigenvalues from diagonal */
    for (int i = 0; i < n; i++) eigenvalues[i] = S[i * n + i];

    /* Sort by descending eigenvalue (bubble) */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (eigenvalues[j] > eigenvalues[i]) {
                float tmp = eigenvalues[i]; eigenvalues[i] = eigenvalues[j]; eigenvalues[j] = tmp;
                for (int k = 0; k < n; k++) {
                    tmp = eigenvectors[k * n + i];
                    eigenvectors[k * n + i] = eigenvectors[k * n + j];
                    eigenvectors[k * n + j] = tmp;
                }
            }

    tensor_free(S);
}

/* ── Economy PCA ── */
axpca_t ax_pca_fit(const float *data, int n_samples, int dim,
                   float variance_ratio, int max_components) {
    axpca_t pca;
    kmemset(&pca, 0, sizeof(pca));
    pca.dim = dim;

    /* Compute mean */
    pca.mean = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!pca.mean) return pca;
    kmemset(pca.mean, 0, (uint64_t)dim * sizeof(float));
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < dim; j++)
            pca.mean[j] += data[i * dim + j];
    float inv_n = 1.0f / (float)n_samples;
    for (int j = 0; j < dim; j++) pca.mean[j] *= inv_n;

    /* Build covariance matrix (capped at dim for memory) */
    int cov_dim = dim;
    if (cov_dim > 256) cov_dim = 256; /* cap for tractable Jacobi */

    float *cov = (float *)tensor_alloc((uint64_t)cov_dim * cov_dim * sizeof(float));
    if (!cov) return pca;
    kmemset(cov, 0, (uint64_t)cov_dim * cov_dim * sizeof(float));

    float *centered = (float *)tensor_alloc((uint64_t)cov_dim * sizeof(float));
    if (!centered) { tensor_free(cov); return pca; }

    for (int s = 0; s < n_samples; s++) {
        for (int j = 0; j < cov_dim; j++)
            centered[j] = data[s * dim + j] - pca.mean[j];
        for (int i = 0; i < cov_dim; i++)
            for (int j = i; j < cov_dim; j++) {
                float v = centered[i] * centered[j];
                cov[i * cov_dim + j] += v;
                if (i != j) cov[j * cov_dim + i] += v;
            }
    }
    float inv_ns = 1.0f / (float)(n_samples > 1 ? n_samples - 1 : 1);
    for (int i = 0; i < cov_dim * cov_dim; i++) cov[i] *= inv_ns;
    tensor_free(centered);

    /* Eigendecomposition */
    float *evals = (float *)tensor_alloc((uint64_t)cov_dim * sizeof(float));
    float *evecs = (float *)tensor_alloc((uint64_t)cov_dim * cov_dim * sizeof(float));
    if (!evals || !evecs) {
        if (evals) tensor_free(evals);
        if (evecs) tensor_free(evecs);
        tensor_free(cov);
        return pca;
    }

    ax_jacobi_eigen(cov, cov_dim, evals, evecs);
    tensor_free(cov);

    /* Total variance */
    pca.total_var = 0.0f;
    for (int i = 0; i < cov_dim; i++)
        pca.total_var += (evals[i] > 0 ? evals[i] : 0);

    /* Determine number of components */
    int nc = 0;
    float cumvar = 0.0f;
    for (int i = 0; i < cov_dim && nc < max_components; i++) {
        if (evals[i] <= 0) break;
        cumvar += evals[i];
        nc++;
        if (pca.total_var > 0 && cumvar / pca.total_var >= variance_ratio) break;
    }
    if (nc < 1) nc = 1;
    pca.n_components = nc;

    /* Copy retained components (row-major: component[i] = evecs column i) */
    pca.components = (float *)tensor_alloc((uint64_t)nc * dim * sizeof(float));
    pca.variances = (float *)tensor_alloc((uint64_t)nc * sizeof(float));
    if (!pca.components || !pca.variances) {
        tensor_free(evals); tensor_free(evecs);
        return pca;
    }
    for (int i = 0; i < nc; i++) {
        pca.variances[i] = evals[i];
        for (int j = 0; j < dim; j++) {
            if (j < cov_dim)
                pca.components[i * dim + j] = evecs[j * cov_dim + i];
            else
                pca.components[i * dim + j] = 0.0f;
        }
    }

    tensor_free(evals);
    tensor_free(evecs);
    return pca;
}

void ax_pca_destroy(axpca_t *pca) {
    if (pca->mean) tensor_free(pca->mean);
    if (pca->components) tensor_free(pca->components);
    if (pca->variances) tensor_free(pca->variances);
    kmemset(pca, 0, sizeof(axpca_t));
}

void ax_pca_project(const axpca_t *pca, const float *x, float *out) {
    for (int i = 0; i < pca->n_components; i++) {
        float s = 0.0f;
        for (int j = 0; j < pca->dim; j++)
            s += (x[j] - pca->mean[j]) * pca->components[i * pca->dim + j];
        out[i] = s;
    }
}

/* ── TwoNN intrinsic dimension estimator ── */
float ax_twonn_estimate(const float *data, int n_samples, int dim) {
    if (n_samples < 3) return 1.0f;

    /* For each point, find distances to 2 nearest neighbors */
    float *mu = (float *)tensor_alloc((uint64_t)n_samples * sizeof(float));
    if (!mu) return 1.0f;

    int valid = 0;
    for (int i = 0; i < n_samples; i++) {
        float d1 = 1e30f, d2 = 1e30f;
        for (int j = 0; j < n_samples; j++) {
            if (j == i) continue;
            float dist2 = 0.0f;
            for (int k = 0; k < dim; k++) {
                float d = data[i * dim + k] - data[j * dim + k];
                dist2 += d * d;
            }
            float dist = ax_sqrtf(dist2);
            if (dist < d1) { d2 = d1; d1 = dist; }
            else if (dist < d2) { d2 = dist; }
        }
        if (d1 > 1e-15f) {
            mu[valid++] = d2 / d1;
        }
    }

    if (valid < 2) { tensor_free(mu); return 1.0f; }

    /* Sort mu (insertion sort, small n) */
    for (int i = 1; i < valid; i++) {
        float key = mu[i]; int j = i - 1;
        while (j >= 0 && mu[j] > key) { mu[j + 1] = mu[j]; j--; }
        mu[j + 1] = key;
    }

    /* Empirical CDF + log-log regression:
     * log(1 - F(mu)) = -d * log(mu) => slope = -d */
    float sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    int count = 0;
    for (int i = 0; i < valid; i++) {
        float F = (float)(i + 1) / (float)(valid + 1);
        if (F >= 0.99f || mu[i] <= 1.0f) continue;
        float x = ax_logf(mu[i]);
        float y = ax_logf(1.0f - F);
        sum_x += x; sum_y += y; sum_xx += x * x; sum_xy += x * y;
        count++;
    }

    tensor_free(mu);

    if (count < 2) return 1.0f;
    float slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x + 1e-20f);
    float dim_est = -slope;
    if (dim_est < 1.0f) dim_est = 1.0f;
    if (dim_est > (float)dim) dim_est = (float)dim;
    return dim_est;
}

/* ── GGUF weight dequantization ── */
/* Helper: F16 → F32 */
static float ax_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else { /* subnormal */
            exp = 127 - 14;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    kmemcpy(&result, &f, 4);
    return result;
}

/* Helper: BF16 → F32 */
static float ax_bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float result;
    kmemcpy(&result, &f, 4);
    return result;
}

void ax_dequant_row(float *out, const void *data, int cols, int ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_F32: {
        const float *f = (const float *)data;
        for (int i = 0; i < cols; i++) out[i] = f[i];
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *h = (const uint16_t *)data;
        for (int i = 0; i < cols; i++) out[i] = ax_fp16_to_f32(h[i]);
        break;
    }
    case GGML_TYPE_BF16: {
        const uint16_t *b = (const uint16_t *)data;
        for (int i = 0; i < cols; i++) out[i] = ax_bf16_to_f32(b[i]);
        break;
    }
    case GGML_TYPE_Q4_0: {
        /* 32 elements per block: 2-byte scale (F16) + 16 bytes packed nibbles */
        typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_blk;
        const q4_0_blk *blk = (const q4_0_blk *)data;
        int nb = cols / 32;
        for (int b = 0; b < nb; b++) {
            float d = ax_fp16_to_f32(blk[b].d);
            for (int j = 0; j < 16; j++) {
                uint8_t packed = blk[b].qs[j];
                out[b * 32 + j]      = (float)((int)(packed & 0x0F) - 8) * d;
                out[b * 32 + j + 16] = (float)((int)(packed >> 4) - 8) * d;
            }
        }
        break;
    }
    case GGML_TYPE_Q8_0: {
        /* 32 elements per block: 2-byte scale (F16) + 32 bytes signed ints */
        typedef struct { uint16_t d; int8_t qs[32]; } q8_0_blk;
        const q8_0_blk *blk = (const q8_0_blk *)data;
        int nb = cols / 32;
        for (int b = 0; b < nb; b++) {
            float d = ax_fp16_to_f32(blk[b].d);
            for (int j = 0; j < 32; j++)
                out[b * 32 + j] = (float)blk[b].qs[j] * d;
        }
        break;
    }
    default:
        /* Unsupported type — zero-fill */
        kmemset(out, 0, (uint64_t)cols * sizeof(float));
        break;
    }
}
