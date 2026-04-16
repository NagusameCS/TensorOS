/* =============================================================================
 * TensorOS Axiomatic Subsystem — Geodesic Autonomous Discovery Pipeline
 *
 * Five-phase pipeline for discovering mathematical axioms from the geometry
 * of a pre-trained language model's weight manifold:
 *
 *   Phase 1: Manifold Identification (PCA + TwoNN intrinsic dimension)
 *   Phase 2: Symmetry Mining (real dequantized weight analysis per head)
 *   Phase 3: Curvature Field (Fisher/covariance metric → Christoffel → Riemann)
 *   Phase 4: Axiom Discovery (geometry-derived candidates + oracle validation)
 *   Phase 5: Geodesic Pilot (geodesic solve with real metric, compare to forward pass)
 *
 * Based on Organic Training Theory: model weights form a Riemannian manifold
 * whose geodesics approximate the transformer forward pass.
 * =============================================================================*/

#ifndef TENSOROS_AXIOM_BETA_H
#define TENSOROS_AXIOM_BETA_H

#include <stdint.h>

/* ─── Configuration ─── */
typedef struct {
    int     embedding_samples;    /* Number of random token embeddings to sample */
    float   pca_variance_ratio;   /* Variance ratio threshold for PCA (e.g. 0.95) */
    int     symmetry_trials;      /* Number of head pairs to test for symmetry */
    int     metric_sample_points; /* Number of points in metric field */
    int     fisher_probes;        /* Number of random probes for Fisher matrix */
    float   fisher_epsilon;       /* Perturbation magnitude for Fisher */
    int     active_iterations;    /* Axiom discovery iterations */
    int     oracle_calls_max;     /* Max forward-pass oracle calls */
    int     geodesic_steps;       /* RK4 geodesic integration steps */
    int     geodesic_test_tokens; /* Number of test tokens for geodesic pilot */
    uint32_t seed;
    int     verbose;
    int     skip_geodesic;        /* Skip Phase 5 (for quick runs) */
    int     use_fisher;           /* Use Fisher metric (slower but more accurate) */
    int     fast_mode;            /* Reduce expensive phase workloads for speed */
    int     reuse_cache;          /* Reuse phase 1-4 geometry cache in-process */
} axiom_beta_config_t;

/* ─── Per-Phase Reports ─── */
typedef struct {
    float intrinsic_dim;
    int   pca_components;
    float pca_variance_explained;
    float total_variance;
} axiom_phase1_report_t;

typedef struct {
    int   n_symmetries_found;
    float avg_symmetry_score;
    float max_symmetry_score;
    int   symmetric_pairs[32][2]; /* Top symmetric head pairs */
    float symmetry_scores[32];
} axiom_phase2_report_t;

typedef struct {
    float avg_scalar_curvature;
    float min_scalar_curvature;
    float max_scalar_curvature;
    float curvature_variance;
    int   metric_field_points;
    int   used_fisher;
} axiom_phase3_report_t;

/* Axiom types discovered */
typedef enum {
    AXIOM_TYPE_METRIC    = 0,  /* g_ij relation (e.g. near-diagonal, block structure) */
    AXIOM_TYPE_SYMMETRY  = 1,  /* Symmetry group element (head permutation invariance) */
    AXIOM_TYPE_CURVATURE = 2,  /* Curvature bound (sectional curvature in range) */
    AXIOM_TYPE_GEODESIC  = 3,  /* Geodesic constraint (path length bounds) */
    AXIOM_TYPE_BOUNDARY  = 4,  /* Boundary condition (embedding norm constraints) */
} axiom_type_t;

typedef struct {
    axiom_type_t type;
    float        confidence;
    float        evidence;      /* Accumulated evidence from oracle */
    char         description[128];
} axiom_entry_t;

typedef struct {
    int           n_axioms_proposed;
    int           n_axioms_accepted;
    int           oracle_calls_used;
    axiom_entry_t axioms[64];
} axiom_phase4_report_t;

typedef struct {
    float avg_cosine_sim;
    float avg_l2_error;
    float geodesic_length;
    int   n_tests;
    int   geodesic_diverged;
    float complexity_ratio;     /* geodesic_length / euclidean_distance */
    int   reused_geometry_cache;
} axiom_phase5_report_t;

/* ─── Full Report ─── */
typedef struct {
    axiom_beta_config_t   config;
    axiom_phase1_report_t phase1;
    axiom_phase2_report_t phase2;
    axiom_phase3_report_t phase3;
    axiom_phase4_report_t phase4;
    axiom_phase5_report_t phase5;
    int                   phases_completed;
} axiom_beta_report_t;

/* ─── API ─── */

/* Return default configuration */
axiom_beta_config_t axiom_beta_default_config(void);

/* Run the full 5-phase axiomatic discovery pipeline.
 * Requires a loaded LLM model (llm_is_loaded() == 1). */
axiom_beta_report_t axiom_beta_run(axiom_beta_config_t config);

/* Write JSON report to buffer. Returns bytes written. */
int axiom_beta_write_json(const axiom_beta_report_t *report,
                          char *buf, int max_buf);

#endif /* TENSOROS_AXIOM_BETA_H */
