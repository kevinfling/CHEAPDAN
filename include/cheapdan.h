#ifndef CHEAPDAN_H
#define CHEAPDAN_H

/*
 * CHEAPDAN — Circulant Hessian Ensemble Adaptive Decomposition with Adaptive Noise
 * Fractional-noise CEEMDAN plugin on top of cheap.h (v0.1.0)
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Kevin Fling
 */

#include "cheap.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * cheap_weights_spectral_lowpass — Hard low-pass cutoff for rapid sifting.
 *
 *   w_k = 1.0 if k < k_cutoff, else 0.0
 *   (rect function in frequency index)
 */
static inline int cheap_weights_spectral_lowpass(int n, int k_cutoff,
                                                 double* restrict weights_out)
{
    if (n < 2 || k_cutoff < 0 || k_cutoff > n || !weights_out) return CHEAP_EINVAL;
    for (int k = 0; k < k_cutoff; ++k) weights_out[k] = 1.0;
    for (int k = k_cutoff; k < n; ++k) weights_out[k] = 0.0;
    return CHEAP_OK;
}

/*
 * cheap_weights_frac_integrator — Fractional integrator for "area-under-curve" IMFs.
 *
 *   w_k = |i ξ_k|^{-α} = |ξ_k|^{-α}   (real-valued magnitude)
 *   ξ_k = π k / n   (DCT-II frequency)
 *   α > 0 → integration (low-pass), α < 0 → differentiation (high-pass).
 *   DC (k=0) clamped to avoid singularity.
 */
static inline int cheap_weights_frac_integrator(int n, double alpha,
                                                double* restrict weights_out)
{
    if (n < 2 || !weights_out) return CHEAP_EINVAL;
    const double inv_n = 1.0 / (double)n;
    for (int k = 0; k < n; ++k) {
        double xi = M_PI * (double)k * inv_n;
        if (k == 0) xi = CHEAP_EPS_LOG;
        weights_out[k] = pow(xi, -alpha);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/*
 * cheap_weights_heat_kernel_gauss — Gaussian heat kernel sifting (true diffusion).
 *
 *   w_k = exp(-t ξ_k²)     ξ_k = π k / n
 *   Different from Laplacian μ_k = 4 sin²(πk/2n) — this is the continuous
 *   Fourier approximation, excellent for smooth physical sifting.
 */
static inline int cheap_weights_heat_kernel_gauss(int n, double t,
                                                  double* restrict weights_out)
{
    if (n < 2 || t < 0.0 || !weights_out) return CHEAP_EINVAL;
    const double inv_n = 1.0 / (double)n;
    weights_out[0] = 1.0;   /* DC never diffuses */
    for (int k = 1; k < n; ++k) {
        double xi = M_PI * (double)k * inv_n;
        weights_out[k] = exp(-t * xi * xi);
    }
    CHEAP_CONTRACT_FINITE_OR_EDOM(weights_out, n);
    return CHEAP_OK;
}

/* ============================================================
 * Internal helpers
 * ============================================================ */

/*
 * cheap_fgn_sample — Generate fractional Gaussian noise (dfBm increments).
 * Re-uses ctx->sqrt_lambda (Flandrin spectrum).
 * Uses a persistent static RNG (initialized once from rdtsc) so successive
 * calls within an ensemble loop produce independent noise vectors.
 */
static inline int cheap_fgn_sample(cheap_ctx* restrict ctx, double* restrict noise_out)
{
    if (!ctx || !ctx->is_initialized || !noise_out) return CHEAP_EINVAL;

    /* One persistent RNG per translation unit — avoids correlated seeds when
     * called in a tight loop where rdtsc increments by only a few hundred cycles. */
    static cheap__rng s_rng;
    static int s_rng_init = 0;
    if (!s_rng_init) {
        cheap__rng_init(&s_rng, cheap_rdtsc());
        s_rng_init = 1;
    }

    double* white = ctx->scratch1;
    for (int i = 0; i < ctx->n; ++i)
        white[i] = cheap__lcg_normal(&s_rng);

    return cheap_apply(ctx, white, ctx->sqrt_lambda, noise_out);
}


typedef enum {
    CHEAP_CEEMDAN_SIFT_LOWPASS,
    CHEAP_CEEMDAN_SIFT_FRAC_INTEGRATOR,
    CHEAP_CEEMDAN_SIFT_HEAT_GAUSS,
    CHEAP_CEEMDAN_SIFT_FRAC_ROUGH   /* existing -2H fractional */
} cheap_ceemdan_sift_mode;

/*
 * cheap_ceemdan_sift_step — kernel-agnostic spectral sifting.
 */
static inline int cheap_ceemdan_sift_step(cheap_ctx* restrict ctx,
                                          const double* restrict candidate,
                                          double* restrict mean_out,
                                          cheap_ceemdan_sift_mode mode,
                                          double param)   /* k_cutoff, alpha, t, etc. */
{
    double* weights = ctx->scratch1;
    int ret = CHEAP_EINVAL;
    switch (mode) {
        case CHEAP_CEEMDAN_SIFT_LOWPASS:
            ret = cheap_weights_spectral_lowpass(ctx->n, (int)param, weights);
            break;
        case CHEAP_CEEMDAN_SIFT_FRAC_INTEGRATOR:
            ret = cheap_weights_frac_integrator(ctx->n, param, weights);
            break;
        case CHEAP_CEEMDAN_SIFT_HEAT_GAUSS:
            ret = cheap_weights_heat_kernel_gauss(ctx->n, param, weights);
            break;
        case CHEAP_CEEMDAN_SIFT_FRAC_ROUGH:
            ret = cheap_weights_fractional(ctx->n, param, weights);   /* param = -2H */
            /* Clamp DC weight to 1: (2*eps)^(-1.4) ~ 2e16 causes trial[0] to overflow. */
            if (ret == CHEAP_OK) weights[0] = 1.0;
            break;
    }
    if (ret != CHEAP_OK) return ret;
    return cheap_apply(ctx, candidate, weights, mean_out);
}

/*
 * cheap_ceemdan_is_imf — Standard stopping criterion.
 * IMF condition: |n_extrema - n_zero_crossings| <= 1 and zero mean.
 */
static inline bool cheap_ceemdan_is_imf(const double* imf, int n, double tol)
{
    int n_ext = 0, n_zero = 0;
    for (int i = 1; i < n-1; ++i) {
        if ((imf[i] > imf[i-1] && imf[i] > imf[i+1]) ||
            (imf[i] < imf[i-1] && imf[i] < imf[i+1])) ++n_ext;
        if (imf[i] * imf[i-1] < 0.0) ++n_zero;
    }
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += imf[i];
    mean /= n;
    return (abs(n_ext - n_zero) <= 1) && (fabs(mean) < tol);
}

/* ============================================================
 * Public API — 1D Fractional CEEMDAN
 * ============================================================ */

/*
 * cheap_ceemdan_frac — Full CEEMDAN with Gupta-Joshi fractional noise.
 *
 * signal:      input of length n
 * n_ensemble:  50-200 typical
 * eps:         noise std (0.2 recommended)
 * H:           Hurst for fGn (0.5 = white, <0.5 = rough)
 * imfs_out:    caller-allocated [max_imfs * n] or NULL (averaged only)
 * n_imfs_out:  returns number of IMFs extracted
 */
static inline int cheap_ceemdan_frac(cheap_ctx* restrict ctx,
                                     const double* restrict signal,
                                     int n_ensemble, double eps, double H,
                                     double* restrict imfs_out,
                                     int* n_imfs_out)
{
    if (!ctx || !signal || n_ensemble < 1 || eps <= 0.0 || H <= 0.0 || H >= 1.0)
        return CHEAP_EINVAL;

    const int n = ctx->n;
    double* residue = ctx->scratch2;

    /* Allocate a private trial buffer so it never aliases ctx->scratch1,
     * which cheap_ceemdan_sift_step uses for the spectral weight array. */
    double* trial = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!trial) return CHEAP_EINVAL;

    /* fGn output buffer — separate from trial so cheap_fgn_sample's
     * cheap_apply(ctx, scratch1, sqrt_lambda, fgn_out) doesn't alias trial. */
    double* fgn = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!fgn) { fftw_free(trial); return CHEAP_EINVAL; }

    memcpy(residue, signal, (size_t)n * sizeof(double));

    int imf_idx = 0;
    int ret = CHEAP_OK;
    while (imf_idx < 12 && !cheap_ceemdan_is_imf(residue, n, 1e-3)) {
        double* this_imf = (imfs_out) ? imfs_out + imf_idx * n : trial;

        for (int i = 0; i < n; ++i) this_imf[i] = 0.0;

        for (int ens = 0; ens < n_ensemble; ++ens) {
            ret = cheap_fgn_sample(ctx, fgn);
            if (ret != CHEAP_OK) goto done;

            double noise_scale = eps / sqrt((double)n_ensemble);
            for (int i = 0; i < n; ++i)
                trial[i] = residue[i] + noise_scale * fgn[i];

            for (int sift = 0; sift < 10; ++sift) {
                /* Use heat-kernel Gaussian sifting: attenuates high frequencies,
                 * gives a smooth local mean suitable for EMD envelope estimation.
                 * t scales with H so rougher noise (low H) uses a tighter smoother. */
                double t = 0.5 * (1.0 - H);
                ret = cheap_ceemdan_sift_step(ctx, trial, ctx->prev_g,
                                              CHEAP_CEEMDAN_SIFT_HEAT_GAUSS, t);
                if (ret != CHEAP_OK) goto done;
                for (int i = 0; i < n; ++i) trial[i] -= ctx->prev_g[i];
                if (cheap_ceemdan_is_imf(trial, n, 1e-4)) break;
            }
            for (int i = 0; i < n; ++i) this_imf[i] += trial[i];
        }
        for (int i = 0; i < n; ++i) this_imf[i] /= (double)n_ensemble;

        for (int i = 0; i < n; ++i) residue[i] -= this_imf[i];
        ++imf_idx;
    }

done:
    fftw_free(trial);
    fftw_free(fgn);
    *n_imfs_out = imf_idx;
    return ret;
}

/* ============================================================
 * 2D separable CEEMDAN (row-column)
 * ============================================================ */

/*
 * cheap_ceemdan_frac_2d — Separable row-then-column fractional CEEMDAN.
 *
 * Uses two 1D contexts: row_ctx->n must equal width, col_ctx->n must equal height.
 * imfs_2d_out: caller-allocated [max_imfs * height * width], row-major per plane.
 * Returns number of 2D IMF planes extracted (up to max_imfs).
 */
static inline int cheap_ceemdan_frac_2d(cheap_ctx* row_ctx, cheap_ctx* col_ctx,
                                        const double* restrict signal_2d,
                                        int height, int width,
                                        int n_ensemble, double eps, double H,
                                        cheap_ceemdan_sift_mode mode, double sift_param,
                                        double* restrict imfs_2d_out,
                                        int max_imfs,
                                        int* n_imfs_out)
{
    if (!row_ctx || !col_ctx || !signal_2d || !imfs_2d_out || !n_imfs_out)
        return CHEAP_EINVAL;
    if (height < 2 || width < 2 || n_ensemble < 1 || max_imfs < 1)
        return CHEAP_EINVAL;
    if (eps <= 0.0 || H <= 0.0 || H >= 1.0)
        return CHEAP_EINVAL;
    if (row_ctx->n != width || col_ctx->n != height)
        return CHEAP_EINVAL;

    const int N = height * width;
    int ret;

    /* Allocate working surfaces with FFTW alignment.
     * row_trial/col_trial are private buffers separate from ctx->scratch1,
     * which cheap_ceemdan_sift_step uses for the spectral weight array. */
    double* residue_2d      = (double*)fftw_malloc((size_t)N * sizeof(double));
    double* row_imf_surface = (double*)fftw_malloc((size_t)N * sizeof(double));
    double* col_buf         = (double*)fftw_malloc((size_t)height * sizeof(double));
    double* col_out         = (double*)fftw_malloc((size_t)height * sizeof(double));
    double* row_trial       = (double*)fftw_malloc((size_t)width  * sizeof(double));
    double* row_fgn         = (double*)fftw_malloc((size_t)width  * sizeof(double));
    double* col_trial       = (double*)fftw_malloc((size_t)height * sizeof(double));
    double* col_fgn         = (double*)fftw_malloc((size_t)height * sizeof(double));

    if (!residue_2d || !row_imf_surface || !col_buf || !col_out ||
        !row_trial || !row_fgn || !col_trial || !col_fgn) {
        fftw_free(residue_2d); fftw_free(row_imf_surface);
        fftw_free(col_buf);    fftw_free(col_out);
        fftw_free(row_trial);  fftw_free(row_fgn);
        fftw_free(col_trial);  fftw_free(col_fgn);
        return CHEAP_EINVAL;
    }

    memcpy(residue_2d, signal_2d, (size_t)N * sizeof(double));

    int imf_idx = 0;

    while (imf_idx < max_imfs) {
        /* Check if residue is already flat (all columns satisfy IMF condition) */
        int all_imf = 1;
        for (int c = 0; c < width && all_imf; ++c) {
            for (int r = 0; r < height; ++r) col_buf[r] = residue_2d[r * width + c];
            if (!cheap_ceemdan_is_imf(col_buf, height, 1e-3)) all_imf = 0;
        }
        if (all_imf) break;

        double* this_plane = imfs_2d_out + imf_idx * N;
        for (int i = 0; i < N; ++i) this_plane[i] = 0.0;

        /* ---- Row pass: extract one IMF from each row of the residue ---- */
        for (int i = 0; i < N; ++i) row_imf_surface[i] = 0.0;

        double noise_scale_r = eps / sqrt((double)n_ensemble);
        for (int r = 0; r < height; ++r) {
            const double* row_in  = residue_2d + r * width;
            double*       row_out = row_imf_surface + r * width;

            for (int ens = 0; ens < n_ensemble; ++ens) {
                ret = cheap_fgn_sample(row_ctx, row_fgn);
                if (ret != CHEAP_OK) goto cleanup;

                for (int i = 0; i < width; ++i)
                    row_trial[i] = row_in[i] + noise_scale_r * row_fgn[i];

                for (int sift = 0; sift < 10; ++sift) {
                    ret = cheap_ceemdan_sift_step(row_ctx, row_trial,
                                                  row_ctx->prev_g, mode, sift_param);
                    if (ret != CHEAP_OK) goto cleanup;
                    for (int i = 0; i < width; ++i)
                        row_trial[i] -= row_ctx->prev_g[i];
                    if (cheap_ceemdan_is_imf(row_trial, width, 1e-4)) break;
                }
                for (int i = 0; i < width; ++i) row_out[i] += row_trial[i];
            }
            for (int i = 0; i < width; ++i) row_out[i] /= (double)n_ensemble;
        }

        /* ---- Column pass: extract one IMF from each column of row_imf_surface ---- */
        double noise_scale_c = eps / sqrt((double)n_ensemble);
        for (int c = 0; c < width; ++c) {
            for (int r = 0; r < height; ++r) col_buf[r] = row_imf_surface[r * width + c];
            for (int r = 0; r < height; ++r) col_out[r] = 0.0;

            for (int ens = 0; ens < n_ensemble; ++ens) {
                ret = cheap_fgn_sample(col_ctx, col_fgn);
                if (ret != CHEAP_OK) goto cleanup;

                for (int i = 0; i < height; ++i)
                    col_trial[i] = col_buf[i] + noise_scale_c * col_fgn[i];

                for (int sift = 0; sift < 10; ++sift) {
                    ret = cheap_ceemdan_sift_step(col_ctx, col_trial,
                                                  col_ctx->prev_g, mode, sift_param);
                    if (ret != CHEAP_OK) goto cleanup;
                    for (int i = 0; i < height; ++i)
                        col_trial[i] -= col_ctx->prev_g[i];
                    if (cheap_ceemdan_is_imf(col_trial, height, 1e-4)) break;
                }
                for (int i = 0; i < height; ++i) col_out[i] += col_trial[i];
            }

            for (int r = 0; r < height; ++r)
                this_plane[r * width + c] = col_out[r] / (double)n_ensemble;
        }

        for (int i = 0; i < N; ++i) residue_2d[i] -= this_plane[i];
        ++imf_idx;
    }

    ret = CHEAP_OK;

cleanup:
    fftw_free(residue_2d); fftw_free(row_imf_surface);
    fftw_free(col_buf);    fftw_free(col_out);
    fftw_free(row_trial);  fftw_free(row_fgn);
    fftw_free(col_trial);  fftw_free(col_fgn);

    *n_imfs_out = imf_idx;
    return ret;
}

#endif /* CHEAPDAN_H */
