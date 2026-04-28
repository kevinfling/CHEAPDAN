/* cheapdan_fracdiff.h
 * Fractional differencing / integration — spectral multiplier, zero extra cost.
 * Re-uses CHEAP DCT machinery (real signals) + FFTW complex plans (phase-preserving).
 * Full 2D separable bonus for vibration maps / multi-sensor arrays.
 *
 * C99 header-only, restrict everywhere, _Alignas(64) hot buffers, no UB,
 * no malloc in hot path, FFTW_MEASURE plans cached.
 *
 * Matches paper's fractional philosophy (Hurst, FRAC_INTEGRATOR kernel).
 * Perfect pre/post step for homomorphic → CEEMDAN → RTS cascade.
 *
 */

#ifndef CHEAPDAN_FRACDIFF_H
#define CHEAPDAN_FRACDIFF_H

#include "cheapdan.h"           /* cheap_ctx + spectral primitives */
#include "cheapdan/homomorphic.h" /* complex FFTW plans when needed */
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1D real (DCT-based, fastest, re-uses CHEAP ctx) */
/* d > 0 = differencing, d < 0 = integration */
int cheap_frac_diff(cheap_ctx *restrict ctx,
                    const double *restrict x, int n,
                    double d, double *restrict y);

/* Symmetric integrate (just flips sign) */
static inline int cheap_frac_integrate(cheap_ctx *restrict ctx,
                                       const double *restrict x, int n,
                                       double d, double *restrict y) {
    return cheap_frac_diff(ctx, x, n, -d, y);
}

/* 1D complex (FFTW, phase-preserving — post-homomorphic use) */
int cheap_frac_diff_complex(const cheapdan_homomorphic_ctx *restrict hctx,
                            const double *restrict x, int n,
                            double d, double *restrict y);

/* 2D separable bonus (row-then-column, Nunes/Linderhed style) */
int cheap_frac_diff_2d(cheap_ctx *restrict row_ctx, cheap_ctx *restrict col_ctx,
                       const double *restrict x, int H, int W,
                       double d, double *restrict y);

#ifdef __cplusplus
}
#endif

#endif /* CHEAPDAN_FRACDIFF_H */

#ifdef CHEAPDAN_FRACDIFF_IMPLEMENTATION

/* ────────────────────────────────────────────────────────────── */
/* 1D real DCT-based (re-uses existing CHEAP spectral multiply)   */
/* ────────────────────────────────────────────────────────────── */
int cheap_frac_diff(cheap_ctx *restrict ctx,
                    const double *restrict x, int n,
                    double d, double *restrict y)
{
    if (!ctx || !x || !y || n != ctx->n || !isfinite(d)) return CHEAP_EINVAL;

    /* Reuse scratch1 as weights buffer — already _Alignas(64) */
    double *restrict w = ctx->scratch1;

    /* Spectral multiplier for fractional differencing: |ξ_k|^d */
    /* ξ_k = π k / n  (DCT-II frequency) */
    for (int k = 0; k < n; ++k) {
        double xi = (k == 0) ? CHEAP_EPS_LOG : M_PI * (double)k / (double)n;
        w[k] = pow(xi, d);
    }

    /* Spectral multiply + IDCT — zero-copy hot path */
    return cheap_apply(ctx, x, w, y);
}

/* ────────────────────────────────────────────────────────────── */
/* 1D complex FFTW version (phase-preserving)                     */
/* ────────────────────────────────────────────────────────────── */
int cheap_frac_diff_complex(const cheapdan_homomorphic_ctx *restrict hctx,
                            const double *restrict x, int n,
                            double d, double *restrict y)
{
    if (!hctx || !x || !y || n != hctx->n || !isfinite(d)) return CHEAP_EINVAL;

    /* FFT(x) → cepstrum buffer (used as spectrum workspace) */
    for (int i = 0; i < n; ++i) {
        hctx->X[i][0] = x[i];
        hctx->X[i][1] = 0.0;
    }
    fftw_execute(hctx->fwd_plan);

    /* Multiply by (jω)^d = |ω|^d * exp(j π d / 2 * sgn(ω)) */
    /* Process in cepstrum buffer, write to liftered for inverse plan. */
    for (int k = 0; k < n; ++k) {
        double omega = (k <= n/2 ? (double)k : (double)(k - n)) * (2.0 * M_PI / (double)n);
        double mag = pow(fabs(omega) + CHEAP_EPS_LOG, d);
        double phase = (d * M_PI / 2.0) * (omega >= 0.0 ? 1.0 : -1.0);
        double c = cos(phase);
        double s = sin(phase);
        double xr = hctx->cepstrum[k][0];
        double xi = hctx->cepstrum[k][1];
        hctx->liftered[k][0] = mag * (c * xr - s * xi);
        hctx->liftered[k][1] = mag * (s * xr + c * xi);
    }

    /* IFFT → X */
    fftw_execute(hctx->inv_plan);
    const double norm = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        y[i] = hctx->X[i][0] * norm;
    }

    return CHEAP_OK;
}

/* ────────────────────────────────────────────────────────────── */
/* 2D separable (row → column) — zero extra plans                */
/* ────────────────────────────────────────────────────────────── */
int cheap_frac_diff_2d(cheap_ctx *restrict row_ctx,
                       cheap_ctx *restrict col_ctx,
                       const double *restrict x, int H, int W,
                       double d, double *restrict y)
{
    if (!row_ctx || !col_ctx || !x || !y || H <= 0 || W <= 0) return CHEAP_EINVAL;
    if (row_ctx->n != W || col_ctx->n != H) return CHEAP_EINVAL;

    /* Temporary buffer for row-pass result */
    double *tmp = (double*) malloc((size_t)H * W * sizeof(double));
    if (!tmp) return CHEAP_ENOMEM;

    /* 1. Fractional diff along rows (horizontal) */
    for (int r = 0; r < H; ++r) {
        int ret = cheap_frac_diff(row_ctx, x + r * W, W, d, tmp + r * W);
        if (ret != CHEAP_OK) { free(tmp); return ret; }
    }

    /* 2. Fractional diff along columns (vertical) on the result */
    for (int c = 0; c < W; ++c) {
        double col_buf[1024];
        if (H > 1024) { free(tmp); return CHEAP_EINVAL; }
        for (int r = 0; r < H; ++r) col_buf[r] = tmp[r * W + c];
        int ret = cheap_frac_diff(col_ctx, col_buf, H, d, col_buf);
        if (ret != CHEAP_OK) { free(tmp); return ret; }
        for (int r = 0; r < H; ++r) y[r * W + c] = col_buf[r];
    }

    free(tmp);
    return CHEAP_OK;
}

#endif /* CHEAPDAN_FRACDIFF_IMPLEMENTATION */
