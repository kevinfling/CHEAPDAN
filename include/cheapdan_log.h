/*
 * CHEAPDAN_LOG — Multiplicative CEEMDAN via log-domain additive decomposition.
 *
 *   log(signal) → additive CEEMDAN → exp(reconstruction)
 *
 * This maps multiplicative modulations into an additive Hilbert space where
 * the standard CEEMDAN sifting applies. Each returned IMF is exponentiated
 * so that the original signal is recovered by pointwise product:
 *
 *   signal[i] ≈ ∏_m imfs_out[m*n + i]
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Kevin Fling
 */

#ifndef CHEAPDAN_LOG_H
#define CHEAPDAN_LOG_H

#include "cheapdan.h"
#include <math.h>

/*
 * cheap_ceemdan_log — Multiplicative sifting on strictly-positive signals.
 *
 * signal:      input of length ctx->n, every element must be > 0
 * n_ensemble:  number of ensemble trials (50–200 typical)
 * eps:         noise std (0.2 recommended)
 * H:           Hurst exponent for fGn (0.5 = white, <0.5 = rougher)
 * imfs_out:    caller-allocated [max_imfs * n]; on success holds exp(log-IMFs)
 * n_imfs_out:  returns number of IMFs extracted
 *
 * Returns CHEAP_EINVAL if any signal[i] <= 0 or if other arguments are invalid.
 */
static inline int cheap_ceemdan_log(cheap_ctx* restrict ctx,
                                    const double* restrict signal,
                                    int n_ensemble, double eps, double H,
                                    double* restrict imfs_out,
                                    int* n_imfs_out)
{
    if (!ctx || !signal || n_ensemble < 1 || eps <= 0.0 || H <= 0.0 || H >= 1.0 ||
        !imfs_out || !n_imfs_out)
        return CHEAP_EINVAL;

    const int n = ctx->n;
    for (int i = 0; i < n; ++i)
        if (signal[i] <= 0.0) return CHEAP_EINVAL;

    /*
     * Write log(signal) into prev_g (safe temporary), then pass it to the
     * additive CEEMDAN.  cheap_ceemdan_frac copies its input into scratch2
     * before touching prev_g, so there is no aliasing hazard.
     */
    double* log_sig = ctx->prev_g;
    for (int i = 0; i < n; ++i)
        log_sig[i] = log(signal[i]);

    int ret = cheap_ceemdan_frac(ctx, log_sig, n_ensemble, eps, H, imfs_out, n_imfs_out);
    if (ret != CHEAP_OK) return ret;

    const int n_imfs = *n_imfs_out;
    for (int m = 0; m < n_imfs; ++m)
        for (int i = 0; i < n; ++i)
            imfs_out[m * n + i] = exp(imfs_out[m * n + i]);

    return CHEAP_OK;
}

/*
 * cheap_ceemdan_log_shift — Multiplicative sifting with automatic positivity shift.
 *
 * If the signal contains zeros or negative values, this routine shifts it by
 *   offset = |min(signal)| + delta
 * so that every sample becomes strictly positive before taking the log.
 *
 * delta:       small positive guard (e.g. 1e-12)
 * offset_out:  if non-NULL, receives the applied offset
 *
 * The returned IMFs live in the *shifted* multiplicative domain:
 *   signal[i] + offset ≈ ∏_m imfs_out[m*n + i]
 */
static inline int cheap_ceemdan_log_shift(cheap_ctx* restrict ctx,
                                          const double* restrict signal,
                                          int n_ensemble, double eps, double H,
                                          double delta,
                                          double* restrict imfs_out,
                                          int* n_imfs_out,
                                          double* offset_out)
{
    if (!ctx || !signal || n_ensemble < 1 || eps <= 0.0 || H <= 0.0 || H >= 1.0 ||
        delta <= 0.0 || !imfs_out || !n_imfs_out)
        return CHEAP_EINVAL;

    const int n = ctx->n;
    double min_val = signal[0];
    for (int i = 1; i < n; ++i)
        if (signal[i] < min_val) min_val = signal[i];

    double offset = 0.0;
    if (min_val <= 0.0)
        offset = fabs(min_val) + delta;

    if (offset_out) *offset_out = offset;

    double* log_sig = ctx->prev_g;
    for (int i = 0; i < n; ++i)
        log_sig[i] = log(signal[i] + offset);

    int ret = cheap_ceemdan_frac(ctx, log_sig, n_ensemble, eps, H, imfs_out, n_imfs_out);
    if (ret != CHEAP_OK) return ret;

    const int n_imfs = *n_imfs_out;
    for (int m = 0; m < n_imfs; ++m)
        for (int i = 0; i < n; ++i)
            imfs_out[m * n + i] = exp(imfs_out[m * n + i]);

    return CHEAP_OK;
}

/*
 * cheap_ceemdan_log_recon — Pointwise product reconstruction from multiplicative IMFs.
 *
 * imfs:    flat array [n_imfs * n] as produced by cheap_ceemdan_log / _shift
 * n_imfs:  number of IMFs
 * n:       signal length
 * recon:   output buffer of length n
 */
static inline void cheap_ceemdan_log_recon(const double* restrict imfs,
                                           int n_imfs, int n,
                                           double* restrict recon)
{
    for (int i = 0; i < n; ++i) {
        double prod = 1.0;
        for (int m = 0; m < n_imfs; ++m)
            prod *= imfs[m * n + i];
        recon[i] = prod;
    }
}

#endif /* CHEAPDAN_LOG_H */
