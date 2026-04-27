#include "cheapdan.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define IMG_H    32
#define IMG_W    32
#define N_ENS    10
#define MAX_IMFS  3

int main(void)
{
    cheap_ctx row_ctx, col_ctx;
    memset(&row_ctx, 0, sizeof(row_ctx));
    memset(&col_ctx, 0, sizeof(col_ctx));
    if (cheap_init(&row_ctx, IMG_W, 0.6) != CHEAP_OK) {
        fprintf(stderr, "cheap_init row_ctx failed\n"); return 1;
    }
    if (cheap_init(&col_ctx, IMG_H, 0.6) != CHEAP_OK) {
        fprintf(stderr, "cheap_init col_ctx failed\n"); return 1;
    }

    const int N = IMG_H * IMG_W;
    double* signal = (double*)fftw_malloc((size_t)N * sizeof(double));
    double* out    = (double*)fftw_malloc((size_t)MAX_IMFS * N * sizeof(double));
    if (!signal || !out) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* 2D signal: trend + high-frequency texture; fails IMF condition on its own */
    for (int r = 0; r < IMG_H; ++r)
        for (int c = 0; c < IMG_W; ++c) {
            double t_r = (double)r / IMG_H;
            double t_c = (double)c / IMG_W;
            signal[r * IMG_W + c] =
                t_r * t_c                                  /* smooth trend */
              + 0.4 * cos(2.0 * M_PI * 4.0 * t_r)        /* low-freq row oscillation */
              + 0.4 * cos(2.0 * M_PI * 4.0 * t_c)        /* low-freq col oscillation */
              + 0.2 * cos(2.0 * M_PI * 10.0 * t_r)       /* high-freq row texture */
              + 0.2 * cos(2.0 * M_PI * 10.0 * t_c);      /* high-freq col texture */
        }

    int n_imfs = 0;
    int rc = cheap_ceemdan_frac_2d(&row_ctx, &col_ctx, signal, IMG_H, IMG_W,
                                    N_ENS, 0.2, 0.6,
                                    CHEAP_CEEMDAN_SIFT_HEAT_GAUSS, 0.05,
                                    out, MAX_IMFS, &n_imfs);
    if (rc != CHEAP_OK) {
        fprintf(stderr, "cheap_ceemdan_frac_2d failed: %d\n", rc); return 1;
    }
    if (n_imfs < 1) {
        fprintf(stderr, "expected n_imfs >= 1, got %d\n", n_imfs); return 1;
    }
    if (n_imfs > MAX_IMFS) {
        fprintf(stderr, "n_imfs %d > MAX_IMFS %d\n", n_imfs, MAX_IMFS); return 1;
    }
    printf("2D decomposition: n_imfs=%d\n", n_imfs);

    double total_energy = 0.0;
    for (int m = 0; m < n_imfs; ++m)
        for (int i = 0; i < N; ++i) {
            if (!isfinite(out[m * N + i])) {
                fprintf(stderr, "non-finite value at plane %d idx %d\n", m, i); return 1;
            }
            total_energy += out[m * N + i] * out[m * N + i];
        }
    if (total_energy <= 0.0) {
        fprintf(stderr, "total IMF energy is zero\n"); return 1;
    }
    printf("Total IMF energy: %.4f\n", total_energy);

    fftw_free(signal);
    fftw_free(out);
    cheap_destroy(&row_ctx);
    cheap_destroy(&col_ctx);

    /* Validate dimension mismatch is rejected (separate scope, fresh ctx) */
    {
        cheap_ctx bad_ctx, col_ctx2;
        memset(&bad_ctx,  0, sizeof(bad_ctx));
        memset(&col_ctx2, 0, sizeof(col_ctx2));
        if (cheap_init(&bad_ctx, IMG_W + 1, 0.6) != CHEAP_OK ||
            cheap_init(&col_ctx2, IMG_H, 0.6) != CHEAP_OK) {
            fprintf(stderr, "cheap_init for mismatch test failed\n"); return 1;
        }
        double* sig2 = (double*)fftw_malloc((size_t)IMG_H * IMG_W * sizeof(double));
        double* out2 = (double*)fftw_malloc((size_t)MAX_IMFS * IMG_H * IMG_W * sizeof(double));
        int nim2 = 0;
        if (!sig2 || !out2) { fprintf(stderr, "alloc failed\n"); return 1; }
        int rc2 = cheap_ceemdan_frac_2d(&bad_ctx, &col_ctx2,
                                        sig2, IMG_H, IMG_W,
                                        N_ENS, 0.2, 0.6,
                                        CHEAP_CEEMDAN_SIFT_HEAT_GAUSS, 0.05,
                                        out2, MAX_IMFS, &nim2);
        if (rc2 != CHEAP_EINVAL) {
            fprintf(stderr, "expected CHEAP_EINVAL for dim mismatch, got %d\n", rc2); return 1;
        }
        fftw_free(sig2);
        fftw_free(out2);
        cheap_destroy(&bad_ctx);
        cheap_destroy(&col_ctx2);
    }
    printf("PASS: test_ceemdan_2d\n");
    return 0;
}
