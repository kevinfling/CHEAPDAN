/* example_2d.c — 2D separable fractional CEEMDAN on a synthetic surface */
#include "cheapdan.h"
#include <stdio.h>
#include <math.h>

int main(void)
{
    const int IMG_H = 64, IMG_W = 64;
    const int N = IMG_H * IMG_W;
    const int MAX_IMFS = 3;

    cheap_ctx row_ctx, col_ctx;
    if (cheap_init(&row_ctx, IMG_W, 0.6) != CHEAP_OK ||
        cheap_init(&col_ctx, IMG_H, 0.6) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    double* signal_2d = (double*)fftw_malloc((size_t)N * sizeof(double));
    double* imfs_2d   = (double*)fftw_malloc((size_t)MAX_IMFS * N * sizeof(double));
    if (!signal_2d || !imfs_2d) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Synthetic 2D signal: sum of two separable cosines */
    for (int r = 0; r < IMG_H; ++r)
        for (int c = 0; c < IMG_W; ++c)
            signal_2d[r * IMG_W + c] =
                cos(2.0 * M_PI * 2.0 * r / IMG_H) * cos(2.0 * M_PI * 2.0 * c / IMG_W)
              + 0.5 * cos(2.0 * M_PI * 8.0 * r / IMG_H) * cos(2.0 * M_PI * 8.0 * c / IMG_W);

    int n_imfs = 0;
    int ret = cheap_ceemdan_frac_2d(&row_ctx, &col_ctx, signal_2d, IMG_H, IMG_W,
                                    50, 0.2, 0.3,
                                    CHEAP_CEEMDAN_SIFT_HEAT_GAUSS, 0.05,
                                    imfs_2d, MAX_IMFS, &n_imfs);

    printf("2D CHEAPDAN: ret=%d, extracted %d IMF plane(s) from %dx%d signal\n",
           ret, n_imfs, IMG_H, IMG_W);

    for (int m = 0; m < n_imfs; ++m) {
        double energy = 0.0;
        for (int i = 0; i < N; ++i) energy += imfs_2d[m * N + i] * imfs_2d[m * N + i];
        printf("  IMF plane[%d]: energy = %.4f\n", m, energy);
    }

    cheap_destroy(&row_ctx);
    cheap_destroy(&col_ctx);
    fftw_free(signal_2d);
    fftw_free(imfs_2d);
    return 0;
}
