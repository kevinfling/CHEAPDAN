/* example_1d.c — 1D fractional CEEMDAN decomposition of a synthetic chirp */
#include "cheapdan.h"
#include <stdio.h>
#include <math.h>

#define N 512

int main(void)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, N, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed\n");
        return 1;
    }

    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        signal[i] = sin(2.0 * M_PI * 3.0  * t)
                  + 0.5 * sin(2.0 * M_PI * 15.0 * t)
                  + 0.2 * sin(2.0 * M_PI * 50.0 * t);
    }

    double imfs[12 * N];
    int n_imfs = 0;
    if (cheap_ceemdan_frac(&ctx, signal, 100, 0.2, 0.7, imfs, &n_imfs) != CHEAP_OK) {
        fprintf(stderr, "decomposition failed\n");
        cheap_destroy(&ctx);
        return 1;
    }

    printf("Extracted %d IMFs from N=%d signal:\n", n_imfs, N);
    for (int m = 0; m < n_imfs; ++m) {
        double energy = 0.0;
        for (int i = 0; i < N; ++i) energy += imfs[m * N + i] * imfs[m * N + i];
        printf("  IMF[%2d]: energy = %10.4f\n", m, energy);
    }

    cheap_destroy(&ctx);
    return 0;
}
