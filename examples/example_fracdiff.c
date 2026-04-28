/* example_fracdiff.c — Fractional differencing of a chirp signal */
#define CHEAPDAN_FRACDIFF_IMPLEMENTATION
#include "cheapdan/fracdiff.h"
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

    /* Chirp signal: frequency increases from 2 Hz to 20 Hz */
    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double f = 2.0 + 18.0 * t;
        signal[i] = sin(2.0 * M_PI * f * t * N / 2.0); /* sweep */
    }

    double diff[N], integrate[N];

    /* Fractional derivative of order d=0.5 */
    if (cheap_frac_diff(&ctx, signal, N, 0.5, diff) != CHEAP_OK) {
        fprintf(stderr, "frac_diff failed\n");
        cheap_destroy(&ctx);
        return 1;
    }

    /* Fractional integral of order d=0.5 (integrate the derivative → approximate identity) */
    if (cheap_frac_integrate(&ctx, diff, N, 0.5, integrate) != CHEAP_OK) {
        fprintf(stderr, "frac_integrate failed\n");
        cheap_destroy(&ctx);
        return 1;
    }

    printf("Fractional d=0.5 derivative of chirp (N=%d):\n", N);
    double err = 0.0, sig_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        err += (integrate[i] - signal[i]) * (integrate[i] - signal[i]);
        sig_energy += signal[i] * signal[i];
    }
    printf("  Round-trip NMSE: %.6e\n", sqrt(err / sig_energy));

    cheap_destroy(&ctx);
    return 0;
}
