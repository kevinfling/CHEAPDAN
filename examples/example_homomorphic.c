/* example_homomorphic.c — Homomorphic cepstral filtering for multiplicative noise */
#define CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION
#include "cheapdan/homomorphic.h"
#include <stdio.h>
#include <math.h>

#define N 512

int main(void)
{
    cheapdan_homomorphic_ctx ctx;
    if (cheapdan_homomorphic_init(&ctx, N, 1e-3) != CHEAP_OK) {
        fprintf(stderr, "homomorphic_init failed\n");
        return 1;
    }

    /* Signal with multiplicative modulation */
    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double envelope = 1.0 + 0.4 * sin(2.0 * M_PI * 2.0 * t);
        double carrier  = 1.0 + 0.2 * sin(2.0 * M_PI * 20.0 * t);
        signal[i] = envelope * carrier;
    }

    double filtered[N];
    double cutoff = 0.0;
    int ret = cheapdan_homomorphic_filter(&ctx, signal, N, filtered, &cutoff);
    if (ret != CHEAP_OK) {
        fprintf(stderr, "homomorphic_filter failed: %d\n", ret);
        cheapdan_homomorphic_destroy(&ctx);
        return 1;
    }

    printf("Homomorphic filtering (N=%d):\n", N);
    printf("  Adaptive quefrency cutoff: %.1f\n", cutoff);

    double var_before = 0.0, var_after = 0.0;
    double mean_before = 0.0, mean_after = 0.0;
    for (int i = 0; i < N; ++i) {
        mean_before += signal[i];
        mean_after  += filtered[i];
    }
    mean_before /= N; mean_after /= N;
    for (int i = 0; i < N; ++i) {
        var_before += (signal[i] - mean_before) * (signal[i] - mean_before);
        var_after  += (filtered[i] - mean_after) * (filtered[i] - mean_after);
    }
    printf("  Variance before: %.6f  after: %.6f\n", var_before / N, var_after / N);

    cheapdan_homomorphic_destroy(&ctx);
    return 0;
}
