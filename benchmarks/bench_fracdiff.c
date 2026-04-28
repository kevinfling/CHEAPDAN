/* bench_fracdiff.c — Throughput benchmark for cheap_frac_diff */
#define CHEAPDAN_FRACDIFF_IMPLEMENTATION
#include "cheapdan/fracdiff.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void bench_one(int n, double d)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, n, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed n=%d\n", n);
        return;
    }

    double* signal = (double*)fftw_malloc((size_t)n * sizeof(double));
    double* out    = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!signal || !out) {
        fftw_free(signal); fftw_free(out);
        cheap_destroy(&ctx);
        return;
    }

    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        signal[i] = sin(2.0 * M_PI * 5.0 * t) + 0.5 * sin(2.0 * M_PI * 17.0 * t);
    }

    const int N_REPS = 100;
    double best = 1e18;
    for (int rep = 0; rep < N_REPS; ++rep) {
        double t0 = now_sec();
        cheap_frac_diff(&ctx, signal, n, d, out);
        double elapsed = now_sec() - t0;
        if (elapsed < best) best = elapsed;
    }

    double throughput = (double)n / best;
    printf("n=%5d  d=%.2f  time=%8.3f us  throughput=%10.0f samples/sec\n",
           n, d, best * 1e6, throughput);

    fftw_free(signal);
    fftw_free(out);
    cheap_destroy(&ctx);
}

int main(void)
{
    printf("CHEAPDAN Fractional Differencing Benchmark\n");
    printf("%s\n", "--------------------------------------------------------------");
    bench_one(256,  0.5);
    bench_one(1024, 0.5);
    bench_one(4096, 0.5);
    bench_one(256,  1.0);
    bench_one(1024, 1.0);
    bench_one(4096, 1.0);
    return 0;
}
