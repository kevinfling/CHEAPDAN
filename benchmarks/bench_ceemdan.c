/* bench_ceemdan.c — Throughput benchmark for cheap_ceemdan_frac */
#include "cheapdan.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void bench_one(int n, int n_ensemble)
{
    cheap_ctx ctx;
    if (cheap_init(&ctx, n, 0.7) != CHEAP_OK) {
        fprintf(stderr, "cheap_init failed n=%d\n", n);
        return;
    }

    double* signal = (double*)fftw_malloc((size_t)n * sizeof(double));
    double* imfs   = (double*)fftw_malloc((size_t)12 * n * sizeof(double));
    if (!signal || !imfs) {
        fprintf(stderr, "alloc failed\n");
        fftw_free(signal);
        fftw_free(imfs);
        cheap_destroy(&ctx);
        return;
    }

    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        signal[i] = sin(2.0 * M_PI * 5.0 * t) + 0.5 * sin(2.0 * M_PI * 17.0 * t);
    }

    int n_imfs = 0;
    const int N_REPS = 3;
    double best = 1e18;

    for (int rep = 0; rep < N_REPS; ++rep) {
        double t0 = now_sec();
        cheap_ceemdan_frac(&ctx, signal, n_ensemble, 0.2, 0.7, imfs, &n_imfs);
        double elapsed = now_sec() - t0;
        if (elapsed < best) best = elapsed;
    }

    double throughput = (double)n / best;
    printf("n=%5d  n_ens=%3d  n_imfs=%2d  time=%7.1f ms  throughput=%10.0f samples/sec\n",
           n, n_ensemble, n_imfs, best * 1e3, throughput);

    fftw_free(signal);
    fftw_free(imfs);
    cheap_destroy(&ctx);
}

int main(void)
{
    printf("CHEAPDAN 1D Fractional CEEMDAN Benchmark\n");
    printf("%-70s\n", "----------------------------------------------------------------------");
    bench_one(256,  100);
    bench_one(1024, 100);
    bench_one(4096, 100);
    return 0;
}
