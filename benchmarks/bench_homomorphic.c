/* bench_homomorphic.c — Throughput benchmark for cheapdan_homomorphic_filter */
#define CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION
#include "cheapdan/homomorphic.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void bench_one(int n)
{
    cheapdan_homomorphic_ctx ctx;
    if (cheapdan_homomorphic_init(&ctx, n, 1e-3) != CHEAP_OK) {
        fprintf(stderr, "homomorphic_init failed n=%d\n", n);
        return;
    }

    double* signal = (double*)fftw_malloc((size_t)n * sizeof(double));
    double* out    = (double*)fftw_malloc((size_t)n * sizeof(double));
    if (!signal || !out) {
        fftw_free(signal); fftw_free(out);
        cheapdan_homomorphic_destroy(&ctx);
        return;
    }

    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        signal[i] = (1.0 + 0.3 * sin(2.0 * M_PI * 2.0 * t))
                  * (1.0 + 0.1 * sin(2.0 * M_PI * 15.0 * t));
    }

    const int N_REPS = 50;
    double best = 1e18;
    for (int rep = 0; rep < N_REPS; ++rep) {
        double t0 = now_sec();
        cheapdan_homomorphic_filter(&ctx, signal, n, out, NULL);
        double elapsed = now_sec() - t0;
        if (elapsed < best) best = elapsed;
    }

    double throughput = (double)n / best;
    printf("n=%5d  time=%8.3f us  throughput=%10.0f samples/sec\n",
           n, best * 1e6, throughput);

    fftw_free(signal);
    fftw_free(out);
    cheapdan_homomorphic_destroy(&ctx);
}

int main(void)
{
    printf("CHEAPDAN Homomorphic Filter Benchmark\n");
    printf("%s\n", "--------------------------------------------------------------");
    bench_one(256);
    bench_one(512);
    bench_one(1024);
    return 0;
}
