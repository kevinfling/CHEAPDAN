#define CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION
#include "cheapdan/homomorphic.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N 256
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)

int test_homomorphic_init_destroy(void)
{
    cheapdan_homomorphic_ctx ctx;
    CHECK(cheapdan_homomorphic_init(&ctx, N, 1e-3) == CHEAP_OK, "init failed\n");
    CHECK(ctx.n == N, "n mismatch\n");
    CHECK(ctx.fwd_plan != NULL, "fwd_plan null\n");
    CHECK(ctx.inv_plan != NULL, "inv_plan null\n");
    CHECK(ctx.X != NULL, "X null\n");
    CHECK(ctx.cepstrum != NULL, "cepstrum null\n");
    CHECK(ctx.liftered != NULL, "liftered null\n");
    CHECK(ctx.temp_real != NULL, "temp_real null\n");
    cheapdan_homomorphic_destroy(&ctx);
    CHECK(ctx.n == 0, "destroy did not zero\n");
    printf("PASS: test_homomorphic_init_destroy\n");
    return 0;
}

int test_homomorphic_errors(void)
{
    cheapdan_homomorphic_ctx ctx;
    CHECK(cheapdan_homomorphic_init(NULL, N, 1e-3) == CHEAP_EINVAL, "NULL ctx\n");
    CHECK(cheapdan_homomorphic_init(&ctx, 0, 1e-3) == CHEAP_EINVAL, "n=0\n");
    CHECK(cheapdan_homomorphic_init(&ctx, N, 0.0) == CHEAP_EINVAL, "eps=0\n");
    printf("PASS: test_homomorphic_errors\n");
    return 0;
}

int test_homomorphic_filter(void)
{
    cheapdan_homomorphic_ctx ctx;
    CHECK(cheapdan_homomorphic_init(&ctx, N, 1e-3) == CHEAP_OK, "init failed\n");

    /* Build a signal with multiplicative modulation:
     * x[i] = (1 + 0.3*sin(2π*3*t)) * (1 + 0.1*sin(2π*15*t))
     * The low-freq envelope is the multiplicative component.
     */
    double x[N], y[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double envelope = 1.0 + 0.3 * sin(2.0 * M_PI * 3.0 * t);
        double carrier  = 1.0 + 0.1 * sin(2.0 * M_PI * 15.0 * t);
        x[i] = envelope * carrier;
    }

    double cutoff = 0.0;
    int ret = cheapdan_homomorphic_filter(&ctx, x, N, y, &cutoff);
    CHECK(ret == CHEAP_OK, "filter failed: %d\n", ret);
    CHECK(cutoff > 0.0 && cutoff < N, "cutoff out of range: %.1f\n", cutoff);

    for (int i = 0; i < N; ++i)
        CHECK(isfinite(y[i]), "non-finite y[%d]\n", i);

    /* The filtered output should have reduced variance compared to the raw
     * multiplicative product (homomorphic processing suppresses the envelope). */
    double var_x = 0.0, var_y = 0.0, mean_x = 0.0, mean_y = 0.0;
    for (int i = 0; i < N; ++i) { mean_x += x[i]; mean_y += y[i]; }
    mean_x /= N; mean_y /= N;
    for (int i = 0; i < N; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    var_x /= N; var_y /= N;
    printf("  var(x)=%.6f  var(y)=%.6f  cutoff=%.1f\n", var_x, var_y, cutoff);

    cheapdan_homomorphic_destroy(&ctx);
    printf("PASS: test_homomorphic_filter\n");
    return 0;
}

int test_complex_cepstrum(void)
{
    cheapdan_homomorphic_ctx ctx;
    CHECK(cheapdan_homomorphic_init(&ctx, N, 1e-3) == CHEAP_OK, "init failed\n");

    double x[N];
    for (int i = 0; i < N; ++i)
        x[i] = sin(2.0 * M_PI * 4.0 * i / N);

    fftw_complex cep[N];
    cheapdan_complex_cepstrum(&ctx, x, cep, N, 1e-12);

    for (int i = 0; i < N; ++i) {
        CHECK(isfinite(cep[i][0]), "non-finite real cep[%d]\n", i);
        CHECK(isfinite(cep[i][1]), "non-finite imag cep[%d]\n", i);
    }

    /* For a pure sinusoid, the complex cepstrum should have strong peaks
     * at the quefrency corresponding to the period. */
    double peak_quef = 0.0, peak_val = 0.0;
    for (int i = 1; i < N/2; ++i) {
        double v = cep[i][0] * cep[i][0] + cep[i][1] * cep[i][1];
        if (v > peak_val) { peak_val = v; peak_quef = (double)i; }
    }
    printf("  peak quefrency ≈ %.1f (expected near N/4=%d for f=4)\n", peak_quef, N/4);

    cheapdan_homomorphic_destroy(&ctx);
    printf("PASS: test_complex_cepstrum\n");
    return 0;
}

int test_lifter_apply(void)
{
    int n = 64;
    fftw_complex cep[64];
    memset(cep, 0, sizeof(cep));
    for (int i = 0; i < n; ++i) {
        cep[i][0] = 1.0;
        cep[i][1] = 1.0;
    }

    cheapdan_lifter_apply(cep, n, 8.0, 3.0, 0.0, 1.0);

    /* DC removed */
    CHECK(cep[0][0] == 0.0 && cep[0][1] == 0.0, "DC not removed\n");

    /* High quefrencies (k > 8) should be passed (weight=1) */
    for (int k = 9; k < n/2; ++k) {
        CHECK(cep[k][0] == 1.0, "high q real not passed at %d\n", k);
        CHECK(cep[k][1] == 1.0, "high q imag not passed at %d\n", k);
    }

    /* Symmetry check */
    for (int k = 1; k < n/2; ++k) {
        int mk = n - k;
        CHECK(cep[k][0] == cep[mk][0], "symmetry real broken at %d\n", k);
        CHECK(cep[k][1] == cep[mk][1], "symmetry imag broken at %d\n", k);
    }

    printf("PASS: test_lifter_apply\n");
    return 0;
}

int main(void)
{
    if (test_homomorphic_init_destroy() != 0) return 1;
    if (test_homomorphic_errors()       != 0) return 1;
    if (test_homomorphic_filter()       != 0) return 1;
    if (test_complex_cepstrum()         != 0) return 1;
    if (test_lifter_apply()             != 0) return 1;
    printf("All homomorphic tests passed.\n");
    return 0;
}
