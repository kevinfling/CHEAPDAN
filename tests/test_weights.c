#include "cheapdan.h"
#include <stdio.h>
#include <math.h>

#define N 64
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)
#define CLOSE(a, b, tol) (fabs((a)-(b)) < (tol))

static int test_spectral_lowpass(void)
{
    double w[N];

    CHECK(cheap_weights_spectral_lowpass(N, 16, w) == CHEAP_OK, "lowpass ok failed\n");
    for (int k = 0;  k < 16; ++k) CHECK(CLOSE(w[k], 1.0, 1e-15), "w[%d]=%f expected 1\n", k, w[k]);
    for (int k = 16; k < N;  ++k) CHECK(CLOSE(w[k], 0.0, 1e-15), "w[%d]=%f expected 0\n", k, w[k]);

    CHECK(cheap_weights_spectral_lowpass(N, 0, w) == CHEAP_OK,  "lowpass k=0 failed\n");
    for (int k = 0; k < N; ++k) CHECK(CLOSE(w[k], 0.0, 1e-15), "w[%d] should be 0\n", k);

    CHECK(cheap_weights_spectral_lowpass(N, N, w) == CHEAP_OK,  "lowpass k=N failed\n");
    for (int k = 0; k < N; ++k) CHECK(CLOSE(w[k], 1.0, 1e-15), "w[%d] should be 1\n", k);

    CHECK(cheap_weights_spectral_lowpass(1, 0, w) == CHEAP_EINVAL, "n<2 should fail\n");
    CHECK(cheap_weights_spectral_lowpass(N, 8, NULL) == CHEAP_EINVAL, "NULL should fail\n");
    CHECK(cheap_weights_spectral_lowpass(N, -1, w) == CHEAP_EINVAL, "k<0 should fail\n");
    CHECK(cheap_weights_spectral_lowpass(N, N+1, w) == CHEAP_EINVAL, "k>n should fail\n");

    printf("PASS: test_spectral_lowpass\n");
    return 0;
}

static int test_frac_integrator(void)
{
    double w[N];

    CHECK(cheap_weights_frac_integrator(N, 1.0, w) == CHEAP_OK, "frac_int ok failed\n");
    CHECK(CLOSE(w[1], (double)N / M_PI, 1e-10), "w[1]=%f expected %f\n", w[1], (double)N/M_PI);
    CHECK(isfinite(w[0]), "w[0] must be finite\n");
    for (int k = 1; k < N-1; ++k)
        CHECK(w[k] >= w[k+1], "not monotone at k=%d: w[%d]=%f w[%d]=%f\n", k, k, w[k], k+1, w[k+1]);

    CHECK(cheap_weights_frac_integrator(N, 0.0, w) == CHEAP_OK, "frac_int alpha=0 failed\n");
    for (int k = 0; k < N; ++k) CHECK(CLOSE(w[k], 1.0, 1e-14), "w[%d]=%f should be 1\n", k, w[k]);

    CHECK(cheap_weights_frac_integrator(1, 1.0, w) == CHEAP_EINVAL, "n<2 should fail\n");
    CHECK(cheap_weights_frac_integrator(N, 1.0, NULL) == CHEAP_EINVAL, "NULL should fail\n");

    printf("PASS: test_frac_integrator\n");
    return 0;
}

static int test_heat_kernel_gauss(void)
{
    double w[N];

    CHECK(cheap_weights_heat_kernel_gauss(N, 1.0, w) == CHEAP_OK, "heat ok failed\n");
    CHECK(CLOSE(w[0], 1.0, 1e-15), "DC should be 1, got %f\n", w[0]);
    for (int k = 0; k < N-1; ++k)
        CHECK(w[k] >= w[k+1], "not monotone at k=%d\n", k);

    CHECK(cheap_weights_heat_kernel_gauss(N, 0.0, w) == CHEAP_OK, "t=0 failed\n");
    for (int k = 0; k < N; ++k) CHECK(CLOSE(w[k], 1.0, 1e-14), "w[%d] should be 1\n", k);

    CHECK(cheap_weights_heat_kernel_gauss(N, 100.0, w) == CHEAP_OK, "large t failed\n");
    CHECK(w[N-1] < 1e-10, "high freq not suppressed: w[N-1]=%e\n", w[N-1]);

    CHECK(cheap_weights_heat_kernel_gauss(N, -0.1, w) == CHEAP_EINVAL, "t<0 should fail\n");
    CHECK(cheap_weights_heat_kernel_gauss(1, 1.0, w) == CHEAP_EINVAL, "n<2 should fail\n");
    CHECK(cheap_weights_heat_kernel_gauss(N, 1.0, NULL) == CHEAP_EINVAL, "NULL should fail\n");

    printf("PASS: test_heat_kernel_gauss\n");
    return 0;
}

int main(void)
{
    if (test_spectral_lowpass()   != 0) return 1;
    if (test_frac_integrator()    != 0) return 1;
    if (test_heat_kernel_gauss()  != 0) return 1;
    printf("All weight tests passed.\n");
    return 0;
}
