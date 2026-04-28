#define CHEAPDAN_FRACDIFF_IMPLEMENTATION
#include "cheapdan/fracdiff.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N 256
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)
#define CLOSE(a, b, tol) (fabs((a)-(b)) < (tol))

int test_fracdiff_identity(void)
{
    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    double x[N], y[N];
    for (int i = 0; i < N; ++i) x[i] = sin(2.0 * M_PI * 3.0 * i / N);

    /* d = 0 → identity (weights = 1 everywhere) */
    CHECK(cheap_frac_diff(&ctx, x, N, 0.0, y) == CHEAP_OK, "frac_diff d=0 failed\n");
    for (int i = 0; i < N; ++i)
        CHECK(CLOSE(y[i], x[i], 1e-12), "identity fail at %d: %.6e vs %.6e\n", i, y[i], x[i]);

    cheap_destroy(&ctx);
    printf("PASS: test_fracdiff_identity\n");
    return 0;
}

int test_fracdiff_derivative(void)
{
    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    /* DCT-II eigenfunction: cos(π * k * (i + 0.5) / N)
     * For k=3, the DCT gives a delta at bin 3; fractional diff scales by (π*3/N)^d */
    const int k = 3;
    double x[N], y[N];
    for (int i = 0; i < N; ++i) {
        x[i] = cos(M_PI * k * (i + 0.5) / N);
    }

    /* d = 1 → should scale the eigenfunction by (πk/N) */
    CHECK(cheap_frac_diff(&ctx, x, N, 1.0, y) == CHEAP_OK, "frac_diff d=1 failed\n");

    double expected_scale = M_PI * k / N;
    double err = 0.0, ref_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = expected_scale * x[i];
        err += (y[i] - ref) * (y[i] - ref);
        ref_energy += ref * ref;
    }
    double nrmse = (ref_energy > 0.0) ? sqrt(err / ref_energy) : 0.0;
    CHECK(nrmse < 0.05, "derivative NRMSE too large: %.4f\n", nrmse);

    cheap_destroy(&ctx);
    printf("PASS: test_fracdiff_derivative (NRMSE=%.4f)\n", nrmse);
    return 0;
}

int test_fracdiff_errors(void)
{
    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    double x[N], y[N];
    memset(x, 0, sizeof(x));

    CHECK(cheap_frac_diff(NULL, x, N, 0.5, y) == CHEAP_EINVAL, "NULL ctx\n");
    CHECK(cheap_frac_diff(&ctx, NULL, N, 0.5, y) == CHEAP_EINVAL, "NULL x\n");
    CHECK(cheap_frac_diff(&ctx, x, N, 0.5, NULL) == CHEAP_EINVAL, "NULL y\n");
    CHECK(cheap_frac_diff(&ctx, x, N + 1, 0.5, y) == CHEAP_EINVAL, "mismatched n\n");
    CHECK(cheap_frac_diff(&ctx, x, N, INFINITY, y) == CHEAP_EINVAL, "infinite d\n");

    cheap_destroy(&ctx);
    printf("PASS: test_fracdiff_errors\n");
    return 0;
}

int test_fracdiff_integrate(void)
{
    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    double x[N], y[N];
    for (int i = 0; i < N; ++i) x[i] = cos(2.0 * M_PI * 3.0 * i / N);

    /* integrate then diff should be approximate identity */
    CHECK(cheap_frac_integrate(&ctx, x, N, 1.0, y) == CHEAP_OK, "integrate failed\n");
    CHECK(cheap_frac_diff(&ctx, y, N, 1.0, y) == CHEAP_OK, "diff after integrate failed\n");

    double err = 0.0, sig_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        err += (y[i] - x[i]) * (y[i] - x[i]);
        sig_energy += x[i] * x[i];
    }
    double nrmse = sqrt(err / sig_energy);
    CHECK(nrmse < 0.05, "round-trip NRMSE too large: %.4f\n", nrmse);

    cheap_destroy(&ctx);
    printf("PASS: test_fracdiff_integrate (NRMSE=%.4f)\n", nrmse);
    return 0;
}

int test_fracdiff_2d(void)
{
    const int H = 32, W = 32;
    cheap_ctx row_ctx, col_ctx;
    CHECK(cheap_init(&row_ctx, W, 0.6) == CHEAP_OK, "row init failed\n");
    CHECK(cheap_init(&col_ctx, H, 0.6) == CHEAP_OK, "col init failed\n");

    double x[H * W], y[H * W];
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            x[r * W + c] = sin(2.0 * M_PI * r / H) * cos(2.0 * M_PI * c / W);

    CHECK(cheap_frac_diff_2d(&row_ctx, &col_ctx, x, H, W, 0.0, y) == CHEAP_OK,
          "2d identity failed\n");

    double err = 0.0;
    for (int i = 0; i < H * W; ++i)
        err += (y[i] - x[i]) * (y[i] - x[i]);
    CHECK(sqrt(err / (H * W)) < 1e-12, "2d identity error too large\n");

    /* Dimension mismatch */
    cheap_ctx bad;
    CHECK(cheap_init(&bad, W + 1, 0.6) == CHEAP_OK, "bad init failed\n");
    CHECK(cheap_frac_diff_2d(&bad, &col_ctx, x, H, W, 0.5, y) == CHEAP_EINVAL,
          "dim mismatch should fail\n");
    cheap_destroy(&bad);

    cheap_destroy(&row_ctx);
    cheap_destroy(&col_ctx);
    printf("PASS: test_fracdiff_2d\n");
    return 0;
}

int main(void)
{
    if (test_fracdiff_identity()  != 0) return 1;
    if (test_fracdiff_derivative() != 0) return 1;
    if (test_fracdiff_errors()    != 0) return 1;
    if (test_fracdiff_integrate() != 0) return 1;
    if (test_fracdiff_2d()        != 0) return 1;
    printf("All fracdiff tests passed.\n");
    return 0;
}
