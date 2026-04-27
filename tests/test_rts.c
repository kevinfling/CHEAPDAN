#define CHEAPDAN_RTS_IMPLEMENTATION
#include "cheapdan_rts.h"
#include "cheapdan.h"
#include <stdio.h>
#include <math.h>

#define N 256
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)

int main(void)
{
    double true_sig[N], noisy_imf[N], smoothed[N];
    for (int i = 0; i < N; ++i)
        true_sig[i] = sin(2.0 * M_PI * 3.0 * i / N);

    cheap__rng rng;
    cheap__rng_init(&rng, 0x12345678ULL);
    for (int i = 0; i < N; ++i)
        noisy_imf[i] = true_sig[i] + 0.2 * cheap__lcg_normal(&rng);

    cheapdan_rts_t rts;
    cheapdan_rts_init(&rts, 0.7);

    /* Verify var_scale = (1-H)^2 = 0.09 was applied */
    double expected_var = pow(1.0 - 0.7, 2.0);
    CHECK(fabs(rts.P_fwd[0][0] - expected_var) < 1e-14,
          "P_fwd[0][0]=%.10f expected %.10f\n", rts.P_fwd[0][0], expected_var);
    printf("P_fwd[0][0] = %.6f (expected %.6f)\n", rts.P_fwd[0][0], expected_var);

    cheapdan_rts_smooth(&rts, noisy_imf, N, smoothed);

    double mse_raw = 0.0, mse_smooth = 0.0;
    for (int i = 0; i < N; ++i) {
        double er = noisy_imf[i] - true_sig[i];
        double es = smoothed[i]  - true_sig[i];
        mse_raw    += er * er;
        mse_smooth += es * es;
    }
    mse_raw    /= N;
    mse_smooth /= N;
    printf("MSE raw=%.6f  MSE smoothed=%.6f\n", mse_raw, mse_smooth);
    CHECK(mse_smooth < mse_raw, "smoother did not reduce MSE: %.6f >= %.6f\n",
          mse_smooth, mse_raw);

    CHECK(isfinite(rts.x_fwd[0]), "rts.x_fwd[0] is not finite\n");
    CHECK(isfinite(rts.x_smooth[0]), "rts.x_smooth[0] is not finite\n");
    printf("x_fwd[0]=%.6f  x_smooth[0]=%.6f\n", rts.x_fwd[0], rts.x_smooth[0]);

    printf("PASS: test_rts\n");
    return 0;
}
