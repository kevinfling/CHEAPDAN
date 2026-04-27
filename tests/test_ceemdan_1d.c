#include "cheapdan.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N          256
#define N_ENSEMBLE  50
#define MAX_IMFS    12

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)

int main(void)
{
    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        signal[i] = sin(2.0 * M_PI * 3.0 * t) + 0.5 * sin(2.0 * M_PI * 15.0 * t);
    }

    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    double imfs[MAX_IMFS * N];
    int n_imfs = 0;
    int frac_ret = cheap_ceemdan_frac(&ctx, signal, N_ENSEMBLE, 0.2, 0.7, imfs, &n_imfs);
    CHECK(frac_ret == CHEAP_OK, "cheap_ceemdan_frac failed: %d\n", frac_ret);
    CHECK(n_imfs > 0, "expected n_imfs > 0, got %d\n", n_imfs);
    printf("Extracted %d IMFs\n", n_imfs);

    for (int m = 0; m < n_imfs; ++m) {
        double energy = 0.0;
        for (int i = 0; i < N; ++i) {
            CHECK(isfinite(imfs[m * N + i]), "non-finite value at IMF %d idx %d\n", m, i);
            energy += imfs[m * N + i] * imfs[m * N + i];
        }
        CHECK(energy > 1e-10, "IMF %d has zero energy\n", m);
        printf("  IMF[%d] energy = %.6f\n", m, energy);
    }

    double recon[N];
    memset(recon, 0, sizeof(recon));
    for (int m = 0; m < n_imfs; ++m)
        for (int i = 0; i < N; ++i)
            recon[i] += imfs[m * N + i];

    double sig_energy = 0.0, recon_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        sig_energy   += signal[i] * signal[i];
        recon_energy += recon[i]  * recon[i];
    }
    CHECK(recon_energy > 0.0, "partial reconstruction has zero energy\n");
    printf("Signal energy: %.4f  Partial-recon energy: %.4f\n", sig_energy, recon_energy);

    cheap_destroy(&ctx);
    printf("PASS: test_ceemdan_1d\n");
    return 0;
}
