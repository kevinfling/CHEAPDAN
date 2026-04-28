#include "cheapdan/log.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N          256
#define N_ENSEMBLE  50
#define MAX_IMFS    12

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); return 1; } } while(0)

int main(void)
{
    /* Build a strictly-positive multiplicative test signal */
    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double mod = 1.0 + 0.3 * sin(2.0 * M_PI * 3.0 * t);
        double osc = 1.0 + 0.1 * sin(2.0 * M_PI * 15.0 * t);
        signal[i] = mod * osc;
        CHECK(signal[i] > 0.0, "signal[%d] <= 0\n", i);
    }

    cheap_ctx ctx;
    CHECK(cheap_init(&ctx, N, 0.7) == CHEAP_OK, "cheap_init failed\n");

    double imfs[MAX_IMFS * N];
    int n_imfs = 0;
    int ret = cheap_ceemdan_log(&ctx, signal, N_ENSEMBLE, 0.2, 0.7, imfs, &n_imfs);
    CHECK(ret == CHEAP_OK, "cheap_ceemdan_log failed: %d\n", ret);
    CHECK(n_imfs > 0, "expected n_imfs > 0, got %d\n", n_imfs);
    printf("Extracted %d multiplicative IMFs\n", n_imfs);

    for (int m = 0; m < n_imfs; ++m) {
        for (int i = 0; i < N; ++i)
            CHECK(isfinite(imfs[m * N + i]), "non-finite value at IMF %d idx %d\n", m, i);
    }

    /* Product reconstruction */
    double recon[N];
    cheap_ceemdan_log_recon(imfs, n_imfs, N, recon);

    double sig_energy = 0.0, err_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        sig_energy  += signal[i] * signal[i];
        double err   = signal[i] - recon[i];
        err_energy  += err * err;
    }
    double nmse = err_energy / sig_energy;
    printf("Signal energy: %.4f  Recon NMSE: %.6e\n", sig_energy, nmse);
    CHECK(nmse < 0.1, "NMSE too large: %.6e\n", nmse);

    /* --- Test shift variant with a non-positive multiplicative signal --- */
    double signal_shift[N];
    for (int i = 0; i < N; ++i)
        signal_shift[i] = signal[i] - 3.0;  /* make it negative, then shift back */

    double imfs2[MAX_IMFS * N];
    int n_imfs2 = 0;
    double offset = 0.0;
    ret = cheap_ceemdan_log_shift(&ctx, signal_shift, N_ENSEMBLE, 0.2, 0.7,
                                  1.0, imfs2, &n_imfs2, &offset);
    CHECK(ret == CHEAP_OK, "cheap_ceemdan_log_shift failed: %d\n", ret);
    CHECK(offset > 0.0, "offset should be positive: %.6e\n", offset);
    printf("Shift variant: offset=%.6e, n_imfs=%d\n", offset, n_imfs2);

    double recon2[N];
    cheap_ceemdan_log_recon(imfs2, n_imfs2, N, recon2);
    double shift_sig_energy = 0.0;
    err_energy = 0.0;
    for (int i = 0; i < N; ++i) {
        double shifted = signal_shift[i] + offset;
        shift_sig_energy += shifted * shifted;
        double err = shifted - recon2[i];
        err_energy += err * err;
    }
    nmse = (shift_sig_energy > 0.0) ? err_energy / shift_sig_energy : 0.0;
    printf("Shifted recon NMSE: %.6e\n", nmse);
    CHECK(nmse < 0.1, "shifted NMSE too large: %.6e\n", nmse);

    cheap_destroy(&ctx);
    printf("PASS: test_log\n");
    return 0;
}
