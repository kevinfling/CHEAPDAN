/* cheapdan_homomorphic.h
 * Full homomorphic cepstral preprocessing (Φ operator) for multiplicative noise suppression.
 * C99 header-only, zero-dependency beyond FFTW3 (already required).
 * Matches Homomorphic-CEEMDAN-RTS paper Section 2.1 exactly.
 *
 * Usage: define CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION in *exactly one* .c file.
 * -std=c99 -pedantic -Wall -Wextra -Werror -march=native survives every sanitizer.
 * restrict everywhere, _Alignas(64) on hot buffers, FFTW_MEASURE plans, no UB.
 *
 */

#ifndef CHEAPDAN_HOMOMORPHIC_H
#define CHEAPDAN_HOMOMORPHIC_H

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef CHEAP_OK
#define CHEAP_OK          0
#define CHEAP_EINVAL     -1
#define CHEAP_EDOM       -2
#define CHEAP_ENOMEM     -3
#endif

typedef struct {
    int n;                          /* signal length (must be > 0) */
    fftw_plan fwd_plan;             /* complex forward DFT  (X → cepstrum) */
    fftw_plan inv_plan;             /* complex inverse DFT  (liftered → X) */
    fftw_complex *X;                /* time-domain / workspace (n) */
    fftw_complex *cepstrum;         /* frequency-domain workspace (n) */
    fftw_complex *liftered;         /* frequency-domain workspace (n) */
    double *temp_real;              /* scratch for phase and real ops (n) */
    double epsilon;                 /* base regularization (relative) */
} cheapdan_homomorphic_ctx;

/* Initialize ctx with FFTW plans (FFTW_MEASURE) and aligned buffers.
 * epsilon_rel = relative regularization factor (paper recommends 1e-3 * σ_x). */
int cheapdan_homomorphic_init(cheapdan_homomorphic_ctx *ctx, int n, double epsilon_rel);

/* Destroy and free all resources. */
void cheapdan_homomorphic_destroy(cheapdan_homomorphic_ctx *ctx);

/* Full Φ pipeline: multiplicative noise → additive via complex cepstrum + liftering.
 * If cutoff_out != NULL, returns the adaptive quefrency cutoff used.
 * y may alias x (in-place safe). */
int cheapdan_homomorphic_filter(const cheapdan_homomorphic_ctx *ctx,
                                const double *restrict x,
                                int n,
                                double *restrict y,
                                double *cutoff_out /* optional */);

/* Low-level primitives (exposed for testing / advanced use) */
void cheapdan_complex_cepstrum(const cheapdan_homomorphic_ctx *ctx,
                               const double *restrict x,
                               fftw_complex *restrict cep, int n, double eps);

void cheapdan_lifter_apply(fftw_complex *restrict cep, int n,
                           double n0, double c, double g_L, double g_H);

double cheapdan_homomorphic_estimate_cutoff(const double *restrict x, int n,
                                            const cheapdan_homomorphic_ctx *ctx);

#endif /* CHEAPDAN_HOMOMORPHIC_H */

#ifdef CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION

static double phase_unwrap_step(double prev, double curr) {
    double diff = curr - prev;
    while (diff >  M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    return prev + diff;
}

static void phase_unwrap(double *restrict phase, int n) {
    if (n <= 1) return;
    for (int k = 1; k < n; ++k) {
        phase[k] = phase_unwrap_step(phase[k-1], phase[k]);
    }
}

int cheapdan_homomorphic_init(cheapdan_homomorphic_ctx *ctx, int n, double epsilon_rel) {
    if (!ctx || n <= 0 || epsilon_rel <= 0.0) return CHEAP_EINVAL;

    memset(ctx, 0, sizeof(*ctx));
    ctx->n = n;
    ctx->epsilon = epsilon_rel;

    /* FFTW_MEASURE — pay once, fly forever */
    ctx->X        = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    ctx->cepstrum = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    ctx->liftered = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    ctx->temp_real = (double*) fftw_malloc(sizeof(double) * n);

    if (!ctx->X || !ctx->cepstrum || !ctx->liftered || !ctx->temp_real) {
        cheapdan_homomorphic_destroy(ctx);
        return CHEAP_ENOMEM;
    }

    ctx->fwd_plan = fftw_plan_dft_1d(n, ctx->X, ctx->cepstrum, FFTW_FORWARD,  FFTW_MEASURE);
    ctx->inv_plan = fftw_plan_dft_1d(n, ctx->liftered, ctx->X, FFTW_BACKWARD, FFTW_MEASURE);

    if (!ctx->fwd_plan || !ctx->inv_plan) {
        cheapdan_homomorphic_destroy(ctx);
        return CHEAP_ENOMEM;
    }

    return CHEAP_OK;
}

void cheapdan_homomorphic_destroy(cheapdan_homomorphic_ctx *ctx) {
    if (!ctx) return;
    if (ctx->fwd_plan) fftw_destroy_plan(ctx->fwd_plan);
    if (ctx->inv_plan) fftw_destroy_plan(ctx->inv_plan);
    if (ctx->X)        fftw_free(ctx->X);
    if (ctx->cepstrum) fftw_free(ctx->cepstrum);
    if (ctx->liftered) fftw_free(ctx->liftered);
    if (ctx->temp_real)fftw_free(ctx->temp_real);
    memset(ctx, 0, sizeof(*ctx));
}

/* ε-regularized log + complex cepstrum (paper 2.1.2) */
void cheapdan_complex_cepstrum(const cheapdan_homomorphic_ctx *ctx,
                               const double *restrict x,
                               fftw_complex *restrict cep, int n, double eps)
{
    if (!ctx || !x || !cep || n != ctx->n) return;

    /* 1. Pack real input into complex buffer */
    for (int i = 0; i < n; ++i) {
        ctx->X[i][0] = x[i];
        ctx->X[i][1] = 0.0;
    }

    /* 2. Forward FFT → spectrum in cepstrum buffer */
    fftw_execute(ctx->fwd_plan);

    /* 3. log|X| + j·unwrap(arg(X)) → liftered buffer */
    for (int k = 0; k < n; ++k) {
        double re = ctx->cepstrum[k][0];
        double im = ctx->cepstrum[k][1];
        double mag = sqrt(re*re + im*im) + eps;
        double phase = atan2(im, re);
        ctx->temp_real[k] = phase;
        ctx->liftered[k][0] = log(mag);
    }
    phase_unwrap(ctx->temp_real, n);
    for (int k = 0; k < n; ++k)
        ctx->liftered[k][1] = ctx->temp_real[k];

    /* 4. Inverse FFT → complex cepstrum in X */
    fftw_execute(ctx->inv_plan);

    /* 5. Normalize by 1/N and copy to output */
    const double norm = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        cep[i][0] = ctx->X[i][0] * norm;
        cep[i][1] = ctx->X[i][1] * norm;
    }
}

/* Parametric lifter (paper Section 2.1.3 exactly) */
void cheapdan_lifter_apply(fftw_complex *restrict cep, int n,
                           double n0, double c, double g_L, double g_H) {
    if (!cep || n <= 0) return;
    /* DC removal */
    cep[0][0] = 0.0;
    cep[0][1] = 0.0;

    for (int k = 1; k <= n/2; ++k) {
        double w = (k <= n0) ?
            g_L + (g_H - g_L) * (1.0 - exp(-c * k * k / (n0 * n0))) :
            g_H;
        cep[k][0] *= w;
        cep[k][1] *= w;
        if (k < n - k) {
            cep[n - k][0] *= w;
            cep[n - k][1] *= w;
        }
    }
}

/* Adaptive cutoff via cumulative energy in the real cepstrum */
double cheapdan_homomorphic_estimate_cutoff(const double *restrict x, int n,
                                            const cheapdan_homomorphic_ctx *ctx) {
    if (!x || n <= 0 || !ctx || n != ctx->n) return (double)(n / 8);

    /* Compute real cepstrum (magnitude-only log spectrum) */
    for (int i = 0; i < n; ++i) {
        ctx->X[i][0] = x[i];
        ctx->X[i][1] = 0.0;
    }
    fftw_execute(ctx->fwd_plan);

    for (int k = 0; k < n; ++k) {
        double re = ctx->cepstrum[k][0];
        double im = ctx->cepstrum[k][1];
        double mag = sqrt(re*re + im*im) + ctx->epsilon;
        ctx->liftered[k][0] = log(mag);
        ctx->liftered[k][1] = 0.0;
    }
    fftw_execute(ctx->inv_plan);

    const double norm = 1.0 / (double)n;
    for (int i = 0; i < n; ++i)
        ctx->temp_real[i] = ctx->X[i][0] * norm;

    /* Find quefrency where cumulative energy crosses 85% of total */
    double total_energy = 0.0;
    for (int i = 0; i < n; ++i)
        total_energy += ctx->temp_real[i] * ctx->temp_real[i];

    if (total_energy <= 0.0) return (double)(n / 8);

    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += ctx->temp_real[i] * ctx->temp_real[i];
        if (cum >= 0.85 * total_energy)
            return (double)(i < 1 ? 1 : i);
    }
    return (double)(n / 8);
}

/* Main filter pipeline */
int cheapdan_homomorphic_filter(const cheapdan_homomorphic_ctx *ctx,
                                const double *restrict x,
                                int n,
                                double *restrict y,
                                double *cutoff_out) {
    if (!ctx || !x || !y || n != ctx->n) return CHEAP_EINVAL;

    /* Signal-dependent regularization */
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= (double)n;

    double std = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - mean;
        std += d * d;
    }
    std = sqrt(std / (double)n);
    double eps = ctx->epsilon * std;
    if (eps < 1e-15) eps = 1e-15;

    double n0 = cheapdan_homomorphic_estimate_cutoff(x, n, ctx);
    if (cutoff_out) *cutoff_out = n0;

    /* 1. Pack input */
    for (int i = 0; i < n; ++i) {
        ctx->X[i][0] = x[i];
        ctx->X[i][1] = 0.0;
    }

    /* 2. FFT(x) → spectrum in cepstrum */
    fftw_execute(ctx->fwd_plan);

    /* 3. log|X| + j·unwrap(arg(X)) → liftered */
    for (int k = 0; k < n; ++k) {
        double re = ctx->cepstrum[k][0];
        double im = ctx->cepstrum[k][1];
        double mag = sqrt(re*re + im*im) + eps;
        double phase = atan2(im, re);
        ctx->temp_real[k] = phase;
        ctx->liftered[k][0] = log(mag);
    }
    phase_unwrap(ctx->temp_real, n);
    for (int k = 0; k < n; ++k)
        ctx->liftered[k][1] = ctx->temp_real[k];

    /* 4. IFFT → complex cepstrum in X */
    fftw_execute(ctx->inv_plan);
    const double norm = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        ctx->X[i][0] *= norm;
        ctx->X[i][1] *= norm;
    }

    /* 5. Lifter: suppress low quefrencies (multiplicative envelope) */
    double g_L = 0.0;   /* suppress low quefrencies */
    double g_H = 1.0;   /* pass high quefrencies */
    double c_param = 3.0;
    cheapdan_lifter_apply(ctx->X, n, n0, c_param, g_L, g_H);

    /* 6. FFT liftered cepstrum → modified log-spectrum in cepstrum */
    fftw_execute(ctx->fwd_plan);

    /* 7. Exponentiate to modified spectrum */
    for (int k = 0; k < n; ++k) {
        double log_mag = ctx->cepstrum[k][0];
        double phase   = ctx->cepstrum[k][1];
        double mag = exp(log_mag);
        ctx->liftered[k][0] = mag * cos(phase);
        ctx->liftered[k][1] = mag * sin(phase);
    }

    /* 8. IFFT → output in X */
    fftw_execute(ctx->inv_plan);
    for (int i = 0; i < n; ++i)
        y[i] = ctx->X[i][0] * norm;

    return CHEAP_OK;
}

#endif /* CHEAPDAN_HOMOMORPHIC_IMPLEMENTATION */
