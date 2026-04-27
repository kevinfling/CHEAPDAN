# CHEAPDAN

**C99 · Header-only · MIT**

Fractional-noise CEEMDAN signal decomposition library. Extracts Intrinsic Mode Functions (IMFs) from 1D and 2D signals using Complete Ensemble EMD with Adaptive Noise, driven by fractional Gaussian noise coloured by the Hurst exponent *H*. Built as a plugin on top of [CHEAP](https://github.com/kevinfling/CHEAP), which provides DCT-based circulant spectral operations.

---

## Mathematical Summary

### Fractional CEEMDAN

Given signal **x** of length *n*, CHEAPDAN iterates the CEEMDAN outer loop (Torres et al. 2011):

```
r₀ = x
For k = 1, 2, ...:
    IMFₖ = (1/L) Σᵢ E₁(rₖ₋₁ + ε · fGnᵢ)     [ensemble average of first EMD mode]
    rₖ   = rₖ₋₁ − IMFₖ
    Stop when rₖ satisfies the IMF condition.
```

where `E₁(·)` denotes extraction of the first mode via spectral sifting and `fGnᵢ` is the *i*-th realisation of fractional Gaussian noise.

### Fractional Gaussian Noise (fGn) Synthesis

Noise is synthesised spectrally via the Flandrin–Wood eigenvalue approximation:

```
fGn = IDCT( sqrt(λₖ) ⊙ z ),    z ~ N(0, I)
λₖ  = |ξₖ|^{2H-2},             ξₖ = π k / n
```

- **H = 0.5**: white noise (standard CEEMDAN)
- **H > 0.5**: long-memory noise (persistent)
- **H < 0.5**: anti-persistent (rough)

### Spectral Sifting Kernels

The sifting step computes `E₁(x) = IDCT(wₖ ⊙ DCT(x))`. Four kernel choices:

| Mode | Weight `wₖ` | Parameter |
|------|-------------|-----------|
| `LOWPASS` | `1` if `k < k_cutoff`, else `0` | `k_cutoff` (integer) |
| `FRAC_INTEGRATOR` | `\|ξₖ\|^{−α}` | `α` (α > 0 = LP, α < 0 = HP) |
| `HEAT_GAUSS` | `exp(−t ξₖ²)` | `t ≥ 0` (diffusion time) |
| `FRAC_ROUGH` | `\|ξₖ\|^{2H−2}` | `−2H` (Hurst roughness) |

### RTS Smoother

`cheapdan_rts.h` provides an optional Rauch-Tung-Striebel smoother for post-processing IMFs. Uses a scalar random-walk state model:

```
Forward:   x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + Kₜ (yₜ − x̂ₜ|ₜ₋₁)
           Kₜ = Pₜ|ₜ₋₁ / (Pₜ|ₜ₋₁ + R)

Backward:  Jₜ = Pₜ|ₜ / Pₜ₊₁|ₜ
           x̂ₜ|N = x̂ₜ|ₜ + Jₜ (x̂ₜ₊₁|N − x̂ₜ₊₁|ₜ)
```

Initial covariance is scaled by `(1−H)²` to encode Hurst-dependent prior uncertainty.

### 2D Separable Extension

`cheap_ceemdan_frac_2d` applies the 1D algorithm separably: first along rows (using `row_ctx`), then along columns of the resulting intermediate surface (using `col_ctx`). This follows Nunes et al. (2003) and Linderhed (2009) for 2D EMD. Each pass yields one IMF plane; the function iterates to extract up to `max_imfs` planes.

---

## Build

### CMake (recommended)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -L cheapdan --output-on-failure
```

**Dependencies:** FFTW3 must be installed:
```sh
apt install libfftw3-dev   # Debian/Ubuntu
brew install fftw          # macOS
```

CHEAP is fetched automatically via CMake FetchContent from `https://github.com/kevinfling/CHEAP` (tag `v0.2.0`).

**Build options:**

| Option | Default | Description |
|--------|---------|-------------|
| `CHEAPDAN_BUILD_TESTS` | ON | Compile test suite |
| `CHEAPDAN_BUILD_BENCHMARKS` | ON | Compile benchmarks |
| `CHEAPDAN_BUILD_EXAMPLES` | ON | Compile examples |

### Manual (single-file)

```sh
gcc -std=c99 -O3 -march=native examples/example_1d.c -Iinclude -I/path/to/cheap/include -lfftw3 -lm -o example_1d
```

---

## API Reference

### `cheapdan.h`

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `cheap_weights_spectral_lowpass` | `(int n, int k_cutoff, double* w)` | `CHEAP_OK` / `CHEAP_EINVAL` | Rect low-pass weights |
| `cheap_weights_frac_integrator` | `(int n, double alpha, double* w)` | `CHEAP_OK` / `CHEAP_EINVAL` / `CHEAP_EDOM` | Fractional integrator weights |
| `cheap_weights_heat_kernel_gauss` | `(int n, double t, double* w)` | `CHEAP_OK` / `CHEAP_EINVAL` / `CHEAP_EDOM` | Gaussian heat kernel weights |
| `cheap_ceemdan_sift_step` | `(ctx, in, out, mode, param)` | `CHEAP_OK` / error | One spectral sift step |
| `cheap_ceemdan_is_imf` | `(const double* x, int n, double tol)` | `bool` | IMF stopping criterion |
| `cheap_ceemdan_frac` | `(ctx, signal, n_ens, eps, H, imfs_out, n_imfs_out)` | `CHEAP_OK` / error | 1D fractional CEEMDAN |
| `cheap_ceemdan_frac_2d` | `(row_ctx, col_ctx, sig, H, W, n_ens, eps, H_hurst, mode, param, imfs_out, max_imfs, n_imfs_out)` | `CHEAP_OK` / error | 2D separable CEEMDAN |

### `cheapdan_rts.h`

Define `CHEAPDAN_RTS_IMPLEMENTATION` in exactly one translation unit before including.

| Function | Signature | Description |
|----------|-----------|-------------|
| `cheapdan_rts_init` | `(cheapdan_rts_t* rts, double H)` | Initialize smoother; sets P = (1−H)²·I |
| `cheapdan_rts_smooth` | `(rts, imf_in, N, imf_smooth_out)` | Forward Kalman + backward RTS pass |

---

## Example

```c
#include "cheapdan.h"
#include <stdio.h>
#include <math.h>

#define N 512

int main(void) {
    cheap_ctx ctx;
    cheap_init(&ctx, N, 0.7);   /* Hurst H=0.7 */

    double signal[N];
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        signal[i] = sin(2.0 * M_PI * 3.0 * t) + 0.5 * sin(2.0 * M_PI * 15.0 * t);
    }

    double imfs[12 * N];
    int n_imfs = 0;
    cheap_ceemdan_frac(&ctx, signal, 100, 0.2, 0.7, imfs, &n_imfs);

    for (int m = 0; m < n_imfs; ++m) {
        double energy = 0.0;
        for (int i = 0; i < N; ++i) energy += imfs[m * N + i] * imfs[m * N + i];
        printf("IMF[%d] energy = %.4f\n", m, energy);
    }

    cheap_destroy(&ctx);
    return 0;
}
```

---

## Benchmark Results

Results from `./build/benchmarks/bench_ceemdan` (Release build, `-O3 -march=native`):

| `n` | `n_ensemble` | `n_imfs` | `time_ms` | `samples/sec` |
|-----|-------------|---------|-----------|--------------|
| 256 | 100 | 12 | 64 | 3,976 |
| 1024 | 100 | 12 | 404 | 2,536 |
| 4096 | 100 | 12 | 2,219 | 1,846 |

*Measured on ARM Cortex-A57 (Jetson TX2), Release build `-O3 -march=native`. Re-run `./build/benchmarks/bench_ceemdan` for your hardware.*

---

## License

MIT License  
Copyright (c) 2026 Kevin Fling
