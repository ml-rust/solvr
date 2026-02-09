# solvr

`solvr` is a scientific computing crate for real-world numerical workflows in Rust.
It provides high-level algorithms on top of `numr` tensor/runtime primitives, with CPU support by default and optional CUDA/WebGPU backends.

## For SciPy Users

If you are coming from SciPy, this map should help:

| SciPy area             | `solvr` module(s)                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `scipy.signal`         | `signal`, `window`                                                                                            |
| `scipy.ndimage`        | `signal` (N-D filters, edge ops), `morphology`, `spatial` (distance transform), `interpolate` (geometric ops) |
| `scipy.interpolate`    | `interpolate`                                                                                                 |
| `scipy.optimize`       | `optimize`                                                                                                    |
| `scipy.integrate`      | `integrate`                                                                                                   |
| `scipy.stats`          | `stats`                                                                                                       |
| `scipy.spatial`        | `spatial`                                                                                                     |
| `scipy.cluster`        | `cluster`                                                                                                     |
| `scipy.sparse.csgraph` | `graph` (feature-gated)                                                                                       |
| `scipy.linalg`         | `linalg`                                                                                                      |

## Module Catalog

- `signal`: convolution, STFT, spectrograms, spectral analysis, N-D filters, edge operations.
- `window`: signal window generators (Hann, Hamming, Blackman, Kaiser, and related windows).
- `interpolate`: 1D and N-D interpolation, splines, PCHIP/Akima, RBF, geometric interpolation and transforms.
- `optimize`: scalar optimization, roots, unconstrained/constrained minimization, least squares, LP/QP/conic/global methods.
- `integrate`: numerical quadrature and ODE solvers (including stiff/non-stiff families and sensitivity support).
- `stats`: continuous/discrete distributions, descriptive stats, hypothesis testing, regression, information metrics.
- `spatial`: KDTree/BallTree, distances, convex hull, Delaunay/Voronoi, rotations, mesh operations.
- `morphology`: binary/gray morphology and region measurements.
- `cluster`: k-means family, DBSCAN, OPTICS, HDBSCAN, GMM/Bayesian GMM, hierarchical and spectral clustering.
- `linalg`: matrix-equation-focused algorithms and advanced linear algebra helpers.
- `graph` (feature-gated): shortest paths, connectivity, centrality, MST, flow, graph matrix algorithms.
- `pde` (feature-gated): finite-difference, finite-element, and spectral PDE tooling.

The crate also re-exports commonly used `numr` types:
`Tensor`, `Runtime`, `RuntimeClient`, `DType`, `Result`, and `Error`.

## Installation

```toml
[dependencies]
solvr = "<latest>"
```

Enable GPU backends as needed:

```toml
[dependencies]
solvr = { version = "<latest>", features = ["cuda"] }
```

```toml
[dependencies]
solvr = { version = "<latest>", features = ["wgpu"] }
```

## Feature Flags

Default features: `["graph", "pde"]`

- `cuda`: enables CUDA backend support through `numr/cuda`.
- `wgpu`: enables WebGPU backend support through `numr/wgpu`.
- `graph`: enables graph algorithms module (also enables `sparse`).
- `pde`: enables PDE module (also enables `sparse`).
- `sparse`: sparse tensor support through `numr/sparse`.
- `f16`: half precision support through `numr/f16`.

## Quick Example

```rust
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use solvr::{ConvMode, ConvolutionAlgorithms, DType, Tensor, WindowFunctions};

fn main() -> solvr::Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let _window = client.hann_window(256, DType::F32, &device)?;

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.5], &[2], &device);
    let _y = client.convolve(&signal, &kernel, ConvMode::Same)?;

    Ok(())
}
```

## Versioning and MSRV

- Current line: pre-1.0 beta (API may evolve).
- MSRV: `1.85` (Rust 2024 edition baseline).

## Development

```bash
cargo fmt --all -- --check
cargo check --all-features
cargo test --lib
```

## Contributing

See `CONTRIBUTING.md` for local workflow and pull request guidance.

## License

Licensed under Apache-2.0. See `LICENSE`.
