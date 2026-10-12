<div align="center">

# solvr

<h3>Scientific computing in Rust. Write once. Run on any backend.</h3>

<p>
Optimization, ODE, DAE, and PDE solvers, interpolation, statistics, signal processing, and spatial
algorithms, built on numr. The crate exposes 13 public modules.
</p>

<p>
  <a href="https://docs.rs/solvr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/solvr"><strong>Crate</strong></a>
  ·
  <a href="#module-catalog"><strong>Modules</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/solvr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/solvr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/solvr">
    <img src="https://img.shields.io/crates/v/solvr" alt="crates.io">
  </a>
  <a href="https://docs.rs/solvr">
    <img src="https://img.shields.io/docsrs/solvr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/solvr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/solvr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/solvr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

solvr is a scientific computing crate for numerical workflows in Rust. It provides algorithms on
[numr](https://github.com/ml-rust/numr) tensors and runtimes. The CPU backend is the default.
CUDA and WebGPU are optional features.

## Installation

Add `solvr` and `numr`. The examples create numr devices and clients.

```bash
cargo add solvr numr
```

For GPU backends, enable the matching feature:

```bash
cargo add solvr --features cuda   # NVIDIA GPUs, CUDA 12.8+ or 13.x
cargo add solvr --features wgpu   # WebGPU
```

## Feature Flags

Default features: `graph`, `pde`.

| Feature | Description | Default |
| ------- | ----------- | ------- |
| `graph` | Graph algorithms module. Enables `sparse`. | ✓ |
| `pde` | PDE module. Enables `sparse`. | ✓ |
| `sparse` | Sparse tensor support through `numr/sparse` | ✗ |
| `cuda` | CUDA backend through `numr/cuda` | ✗ |
| `wgpu` | WebGPU backend through `numr/wgpu` | ✗ |
| `f16` | Half-precision support through `numr/f16` | ✗ |

WebGPU computes in `F32` only. The CUDA backend needs the CUDA toolchain that numr requires.

## Quick Start

```rust
use numr::dtype::DType;
use numr::ops::RandomOps;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice};
use solvr::signal::{ConvMode, ConvolutionAlgorithms};
use solvr::window::WindowFunctions;

fn main() -> solvr::Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    // Create a Hann window
    let _window = client.hann_window(256, DType::F32, &device)?;

    // Convolve two random signals
    let signal = client.randn(&[256], DType::F32)?;
    let kernel = client.randn(&[32], DType::F32)?;
    let _result = client.convolve(&signal, &kernel, ConvMode::Same)?;

    Ok(())
}
```

The algorithm calls stay the same on any backend that implements the operation. Only the device and
client types change.

## Module Catalog

| Module | Contents |
| ------ | -------- |
| `signal` | Convolution, correlation, STFT and inverse STFT, spectrograms, spectral analysis, N-D filters, edge operations |
| `window` | Window generators: Hann, Hamming, Blackman, Kaiser, and related windows |
| `interpolate` | 1D and N-D interpolation, splines, PCHIP, Akima, RBF, geometric transforms |
| `optimize` | Scalar optimization, root finding, minimization, least squares, LP, QP, conic, global methods |
| `integrate` | Numerical quadrature and ODE, DAE, and BVP solvers |
| `stats` | Distributions, descriptive statistics, hypothesis tests, regression, information metrics |
| `spatial` | KDTree, BallTree, distances, convex hull, Delaunay, Voronoi, rotations, mesh operations, distance transforms |
| `morphology` | Binary and grayscale morphology, connected-region labeling, region measurements |
| `cluster` | k-means family, DBSCAN, OPTICS, HDBSCAN, GMM and Bayesian GMM, hierarchical, spectral |
| `linalg` | Matrix-equation solvers (Lyapunov, Riccati, Sylvester), sparse QR, advanced linear algebra helpers |
| `graph` (feature `graph`) | Shortest paths, connectivity, centrality, MST, flow, graph matrix algorithms |
| `pde` (feature `pde`) | Finite-difference, finite-element, and spectral PDE tooling |

The full algorithm list for each module is in [docs/modules.md](docs/modules.md).

The crate re-exports `Tensor`, `Runtime`, `RuntimeClient`, `DType`, `Result`, and `Error` from numr.

## For SciPy Users

| SciPy area | `solvr` module(s) |
| ---------- | ----------------- |
| `scipy.signal` | `signal`, `window` |
| `scipy.ndimage` | `signal` (N-D filters, edge operations), `morphology`, `spatial` (distance transform), `interpolate` (geometric transforms) |
| `scipy.interpolate` | `interpolate` |
| `scipy.optimize` | `optimize` |
| `scipy.integrate` | `integrate` |
| `scipy.stats` | `stats` |
| `scipy.spatial` | `spatial` |
| `scipy.cluster` | `cluster` |
| `scipy.sparse.csgraph` | `graph` (feature-gated) |
| `scipy.linalg` | `linalg` |

## The Stack

- [numr](https://github.com/ml-rust/numr): tensors, FFT, linear algebra, and the `Runtime` abstraction.
- **solvr**: scientific and numerical algorithms built on numr.
- [boostr](https://github.com/ml-rust/boostr): ML primitives built on numr.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
