<div align="center">

# solvr

<h3>Scientific computing library in Rust. Write once. Run on any backend.</h3>

<p>
SciPy/scikit-learn/scikit-image-class breadth — optimization, ODE/DAE/PDE solvers,
interpolation, statistics, signal processing, and spatial algorithms — built on
<a href="https://github.com/ml-rust/numr">numr</a> so the same code runs on CPU, CUDA, and WebGPU.
</p>

<p>
  <a href="https://docs.rs/solvr"><strong>Docs</strong></a>
  · 
  <a href="https://crates.io/crates/solvr"><strong>Crate</strong></a>
  ·
  <a href="#module-catalog"><strong>Modules</strong></a>
  ·
  <a href="#quick-example"><strong>Example</strong></a>
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

[solvr](https://crates.io/crates/solvr) is a scientific computing crate for real-world numerical workflows in Rust.
It provides high-level algorithms on top of [numr](https://github.com/ml-rust/numr) tensor/runtime primitives, with CPU support by default and optional CUDA/WebGPU backends.

## Why solvr

- **One codebase, every backend.** Write once against numr's `Runtime`; run on CPU (SIMD), CUDA, or WebGPU by switching a feature flag — no per-device dispatch, no rewrite.
- **Deploy anywhere without rewriting.** The same numerical code goes from a laptop to a CUDA server to a WebGPU/edge target. The backend is chosen at build/runtime, never baked into your algorithm.
- **Backends are a foundation concern.** Hardware support lives in numr, so new backends added there flow up to solvr automatically — the abstraction is built in, not bolted on per device.
- **The heavy lifting, handled.** Production-grade solvers for optimization, ODE/DAE/PDE, statistics, signal, and spatial problems — so you build your application instead of the numerical plumbing.

## Who it's for

- **ML engineers & data scientists** — clustering (DBSCAN, OPTICS, HDBSCAN, GMM, spectral), global/constrained optimization, GPU-accelerated without a Python runtime.
- **Scientific computing & quantitative researchers** — ODE/DAE/BVP solvers (Radau, LSODA, BDF), a native PDE suite, and matrix-equation solvers (Lyapunov, Riccati, Sylvester).
- **Signal, audio & imaging engineers** — convolution, STFT, spectrograms, Lomb-Scargle, wavelets, FIR/IIR design, and mathematical morphology.
- **Graphics, GIS & geometry developers** — KD-trees/Ball-trees, convex hull, Delaunay/Voronoi, rotations, and distance transforms.
- **WebAssembly & edge developers** — the WebGPU backend targets consumer GPUs (Vulkan/Metal/DX12) and the browser, with no CUDA dependency.

## solvr vs SciPy / CuPy

solvr is not a drop-in replacement for SciPy's breadth or maturity. It's the better fit when **Rust-native integration, single-binary deployment, and vendor-agnostic GPU** matter — and it covers ground that core SciPy and CuPy leave to scattered third-party tools.

|                                 | Python (SciPy / CuPy)                       | solvr                                     |
| ------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Language / runtime              | Python + C/C++/Cython; needs an interpreter | Rust; compiles to a single static binary  |
| GPU backends                    | CUDA / ROCm only (CuPy); SciPy is CPU       | CPU, CUDA, **and** WebGPU through one API |
| In-browser / WASM               | Limited                                     | WebGPU-ready (no CUDA dependency)         |
| PDE suite (FD / FEM / spectral) | Not in core (external: FEniCS, FiPy, …)     | Built-in                                  |
| DAE solvers                     | Not in core                                 | Built-in                                  |
| ODE adjoint sensitivity         | External (JAX/diffrax, torchdiffeq)         | Built-in                                  |
| Conic optimization (SDP / SOCP) | Not in core (external: CVXPY, …)            | Built-in                                  |

**When SciPy or CuPy is the better choice:** interactive exploration in notebooks, the deepest possible algorithm coverage and edge-case maturity, or tight interop with the Python ML ecosystem (PyTorch, JAX, Hugging Face, DLPack).

## The stack

solvr is one layer of a Rust numerical stack:

- [numr](https://github.com/ml-rust/numr) — foundational tensors, FFT, linear algebra, and the multi-backend `Runtime` (plus HPC/distributed support).
- **solvr** — scientific and numerical algorithms built on numr.
- [boostr](https://github.com/ml-rust/boostr) — AI/ML primitives built on numr.

Anything built on numr inherits its backends as they are added.

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
- `integrate`: numerical quadrature and ODE/DAE/BVP solvers (including stiff/non-stiff families and sensitivity support).
- `stats`: continuous/discrete distributions, descriptive stats, hypothesis testing, regression, information metrics.
- `spatial`: KDTree/BallTree, distances, convex hull, Delaunay/Voronoi, rotations, mesh operations.
- `morphology`: binary/gray morphology and region measurements.
- `cluster`: k-means family, DBSCAN, OPTICS, HDBSCAN, GMM/Bayesian GMM, hierarchical and spectral clustering.
- `linalg`: matrix-equation solvers (Lyapunov/Riccati/Sylvester), sparse QR, and advanced linear algebra helpers.
- `graph` (feature-gated): shortest paths, connectivity, centrality, MST, flow, graph matrix algorithms.
- `pde` (feature-gated): finite-difference, finite-element, and spectral PDE tooling.

The crate also re-exports commonly used [numr](https://github.com/ml-rust/numr) types:
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

Switching the example to a GPU backend means swapping the client/runtime — the algorithm code stays the same.

## Development

```bash
cargo fmt --all -- --check
cargo check --all-features
cargo test --lib
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local workflow and pull request guidance.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
