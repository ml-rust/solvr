# Contributing

Thanks for contributing to [solvr](https://crates.io/crates/solvr). This guide covers the architecture conventions and quality gates.

## Before you start

- A recent stable Rust toolchain.
- A clean working tree before you open a pull request.

Useful contributions:

- **Missing algorithms**: methods that SciPy, scikit-learn, or scikit-image have and solvr lacks, or new methods in an existing module.
- Bug fixes, numerical-accuracy improvements, and additional backend coverage.

Open an issue before you write a non-trivial algorithm. Describe the method, its reference, and the crate it belongs in. Small, self-contained fixes can go straight to a pull request.

### Which crate

Place work by what it is:

| Crate | Scope |
| --- | --- |
| [numr](https://github.com/ml-rust/numr) | Foundational primitives: tensor ops, dtypes, the `Runtime`/backend abstraction and new backends, FFT, core linear algebra (matmul, LU/QR/SVD/eigen, `solve`), special functions, basic descriptive statistics. |
| **solvr** (this crate) | Scientific and solving algorithms built from numr primitives: optimization, ODE/DAE/BVP/PDE, interpolation, advanced statistics (distributions, tests, regression), signal processing, spatial, clustering, graphs, morphology, matrix-equation solvers. |
| [boostr](https://github.com/ml-rust/boostr) | AI/ML building blocks: attention, positional encodings, mixture-of-experts, quantization, neural-network layers, training and inference machinery. |

Quick test:

- A low-level primitive (tensor op, FFT, linear-algebra factorization, special function) or a new backend: **numr**.
- A domain solver a scientist or engineer reaches for: **solvr**.
- Makes sense only for neural networks or LLMs: **boostr**.

When in doubt, propose the work in an issue. A primitive that several higher layers reuse belongs in numr. Do not duplicate it in solvr.

## Architecture

solvr is backend-agnostic. The same algorithm runs on CPU, CUDA, and WebGPU through numr's `Runtime` abstraction. Write each algorithm once, generically. Each backend is a thin delegation.

### Runtime-generic algorithms

- Be generic over `R: Runtime`. Operate on `Tensor<R>`, never on `&[f64]`, `Vec<f64>`, or struct fields of those types.
- Build computation from numr operations, not scalar `for` loops. numr uses SIMD on CPU and kernels on GPU. Scalar loops are slower and not portable.
- Support `F32` and `F64`. Respect backend dtype limits, such as WebGPU being F32-only. Return a clear error instead of degrading silently.
- If numr lacks a primitive you need, add it to numr. Do not work around it with a host-side loop.

### The `impl_generic` pattern

Each module places the algorithm in exactly one location:

```
src/<module>/
├── mod.rs            # ONLY `pub mod` + `pub use`
├── traits/           # trait definitions + option/result types
├── impl_generic/     # the algorithm: `fn <algo>_impl<R, C>(...)`
├── cpu/              # `impl Trait for CpuClient` — delegates to *_impl
├── cuda/             # `impl Trait for CudaClient` — delegates to *_impl
└── wgpu/             # `impl Trait for WgpuClient` — delegates to *_impl
```

- One algorithm is one file. Use the **same file name** under `traits/`, `impl_generic/`, `cpu/`, `cuda/`, and `wgpu/`.
- `mod.rs` contains only `pub mod` and `pub use`. It holds no logic, traits, or types.
- Backend files are thin. Each implements its trait by calling the generic `*_impl` function and nothing else.
- Add files for a new algorithm. Do not expand existing files.

### No GPU↔CPU transfers in hot paths

Host/device transfers cost far more than the computation. Inside algorithms, never call `tensor.to_vec()` or `Tensor::from_slice(...)`. Two transfers are allowed:

- At the public API boundary, for user input and returned output.
- A single scalar pulled to the host for a convergence or control-flow check.

Keep state in `Tensor<R>`. Keep loops on the device with numr ops.

## Build

```bash
cargo build                      # CPU (default)
cargo build --features cuda      # CUDA (requires a CUDA 12.8+ or 13.x toolchain)
cargo build --features wgpu      # WebGPU
cargo build --features sparse    # sparse-tensor-backed modules
```

The `graph` and `pde` features enable `sparse`. Both features are default features.

## Testing

- Put unit tests in the same file as the code, in `#[cfg(test)] mod tests`.
- Test numerical correctness against an analytic or reference result, not only that the call returns `Ok`.
- A backend-specific test skips gracefully when no device is available.
- Run the suite on each backend you can. CUDA and WebGPU paths run the same generic code and catch backend-specific bugs.

```bash
cargo test --release                       # CPU
cargo test --release --features cuda,sparse
cargo test --release --features wgpu
```

## Quality checks

Run these before you submit. CI runs clippy with `-D warnings`, so treat a warning as an error.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features f16,sparse,graph,pde -- -D warnings
cargo test --release
```

If you touch a GPU backend, also run clippy with `--features cuda` and `--features wgpu`.

## Pull requests

- Keep each pull request focused and scoped.
- Keep the module structure and `impl_generic` pattern.
- Include tests for behavioral changes. Check numerical parity across backends where it applies.
- Update docs when public APIs or features change.
- Avoid `.unwrap()` in library code. Return a typed error with context.

## Commit messages

Use Conventional Commits with a clear, imperative summary:

```
feat(integrate): add automatic Jacobian sparsity detection for BDF and Radau
fix(spatial): correct single-vector rotation to apply R instead of R^T
```
