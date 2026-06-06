# Contributing

Thanks for contributing to [solvr](https://crates.io/crates/solvr). This guide
covers the architecture conventions and quality gates the project expects.

## Prerequisites

- A recent stable Rust toolchain.
- A clean working tree before opening a pull request.

## What to contribute

The most valuable contributions are usually **missing algorithms** — coverage
that SciPy/scikit-learn/scikit-image have but solvr does not yet, or new methods
within an existing module. Bug fixes, numerical-accuracy improvements, and
additional backend coverage are equally welcome.

Before writing a non-trivial algorithm, **open an issue first** describing what
you want to add, the method/reference, and which crate it belongs in (see
below). This avoids duplicated effort and lets us agree on placement and API up
front. Small, self-contained fixes can go straight to a pull request.

## Which crate: numr, solvr, or boostr

solvr is one layer of a stack, and a contribution only belongs here if it fits
this layer. Place new work by what it _is_, not where it's convenient:

- **[numr](https://github.com/ml-rust/numr)** — foundational primitives that
  everything else builds on: tensor ops, dtypes, the `Runtime`/backend
  abstraction (and **new backends** themselves), FFT, core linear algebra
  (matmul, LU/QR/SVD/eigen, `solve`), special functions, and basic descriptive
  statistics. If it's a building block reused across domains, or it adds/touches
  a hardware backend, it goes in numr.
- **solvr** (this crate) — complete _scientific/solving_ algorithms composed
  from numr primitives: optimization, ODE/DAE/BVP/PDE, interpolation, advanced
  statistics (distributions, tests, regression), signal processing, spatial,
  clustering, graphs, morphology, and matrix-equation solvers.
- **[boostr](https://github.com/ml-rust/boostr)** — AI/ML-specific building
  blocks: attention, positional encodings, mixture-of-experts, quantization,
  neural-network layers, and training/inference machinery.

Quick test:

- Is it a low-level primitive (a tensor op, an FFT, a linear-algebra
  factorization, a special function) or a new backend? → **numr**.
- Is it a domain solver a scientist/engineer would reach for? → **solvr**.
- Does it only make sense for neural networks / LLMs? → **boostr**.

When in doubt, propose it in an issue and we'll help place it. A primitive that
several higher layers would reuse should live in numr so the whole stack
benefits, rather than being duplicated in solvr.

## Architecture

solvr is backend-agnostic: the same algorithm runs on CPU, CUDA, and WebGPU
through [numr](https://github.com/ml-rust/numr)'s `Runtime` abstraction. Every
algorithm is written **once**, generically, and each backend is a thin
delegation. Please follow this structure when adding or changing algorithms.

### Runtime-generic algorithms

- Be generic over `R: Runtime`; operate on `Tensor<R>`, never on `&[f64]` /
  `Vec<f64>` parameters or struct fields.
- Build computation out of numr operations rather than scalar `for` loops — numr
  uses SIMD on CPU and kernels on GPU, so scalar loops are both slower and not
  portable across backends.
- Support multiple dtypes (`F32`/`F64`). Respect backend dtype limits (for
  example, the WebGPU backend is F32-only) and surface a clear error rather than
  silently degrading.
- If a primitive you need does not exist in numr, add it to numr instead of
  working around it with a host-side loop.

### The `impl_generic` pattern

Each module is laid out so the algorithm exists in exactly one place:

```
src/<module>/
├── mod.rs            # ONLY `pub mod` + `pub use`
├── traits/           # trait definitions + option/result types
├── impl_generic/     # the algorithm: `fn <algo>_impl<R, C>(...)`
├── cpu/              # `impl Trait for CpuClient` — delegates to *_impl
├── cuda/             # `impl Trait for CudaClient` — delegates to *_impl
└── wgpu/             # `impl Trait for WgpuClient` — delegates to *_impl
```

- One algorithm = one file, with the **same file name** under `traits/`,
  `impl_generic/`, `cpu/`, `cuda/`, and `wgpu/`.
- `mod.rs` contains only `pub mod` / `pub use` — no logic, traits, or types.
- Backend files (`cpu/`, `cuda/`, `wgpu/`) are thin: they implement the trait by
  calling the generic `*_impl` function and nothing else.
- Adding an algorithm means adding new files, not expanding existing ones.

### No GPU↔CPU transfers in hot paths

Host/device transfers cost far more than the computation itself. Inside
algorithms, do **not** call `tensor.to_vec()` or `Tensor::from_slice(...)`.
The only acceptable transfers are:

- at the public API boundary (user-provided input / returned output), and
- a single scalar pulled to the host for a convergence/control-flow check.

Keep state in `Tensor<R>`, and keep loops on-device using numr ops.

## Building with backends

```bash
cargo build                      # CPU (default)
cargo build --features cuda      # CUDA (requires a CUDA 12.x toolchain)
cargo build --features wgpu      # WebGPU
cargo build --features sparse    # sparse-tensor-backed modules
```

The `graph` and `pde` modules require `sparse` (enabled by default).

## Testing

- Put unit tests in the same file as the code under test
  (`#[cfg(test)] mod tests`).
- Test numerical correctness against an analytic or reference result, not just
  that the call returns `Ok`.
- A backend-specific test should skip gracefully when no device is available
  rather than fail.
- Run the suite on each backend you can; CUDA/WebGPU paths exercise the same
  generic code but catch backend-specific issues.

```bash
cargo test --release                       # CPU
cargo test --release --features cuda,sparse
cargo test --release --features wgpu
```

## Local Quality Checks

Run these before submitting. Clippy is run with `-D warnings` to match CI, so a
warning is a failure — treat it as one locally too.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features f16,sparse,graph,pde -- -D warnings
cargo test --release
```

If you touch GPU backends, also run clippy with `--features cuda` and
`--features wgpu`.

## Pull Request Guidelines

- Keep PRs focused and scoped.
- Preserve the module structure and `impl_generic` pattern described above.
- Include tests for behavioral changes; verify numerical parity across backends
  where applicable.
- Update docs when public APIs or features change.
- Avoid `.unwrap()` in library code — return a typed error with context.

## Commit Messages

Use Conventional Commits with a clear, imperative summary, for example:

```
feat(integrate): add automatic Jacobian sparsity detection for BDF and Radau
fix(spatial): correct single-vector rotation to apply R instead of R^T
```
