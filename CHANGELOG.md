# Changelog

All notable changes to solvr are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
solvr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tags `v0.1.0` through `v0.2.0` predate this file. Every entry is a delta against the one below it.

---

## [Unreleased]

## [0.3.1] - 2026-11-06

### Changed

- `README`, crate description: state what the crate does.

## [0.3.0] — 2026-11-01

solvr moves to numr 0.14 and MSRV 1.95. Both QP solvers return correct solutions and multipliers.

### Breaking

- **numr 0.14** — solvr depends on `numr = "0.14"`, up from 0.6. The re-exported `Tensor`, `Runtime`, `RuntimeClient`, `DType`, `Error` and `Result` change with it. Callers that name them must use the same numr version.
- **`Tensor::from_slice` returns `Result`** — numr's constructor is fallible. Code that builds tensors for solvr must handle the `Result`.
- **MSRV 1.95** — `rust-version` was 1.88.
- **`QpResult::dual_ineq` from the active-set solver** — a converged solve with inequality constraints returns `Some` multipliers `[m_ineq]`. Before, it returned `None`.

### Added

- **`QpResult` sign convention** — `dual_eq` holds `y` and `dual_ineq` holds `z` of `L = f(x) + y'(A_eq*x - b_eq) - z'(A_ineq*x - b_ineq)`. At the optimum `Q*x + c + A_eq'*y - A_ineq'*z = 0` and `z >= 0`. Both solvers follow it.

### Fixed

- **Interior-point QP with equality and inequality constraints** — `solve_qp` returns a point with `A_eq*x = b_eq` and reports `dual_eq`. Before, the solver ignored the equality constraints whenever inequalities were present and returned `dual_eq: None`.
- **Interior-point QP convergence** — a solve converges only when the complementarity gap and the dual, equality and inequality residuals fall below `tol`. Before, the gap alone decided, so an infeasible point reported `converged: true`. Before, a sign error in the slack and multiplier steps also kept the iterates off the central path.
- **Active-set QP constraint drop** — the solver drops the active row with the most negative inequality multiplier. Before, it dropped the first active row and tested the multiplier sign backwards. Such a solve cycled and never converged.
- **Active-set QP with dependent working constraints** — dependent rows give the minimum-norm solution. Before, a singular KKT matrix returned an error.
