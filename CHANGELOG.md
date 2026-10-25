# Changelog

All notable changes to solvr are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
solvr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tags `v0.1.0` through `v0.2.0` predate this file. Every entry is a delta against the one below it.

---

## [Unreleased]

### Breaking

- **numr 0.6 to 0.13** — solvr depends on `numr = "0.13"`. solvr re-exports `Tensor`, `Runtime`, `RuntimeClient`, `DType`, `Error` and `Result` from numr, so these types change with it. A caller that names them must depend on the same numr version. Every solvr API that takes or returns a `Tensor<R>` carries the numr 0.13 type.
- **`Tensor::from_slice` returns `Result`** — numr's constructor is fallible. Code that builds tensors for solvr calls must handle the `Result`.
- **MSRV 1.95** — `rust-version` is 1.95, the minimum numr requires. It was 1.88.
- **`QpResult::dual_ineq` from the active-set solver** — a converged solve returns `Some` multipliers `[m_ineq]` when the problem has inequality constraints. It returned `None` before.

### Fixed

- **Interior-point QP with equality and inequality constraints** — the Newton step solves the full KKT system with the equality rows, and the dual residual carries the `A_eq'*y` term. `solve_qp` with both constraint kinds returns a point with `A_eq*x = b_eq` and reports `dual_eq`. Before, the solver ignored the equality constraints whenever inequalities were present and returned `dual_eq: None`.
- **Interior-point QP Newton step** — the slack and inequality-multiplier steps follow the linearized conditions `A*dx - ds = -r_primal` and `Z*ds + S*dz = -r_comp`. Before, both carried a sign error, so the iterates did not follow the central path.
- **Interior-point QP convergence test** — a solve converges when the complementarity gap and the dual, equality and inequality residuals, each relative to `1 + ||c||` or `1 + ||b||`, fall below `tol`. Before, the gap alone decided, so an infeasible point could report `converged: true`.
- **Active-set QP drops the wrong constraint** — the argmin of the inequality multipliers is read as `i64`, the dtype numr returns. Before, solvr read it as `f64`, so the index was always 0. The solver dropped the first active row instead of the one with the most negative multiplier. Such a solve can cycle and never converge.
- **Active-set QP multiplier sign** — the inequality multipliers are negated before the optimality test, because the KKT rows `Q d + A' lambda = -g` define `lambda` with the opposite sign. Before, the test used the opposite sign: it dropped rows with a positive multiplier and kept rows with a negative one.
- **Active-set QP with dependent or underdetermined working constraints** — the KKT system is solved by least squares, so dependent rows give the exact minimum-norm solution. Before, a singular KKT matrix returned an error. A working set with no inequality row skips the drop step.

### Added

- **`QpResult` sign convention** — `dual_eq` holds `y` and `dual_ineq` holds `z` of the Lagrangian `L = f(x) + y'(A_eq*x - b_eq) - z'(A_ineq*x - b_ineq)`. At the optimum `Q*x + c + A_eq'*y - A_ineq'*z = 0` and `z >= 0`. Both solvers follow it.
- **Tests** — active-set QP with fewer equalities than variables, dependent equalities, one active inequality, and two active inequalities where only the second multiplier is negative. A test pins the `dual_eq` sign on both solvers. An equality-plus-inequality QP with a hand-computed optimum and multipliers runs on both solvers and compares them. Interior-point tests cover an equality with inequalities and an inequality alone. Smooth bivariate spline tests cover collinear points and fewer distinct points than coefficients. Anderson acceleration tests cover an identical residual history and a history shorter than the unknown count.
- **`docs/modules.md`** — a reference page that lists the algorithms in each module.

### Changed

- **README and CONTRIBUTING** — a shorter README with corrected quick-start imports. CONTRIBUTING shows crate placement as a table and lists the current CUDA toolchains and default features.
- **Features** — the commented-out `rocm` feature line is removed from `Cargo.toml`.
