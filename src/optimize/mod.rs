//! Optimization algorithms for solvr.
//!
//! This module provides root finding, minimization, and other optimization methods.
//!
//! # Modules
//!
//! - [`scalar`] - Univariate (1D) root finding and minimization
//! - [`roots`] - Multivariate root finding (systems of nonlinear equations)
//! - [`minimize`] - Multivariate unconstrained minimization
//! - [`least_squares`] - Nonlinear least squares and curve fitting
//! - [`global`] - Global optimization (escaping local minima)
//! - [`linprog`] - Linear programming (Simplex, MILP)
//!
//! # Quick Start
//!
//! ## Scalar Root Finding
//!
//! ```ignore
//! use solvr::optimize::scalar::{bisect, ScalarOptions};
//!
//! // Find root of f(x) = x^2 - 4 in [1, 3]
//! let result = bisect(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default())?;
//! assert!((result.root - 2.0).abs() < 1e-10);
//! ```
//!
//! ## Multivariate Root Finding
//!
//! ```ignore
//! use solvr::optimize::roots::{newton_system, RootOptions};
//!
//! // Solve system: x + y = 3, 2x - y = 0
//! let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];
//! let result = newton_system(f, &[0.0, 0.0], &RootOptions::default())?;
//! // Solution: x = 1, y = 2
//! ```
//!
//! ## Multivariate Minimization
//!
//! ```ignore
//! use solvr::optimize::minimize::{bfgs, MinimizeOptions};
//!
//! // Minimize Rosenbrock function
//! let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
//! let result = bfgs(f, &[0.0, 0.0], &MinimizeOptions::default())?;
//! // Minimum at (1, 1)
//! ```
//!
//! ## Curve Fitting
//!
//! ```ignore
//! use solvr::optimize::least_squares::{curve_fit, LeastSquaresOptions};
//!
//! // Fit y = a * exp(-b * x)
//! let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();
//! let result = curve_fit(model, &x_data, &y_data, &[1.0, 1.0], &LeastSquaresOptions::default())?;
//! ```
//!
//! ## Global Optimization
//!
//! ```ignore
//! use solvr::optimize::global::{differential_evolution, GlobalOptions};
//!
//! // Find global minimum of Rastrigin function (many local minima)
//! let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
//! let result = differential_evolution(rastrigin, &bounds, &GlobalOptions::default())?;
//! ```
//!
//! ## Linear Programming
//!
//! ```ignore
//! use solvr::optimize::linprog::{linprog, LinearConstraints, LinProgOptions};
//!
//! // Minimize -x - 2y subject to: x + y <= 4, x,y >= 0
//! let c = vec![-1.0, -2.0];
//! let constraints = LinearConstraints {
//!     a_ub: Some(vec![vec![1.0, 1.0]]),
//!     b_ub: Some(vec![4.0]),
//!     bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
//!     ..Default::default()
//! };
//! let result = linprog(&c, &constraints, &LinProgOptions::default())?;
//! ```

pub mod error;
pub mod global;
pub mod least_squares;
pub mod linprog;
pub mod minimize;
pub mod roots;
pub mod scalar;
pub(crate) mod utils;

pub use error::{OptimizeError, OptimizeResult};
pub use global::{
    GlobalOptions, GlobalResult, basinhopping, differential_evolution, dual_annealing,
    simulated_annealing,
};
pub use least_squares::{
    LeastSquaresOptions, LeastSquaresResult, curve_fit, least_squares, leastsq,
};
pub use linprog::{
    LinProgOptions, LinProgResult, LinearConstraints, MilpOptions, MilpResult, linprog, milp,
};
pub use minimize::{
    MinimizeOptions, MultiMinimizeResult, bfgs, conjugate_gradient, nelder_mead, powell,
};
pub use roots::{MultiRootResult, RootOptions, broyden1, levenberg_marquardt, newton_system};
pub use scalar::{
    MinimizeResult, RootResult, ScalarOptions, bisect, brentq, minimize_scalar_bounded,
    minimize_scalar_brent, minimize_scalar_golden, newton, ridder, secant,
};
