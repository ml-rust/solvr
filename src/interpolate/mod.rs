//! Interpolation methods for solvr
//!
//! This module provides trait-based interpolation algorithms that work across all numr backends
//! (CPU, CUDA, WebGPU). All methods use numr's `RuntimeClient` trait for backend-agnostic
//! implementation.
//!
//! # Module Organization
//!
//! - `traits/` - Algorithm trait definitions (one file per algorithm)
//! - `impl_generic/` - Generic algorithm implementations (one file per algorithm)
//! - `cpu/`, `cuda/`, `wgpu/` - Backend trait implementations (one file per algorithm)
//! - `error.rs` - Error types
//! - `hermite_core.rs` - Shared Hermite interpolation utilities
//!
//! # Algorithms
//!
//! ## 1D Interpolation
//!
//! - `Interp1dAlgorithms` - 1D interpolation (linear, nearest, cubic)
//! - `CubicSplineAlgorithms` - Cubic spline interpolation with boundary conditions
//! - `PchipAlgorithms` - Monotonicity-preserving PCHIP interpolation
//! - `AkimaAlgorithms` - Outlier-robust Akima spline interpolation
//!
//! ## N-D Interpolation
//!
//! - `InterpNdAlgorithms` - N-dimensional interpolation on rectilinear grids
//!
//! # Using the Traits
//!
//! Import the trait and call its methods on your client:
//!
//! ```ignore
//! use solvr::interpolate::traits::{Interp1dAlgorithms, InterpMethod};
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
//! let y = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);
//! let x_new = Tensor::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
//!
//! let y_new = client.interp1d(&x, &y, &x_new, InterpMethod::Linear)?;
//! ```

mod error;
mod hermite_core;

// Backend modules
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
mod wgpu;

// Public modules for internal access
pub mod impl_generic;
pub mod traits;

// Public API: traits and types
pub use error::{InterpolateError, InterpolateResult};
pub use traits::bezier_curve::{BezierCurve, BezierCurveAlgorithms};
pub use traits::bezier_surface::{BezierSurface, BezierSurfaceAlgorithms};
pub use traits::bspline::{BSpline, BSplineBoundary};
pub use traits::bspline_curve::{BSplineCurve, BSplineCurveAlgorithms};
pub use traits::bspline_surface::{BSplineSurface, BSplineSurfaceAlgorithms};
pub use traits::clough_tocher::CloughTocher2D;
pub use traits::cubic_spline::SplineBoundary;
pub use traits::geometric::{GeometricTransformAlgorithms, InterpolationOrder};
pub use traits::interp1d::InterpMethod;
pub use traits::interpnd::{ExtrapolateMode, InterpNdMethod};
pub use traits::nurbs_curve::{NurbsCurve, NurbsCurveAlgorithms};
pub use traits::nurbs_surface::{NurbsSurface, NurbsSurfaceAlgorithms};
pub use traits::rbf::{RbfKernel, RbfModel};
pub use traits::rect_bivariate_spline::BivariateSpline;
pub use traits::scattered::ScatteredMethod;
pub use traits::{
    AkimaAlgorithms, BSplineAlgorithms, CloughTocher2DAlgorithms, CubicSplineAlgorithms,
    Interp1dAlgorithms, InterpNdAlgorithms, PchipAlgorithms, RbfAlgorithms,
    RectBivariateSplineAlgorithms, ScatteredInterpAlgorithms, SmoothBivariateSplineAlgorithms,
};
