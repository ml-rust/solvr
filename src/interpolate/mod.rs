//! Interpolation methods for solvr
//!
//! This module provides interpolation algorithms that work across all numr backends
//! (CPU, CUDA, WebGPU). All methods use numr's `RuntimeClient` trait for backend-agnostic
//! implementation.
//!
//! # Module Organization
//!
//! - [`Interp1d`] - 1D interpolation (linear, nearest, cubic)
//! - [`CubicSpline`] - Cubic spline interpolation with various boundary conditions
//! - [`PchipInterpolator`] - Monotonicity-preserving PCHIP interpolation
//! - [`Akima1DInterpolator`] - Outlier-robust Akima spline interpolation
//!
//! # Choosing an Interpolator
//!
//! | Interpolator         | Continuity | Monotonicity | Outlier Robust | Best For                  |
//! |----------------------|------------|--------------|----------------|---------------------------|
//! | `Interp1d::Linear`   | C0         | Preserved    | Yes            | Simple interpolation      |
//! | `Interp1d::Cubic`    | C1         | No           | No             | Smooth curves             |
//! | `CubicSpline`        | C2         | No           | No             | Very smooth curves        |
//! | `PchipInterpolator`  | C1         | Preserved    | Moderate       | Monotonic data            |
//! | `Akima1DInterpolator`| C1         | No           | Yes            | Data with outliers        |
//!
//! # Example
//!
//! ```ignore
//! use solvr::interpolate::{Interp1d, InterpMethod};
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Known data points
//! let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device);
//! let y = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device);
//!
//! // Create interpolator
//! let interp = Interp1d::new(&client, &x, &y, InterpMethod::Linear)?;
//!
//! // Evaluate at new points
//! let x_new = Tensor::from_slice(&[0.5, 1.5, 2.5], &[3], &device);
//! let y_new = interp.evaluate(&client, &x_new)?;
//! ```

mod akima;
mod cubic_spline;
mod error;
mod hermite_core;
mod interp1d;
mod pchip;

pub use akima::Akima1DInterpolator;
pub use cubic_spline::{CubicSpline, SplineBoundary};
pub use error::{InterpolateError, InterpolateResult};
pub use interp1d::{Interp1d, InterpMethod};
pub use pchip::PchipInterpolator;
