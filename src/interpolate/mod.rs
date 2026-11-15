//! Tensor interpolation across Numr CPU, CUDA, and WebGPU backends.
//!
//! # Using the Traits
//!
//! Import the trait and call its methods on your client:
//!
//! ```
//! # use numr::runtime::cpu::{CpuClient, CpuDevice};
//! # use numr::tensor::Tensor;
//! use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};
//! # let device = CpuDevice::new();
//! # let client = CpuClient::new(device.clone());
//! # let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4], &device)?;
//! # let y = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], &[4], &device)?;
//! # let x_new = Tensor::from_slice(&[0.5, 1.5, 2.5], &[3], &device)?;
//! let y_new = client.interp1d(&x, &y, &x_new, InterpMethod::Linear)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
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

pub use error::{InterpolateError, InterpolateResult};
pub use traits::{
    AkimaAlgorithms, BSpline, BSplineAlgorithms, BSplineBoundary, BSplineCurve,
    BSplineCurveAlgorithms, BSplineSurface, BSplineSurfaceAlgorithms, BezierCurve,
    BezierCurveAlgorithms, BezierSurface, BezierSurfaceAlgorithms, BivariateSpline, CloughTocher2D,
    CloughTocher2DAlgorithms, CubicSplineAlgorithms, ExtrapolateMode, GeometricTransformAlgorithms,
    Interp1dAlgorithms, InterpMethod, InterpNdAlgorithms, InterpNdMethod, InterpolationOrder,
    LinearInterp1dDiagnostics, LinearInterp1dReceipt, NurbsCurve, NurbsCurveAlgorithms,
    NurbsSurface, NurbsSurfaceAlgorithms, PchipAlgorithms, RbfAlgorithms, RbfKernel, RbfModel,
    RectBivariateSplineAlgorithms, ScatteredInterpAlgorithms, ScatteredMethod,
    SmoothBivariateSplineAlgorithms, SplineBoundary,
};
