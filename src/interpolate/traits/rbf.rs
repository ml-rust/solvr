//! Radial Basis Function interpolation algorithm trait.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// RBF kernel function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// r^2 * ln(r), conditionally positive definite of order 2.
    ThinPlateSpline,
    /// sqrt(1 + (r/epsilon)^2)
    Multiquadric,
    /// 1/sqrt(1 + (r/epsilon)^2)
    InverseMultiquadric,
    /// exp(-(r/epsilon)^2)
    Gaussian,
    /// r
    Linear,
    /// r^3
    Cubic,
    /// r^5
    Quintic,
}

/// A fitted RBF interpolation model.
#[derive(Debug, Clone)]
pub struct RbfModel<R: Runtime> {
    /// Center points, shape [n, d].
    pub centers: Tensor<R>,
    /// Weights for each center, shape [n] or [n, m] for multi-output.
    pub weights: Tensor<R>,
    /// Kernel function.
    pub kernel: RbfKernel,
    /// Shape parameter (used by Gaussian, Multiquadric, InverseMultiquadric).
    pub epsilon: f64,
    /// Polynomial augmentation coefficients, if any.
    pub poly_coeffs: Option<Tensor<R>>,
    /// Dimension of input points.
    pub dim: usize,
}

/// Radial Basis Function interpolation algorithms.
pub trait RbfAlgorithms<R: Runtime> {
    /// Fit an RBF interpolant to scattered data.
    ///
    /// # Arguments
    /// * `points` - Data point coordinates, shape [n, d]
    /// * `values` - Values at data points, shape [n] or [n, m]
    /// * `kernel` - RBF kernel function
    /// * `epsilon` - Shape parameter (None = auto-select)
    /// * `smoothing` - Smoothing parameter (0 = exact interpolation)
    fn rbf_fit(
        &self,
        points: &Tensor<R>,
        values: &Tensor<R>,
        kernel: RbfKernel,
        epsilon: Option<f64>,
        smoothing: f64,
    ) -> InterpolateResult<RbfModel<R>>;

    /// Evaluate an RBF interpolant at query points.
    ///
    /// # Arguments
    /// * `model` - Fitted RBF model
    /// * `query` - Query point coordinates, shape [m, d]
    fn rbf_evaluate(&self, model: &RbfModel<R>, query: &Tensor<R>) -> InterpolateResult<Tensor<R>>;
}
