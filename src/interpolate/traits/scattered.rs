//! Scattered data interpolation algorithm trait.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Method for scattered data interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatteredMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Linear interpolation based on Delaunay triangulation (2D only).
    Linear,
}

/// Scattered data interpolation algorithms.
pub trait ScatteredInterpAlgorithms<R: Runtime<DType = DType>> {
    /// Interpolate scattered data at query points.
    ///
    /// # Arguments
    /// * `points` - Known data point coordinates, shape `[n, d]`
    /// * `values` - Known values at data points, shape `[n]`
    /// * `xi` - Query point coordinates, shape `[m, d]`
    /// * `method` - Interpolation method
    fn griddata(
        &self,
        points: &Tensor<R>,
        values: &Tensor<R>,
        xi: &Tensor<R>,
        method: ScatteredMethod,
    ) -> InterpolateResult<Tensor<R>>;
}
