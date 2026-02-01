//! N-dimensional grid interpolation algorithm trait.

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Interpolation method for N-dimensional grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpNdMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Multilinear interpolation (default).
    #[default]
    Linear,
}

/// Behavior when query points are outside the grid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtrapolateMode {
    /// Return an error for out-of-bounds queries.
    #[default]
    Error,
    /// Return NaN for out-of-bounds queries.
    Nan,
    /// Extrapolate beyond grid bounds (use boundary values for nearest).
    Extrapolate,
}

/// N-dimensional grid interpolation algorithm.
///
/// Supports interpolation on rectilinear (orthogonal) N-dimensional grids
/// with various extrapolation modes.
pub trait InterpNdAlgorithms<R: Runtime> {
    /// Evaluate N-dimensional interpolation at query points.
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of 1D tensors (coordinate arrays for each dimension)
    /// * `values` - N-dimensional tensor of grid values
    /// * `xi` - Query points as 2D tensor of shape [n_points, ndim]
    /// * `method` - Interpolation method (Nearest or Linear)
    /// * `extrapolate` - How to handle out-of-bounds queries
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated values at query points.
    fn interpnd(
        &self,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        xi: &Tensor<R>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Tensor<R>>;
}
