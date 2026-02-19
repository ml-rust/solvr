//! Geometric transform algorithm traits.
//!
//! Provides coordinate mapping and affine transformation algorithms for
//! N-dimensional arrays, equivalent to scipy.ndimage geometric transforms.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Interpolation order for geometric transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationOrder {
    /// Nearest neighbor interpolation (order 0).
    Nearest,
    /// Bilinear/trilinear interpolation (order 1).
    #[default]
    Linear,
}

/// Algorithmic contract for geometric transform operations.
///
/// All backends implementing geometric transforms MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait GeometricTransformAlgorithms<R: Runtime<DType = DType>> {
    /// Map coordinates through interpolation.
    ///
    /// Given arrays of coordinates, interpolates values from the input at those
    /// coordinates. This is the core function that all other transforms use.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `coordinates` - Tensor of shape `[ndim, ...output_shape]` where each
    ///   slice along dim 0 gives the coordinates for that axis
    /// * `order` - Interpolation order
    /// * `cval` - Value for out-of-bounds coordinates (default: 0.0)
    ///
    /// # Returns
    ///
    /// Output tensor with shape matching `coordinates[0].shape()`
    fn map_coordinates(
        &self,
        input: &Tensor<R>,
        coordinates: &Tensor<R>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<R>>;

    /// Apply an affine transformation to an N-dimensional array.
    ///
    /// Maps output coordinates to input coordinates via: input_coords = matrix @ output_coords + offset
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `matrix` - Transformation matrix of shape `[ndim, ndim]`
    /// * `offset` - Translation offset of shape `[ndim]`
    /// * `output_shape` - Shape of the output tensor (if None, same as input)
    /// * `order` - Interpolation order
    /// * `cval` - Fill value for out-of-bounds
    fn affine_transform(
        &self,
        input: &Tensor<R>,
        matrix: &Tensor<R>,
        offset: &Tensor<R>,
        output_shape: Option<&[usize]>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<R>>;

    /// Zoom (rescale) an N-dimensional array.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `zoom` - Zoom factor for each axis. Values > 1 enlarge, < 1 shrink.
    /// * `order` - Interpolation order
    fn zoom(
        &self,
        input: &Tensor<R>,
        zoom: &[f64],
        order: InterpolationOrder,
    ) -> InterpolateResult<Tensor<R>>;

    /// Rotate an array by the given angle.
    ///
    /// Rotation is applied in the plane defined by the two given axes.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (at least 2D)
    /// * `angle` - Rotation angle in degrees (counter-clockwise)
    /// * `axes` - Tuple of two axes defining the rotation plane (default: last two)
    /// * `reshape` - If true, output shape is adjusted to contain the full rotated image
    /// * `order` - Interpolation order
    /// * `cval` - Fill value for out-of-bounds
    fn rotate(
        &self,
        input: &Tensor<R>,
        angle: f64,
        axes: (usize, usize),
        reshape: bool,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<R>>;

    /// Shift an array by the given amount.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `shift` - Shift amount for each axis
    /// * `order` - Interpolation order
    /// * `cval` - Fill value for out-of-bounds
    fn shift(
        &self,
        input: &Tensor<R>,
        shift: &[f64],
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<R>>;
}
