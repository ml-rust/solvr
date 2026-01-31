//! N-dimensional interpolation on regular grids.
//!
//! This module provides interpolation over N-dimensional rectilinear grids,
//! where each dimension has its own 1D coordinate array.
//!
//! # Example
//!
//! ```ignore
//! use solvr::interpolate::{RegularGridInterpolator, InterpNdMethod};
//!
//! // Create a 2D grid: x = [0, 1, 2], y = [0, 1]
//! let x = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
//! let y = Tensor::from_slice(&[0.0, 1.0], &[2], &device);
//! // Values on 2x3 grid (shape matches [len(y), len(x)])
//! let z = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0, 2.0, 3.0], &[2, 3], &device);
//!
//! let interp = RegularGridInterpolator::new(&client, &[&x, &y], &z, InterpNdMethod::Linear)?;
//! let result = interp.evaluate(&client, &query_points)?;
//! ```

mod evaluate;

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
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

/// N-dimensional interpolation on a regular (rectilinear) grid.
///
/// Supports any number of dimensions. For a grid with dimensions [d0, d1, ..., dn-1],
/// you provide n coordinate arrays where the i-th array has length di, and a values
/// tensor with shape [d0, d1, ..., dn-1].
pub struct RegularGridInterpolator<R: Runtime> {
    /// Coordinate arrays for each dimension.
    pub(crate) points: Vec<Vec<f64>>,
    /// Grid values (N-dimensional).
    pub(crate) values: Vec<f64>,
    /// Shape of the values array.
    pub(crate) shape: Vec<usize>,
    /// Number of dimensions.
    pub(crate) n_dims: usize,
    /// Interpolation method.
    pub(crate) method: InterpNdMethod,
    /// How to handle out-of-bounds queries.
    pub(crate) extrapolate: ExtrapolateMode,
    /// Device for output tensors.
    pub(crate) device: R::Device,
}

impl<R: Runtime> RegularGridInterpolator<R> {
    /// Create a new N-dimensional grid interpolator.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client
    /// * `points` - Slice of 1D tensors, one per dimension. Each must be strictly increasing.
    /// * `values` - N-dimensional tensor of values. Shape must match [len(points[0]), ...]
    /// * `method` - Interpolation method (Nearest or Linear)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any coordinate array has fewer than 2 points
    /// - Coordinate arrays are not strictly increasing
    /// - Values shape doesn't match coordinate array lengths
    pub fn new<C: RuntimeClient<R>>(
        _client: &C,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        method: InterpNdMethod,
    ) -> InterpolateResult<Self> {
        Self::with_extrapolate(_client, points, values, method, ExtrapolateMode::Error)
    }

    /// Create a new interpolator with custom extrapolation behavior.
    pub fn with_extrapolate<C: RuntimeClient<R>>(
        _client: &C,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        let ndim = points.len();
        if ndim == 0 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "points".to_string(),
                message: "At least one dimension required".to_string(),
            });
        }

        let values_shape = values.shape();
        if values_shape.len() != ndim {
            return Err(InterpolateError::DimensionMismatch {
                expected: ndim,
                actual: values_shape.len(),
                context: "RegularGridInterpolator::new (values dimensions)".to_string(),
            });
        }

        let mut point_vecs = Vec::with_capacity(ndim);
        let mut shape = Vec::with_capacity(ndim);

        for (dim, &pts) in points.iter().enumerate() {
            let pts_shape = pts.shape();
            if pts_shape.len() != 1 {
                return Err(InterpolateError::InvalidParameter {
                    parameter: format!("points[{}]", dim),
                    message: "Coordinate arrays must be 1D".to_string(),
                });
            }

            let n = pts_shape[0];
            if n < 2 {
                return Err(InterpolateError::InsufficientData {
                    required: 2,
                    actual: n,
                    context: format!("RegularGridInterpolator dimension {}", dim),
                });
            }

            if n != values_shape[dim] {
                return Err(InterpolateError::ShapeMismatch {
                    expected: n,
                    actual: values_shape[dim],
                    context: format!(
                        "RegularGridInterpolator dimension {} (points vs values)",
                        dim
                    ),
                });
            }

            let pts_data: Vec<f64> = pts.to_vec();

            // Check strictly increasing
            for i in 1..n {
                if pts_data[i] <= pts_data[i - 1] {
                    return Err(InterpolateError::NotMonotonic {
                        context: format!("RegularGridInterpolator dimension {}", dim),
                    });
                }
            }

            shape.push(n);
            point_vecs.push(pts_data);
        }

        let values_data: Vec<f64> = values.to_vec();
        let device = values.device().clone();

        Ok(Self {
            points: point_vecs,
            values: values_data,
            shape,
            n_dims: ndim,
            method,
            extrapolate,
            device,
        })
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.n_dims
    }

    /// Returns the shape of the grid.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the domain bounds as [(min, max), ...] for each dimension.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|pts| (pts[0], pts[pts.len() - 1]))
            .collect()
    }

    /// Returns the interpolation method.
    pub fn method(&self) -> InterpNdMethod {
        self.method
    }

    /// Returns the extrapolation mode.
    pub fn extrapolate_mode(&self) -> ExtrapolateMode {
        self.extrapolate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_constructor() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        assert_eq!(interp.ndim(), 1);
        assert_eq!(interp.shape(), &[3]);
    }

    #[test]
    fn test_dimension_validation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &z, InterpNdMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::DimensionMismatch { .. })
        ));

        let result =
            RegularGridInterpolator::new(&client, &[&x, &y], &z, InterpNdMethod::Linear).unwrap();
        assert_eq!(result.ndim(), 2);
    }

    #[test]
    fn test_shape_mismatch() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_not_monotonic() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_extrapolate_nan() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let interp = RegularGridInterpolator::with_extrapolate(
            &client,
            &[&x],
            &y,
            InterpNdMethod::Linear,
            ExtrapolateMode::Nan,
        )
        .unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!(result_data[0].is_nan());
    }

    #[test]
    fn test_extrapolate_clamp() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0], &[2], &device);

        let interp = RegularGridInterpolator::with_extrapolate(
            &client,
            &[&x],
            &y,
            InterpNdMethod::Linear,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[-1.0, 2.0], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 10.0).abs() < 1e-10);
        assert!((result_data[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[-1.0, 0.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0; 6], &[2, 3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        let bounds = interp.bounds();
        assert_eq!(bounds.len(), 2);
        assert!((bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((bounds[0].1 - 0.0).abs() < 1e-10);
        assert!((bounds[1].0 - 1.0).abs() < 1e-10);
        assert!((bounds[1].1 - 3.0).abs() < 1e-10);
    }
}
