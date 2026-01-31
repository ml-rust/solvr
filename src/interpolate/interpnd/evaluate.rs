//! N-dimensional grid interpolation evaluation methods.

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::{ExtrapolateMode, InterpNdMethod, RegularGridInterpolator};

impl<R: Runtime> RegularGridInterpolator<R> {
    /// Evaluate the interpolant at query points.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client
    /// * `xi` - Query points as 2D tensor of shape [n_points, ndim]
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated values with length n_points.
    pub fn evaluate<C: RuntimeClient<R>>(
        &self,
        _client: &C,
        xi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let xi_shape = xi.shape();
        if xi_shape.len() != 2 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "xi".to_string(),
                message: format!(
                    "Query points must be 2D [n_points, ndim], got {:?}",
                    xi_shape
                ),
            });
        }

        let n_points = xi_shape[0];
        let query_ndim = xi_shape[1];

        if query_ndim != self.n_dims {
            return Err(InterpolateError::DimensionMismatch {
                expected: self.n_dims,
                actual: query_ndim,
                context: "RegularGridInterpolator::evaluate (query dimensions)".to_string(),
            });
        }

        let xi_data: Vec<f64> = xi.to_vec();
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point: Vec<f64> = (0..self.n_dims)
                .map(|d| xi_data[i * self.n_dims + d])
                .collect();

            let value = self.evaluate_single(&point)?;
            results.push(value);
        }

        Ok(Tensor::from_slice(&results, &[n_points], &self.device))
    }

    /// Evaluate at a single point.
    pub(super) fn evaluate_single(&self, point: &[f64]) -> InterpolateResult<f64> {
        // Find intervals and check bounds for each dimension
        let mut indices = Vec::with_capacity(self.n_dims);
        let mut fracs = Vec::with_capacity(self.n_dims);

        for (d, (&x, pts)) in point.iter().zip(self.points.iter()).enumerate() {
            let n = pts.len();

            // Check bounds
            if x < pts[0] || x > pts[n - 1] {
                match self.extrapolate {
                    ExtrapolateMode::Error => {
                        return Err(InterpolateError::OutOfDomainNd {
                            dimension: d,
                            point: x,
                            min: pts[0],
                            max: pts[n - 1],
                            context: "RegularGridInterpolator::evaluate".to_string(),
                        });
                    }
                    ExtrapolateMode::Nan => {
                        return Ok(f64::NAN);
                    }
                    ExtrapolateMode::Extrapolate => {
                        // Clamp to boundary
                        let clamped = x.clamp(pts[0], pts[n - 1]);
                        let (idx, frac) = self.find_interval(pts, clamped);
                        indices.push(idx);
                        fracs.push(frac);
                        continue;
                    }
                }
            }

            let (idx, frac) = self.find_interval(pts, x);
            indices.push(idx);
            fracs.push(frac);
        }

        match self.method {
            InterpNdMethod::Nearest => self.interp_nearest(&indices, &fracs),
            InterpNdMethod::Linear => self.interp_linear(&indices, &fracs),
        }
    }

    /// Find interval index and fractional position for a value.
    pub(super) fn find_interval(&self, pts: &[f64], x: f64) -> (usize, f64) {
        let n = pts.len();

        // Binary search for interval
        let mut lo = 0;
        let mut hi = n - 1;

        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if pts[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Handle edge case: x exactly at upper bound
        if lo == n - 2 && x == pts[n - 1] {
            return (lo, 1.0);
        }

        let frac = (x - pts[lo]) / (pts[lo + 1] - pts[lo]);
        (lo, frac)
    }

    /// Nearest neighbor interpolation.
    fn interp_nearest(&self, indices: &[usize], fracs: &[f64]) -> InterpolateResult<f64> {
        // Round to nearest grid point
        let mut idx = Vec::with_capacity(self.n_dims);
        for d in 0..self.n_dims {
            let i = if fracs[d] < 0.5 {
                indices[d]
            } else {
                (indices[d] + 1).min(self.shape[d] - 1)
            };
            idx.push(i);
        }

        Ok(self.get_value(&idx))
    }

    /// Multilinear interpolation.
    fn interp_linear(&self, indices: &[usize], fracs: &[f64]) -> InterpolateResult<f64> {
        // Number of vertices in the hypercube: 2^ndim
        let n_vertices = 1 << self.n_dims;
        let mut result = 0.0;

        for vertex in 0..n_vertices {
            // Build index for this vertex and compute weight
            let mut idx = Vec::with_capacity(self.n_dims);
            let mut weight = 1.0;

            for d in 0..self.n_dims {
                let use_upper = (vertex >> d) & 1 == 1;
                if use_upper {
                    idx.push((indices[d] + 1).min(self.shape[d] - 1));
                    weight *= fracs[d];
                } else {
                    idx.push(indices[d]);
                    weight *= 1.0 - fracs[d];
                }
            }

            result += weight * self.get_value(&idx);
        }

        Ok(result)
    }

    /// Get value at a multi-dimensional index (row-major order).
    pub(super) fn get_value(&self, idx: &[usize]) -> f64 {
        let mut flat_idx = 0;
        let mut stride = 1;

        // Row-major: last dimension varies fastest
        for d in (0..self.n_dims).rev() {
            flat_idx += idx[d] * stride;
            stride *= self.shape[d];
        }

        self.values[flat_idx]
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
    fn test_1d_linear() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_2d_linear() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 1.0, 2.0], &[2, 2], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0, 30.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Nearest).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.3, 0.7, 1.4], &[3, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 10.0).abs() < 1e-10);
        assert!((result_data[1] - 20.0).abs() < 1e-10);
        assert!((result_data[2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_out_of_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1, 1], &device);
        let result = interp.evaluate(&client, &xi);

        assert!(matches!(
            result,
            Err(InterpolateError::OutOfDomainNd { .. })
        ));
    }

    #[test]
    fn test_3d_interpolation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let values = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0],
            &[2, 2, 2],
            &device,
        );

        let interp =
            RegularGridInterpolator::new(&client, &[&x, &y, &z], &values, InterpNdMethod::Linear)
                .unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5, 0.5], &[1, 3], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.5).abs() < 1e-10);
    }
}
