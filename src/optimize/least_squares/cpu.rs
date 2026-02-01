//! CPU implementation of least squares algorithms.
//!
//! This module implements the [`LeastSquaresAlgorithms`] trait for CPU
//! by delegating to the generic implementations in `impl_generic/least_squares/`.

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::least_squares::{least_squares_impl, leastsq_impl};
use crate::optimize::least_squares::{
    LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresResult, LeastSquaresTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

impl LeastSquaresAlgorithms<CpuRuntime> for CpuClient {
    fn leastsq<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = leastsq_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "leastsq", e.to_string())
        })?;
        Ok(LeastSquaresTensorResult {
            x: result.x,
            residuals: result.residuals,
            cost: result.cost,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }

    fn least_squares<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        bounds: Option<(&Tensor<CpuRuntime>, &Tensor<CpuRuntime>)>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = least_squares_impl(self, f, x0, bounds, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "least_squares", e.to_string())
        })?;
        Ok(LeastSquaresTensorResult {
            x: result.x,
            residuals: result.residuals,
            cost: result.cost,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}

// ============================================================================
// Convenience Functions (Scalar API)
// ============================================================================

/// Levenberg-Marquardt algorithm for nonlinear least squares.
///
/// This is a convenience function that wraps the tensor-based implementation.
///
/// # Arguments
/// * `f` - Residual function f: R^n -> R^m
/// * `x0` - Initial parameter guess
/// * `options` - Solver options
pub fn leastsq<F>(
    f: F,
    x0: &[f64],
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "leastsq: empty initial guess".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
        let x_data: Vec<f64> = x.to_vec();
        let result = f(&x_data);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &result,
            &[result.len()],
            &device,
        ))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);
    let result = client
        .leastsq(tensor_f, &x0_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(LeastSquaresResult {
        x: result.x.to_vec(),
        residuals: result.residuals.to_vec(),
        cost: result.cost,
        iterations: result.iterations,
        nfev: result.nfev,
        converged: result.converged,
    })
}

/// Bounded Levenberg-Marquardt algorithm.
///
/// This is a convenience function that wraps the tensor-based implementation.
pub fn least_squares<F>(
    f: F,
    x0: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "least_squares: empty initial guess".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
        let x_data: Vec<f64> = x.to_vec();
        let result = f(&x_data);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &result,
            &[result.len()],
            &device,
        ))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);

    let bounds_tensors = match bounds {
        Some((lower, upper)) => {
            if lower.len() != n || upper.len() != n {
                return Err(OptimizeError::InvalidInput {
                    context: "least_squares: bounds dimension mismatch".to_string(),
                });
            }
            let lower_tensor = Tensor::<CpuRuntime>::from_slice(lower, &[n], &device);
            let upper_tensor = Tensor::<CpuRuntime>::from_slice(upper, &[n], &device);
            Some((lower_tensor, upper_tensor))
        }
        None => None,
    };

    let bounds_refs = bounds_tensors.as_ref().map(|(l, u)| (l, u));
    let result = client
        .least_squares(tensor_f, &x0_tensor, bounds_refs, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(LeastSquaresResult {
        x: result.x.to_vec(),
        residuals: result.residuals.to_vec(),
        cost: result.cost,
        iterations: result.iterations,
        nfev: result.nfev,
        converged: result.converged,
    })
}

/// Fit a model function to data using nonlinear least squares.
pub fn curve_fit<F>(
    model: F,
    x_data: &[f64],
    y_data: &[f64],
    p0: &[f64],
    options: &LeastSquaresOptions,
) -> OptimizeResult<LeastSquaresResult>
where
    F: Fn(f64, &[f64]) -> f64,
{
    if x_data.len() != y_data.len() {
        return Err(OptimizeError::InvalidInput {
            context: "curve_fit: x_data and y_data must have same length".to_string(),
        });
    }

    if x_data.is_empty() {
        return Err(OptimizeError::InvalidInput {
            context: "curve_fit: data arrays are empty".to_string(),
        });
    }

    let residual_fn = |params: &[f64]| -> Vec<f64> {
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| model(x, params) - y)
            .collect()
    };

    leastsq(residual_fn, p0, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leastsq_linear_fit() {
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        let result =
            leastsq(residual, &[0.0, 0.0], &LeastSquaresOptions::default()).expect("leastsq failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_leastsq_exponential_fit() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] * (-p[1] * x).exp() - y)
                .collect()
        };

        let result =
            leastsq(residual, &[1.0, 1.0], &LeastSquaresOptions::default()).expect("leastsq failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_least_squares_bounded() {
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p[0] + p[1] * x - y)
                .collect()
        };

        let lower = vec![-10.0, 0.0];
        let upper = vec![10.0, 1.5];

        let result = least_squares(
            residual,
            &[1.0, 1.0],
            Some((&lower, &upper)),
            &LeastSquaresOptions::default(),
        )
        .expect("least_squares failed");

        assert!(result.x[1] <= 1.5 + 1e-6);
        assert!(result.x[1] >= 0.0 - 1e-6);
    }

    #[test]
    fn test_curve_fit_exponential() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();

        let result = curve_fit(model, &x_data, &y_data, &[1.0, 1.0], &LeastSquaresOptions::default())
            .expect("curve_fit failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_empty_input() {
        let result = leastsq(|_: &[f64]| vec![], &[], &LeastSquaresOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }
}
