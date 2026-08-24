//! CPU implementation of least squares algorithms.
//!
//! Implements the [`LeastSquaresAlgorithms`] trait for CPU runtime.
//! All implementations delegate to the generic implementations in `impl_generic/least_squares/`.

use crate::optimize::least_squares::impl_generic::{least_squares_impl, leastsq_impl};
use crate::optimize::least_squares::traits::{
    LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
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
        let result = leastsq_impl(self, f, x0, options)
            .map_err(|e| numr::error::Error::backend_limitation("cpu", "leastsq", e.to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_leastsq_linear_fit() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
            let p_data: Vec<f64> = p.to_vec();
            let residuals: Vec<f64> = x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p_data[0] + p_data[1] * x - y)
                .collect();
            Tensor::from_slice(&residuals, &[residuals.len()], p.device())
        };

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let result = client
            .leastsq(residual, &x0, &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        let x_result: Vec<f64> = result.x.to_vec();
        assert!((x_result[0] - 1.0).abs() < 1e-4);
        assert!((x_result[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_leastsq_exponential_fit() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * (-0.5 * x).exp()).collect();

        let residual = |p: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
            let p_data: Vec<f64> = p.to_vec();
            let residuals: Vec<f64> = x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p_data[0] * (-p_data[1] * x).exp() - y)
                .collect();
            Tensor::from_slice(&residuals, &[residuals.len()], p.device())
        };

        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let result = client
            .leastsq(residual, &x0, &LeastSquaresOptions::default())
            .expect("leastsq failed");

        assert!(result.converged);
        let x_result: Vec<f64> = result.x.to_vec();
        assert!((x_result[0] - 2.0).abs() < 1e-4);
        assert!((x_result[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_least_squares_bounded() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 3.0, 5.0, 7.0, 9.0];

        let residual = |p: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
            let p_data: Vec<f64> = p.to_vec();
            let residuals: Vec<f64> = x_data
                .iter()
                .zip(y_data.iter())
                .map(|(&x, &y)| p_data[0] + p_data[1] * x - y)
                .collect();
            Tensor::from_slice(&residuals, &[residuals.len()], p.device())
        };

        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[-10.0, 0.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 1.5], &[2], &device).unwrap();

        let result = client
            .least_squares(
                residual,
                &x0,
                Some((&lower, &upper)),
                &LeastSquaresOptions::default(),
            )
            .expect("least_squares failed");

        let x_result: Vec<f64> = result.x.to_vec();
        assert!(x_result[1] <= 1.5 + 1e-6);
        assert!(x_result[1] >= 0.0 - 1e-6);
    }
}
