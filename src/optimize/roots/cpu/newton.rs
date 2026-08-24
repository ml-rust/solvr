//! CPU implementation of Newton's method for root finding.

use crate::optimize::roots::impl_generic::newton_system_impl;
use crate::optimize::roots::traits::NewtonSystemAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl NewtonSystemAlgorithms<CpuRuntime> for CpuClient {
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = newton_system_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "newton_system", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn linear_system(x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let data: Vec<f64> = x.to_vec();
        let residuals = vec![data[0] + data[1] - 3.0, 2.0 * data[0] - data[1]];
        Tensor::from_slice(&residuals, &[2], x.device())
    }

    fn quadratic_system(x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let data: Vec<f64> = x.to_vec();
        let residuals = vec![
            data[0] * data[0] + data[1] * data[1] - 1.0,
            data[0] - data[1],
        ];
        Tensor::from_slice(&residuals, &[2], x.device())
    }

    #[test]
    fn test_newton_system_linear() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let result = client
            .newton_system(linear_system, &x0, &RootOptions::default())
            .expect("newton_system failed");

        assert!(result.converged);
        let x_data: Vec<f64> = result.x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-6);
        assert!((x_data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_quadratic() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[2], &device).unwrap();
        let result = client
            .newton_system(quadratic_system, &x0, &RootOptions::default())
            .expect("newton_system failed");

        assert!(result.converged);
        let x_data: Vec<f64> = result.x.to_vec();
        let expected = 1.0 / (2.0_f64).sqrt();
        assert!((x_data[0] - expected).abs() < 1e-6);
        assert!((x_data[1] - expected).abs() < 1e-6);
    }
}
