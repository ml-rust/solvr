//! CPU implementation of Broyden's method for root finding.

use crate::optimize::roots::impl_generic::broyden1_impl;
use crate::optimize::roots::traits::Broyden1Algorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl Broyden1Algorithms<CpuRuntime> for CpuClient {
    fn broyden1<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = broyden1_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "broyden1", e.to_string())
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

    #[test]
    fn test_broyden1_linear() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let result = client
            .broyden1(linear_system, &x0, &RootOptions::default())
            .expect("broyden1 failed");

        assert!(result.converged);
        let x_data: Vec<f64> = result.x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 2.0).abs() < 1e-5);
    }
}
