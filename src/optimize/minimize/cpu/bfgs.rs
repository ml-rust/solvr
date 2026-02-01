//! CPU implementation of BFGS quasi-Newton method.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::MinimizeOptions;
use super::super::impl_generic::bfgs_impl;
use super::super::impl_generic::TensorMinimizeResult;

impl crate::optimize::OptimizationAlgorithms<CpuRuntime> for CpuClient {
    fn bfgs<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        bfgs_impl(self, f, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_bfgs_cpu() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device);

        let result = client
            .bfgs(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    Ok(data.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &MinimizeOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-6);
    }
}
