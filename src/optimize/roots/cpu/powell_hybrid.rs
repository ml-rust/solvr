//! CPU implementation of Powell hybrid method.

use crate::optimize::roots::impl_generic::powell_hybrid_impl;
use crate::optimize::roots::traits::PowellHybridAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl PowellHybridAlgorithms<CpuRuntime> for CpuClient {
    fn powell_hybrid<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = powell_hybrid_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "powell_hybrid", e.to_string())
        })?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_powell_hybrid_cpu() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();

        let result = client
            .powell_hybrid(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let residuals = vec![data[0] + data[1] - 3.0, 2.0 * data[0] - data[1]];
                    Tensor::from_slice(&residuals, &[2], x.device())
                },
                &x0,
                &RootOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 1e-6);
        assert!((sol[1] - 2.0).abs() < 1e-6);
    }
}
