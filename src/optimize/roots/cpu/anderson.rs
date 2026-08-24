//! CPU implementation of Anderson mixing.

use crate::optimize::roots::RootTensorResult;
use crate::optimize::roots::impl_generic::anderson_impl;
use crate::optimize::roots::traits::anderson::{AndersonAlgorithms, AndersonOptions};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AndersonAlgorithms<CpuRuntime> for CpuClient {
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<CpuRuntime>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        G: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = anderson_impl(self, g, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "anderson", e.to_string())
        })?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_anderson_cpu() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Fixed point of cos(x) at ~0.7391
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();

        let result = client
            .anderson(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let r: Vec<f64> = data.iter().map(|xi| xi.cos()).collect();
                    Tensor::from_slice(&r, x.shape(), x.device())
                },
                &x0,
                &AndersonOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 0.7390851332).abs() < 1e-6);
    }
}
