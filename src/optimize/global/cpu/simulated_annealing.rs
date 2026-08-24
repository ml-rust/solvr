//! CPU implementation of simulated annealing.

use numr::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::simulated_annealing::simulated_annealing_impl;
use crate::optimize::global::traits::SimulatedAnnealingAlgorithms;
use crate::optimize::global::traits::simulated_annealing::SimulatedAnnealingResult;
use numr::runtime::cpu::CpuClient;

impl SimulatedAnnealingAlgorithms<CpuRuntime> for CpuClient {
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<SimulatedAnnealingResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = simulated_annealing_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| {
                numr::error::Error::backend_limitation("cpu", "simulated_annealing", e.to_string())
            })?;
        Ok(SimulatedAnnealingResult {
            x: result.x,
            fun: result.fun,
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

    fn sphere_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(data.iter().map(|&xi| xi * xi).sum())
    }

    #[test]
    fn test_simulated_annealing_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();

        let opts = GlobalOptions {
            max_iter: 10000,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .simulated_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("SA failed");
        assert!(result.fun < 1.0);
    }
}
