//! CPU implementation of differential evolution.

use numr::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::differential_evolution::differential_evolution_impl;
use crate::optimize::global::traits::DifferentialEvolutionAlgorithms;
use crate::optimize::global::traits::differential_evolution::DifferentialEvolutionResult;
use numr::runtime::cpu::CpuClient;

impl DifferentialEvolutionAlgorithms<CpuRuntime> for CpuClient {
    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<DifferentialEvolutionResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = differential_evolution_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| {
                numr::error::Error::backend_limitation(
                    "cpu",
                    "differential_evolution",
                    e.to_string(),
                )
            })?;
        Ok(DifferentialEvolutionResult {
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
    fn test_de_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();

        let opts = GlobalOptions {
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .differential_evolution(sphere_tensor, &lower, &upper, &opts)
            .expect("DE failed");
        assert!(result.fun < 1e-6);
    }
}
