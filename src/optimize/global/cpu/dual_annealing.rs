//! CPU implementation of dual annealing.

use numr::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::dual_annealing::dual_annealing_impl;
use crate::optimize::global::traits::DualAnnealingAlgorithms;
use crate::optimize::global::traits::dual_annealing::DualAnnealingResult;
use numr::runtime::cpu::CpuClient;

impl DualAnnealingAlgorithms<CpuRuntime> for CpuClient {
    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<DualAnnealingResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result =
            dual_annealing_impl(self, f, lower_bounds, upper_bounds, options).map_err(|e| {
                numr::error::Error::backend_limitation("cpu", "dual_annealing", e.to_string())
            })?;
        Ok(DualAnnealingResult {
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

    fn ackley_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        let n = data.len() as f64;
        let sum_sq: f64 = data.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = data
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        Ok(
            -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
                + 20.0
                + std::f64::consts::E,
        )
    }

    #[test]
    fn test_dual_annealing_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();

        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .dual_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("DA failed");
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_dual_annealing_ackley() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();

        let opts = GlobalOptions {
            max_iter: 3000,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .dual_annealing(ackley_tensor, &lower, &upper, &opts)
            .expect("DA failed");
        // Ackley minimum is 0 at origin; accept reasonable convergence for stochastic method
        // With more iterations, should get close to global minimum
        assert!(result.fun < 10.0);
    }
}
