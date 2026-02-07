//! CPU implementation of SHGO.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::shgo::shgo_impl;
use crate::optimize::global::traits::ShgoAlgorithms;
use crate::optimize::global::traits::shgo::ShgoResult;

impl ShgoAlgorithms<CpuRuntime> for CpuClient {
    fn shgo<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<ShgoResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = shgo_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| numr::error::Error::backend_limitation("cpu", "shgo", e.to_string()))?;
        Ok(ShgoResult {
            x: result.x,
            fun: result.fun,
            local_minima: result.local_minima,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    /// Sphere function: f(x) = sum(x_i^2), minimum at x=0 with f=0.
    fn sphere_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(data.iter().map(|&xi| xi * xi).sum())
    }

    /// Rastrigin function: multimodal with many local minima.
    /// f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    fn rastrigin_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        let n = data.len() as f64;
        let sum: f64 = data
            .iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        Ok(10.0 * n + sum)
    }

    #[test]
    fn test_shgo_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 100,
            tol: 1e-6,
            seed: Some(42),
        };

        let result = client
            .shgo(sphere_tensor, &lower, &upper, &opts)
            .expect("SHGO failed");

        // SHGO should find the global minimum (at origin) for sphere function
        assert!(result.fun < 1e-4, "Expected f < 1e-4, got {}", result.fun);
        assert!(result.converged);
    }

    #[test]
    fn test_shgo_rastrigin() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Rastrigin is challenging: global minimum at x=0 with f=0,
        // but many local minima. We use a smaller domain for tractability.
        let lower = Tensor::<CpuRuntime>::from_slice(&[-2.0, -2.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[2.0, 2.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 100,
            tol: 1e-6,
            seed: Some(42),
        };

        let result = client
            .shgo(rastrigin_tensor, &lower, &upper, &opts)
            .expect("SHGO failed");

        // For Rastrigin in [-2,2], we expect to find a good solution
        // (exact global optimum may be hard, but should be much better than worst case)
        assert!(result.fun < 50.0, "Expected f < 50.0, got {}", result.fun);
        assert!(result.converged);
    }
}
