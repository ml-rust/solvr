//! CPU implementation of basin-hopping.

use numr::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::basinhopping::basinhopping_impl;
use crate::optimize::global::traits::BasinHoppingAlgorithms;
use crate::optimize::global::traits::basin_hopping::BasinHoppingResult;
use numr::runtime::cpu::CpuClient;

impl BasinHoppingAlgorithms<CpuRuntime> for CpuClient {
    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<BasinHoppingResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result =
            basinhopping_impl(self, f, x0, lower_bounds, upper_bounds, options).map_err(|e| {
                numr::error::Error::backend_limitation("cpu", "basinhopping", e.to_string())
            })?;
        Ok(BasinHoppingResult {
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

    fn rosenbrock_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        let mut sum = 0.0;
        for i in 0..data.len() - 1 {
            sum += 100.0 * (data[i + 1] - data[i] * data[i]).powi(2) + (1.0 - data[i]).powi(2);
        }
        Ok(sum)
    }

    #[test]
    fn test_basinhopping_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 2.0, 2.0], &[3], &device).unwrap();
        let lower =
            Tensor::<CpuRuntime>::from_slice(&[-10.0, -10.0, -10.0], &[3], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0, 10.0], &[3], &device).unwrap();

        let opts = GlobalOptions {
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .basinhopping(sphere_tensor, &x0, &lower, &upper, &opts)
            .expect("basinhopping failed");
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_basinhopping_rosenbrock() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[-10.0, -10.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0], &[2], &device).unwrap();

        let opts = GlobalOptions {
            max_iter: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .basinhopping(rosenbrock_tensor, &x0, &lower, &upper, &opts)
            .expect("basinhopping failed");
        assert!(result.fun < 0.1);
    }
}
