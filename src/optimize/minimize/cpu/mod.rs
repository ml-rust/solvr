//! CPU implementations of multivariate minimization algorithms.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::MinimizeOptions;

use super::impl_generic::{
    TensorMinimizeResult, bfgs_impl, conjugate_gradient_impl, nelder_mead_impl, powell_impl,
};
use crate::optimize::impl_generic::scalar::{
    bisect_impl, brentq_impl, minimize_scalar_brent_impl, newton_impl,
};
use crate::optimize::scalar::{MinimizeResult, RootResult, ScalarOptions};

impl crate::optimize::OptimizationAlgorithms<CpuRuntime> for CpuClient {
    // Scalar root finding methods
    fn bisect<F>(&self, f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64,
    {
        bisect_impl(f, a, b, options)
    }

    fn brentq<F>(&self, f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64,
    {
        brentq_impl(f, a, b, options)
    }

    fn newton<F, DF>(
        &self,
        f: F,
        df: DF,
        x0: f64,
        options: &ScalarOptions,
    ) -> OptimizeResult<RootResult>
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64,
    {
        newton_impl(f, df, x0, options)
    }

    fn minimize_scalar_brent<F>(
        &self,
        f: F,
        bracket: Option<(f64, f64, f64)>,
        options: &ScalarOptions,
    ) -> OptimizeResult<MinimizeResult>
    where
        F: Fn(f64) -> f64,
    {
        minimize_scalar_brent_impl(f, bracket, options)
    }

    // Multivariate minimization methods
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

    fn nelder_mead<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        nelder_mead_impl(self, f, x0, options)
    }

    fn powell<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        powell_impl(self, f, x0, options)
    }

    fn conjugate_gradient<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &MinimizeOptions,
    ) -> OptimizeResult<TensorMinimizeResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        conjugate_gradient_impl(self, f, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::OptimizationAlgorithms;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_bfgs_cpu() {
        let (device, client) = setup();
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

    #[test]
    fn test_nelder_mead_cpu() {
        let (device, client) = setup();
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device);

        let result = client
            .nelder_mead(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    Ok((data[0] - 1.0).powi(2) + (data[1] - 2.0).powi(2))
                },
                &x0,
                &MinimizeOptions::default(),
            )
            .unwrap();

        assert!(result.fun < 1e-6);
    }
}
