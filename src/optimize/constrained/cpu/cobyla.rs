//! CPU implementation of COBYLA.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::cobyla_impl;
use crate::optimize::constrained::traits::{
    Bounds, CobylaAlgorithms, ConstrainedOptions, ConstrainedResult, Constraint,
};
use crate::optimize::error::OptimizeResult;

impl CobylaAlgorithms<CpuRuntime> for CpuClient {
    fn cobyla<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        constraints: &[Constraint<'_, CpuRuntime>],
        bounds: &Bounds<CpuRuntime>,
        options: &ConstrainedOptions,
    ) -> OptimizeResult<ConstrainedResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        cobyla_impl(self, f, x0, constraints, bounds, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_cobyla_unconstrained() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();

        let result = client
            .cobyla(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[],
                &Bounds::default(),
                &ConstrainedOptions {
                    max_iter: 500,
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(
            result.fun < 1.0,
            "COBYLA should reduce objective, got {}",
            result.fun
        );
    }

    #[test]
    fn test_cobyla_with_bounds() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x0 = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0], &[2], &device).unwrap();

        let result = client
            .cobyla(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[],
                &Bounds {
                    lower: Some(lower),
                    upper: Some(upper),
                },
                &ConstrainedOptions {
                    max_iter: 500,
                    ..Default::default()
                },
            )
            .unwrap();

        let sol: Vec<f64> = result.x.to_vec();
        // Should be near lower bounds [1, 1]
        assert!(sol[0] <= 2.0, "x[0] = {} should be near 1.0", sol[0]);
        assert!(sol[1] <= 2.0, "x[1] = {} should be near 1.0", sol[1]);
    }
}
