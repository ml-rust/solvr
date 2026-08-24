//! CPU implementation of trust-constr.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::trust_constr_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, TrustConstrAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl TrustConstrAlgorithms<CpuRuntime> for CpuClient {
    fn trust_constr<F>(
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
        trust_constr_impl(self, f, x0, constraints, bounds, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_trust_constr_unconstrained() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();

        let result = client
            .trust_constr(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[],
                &Bounds::default(),
                &ConstrainedOptions::default(),
            )
            .unwrap();

        assert!(
            result.fun < 1e-4,
            "trust_constr should minimize, got {}",
            result.fun
        );
    }

    #[test]
    fn test_trust_constr_with_bounds() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let x0 = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device).unwrap();

        let result = client
            .trust_constr(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[],
                &Bounds {
                    lower: Some(
                        Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap(),
                    ),
                    upper: Some(
                        Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0], &[2], &device).unwrap(),
                    ),
                },
                &ConstrainedOptions {
                    max_iter: 200,
                    ..Default::default()
                },
            )
            .unwrap();

        let sol: Vec<f64> = result.x.to_vec();
        assert!(sol[0] >= 0.9 && sol[0] <= 1.5, "x[0] = {}", sol[0]);
        assert!(sol[1] >= 0.9 && sol[1] <= 1.5, "x[1] = {}", sol[1]);
    }
}
