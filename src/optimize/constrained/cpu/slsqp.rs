//! CPU implementation of SLSQP.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::slsqp_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, SlsqpAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl SlsqpAlgorithms<CpuRuntime> for CpuClient {
    fn slsqp<F>(
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
        slsqp_impl(self, f, x0, constraints, bounds, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::constrained::traits::ConstraintType;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_slsqp_unconstrained() {
        let (device, client) = setup();
        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();

        let result = client
            .slsqp(
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

        assert!(result.converged);
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_slsqp_with_bounds() {
        let (device, client) = setup();
        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[2], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0], &[2], &device).unwrap();

        let bounds = Bounds {
            lower: Some(lower),
            upper: Some(upper),
        };

        let result = client
            .slsqp(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[],
                &bounds,
                &ConstrainedOptions::default(),
            )
            .unwrap();

        // Should find a solution within bounds
        let sol: Vec<f64> = result.x.to_vec();
        assert!(sol[0] >= 1.0 - 1e-6);
        assert!(sol[1] >= 1.0 - 1e-6);
        assert!(sol[0] <= 10.0 + 1e-6);
        assert!(sol[1] <= 10.0 + 1e-6);
    }

    #[test]
    fn test_slsqp_equality_constraint() {
        let (device, client) = setup();
        // Minimize x^2 + y^2 subject to x + y = 1
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[2], &device).unwrap();

        let constraint = Constraint {
            kind: ConstraintType::Equality,
            fun: &|x: &Tensor<CpuRuntime>| {
                let d: Vec<f64> = x.to_vec();
                let val = d[0] + d[1] - 1.0;
                Tensor::<CpuRuntime>::from_slice(&[val], &[1], x.device())
            },
            jac: None,
        };

        let result = client
            .slsqp(
                |x| {
                    let d: Vec<f64> = x.to_vec();
                    Ok(d.iter().map(|xi| xi * xi).sum())
                },
                &x0,
                &[constraint],
                &Bounds::default(),
                &ConstrainedOptions {
                    max_iter: 200,
                    ..Default::default()
                },
            )
            .unwrap();

        // Minimum at (0.5, 0.5) with f = 0.5
        let sol: Vec<f64> = result.x.to_vec();
        assert!(
            (sol[0] - 0.5).abs() < 0.1,
            "x[0] = {} expected ~0.5",
            sol[0]
        );
        assert!(
            (sol[1] - 0.5).abs() < 0.1,
            "x[1] = {} expected ~0.5",
            sol[1]
        );
    }
}
