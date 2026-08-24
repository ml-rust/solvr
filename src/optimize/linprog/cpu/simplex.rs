//! CPU implementation of simplex linear programming.

use crate::optimize::linprog::impl_generic::TensorLinearConstraints;
use crate::optimize::linprog::impl_generic::simplex_impl;
use crate::optimize::linprog::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl LinProgAlgorithms<CpuRuntime> for CpuClient {
    fn linprog(
        &self,
        c: &Tensor<CpuRuntime>,
        constraints: &LinProgTensorConstraints<CpuRuntime>,
        options: &LinProgOptions,
    ) -> Result<LinProgTensorResult<CpuRuntime>> {
        let tensor_constraints = TensorLinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            lower_bounds: constraints.lower_bounds.clone(),
            upper_bounds: constraints.upper_bounds.clone(),
        };

        let result = simplex_impl(self, c, &tensor_constraints, options)
            .map_err(|e| numr::error::Error::backend_limitation("cpu", "linprog", e.to_string()))?;

        Ok(LinProgTensorResult {
            x: result.x,
            fun: result.fun,
            success: result.success,
            nit: result.nit,
            message: result.message,
            slack: result.slack,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_linprog_simple() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Maximize x + 2y (minimize -x - 2y)
        // Subject to:
        //   x + y <= 4
        //   x <= 2
        //   y <= 3
        //   x, y >= 0
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -2.0], &[2], &device).unwrap();

        let a_ub =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device)
                .unwrap();
        let b_ub = Tensor::<CpuRuntime>::from_slice(&[4.0, 2.0, 3.0], &[3], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let upper =
            Tensor::<CpuRuntime>::from_slice(&[f64::INFINITY, f64::INFINITY], &[2], &device)
                .unwrap();

        let constraints = LinProgTensorConstraints {
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: None,
            b_eq: None,
            lower_bounds: Some(lower),
            upper_bounds: Some(upper),
        };

        let result = client
            .linprog(&c, &constraints, &LinProgOptions::default())
            .expect("linprog failed");

        assert!(result.success);
        assert!((result.fun - (-7.0)).abs() < 0.1);
    }

    #[test]
    fn test_linprog_with_equality() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Minimize x + y
        // Subject to: x + y = 2, x >= 0, y >= 0
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();

        let a_eq = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device).unwrap();
        let b_eq = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let upper =
            Tensor::<CpuRuntime>::from_slice(&[f64::INFINITY, f64::INFINITY], &[2], &device)
                .unwrap();

        let constraints = LinProgTensorConstraints {
            a_ub: None,
            b_ub: None,
            a_eq: Some(a_eq),
            b_eq: Some(b_eq),
            lower_bounds: Some(lower),
            upper_bounds: Some(upper),
        };

        let result = client
            .linprog(&c, &constraints, &LinProgOptions::default())
            .expect("linprog failed");

        assert!(result.success);
        assert!((result.fun - 2.0).abs() < 0.1);
    }
}
