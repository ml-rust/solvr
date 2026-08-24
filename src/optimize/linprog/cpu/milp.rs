//! CPU implementation of mixed-integer linear programming.

use crate::optimize::linprog::impl_generic::{
    MilpOptionsInternal, TensorLinearConstraints, milp_impl,
};
use crate::optimize::linprog::{
    LinProgTensorConstraints, MilpAlgorithms, MilpOptions, MilpTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MilpAlgorithms<CpuRuntime> for CpuClient {
    fn milp(
        &self,
        c: &Tensor<CpuRuntime>,
        constraints: &LinProgTensorConstraints<CpuRuntime>,
        integrality: &Tensor<CpuRuntime>,
        options: &MilpOptions,
    ) -> Result<MilpTensorResult<CpuRuntime>> {
        // Convert constraints to impl_generic format
        let internal_constraints = TensorLinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            lower_bounds: constraints.lower_bounds.clone(),
            upper_bounds: constraints.upper_bounds.clone(),
        };

        let internal_options = MilpOptionsInternal {
            max_nodes: options.max_nodes,
            int_tol: options.int_tol,
            gap_tol: options.gap_tol,
            lp_options: options.lp_options.clone(),
        };

        let result = milp_impl(
            self,
            c,
            &internal_constraints,
            integrality,
            &internal_options,
        )
        .map_err(|e| numr::error::Error::backend_limitation("cpu", "milp", e.to_string()))?;

        Ok(MilpTensorResult {
            x: result.x,
            fun: result.fun,
            success: result.success,
            nodes: result.nodes,
            gap: result.gap,
            message: result.message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_milp_simple() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Maximize x + 2y (minimize -x - 2y)
        // Subject to: x + y <= 4, x <= 2.5, x, y >= 0, x and y integer
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -2.0], &[2], &device).unwrap();

        let a_ub =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0, 0.0], &[2, 2], &device).unwrap();
        let b_ub = Tensor::<CpuRuntime>::from_slice(&[4.0, 2.5], &[2], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let upper =
            Tensor::<CpuRuntime>::from_slice(&[f64::INFINITY, f64::INFINITY], &[2], &device)
                .unwrap();
        let integrality = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();

        let constraints = LinProgTensorConstraints {
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: None,
            b_eq: None,
            lower_bounds: Some(lower),
            upper_bounds: Some(upper),
        };

        let result = client
            .milp(&c, &constraints, &integrality, &MilpOptions::default())
            .expect("milp failed");

        assert!(result.success);
        assert!(result.fun <= -6.0 + 0.1);

        let x_data: Vec<f64> = result.x.to_vec();
        for (i, &is_int) in [1.0, 1.0].iter().enumerate() {
            if is_int > 0.5 {
                assert!((x_data[i] - x_data[i].round()).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_milp_binary() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Binary knapsack: Maximize 3x + 4y, subject to 2x + 3y <= 5, 0 <= x,y <= 1
        let c = Tensor::<CpuRuntime>::from_slice(&[-3.0, -4.0], &[2], &device).unwrap();

        let a_ub = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0], &[1, 2], &device).unwrap();
        let b_ub = Tensor::<CpuRuntime>::from_slice(&[5.0], &[1], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let upper = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let integrality = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();

        let constraints = LinProgTensorConstraints {
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: None,
            b_eq: None,
            lower_bounds: Some(lower),
            upper_bounds: Some(upper),
        };

        let result = client
            .milp(&c, &constraints, &integrality, &MilpOptions::default())
            .expect("milp failed");

        assert!(result.success);
        assert!((result.fun - (-7.0)).abs() < 0.1);
    }

    #[test]
    fn test_milp_mixed() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Mixed: Minimize -x - y, x + y <= 2.5, x integer, y continuous
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -1.0], &[2], &device).unwrap();

        let a_ub = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device).unwrap();
        let b_ub = Tensor::<CpuRuntime>::from_slice(&[2.5], &[1], &device).unwrap();
        let lower = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let upper =
            Tensor::<CpuRuntime>::from_slice(&[f64::INFINITY, f64::INFINITY], &[2], &device)
                .unwrap();
        let integrality = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device).unwrap();

        let constraints = LinProgTensorConstraints {
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            a_eq: None,
            b_eq: None,
            lower_bounds: Some(lower),
            upper_bounds: Some(upper),
        };

        let result = client
            .milp(&c, &constraints, &integrality, &MilpOptions::default())
            .expect("milp failed");

        assert!(result.success);
        assert!(result.fun <= -2.5 + 0.1);

        let x_data: Vec<f64> = result.x.to_vec();
        assert!((x_data[0] - x_data[0].round()).abs() < 0.01);
    }
}
