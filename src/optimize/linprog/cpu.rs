//! CPU implementation of linear programming algorithms.
//!
//! This module implements the [`LinProgAlgorithms`] trait for CPU
//! by delegating to the generic implementations in `impl_generic/linprog/`.

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::linprog::{simplex_impl, TensorLinearConstraints};
use crate::optimize::linprog::{
    validate_constraints, LinearConstraints, LinProgAlgorithms, LinProgOptions, LinProgResult,
    LinProgTensorConstraints, LinProgTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
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

        let result = simplex_impl(self, c, &tensor_constraints, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "linprog", e.to_string())
        })?;

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

// ============================================================================
// Convenience Functions (Scalar API)
// ============================================================================

/// Solve a linear programming problem using the Simplex method.
///
/// This is a convenience function that wraps the tensor-based implementation.
///
/// Minimize: c^T * x
/// Subject to:
///   A_ub * x <= b_ub
///   A_eq * x == b_eq
///   bounds.0 <= x <= bounds.1
pub fn linprog(
    c: &[f64],
    constraints: &LinearConstraints,
    options: &LinProgOptions,
) -> OptimizeResult<LinProgResult> {
    let n = c.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "linprog: empty objective vector".to_string(),
        });
    }

    validate_constraints(n, constraints)?;

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let c_tensor = Tensor::<CpuRuntime>::from_slice(c, &[n], &device);

    // Convert constraints to tensors
    let a_ub = constraints.a_ub.as_ref().map(|a| {
        let rows = a.len();
        let flat: Vec<f64> = a.iter().flatten().cloned().collect();
        Tensor::<CpuRuntime>::from_slice(&flat, &[rows, n], &device)
    });

    let b_ub = constraints
        .b_ub
        .as_ref()
        .map(|b| Tensor::<CpuRuntime>::from_slice(b, &[b.len()], &device));

    let a_eq = constraints.a_eq.as_ref().map(|a| {
        let rows = a.len();
        let flat: Vec<f64> = a.iter().flatten().cloned().collect();
        Tensor::<CpuRuntime>::from_slice(&flat, &[rows, n], &device)
    });

    let b_eq = constraints
        .b_eq
        .as_ref()
        .map(|b| Tensor::<CpuRuntime>::from_slice(b, &[b.len()], &device));

    let (lower_bounds, upper_bounds) = match &constraints.bounds {
        Some(bounds) => {
            let lower: Vec<f64> = bounds.iter().map(|&(l, _)| l).collect();
            let upper: Vec<f64> = bounds.iter().map(|&(_, u)| u).collect();
            (
                Some(Tensor::<CpuRuntime>::from_slice(&lower, &[n], &device)),
                Some(Tensor::<CpuRuntime>::from_slice(&upper, &[n], &device)),
            )
        }
        None => (None, None),
    };

    let tensor_constraints = LinProgTensorConstraints {
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        lower_bounds,
        upper_bounds,
    };

    let result = client
        .linprog(&c_tensor, &tensor_constraints, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    // Handle empty slack tensor (when no inequality constraints)
    let slack = if result.slack.numel() == 0 {
        Vec::new()
    } else {
        result.slack.to_vec()
    };

    Ok(LinProgResult {
        x: result.x.to_vec(),
        fun: result.fun,
        success: result.success,
        nit: result.nit,
        message: result.message,
        slack,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linprog_simple() {
        let c = vec![-1.0, -2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 2.0, 3.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - (-7.0)).abs() < 0.1);
    }

    #[test]
    fn test_linprog_with_equality() {
        let c = vec![1.0, 1.0];
        let constraints = LinearConstraints {
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![2.0]),
            bounds: Some(vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default()).expect("linprog failed");
        assert!(result.success);
        assert!((result.fun - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_linprog_empty_objective() {
        let result = linprog(&[], &LinearConstraints::default(), &LinProgOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let constraints = LinearConstraints {
            a_ub: Some(vec![vec![1.0]]),
            b_ub: Some(vec![1.0]),
            ..Default::default()
        };

        let result = linprog(&c, &constraints, &LinProgOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }
}
