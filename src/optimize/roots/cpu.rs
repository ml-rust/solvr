//! CPU implementation of root finding algorithms.
//!
//! This module implements the [`RootFindingAlgorithms`] trait for CPU
//! by delegating to the generic implementations in `impl_generic/roots/`.

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::{broyden1_impl, levenberg_marquardt_impl, newton_system_impl};
use crate::optimize::roots::{
    MultiRootResult, RootFindingAlgorithms, RootOptions, RootTensorResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

impl RootFindingAlgorithms<CpuRuntime> for CpuClient {
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = newton_system_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "newton_system", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }

    fn broyden1<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = broyden1_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "broyden1", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }

    fn levenberg_marquardt<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = levenberg_marquardt_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "levenberg_marquardt", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }
}

// ============================================================================
// Convenience Functions (Scalar API)
// ============================================================================

/// Newton's method for systems of nonlinear equations.
///
/// This is a convenience function that wraps the tensor-based implementation.
pub fn newton_system<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "newton_system: empty initial guess".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
        let x_data: Vec<f64> = x.to_vec();
        let result = f(&x_data);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &result,
            &[result.len()],
            &device,
        ))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);
    let result = client
        .newton_system(tensor_f, &x0_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(MultiRootResult {
        x: result.x.to_vec(),
        fun: result.fun.to_vec(),
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        converged: result.converged,
    })
}

/// Broyden's method (rank-1 update) for systems of nonlinear equations.
///
/// This is a convenience function that wraps the tensor-based implementation.
pub fn broyden1<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "broyden1: empty initial guess".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
        let x_data: Vec<f64> = x.to_vec();
        let result = f(&x_data);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &result,
            &[result.len()],
            &device,
        ))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);
    let result = client
        .broyden1(tensor_f, &x0_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(MultiRootResult {
        x: result.x.to_vec(),
        fun: result.fun.to_vec(),
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        converged: result.converged,
    })
}

/// Levenberg-Marquardt algorithm for systems of nonlinear equations.
///
/// This is a convenience function that wraps the tensor-based implementation.
pub fn levenberg_marquardt<F>(
    f: F,
    x0: &[f64],
    options: &RootOptions,
) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "levenberg_marquardt: empty initial guess".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
        let x_data: Vec<f64> = x.to_vec();
        let result = f(&x_data);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &result,
            &[result.len()],
            &device,
        ))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);
    let result = client
        .levenberg_marquardt(tensor_f, &x0_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(MultiRootResult {
        x: result.x.to_vec(),
        fun: result.fun.to_vec(),
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        converged: result.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_system_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result =
            newton_system(f, &[0.0, 0.0], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_quadratic() {
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];

        let result =
            newton_system(f, &[0.5, 0.5], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        let expected = 1.0 / (2.0_f64).sqrt();
        assert!((result.x[0] - expected).abs() < 1e-6);
        assert!((result.x[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_empty_input() {
        let f = |_: &[f64]| vec![];
        let result = newton_system(f, &[], &RootOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_broyden1_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = broyden1(f, &[0.0, 0.0], &RootOptions::default()).expect("broyden1 failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_levenberg_marquardt_linear() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = levenberg_marquardt(f, &[0.0, 0.0], &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compare_methods() {
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1]];

        let newton_result =
            newton_system(f, &[1.0, 1.0], &RootOptions::default()).expect("newton failed");
        let broyden_result =
            broyden1(f, &[1.0, 1.0], &RootOptions::default()).expect("broyden failed");
        let lm_result =
            levenberg_marquardt(f, &[1.0, 1.0], &RootOptions::default()).expect("lm failed");

        let expected = (2.0_f64).sqrt();

        assert!(newton_result.converged);
        assert!((newton_result.x[0] - expected).abs() < 1e-5);

        assert!(broyden_result.converged);
        assert!((broyden_result.x[0] - expected).abs() < 1e-5);

        assert!(lm_result.converged);
        assert!((lm_result.x[0] - expected).abs() < 1e-4);
    }
}
