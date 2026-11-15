//! Shared multivariate root options.

use super::root_result::TensorRootResult;

/// Alias for TensorRootResult for convenience
pub type RootTensorResult<R> = TensorRootResult<R>;

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct RootOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (norm of F(x))
    pub tol: f64,
    /// Tolerance for step size
    pub x_tol: f64,
    /// Step size for finite difference Jacobian approximation
    pub eps: f64,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            x_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = RootOptions::default();
        assert_eq!(opts.max_iter, 100);
        assert!((opts.tol - 1e-8).abs() < 1e-12);
    }
}
