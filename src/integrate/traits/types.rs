//! Types for integration algorithms.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of subdivisions (default: 50)
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            limit: 50,
        }
    }
}

/// Result of adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadResult<R: Runtime> {
    /// Computed integral value (0-D tensor)
    pub integral: Tensor<R>,
    /// Estimated absolute error
    pub error: f64,
    /// Number of function evaluations
    pub neval: usize,
    /// Whether integration converged
    pub converged: bool,
}

/// Options for Romberg integration.
#[derive(Debug, Clone)]
pub struct RombergOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of extrapolation levels (default: 20)
    pub max_levels: usize,
}

impl Default for RombergOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            max_levels: 20,
        }
    }
}
