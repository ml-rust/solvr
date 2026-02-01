//! Statistical distributions, hypothesis tests, and descriptive statistics.
//!
//! This module provides a comprehensive statistics library with full GPU acceleration
//! via numr's multi-runtime architecture.
//!
//! # Runtime-Generic API
//!
//! Statistics algorithms are organized into three focused traits:
//! - [`DescriptiveStatisticsAlgorithms`] - Computing statistics (mean, variance, skewness, etc.)
//! - [`HypothesisTestingAlgorithms`] - Statistical hypothesis tests (t-tests, correlations)
//! - [`RegressionAlgorithms`] - Regression analysis (linear regression)
//!
//! All are generic over numr's `Runtime`, so the same code works on CPU, CUDA, and WebGPU.
//!
//! ```ignore
//! use solvr::stats::DescriptiveStatisticsAlgorithms;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Create a tensor
//! let data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], &device);
//!
//! // Compute descriptive statistics - works on any backend!
//! let stats = client.describe(&data).unwrap();
//! ```
//!
//! # Distributions
//!
//! Distributions have both scalar and batch (tensor) methods:
//!
//! ```ignore
//! use solvr::stats::{Normal, ContinuousDistribution};
//!
//! let n = Normal::standard();
//!
//! // Scalar - for single values
//! let p = n.pdf(0.0);
//!
//! // Batch - for tensor operations (GPU-accelerated)
//! let x = client.from_slice(&[0.0, 1.0, 2.0], &device).unwrap();
//! let p_batch = n.pdf_tensor(&x, &client).unwrap();
//! ```

// Backend implementations
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
mod wgpu;

// Shared generic implementations
mod helpers;
mod impl_generic;

// Traits
mod traits;

// Core modules
mod continuous;
mod discrete;
mod distribution;
mod error;

use numr::dtype::DType;
use numr::error::Result;
use numr::tensor::Tensor;

// Public API: Trait exports
pub use traits::{
    DescriptiveStatisticsAlgorithms, HypothesisTestingAlgorithms, RegressionAlgorithms,
};

// Public API: Distribution traits and types
pub use distribution::{ContinuousDistribution, DiscreteDistribution, Distribution};
pub use error::{StatsError, StatsResult};

// ============================================================================
// LinregressResult - Linear regression result
// ============================================================================

/// Result of simple linear regression.
#[derive(Debug, Clone, Copy)]
pub struct LinregressResult {
    /// Slope of the regression line
    pub slope: f64,
    /// Y-intercept of the regression line
    pub intercept: f64,
    /// Pearson correlation coefficient
    pub rvalue: f64,
    /// Two-sided p-value for hypothesis test (slope = 0)
    pub pvalue: f64,
    /// Standard error of the slope estimate
    pub stderr: f64,
    /// Standard error of the intercept estimate
    pub intercept_stderr: f64,
}

// Public API: Continuous distributions
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FDistribution, Gamma, Gumbel, GumbelMin, Laplace,
    LogNormal, Normal, Pareto, StudentT, Uniform, Weibull,
};

// Public API: Discrete distributions
pub use discrete::{
    Binomial, DiscreteUniform, Geometric, Hypergeometric, NegativeBinomial, Poisson,
};

// ============================================================================
// TensorDescriptiveStats - Runtime-generic descriptive statistics result
// ============================================================================

/// Descriptive statistics computed from a tensor.
///
/// All numeric fields are returned as tensors to preserve dtype information
/// and enable further tensor operations on the results.
#[derive(Debug, Clone)]
pub struct TensorDescriptiveStats<R: numr::runtime::Runtime> {
    /// Number of observations
    pub nobs: usize,
    /// Minimum value (scalar tensor)
    pub min: Tensor<R>,
    /// Maximum value (scalar tensor)
    pub max: Tensor<R>,
    /// Arithmetic mean (scalar tensor)
    pub mean: Tensor<R>,
    /// Variance with Bessel's correction (scalar tensor)
    pub variance: Tensor<R>,
    /// Standard deviation (scalar tensor)
    pub std: Tensor<R>,
    /// Skewness - Fisher's definition (scalar tensor)
    pub skewness: Tensor<R>,
    /// Excess kurtosis - Fisher's definition (scalar tensor)
    pub kurtosis: Tensor<R>,
}

// ============================================================================
// TensorTestResult - Runtime-generic hypothesis test result
// ============================================================================

/// Result of a statistical hypothesis test.
#[derive(Debug, Clone)]
pub struct TensorTestResult<R: numr::runtime::Runtime> {
    /// Test statistic value (scalar tensor)
    pub statistic: Tensor<R>,
    /// P-value: probability of obtaining result at least as extreme (scalar tensor)
    pub pvalue: Tensor<R>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that dtype is suitable for statistics operations.
pub fn validate_stats_dtype(dtype: numr::dtype::DType) -> Result<()> {
    use numr::error::Error;

    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "statistics (requires F32 or F64)",
        }),
    }
}
