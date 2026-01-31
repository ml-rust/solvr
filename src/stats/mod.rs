//! Statistical distributions, hypothesis tests, and descriptive statistics.
//!
//! This module provides a comprehensive statistics library with full GPU acceleration
//! via numr's multi-runtime architecture.
//!
//! # Runtime-Generic API
//!
//! All statistics algorithms implement the [`StatisticsAlgorithms`] trait, which is
//! generic over numr's `Runtime`. This means the same code works on CPU, CUDA, and WebGPU.
//!
//! ```ignore
//! use solvr::stats::StatisticsAlgorithms;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//! use numr::runtime::RuntimeClient;
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Create a tensor
//! let data = client.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &device).unwrap();
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
//!
//! # Hypothesis Tests
//!
//! ```ignore
//! use solvr::stats::StatisticsAlgorithms;
//!
//! let data = client.from_slice(&[1.2, 1.5, 1.3, 1.4, 1.6], &device).unwrap();
//! let result = client.ttest_1samp(&data, 1.0).unwrap();
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

// Core modules
mod continuous;
mod discrete;
mod distribution;
mod error;

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

// Public API: Result types

// ============================================================================
// TensorDescriptiveStats - Runtime-generic descriptive statistics result
// ============================================================================

/// Descriptive statistics computed from a tensor.
///
/// All numeric fields are returned as tensors to preserve dtype information
/// and enable further tensor operations on the results.
#[derive(Debug, Clone)]
pub struct TensorDescriptiveStats<R: Runtime> {
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
pub struct TensorTestResult<R: Runtime> {
    /// Test statistic value (scalar tensor)
    pub statistic: Tensor<R>,
    /// P-value: probability of obtaining result at least as extreme (scalar tensor)
    pub pvalue: Tensor<R>,
}

// ============================================================================
// StatisticsAlgorithms Trait
// ============================================================================

/// Runtime-generic statistical algorithms.
///
/// This trait extends numr's `TensorOps` with statistical functions that work on tensors.
/// All backends (CPU, CUDA, WebGPU) implement this trait, enabling GPU-accelerated
/// statistics with identical APIs.
///
/// # Example
///
/// ```ignore
/// use solvr::stats::StatisticsAlgorithms;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
///
/// let data = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
/// let stats = client.describe(&data).unwrap();
/// ```
pub trait StatisticsAlgorithms<R: Runtime>: TensorOps<R> {
    // ========================================================================
    // Descriptive Statistics
    // ========================================================================

    /// Compute comprehensive descriptive statistics for a 1D tensor.
    ///
    /// Returns min, max, mean, variance, std, skewness, and kurtosis as tensors.
    fn describe(&self, x: &Tensor<R>) -> Result<TensorDescriptiveStats<R>>;

    /// Compute the p-th percentile.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `p` - Percentile to compute (0-100)
    fn percentile(&self, x: &Tensor<R>, p: f64) -> Result<Tensor<R>>;

    /// Compute the median (50th percentile).
    fn median(&self, x: &Tensor<R>) -> Result<Tensor<R>> {
        StatisticsAlgorithms::percentile(self, x, 50.0)
    }

    /// Compute the interquartile range (Q3 - Q1).
    fn iqr(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute skewness (Fisher's definition).
    fn skewness(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute excess kurtosis (Fisher's definition).
    fn kurtosis(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute z-scores (standardized values).
    fn zscore(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute standard error of the mean.
    fn sem(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Hypothesis Tests
    // ========================================================================

    /// One-sample t-test.
    ///
    /// Tests whether the mean of a sample differs from a specified value.
    fn ttest_1samp(&self, x: &Tensor<R>, popmean: f64) -> Result<TensorTestResult<R>>;

    /// Independent two-sample t-test (Welch's t-test).
    ///
    /// Tests whether two independent samples have different means.
    /// Uses Welch's correction for unequal variances.
    fn ttest_ind(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<TensorTestResult<R>>;

    /// Paired t-test.
    ///
    /// Tests whether the mean difference between paired samples is zero.
    fn ttest_rel(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<TensorTestResult<R>>;

    /// Pearson correlation coefficient.
    ///
    /// Returns correlation coefficient and p-value for testing non-correlation.
    fn pearsonr(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<TensorTestResult<R>>;

    /// Spearman rank correlation coefficient.
    ///
    /// Computes Pearson correlation on ranked data.
    fn spearmanr(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<TensorTestResult<R>>;

    // ========================================================================
    // Regression
    // ========================================================================

    /// Simple linear regression.
    ///
    /// Fits y = slope * x + intercept using ordinary least squares.
    fn linregress(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<LinregressResult>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that dtype is suitable for statistics operations.
pub fn validate_stats_dtype(dtype: DType) -> Result<()> {
    use numr::error::Error;

    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "statistics (requires F32 or F64)",
        }),
    }
}
