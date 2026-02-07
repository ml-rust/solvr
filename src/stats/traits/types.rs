//! Types for statistical algorithms.

use numr::dtype::DType;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

/// Result of a statistical hypothesis test.
#[derive(Debug, Clone)]
pub struct TensorTestResult<R: Runtime> {
    /// Test statistic value (scalar tensor)
    pub statistic: Tensor<R>,
    /// P-value: probability of obtaining result at least as extreme (scalar tensor)
    pub pvalue: Tensor<R>,
}

/// Result of robust regression (Theil-Sen or Siegel slopes).
#[derive(Debug, Clone)]
pub struct RobustRegressionResult<R: Runtime> {
    /// Slope estimate (scalar tensor)
    pub slope: Tensor<R>,
    /// Intercept estimate (scalar tensor)
    pub intercept: Tensor<R>,
    /// Lower confidence interval bound for slope (scalar tensor)
    pub low_slope: Tensor<R>,
    /// Upper confidence interval bound for slope (scalar tensor)
    pub high_slope: Tensor<R>,
}

/// Method for computing the center in Levene's test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeveneCenter {
    /// Use the mean (classical Levene's test)
    Mean,
    /// Use the median (Brown-Forsythe test, more robust)
    Median,
    /// Use the 10% trimmed mean
    TrimmedMean,
}

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
