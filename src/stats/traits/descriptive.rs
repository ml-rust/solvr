//! Descriptive statistics algorithms.

use crate::stats::TensorDescriptiveStats;
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Descriptive statistics algorithms for tensors.
///
/// Provides methods for computing comprehensive statistical summaries
/// of tensor data, including central tendency, dispersion, and shape measures.
pub trait DescriptiveStatisticsAlgorithms<R: Runtime>: TensorOps<R> {
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
        DescriptiveStatisticsAlgorithms::percentile(self, x, 50.0)
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
}
