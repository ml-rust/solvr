//! Hypothesis testing algorithms.

use super::TensorTestResult;
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Hypothesis testing algorithms for tensors.
///
/// Provides methods for conducting statistical hypothesis tests on tensor data,
/// including t-tests and correlation tests.
pub trait HypothesisTestingAlgorithms<R: Runtime>: TensorOps<R> {
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
}
