//! Hypothesis testing algorithms.

use super::{LeveneCenter, TensorTestResult};
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Hypothesis testing algorithms for tensors.
///
/// Provides methods for conducting statistical hypothesis tests on tensor data,
/// including t-tests, correlation tests, ANOVA, non-parametric tests, and normality tests.
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

    /// One-way ANOVA (F-test).
    ///
    /// Tests whether two or more groups have the same population mean.
    /// Returns F-statistic and p-value.
    fn f_oneway(&self, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>;

    /// Kruskal-Wallis H-test (non-parametric one-way ANOVA).
    ///
    /// Tests whether samples from two or more groups come from the same distribution.
    /// Uses ranks instead of raw values. P-value from chi-squared distribution.
    fn kruskal(&self, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>;

    /// Friedman chi-squared test for repeated measures.
    ///
    /// Non-parametric test for differences among repeated measurements.
    /// Each row is a subject, each column is a treatment.
    ///
    /// # Arguments
    ///
    /// * `groups` - Slice of tensors, one per treatment (all same length = number of subjects)
    fn friedmanchisquare(&self, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>;

    /// Shapiro-Wilk test for normality.
    ///
    /// Tests whether a sample comes from a normal distribution.
    /// Best for small to moderate sample sizes (n <= 5000).
    fn shapiro(&self, x: &Tensor<R>) -> Result<TensorTestResult<R>>;

    /// D'Agostino-Pearson omnibus test for normality.
    ///
    /// Tests normality based on skewness and kurtosis.
    /// Requires n >= 20.
    fn normaltest(&self, x: &Tensor<R>) -> Result<TensorTestResult<R>>;

    /// Levene's test for equality of variances.
    ///
    /// Tests whether two or more groups have equal variances.
    /// The `center` parameter controls robustness:
    /// - `Mean`: classical Levene's test
    /// - `Median`: Brown-Forsythe test (recommended, more robust)
    /// - `TrimmedMean`: uses 10% trimmed mean
    fn levene(&self, groups: &[&Tensor<R>], center: LeveneCenter) -> Result<TensorTestResult<R>>;

    /// Bartlett's test for equality of variances.
    ///
    /// Tests whether two or more groups have equal variances.
    /// Assumes normal data; use Levene's test for non-normal data.
    fn bartlett(&self, groups: &[&Tensor<R>]) -> Result<TensorTestResult<R>>;
}
