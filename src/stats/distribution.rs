//! Distribution traits defining the common interface.
//!
//! All distributions support both scalar and batch (tensor) operations.
//! The tensor methods work on all numr backends (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! Tensor methods use numr's tensor operations directly - data stays on device.
//! Each distribution implements its tensor methods using the appropriate numr ops:
//! - `TensorOps<R>` for basic operations (exp, log, sqrt, etc.)
//! - `ScalarOps<R>` for scalar-tensor operations (add_scalar, mul_scalar, etc.)
//! - `SpecialFunctions<R>` for special functions (erf, gamma, betainc, etc.)
//!
//! This ensures GPU tensors stay on GPU throughout computation - no transfers.
use crate::DType;

use crate::stats::StatsResult;
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Common interface for all probability distributions.
pub trait Distribution {
    /// Get the mean of the distribution.
    fn mean(&self) -> f64;

    /// Get the variance of the distribution.
    fn var(&self) -> f64;

    /// Get the standard deviation of the distribution.
    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Get the Shannon entropy of the distribution (in nats).
    fn entropy(&self) -> f64;

    /// Get the median of the distribution.
    fn median(&self) -> f64;

    /// Get the mode of the distribution.
    fn mode(&self) -> f64;

    /// Get the skewness of the distribution.
    fn skewness(&self) -> f64;

    /// Get the excess kurtosis of the distribution.
    fn kurtosis(&self) -> f64;
}

/// Interface for continuous probability distributions.
///
/// Provides both scalar methods (for single values) and tensor methods
/// (for batch GPU-accelerated computation).
///
/// # Scalar Methods
/// - `pdf(x)` - Probability density at x
/// - `cdf(x)` - Cumulative probability P(X ≤ x)
/// - `ppf(p)` - Quantile function (inverse CDF)
///
/// # Tensor Methods (Batch)
/// - `pdf_tensor(&x, &client)` - PDF for all elements in tensor
/// - `cdf_tensor(&x, &client)` - CDF for all elements in tensor
/// - `ppf_tensor(&p, &client)` - PPF for all elements in tensor
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use solvr::stats::{Normal, ContinuousDistribution};
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
/// use numr::tensor::Tensor;
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let n = Normal::standard();
///
/// // Scalar
/// let p = n.pdf(0.0);
///
/// // Batch (tensor)
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3], &device)?;
/// let p_batch = n.pdf_tensor(&x, &client)?;
/// # Ok(())
/// # }
/// ```
pub trait ContinuousDistribution: Distribution {
    /// Probability density function.
    ///
    /// Returns the probability density at point `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Log of the probability density function.
    ///
    /// More numerically stable than `pdf(x).ln()` for extreme values.
    fn log_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }

    /// Cumulative distribution function.
    ///
    /// Returns P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;

    /// Survival function.
    ///
    /// Returns P(X > x) = 1 - CDF(x).
    /// More accurate than `1.0 - cdf(x)` for values close to 1.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Log of the cumulative distribution function.
    fn log_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// Percent point function (quantile function / inverse CDF).
    ///
    /// Returns the value x such that P(X ≤ x) = p.
    fn ppf(&self, p: f64) -> StatsResult<f64>;

    /// Inverse survival function.
    ///
    /// Returns the value x such that P(X > x) = p.
    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }

    /// Interval containing a given probability mass.
    ///
    /// Returns (a, b) such that P(a ≤ X ≤ b) = alpha, centered around the median.
    fn interval(&self, alpha: f64) -> StatsResult<(f64, f64)> {
        let q = (1.0 - alpha) / 2.0;
        let lower = self.ppf(q)?;
        let upper = self.ppf(1.0 - q)?;
        Ok((lower, upper))
    }

    // ========================================================================
    // Tensor (Batch) Methods
    //
    // These methods use numr tensor operations directly. Data stays on device -
    // no GPU→CPU→GPU transfers. Each distribution must implement these methods
    // using the appropriate numr ops for that distribution's formulas.
    // ========================================================================

    /// Batch probability density function.
    ///
    /// Computes PDF for all elements in the input tensor.
    /// Works on all backends (CPU, CUDA, WebGPU) with full acceleration.
    ///
    /// # Implementation
    ///
    /// Each distribution implements this using numr tensor ops. For example,
    /// Normal uses `exp`, `mul_scalar`, `sub_scalar` to compute the PDF formula.
    fn pdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>;

    /// Batch log probability density function.
    ///
    /// More numerically stable than `log(pdf_tensor(x))` for extreme values.
    fn log_pdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>;

    /// Batch cumulative distribution function.
    ///
    /// Computes CDF for all elements in the input tensor.
    fn cdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch survival function.
    ///
    /// Computes SF = 1 - CDF for all elements.
    fn sf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch log cumulative distribution function.
    fn log_cdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch percent point function (quantile function / inverse CDF).
    ///
    /// Computes PPF for all probability values in the input tensor.
    /// Values should be in [0, 1].
    fn ppf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        p: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch inverse survival function.
    ///
    /// Returns x such that SF(x) = p for all elements.
    fn isf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        p: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;
}

/// Interface for discrete probability distributions.
///
/// Provides both scalar methods (for single values) and tensor methods
/// (for batch GPU-accelerated computation).
///
/// # Scalar Methods
/// - `pmf(k)` - Probability mass P(X = k)
/// - `cdf(k)` - Cumulative probability P(X ≤ k)
/// - `ppf(p)` - Quantile function (smallest k with CDF(k) ≥ p)
///
/// # Tensor Methods (Batch)
/// - `pmf_tensor(&k, &client)` - PMF for all elements in tensor
/// - `cdf_tensor(&k, &client)` - CDF for all elements in tensor
/// - `ppf_tensor(&p, &client)` - PPF for all elements in tensor
pub trait DiscreteDistribution: Distribution {
    /// Probability mass function.
    ///
    /// Returns P(X = k).
    fn pmf(&self, k: u64) -> f64;

    /// Log of the probability mass function.
    fn log_pmf(&self, k: u64) -> f64 {
        self.pmf(k).ln()
    }

    /// Cumulative distribution function.
    ///
    /// Returns P(X ≤ k).
    fn cdf(&self, k: u64) -> f64;

    /// Survival function.
    ///
    /// Returns P(X > k) = 1 - CDF(k).
    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }

    /// Percent point function (quantile function).
    ///
    /// Returns the smallest k such that P(X ≤ k) ≥ p.
    fn ppf(&self, p: f64) -> StatsResult<u64>;

    /// Inverse survival function.
    fn isf(&self, p: f64) -> StatsResult<u64> {
        self.ppf(1.0 - p)
    }

    // ========================================================================
    // Tensor (Batch) Methods
    //
    // Discrete distributions are inherently harder to vectorize since they
    // often involve integer-based branching. The tensor methods here use
    // numr ops where possible, falling back to scalar loops for complex cases.
    // ========================================================================

    /// Batch probability mass function.
    ///
    /// Computes PMF for all elements in the input tensor.
    /// The tensor should contain non-negative integer values (stored as floats).
    fn pmf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        k: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch log probability mass function.
    fn log_pmf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        k: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch cumulative distribution function.
    ///
    /// Computes CDF for all elements in the input tensor.
    fn cdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        k: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch survival function.
    fn sf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        k: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;

    /// Batch percent point function (quantile function).
    ///
    /// Computes PPF for all probability values in the input tensor.
    /// Returns the smallest k such that CDF(k) >= p.
    fn ppf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        p: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_trait_bounds() {
        // This test verifies the trait definition compiles correctly.
        // Actual distribution tests are in the specific distribution modules.
        fn _check_continuous_bounds<D: ContinuousDistribution>(d: &D, x: f64) -> f64 {
            d.pdf(x) + d.cdf(x) + d.sf(x)
        }

        fn _check_discrete_bounds<D: DiscreteDistribution>(d: &D, k: u64) -> f64 {
            d.pmf(k) + d.cdf(k) + d.sf(k)
        }
    }
}
