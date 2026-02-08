//! Exponential distribution.

use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Exponential distribution.
///
/// The exponential distribution with rate parameter λ has PDF:
///
/// f(x) = λ exp(-λx)  for x ≥ 0
///
/// Alternatively parameterized by scale β = 1/λ:
///
/// f(x) = (1/β) exp(-x/β)
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Exponential, ContinuousDistribution, Distribution};
///
/// // Rate = 2 (mean = 0.5)
/// let e = Exponential::new(2.0).unwrap();
/// assert!((e.mean() - 0.5).abs() < 1e-10);
///
/// // From scale parameter
/// let e = Exponential::from_scale(0.5).unwrap();
/// assert!((e.rate() - 2.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    /// Rate parameter (λ)
    lambda: f64,
}

impl Exponential {
    /// Create a new exponential distribution with given rate parameter.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if lambda is not positive.
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if lambda <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "lambda".to_string(),
                value: lambda,
                reason: "must be positive".to_string(),
            });
        }
        if !lambda.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "lambda".to_string(),
                value: lambda,
                reason: "must be finite".to_string(),
            });
        }
        Ok(Self { lambda })
    }

    /// Create an exponential distribution from scale parameter β = 1/λ.
    pub fn from_scale(scale: f64) -> StatsResult<Self> {
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "must be positive".to_string(),
            });
        }
        Self::new(1.0 / scale)
    }

    /// Create a standard exponential distribution (λ = 1).
    pub fn standard() -> Self {
        Self { lambda: 1.0 }
    }

    /// Get the rate parameter λ.
    pub fn rate(&self) -> f64 {
        self.lambda
    }

    /// Get the scale parameter β = 1/λ.
    pub fn scale(&self) -> f64 {
        1.0 / self.lambda
    }
}

impl Distribution for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn var(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }

    fn entropy(&self) -> f64 {
        1.0 - self.lambda.ln()
    }

    fn median(&self) -> f64 {
        2.0_f64.ln() / self.lambda
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn kurtosis(&self) -> f64 {
        6.0 // Excess kurtosis
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.lambda.ln() - self.lambda * x
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            // Use -expm1(-λx) for numerical stability when x is small
            -(-self.lambda * x).exp_m1()
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-self.lambda * x).exp()
        }
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // x = -ln(1-p) / λ = -ln1p(-p) / λ
        Ok(-(-p).ln_1p() / self.lambda)
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(f64::INFINITY);
        }
        if p == 1.0 {
            return Ok(0.0);
        }
        // x = -ln(p) / λ
        Ok(-p.ln() / self.lambda)
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // f(x) = λ * exp(-λ*x)
        let neg_lambda_x = client.mul_scalar(x, -self.lambda)?;
        let exp_term = client.exp(&neg_lambda_x)?;
        client.mul_scalar(&exp_term, self.lambda)
    }

    fn log_pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // log(f(x)) = ln(λ) - λ*x
        let lambda_x = client.mul_scalar(x, self.lambda)?;
        client.sub_scalar(&lambda_x, -self.lambda.ln())
    }

    fn cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // CDF(x) = 1 - exp(-λ*x)
        let neg_lambda_x = client.mul_scalar(x, -self.lambda)?;
        let exp_term = client.exp(&neg_lambda_x)?;
        client.rsub_scalar(&exp_term, 1.0)
    }

    fn sf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // SF(x) = exp(-λ*x)
        let neg_lambda_x = client.mul_scalar(x, -self.lambda)?;
        client.exp(&neg_lambda_x)
    }

    fn log_cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // log(CDF(x)) = log(1 - exp(-λ*x))
        let cdf = self.cdf_tensor(x, client)?;
        client.log(&cdf)
    }

    fn ppf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PPF(p) = -ln(1-p) / λ
        let one_minus_p = client.rsub_scalar(p, 1.0)?;
        let ln_term = client.log(&one_minus_p)?;
        let neg_ln = client.mul_scalar(&ln_term, -1.0)?;
        client.mul_scalar(&neg_ln, 1.0 / self.lambda)
    }

    fn isf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // ISF(p) = -ln(p) / λ
        let ln_p = client.log(p)?;
        let neg_ln = client.mul_scalar(&ln_p, -1.0)?;
        client.mul_scalar(&neg_ln, 1.0 / self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_creation() {
        let e = Exponential::new(2.0).unwrap();
        assert!((e.rate() - 2.0).abs() < 1e-10);
        assert!((e.scale() - 0.5).abs() < 1e-10);

        let e = Exponential::from_scale(0.5).unwrap();
        assert!((e.rate() - 2.0).abs() < 1e-10);

        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-1.0).is_err());
        assert!(Exponential::from_scale(0.0).is_err());
    }

    #[test]
    fn test_exponential_pdf() {
        let e = Exponential::new(2.0).unwrap();

        // PDF at 0 should be λ
        assert!((e.pdf(0.0) - 2.0).abs() < 1e-10);

        // PDF at x = ln(2)/λ should be λ/2
        let x = 2.0_f64.ln() / 2.0;
        assert!((e.pdf(x) - 1.0).abs() < 1e-10);

        // PDF at negative x is 0
        assert!((e.pdf(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_cdf() {
        let e = Exponential::new(2.0).unwrap();

        assert!((e.cdf(0.0) - 0.0).abs() < 1e-10);

        // CDF at median should be 0.5
        let median = 2.0_f64.ln() / 2.0;
        assert!((e.cdf(median) - 0.5).abs() < 1e-10);

        // CDF at x < 0 is 0
        assert!((e.cdf(-1.0) - 0.0).abs() < 1e-10);

        // Check specific value
        assert!((e.cdf(1.0) - (1.0 - (-2.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_ppf() {
        let e = Exponential::new(2.0).unwrap();

        assert!((e.ppf(0.0).unwrap() - 0.0).abs() < 1e-10);

        // PPF(0.5) should give median
        let expected_median = 2.0_f64.ln() / 2.0;
        assert!((e.ppf(0.5).unwrap() - expected_median).abs() < 1e-10);

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = e.ppf(p).unwrap();
            assert!((e.cdf(x) - p).abs() < 1e-10, "Failed for p={}", p);
        }
    }

    #[test]
    fn test_exponential_moments() {
        let e = Exponential::new(2.0).unwrap();

        assert!((e.mean() - 0.5).abs() < 1e-10);
        assert!((e.var() - 0.25).abs() < 1e-10);
        assert!((e.std() - 0.5).abs() < 1e-10);
        assert!((e.mode() - 0.0).abs() < 1e-10);
        assert!((e.skewness() - 2.0).abs() < 1e-10);
        assert!((e.kurtosis() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_entropy() {
        let e = Exponential::new(2.0).unwrap();
        // H = 1 - ln(λ) = 1 - ln(2)
        assert!((e.entropy() - (1.0 - 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_sf() {
        let e = Exponential::new(2.0).unwrap();

        assert!((e.sf(0.0) - 1.0).abs() < 1e-10);
        assert!((e.sf(e.median()) - 0.5).abs() < 1e-10);

        // SF + CDF = 1
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((e.sf(x) + e.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exponential_isf() {
        let e = Exponential::new(2.0).unwrap();

        // ISF(0.5) should give median
        assert!((e.isf(0.5).unwrap() - e.median()).abs() < 1e-10);

        // ISF(p) = PPF(1-p)
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let isf = e.isf(p).unwrap();
            let ppf = e.ppf(1.0 - p).unwrap();
            assert!((isf - ppf).abs() < 1e-10);
        }
    }
}
