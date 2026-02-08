//! Laplace (double exponential) distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Laplace (double exponential) distribution.
///
/// The Laplace distribution is a continuous probability distribution with PDF:
///
/// f(x; μ, b) = 1/(2b) * exp(-|x - μ|/b)
///
/// where:
/// - μ is the location parameter (mean, median, mode)
/// - b > 0 is the scale parameter
///
/// It can be thought of as two exponential distributions spliced at the mean,
/// or as the distribution of the difference of two i.i.d. exponential random variables.
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Laplace, ContinuousDistribution};
///
/// let l = Laplace::new(0.0, 1.0).unwrap();  // standard Laplace
/// println!("PDF at 0: {}", l.pdf(0.0));     // 0.5
/// println!("CDF at 0: {}", l.cdf(0.0));     // 0.5
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Laplace {
    /// Location parameter (mean)
    loc: f64,
    /// Scale parameter
    scale: f64,
}

impl Laplace {
    /// Create a new Laplace distribution.
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (mean, median, mode)
    /// * `scale` - Scale parameter (must be > 0)
    pub fn new(loc: f64, scale: f64) -> StatsResult<Self> {
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "scale parameter must be positive".to_string(),
            });
        }
        Ok(Self { loc, scale })
    }

    /// Create the standard Laplace distribution (loc=0, scale=1).
    pub fn standard() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }

    /// Get the location parameter.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution for Laplace {
    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        2.0 * self.scale * self.scale
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(2be) = 1 + ln(2b)
        1.0 + (2.0 * self.scale).ln()
    }

    fn median(&self) -> f64 {
        self.loc
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        // Excess kurtosis is 3
        3.0
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        (-z).exp() / (2.0 * self.scale)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        -z - (2.0 * self.scale).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            1.0 - 0.5 * z.exp()
        } else {
            0.5 * (-z).exp()
        }
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }

        let x = if p <= 0.5 {
            self.loc + self.scale * (2.0 * p).ln()
        } else {
            self.loc - self.scale * (2.0 * (1.0 - p)).ln()
        };
        Ok(x)
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // f(x) = 1/(2b) * exp(-|x - μ|/b)
        let centered = client.sub_scalar(x, self.loc)?;
        let abs_centered = client.abs(&centered)?;
        let z = client.mul_scalar(&abs_centered, -1.0 / self.scale)?;
        let exp_term = client.exp(&z)?;
        client.mul_scalar(&exp_term, 1.0 / (2.0 * self.scale))
    }

    fn log_pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // log(f(x)) = -|x - μ|/b - ln(2b)
        let centered = client.sub_scalar(x, self.loc)?;
        let abs_centered = client.abs(&centered)?;
        let z = client.mul_scalar(&abs_centered, -1.0 / self.scale)?;
        let constant = -(2.0 * self.scale).ln();
        client.add_scalar(&z, constant)
    }

    fn cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // CDF(x) = 0.5 * exp(z) if z < 0, else 1 - 0.5*exp(-z) where z = (x-μ)/b
        let centered = client.sub_scalar(x, self.loc)?;
        let z = client.mul_scalar(&centered, 1.0 / self.scale)?;

        // For positive z: 1 - 0.5*exp(-z) = 0.5 + 0.5*(1 - exp(-z))
        // For negative z: 0.5*exp(z)
        // Approximate: 0.5 + 0.5*sign(z)*(1 - exp(-|z|))
        // = 0.5 + 0.5*(2*sigmoid(2*z) - 1) where sigmoid(t) = 1/(1+exp(-t))
        // For now compute: 0.5 - 0.5*exp(z) for z < 0, and 1 - 0.5*exp(-z) for z >= 0
        // Simpler formula: CDF(x) = 0.5*(1 + sign(z)*(-1 + 2*exp(min(z,0))))
        // = 0.5 + 0.5*sign(z) - 0.5*exp(z) for z < 0
        // = 1 - 0.5*exp(-z) for z >= 0
        // We can use: if z < 0 then 0.5*exp(z) else 1 - 0.5*exp(-z)
        // = 0.5*exp(min(z, -z)) if z < 0 else with adjusted sign
        // Let's just compute as: where z < 0: 0.5*exp(z); where z >= 0: 1 - 0.5*exp(-z)
        // For tensor computation without branching, use smooth approximation or compute both

        // For z < 0: 0.5*exp(z), For z > 0: 1 - 0.5*exp(-z)
        // Blend using: 0.5*exp(z) + 0.5*(1 - exp(-z))*heaviside(z)
        // Where heaviside(z) ≈ sigmoid(kz) for large k
        // Use the closed-form for both cases:
        // For z < 0: 0.5*exp(z), for z >= 0: 1 - 0.5*exp(-z)
        // We can unify as: 0.5 + 0.5*sign(z)*(1 - exp(-|z|))
        // which equals: 0.5*(1 + sign(z)) - 0.5*sign(z)*exp(-|z|)
        // Simplified: we compute both branches and blend with sign

        let abs_z = client.abs(&z)?;
        let neg_abs_z = client.mul_scalar(&abs_z, -1.0)?;
        let exp_neg_abs = client.exp(&neg_abs_z)?;
        let sign_z = client.sign(&z)?;

        // result = 0.5 + 0.5*sign(z) - sign(z)*0.5*exp(-|z|)
        //        = 0.5 + 0.5*sign(z)*(1 - exp(-|z|))
        let one_minus_exp = client.add_scalar(&client.mul_scalar(&exp_neg_abs, -1.0)?, 1.0)?;
        let signed_term = client.mul(&sign_z, &one_minus_exp)?;
        let half_signed = client.mul_scalar(&signed_term, 0.5)?;
        client.add_scalar(&half_signed, 0.5)
    }

    fn sf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // SF(x) = 1 - CDF(x) = 0.5*exp(|x-μ|/b) - 0.5 = 0.5*(exp(-z) - 1) for z < 0, 0.5*exp(-z) for z > 0
        // where z = (x-μ)/b = 0.5*exp(-|z|)
        let centered = client.sub_scalar(x, self.loc)?;
        let z = client.mul_scalar(&centered, 1.0 / self.scale)?;
        let abs_z = client.abs(&z)?;
        let exp_neg_abs = client.exp(&client.mul_scalar(&abs_z, -1.0)?)?;
        client.mul_scalar(&exp_neg_abs, 0.5)
    }

    fn log_cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // log(CDF) = log(cdf(x))
        let cdf = self.cdf_tensor(x, client)?;
        client.log(&cdf)
    }

    fn ppf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PPF(p) = μ + b*ln(2p) if p <= 0.5, else μ - b*ln(2(1-p))
        // Unified formula: PPF(p) = μ - b*sign(p - 0.5)*ln(2*min(p, 1-p))

        // Compute p - 0.5
        let p_minus_half = client.add_scalar(p, -0.5)?;
        let sign_p = client.sign(&p_minus_half)?;

        // Compute 1 - p
        let neg_p = client.mul_scalar(p, -1.0)?;
        let one_minus_p = client.add_scalar(&neg_p, 1.0)?;

        // Compute min(p, 1-p)
        let min_p = client.minimum(p, &one_minus_p)?;

        // Compute 2*min(p, 1-p) and its log
        let two_min_p = client.mul_scalar(&min_p, 2.0)?;
        let ln_two_min_p = client.log(&two_min_p)?;

        // result = μ - b*sign(p - 0.5)*ln(2*min(p, 1-p))
        let signed_ln = client.mul(&sign_p, &ln_two_min_p)?;
        let scaled = client.mul_scalar(&signed_ln, self.scale)?;
        let neg_scaled = client.mul_scalar(&scaled, -1.0)?;
        client.add_scalar(&neg_scaled, self.loc)
    }

    fn isf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // ISF(p) = PPF(1 - p)
        let one_minus_p = client.rsub_scalar(p, 1.0)?;
        self.ppf_tensor(&one_minus_p, client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_creation() {
        assert!(Laplace::new(0.0, 1.0).is_ok());
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Laplace::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_laplace_pdf() {
        let l = Laplace::standard();

        // PDF at 0 is 0.5
        assert!((l.pdf(0.0) - 0.5).abs() < 1e-10);

        // Symmetric
        assert!((l.pdf(1.0) - l.pdf(-1.0)).abs() < 1e-10);

        // PDF at 1 is 0.5 * exp(-1) ≈ 0.184
        assert!((l.pdf(1.0) - 0.5 * (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_cdf() {
        let l = Laplace::standard();

        // CDF at median is 0.5
        assert!((l.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF at 0 from both sides
        assert!((l.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF(-1) = 0.5 * exp(-1) ≈ 0.184
        assert!((l.cdf(-1.0) - 0.5 * (-1.0_f64).exp()).abs() < 1e-10);

        // CDF(1) = 1 - 0.5 * exp(-1) ≈ 0.816
        assert!((l.cdf(1.0) - (1.0 - 0.5 * (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_ppf() {
        let l = Laplace::standard();

        // PPF(0.5) = 0 (median)
        assert!((l.ppf(0.5).unwrap() - 0.0).abs() < 1e-10);

        // Round-trip
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = l.cdf(x);
            assert!((l.ppf(p).unwrap() - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplace_moments() {
        let l = Laplace::new(5.0, 2.0).unwrap();

        assert!((l.mean() - 5.0).abs() < 1e-10);
        assert!((l.var() - 8.0).abs() < 1e-10); // 2 * 2^2 = 8
        assert!((l.skewness() - 0.0).abs() < 1e-10);
        assert!((l.kurtosis() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_sf() {
        let l = Laplace::standard();

        // SF + CDF = 1
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!((l.sf(x) + l.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplace_entropy() {
        let l = Laplace::standard();

        // Entropy of standard Laplace is 1 + ln(2)
        assert!((l.entropy() - (1.0 + 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_median_mode() {
        let l = Laplace::new(3.0, 2.0).unwrap();

        assert!((l.median() - 3.0).abs() < 1e-10);
        assert!((l.mode() - 3.0).abs() < 1e-10);
    }
}
