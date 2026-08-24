//! Inverse Gamma distribution.
use crate::DType;

use super::special;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Inverse Gamma distribution.
///
/// The inverse gamma distribution with shape α and scale β has PDF:
///
/// f(x) = (β^α / Γ(α)) x^(-α-1) exp(-β/x)  for x > 0
///
/// If X ~ Gamma(α, β), then 1/X ~ InverseGamma(α, β).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use solvr::stats::{InverseGamma, ContinuousDistribution, Distribution};
///
/// // Shape = 2, scale = 1
/// let ig = InverseGamma::new(2.0, 1.0)?;
/// assert!((ig.mean() - 1.0).abs() < 1e-10);
///
/// // From shape and scale
/// let ig = InverseGamma::from_shape_scale(3.0, 2.0)?;
/// assert!((ig.mean() - 1.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct InverseGamma {
    /// Shape parameter (α)
    alpha: f64,
    /// Scale parameter (β)
    beta: f64,
    /// Log of normalizing constant: α*ln(β) - ln(Γ(α))
    log_norm: f64,
}

impl InverseGamma {
    /// Create a new inverse gamma distribution with shape α and scale β.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter (must be positive)
    /// * `beta` - Scale parameter (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are not positive.
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha,
                reason: "shape must be positive".to_string(),
            });
        }
        if beta <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "beta".to_string(),
                value: beta,
                reason: "scale must be positive".to_string(),
            });
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "alpha/beta".to_string(),
                value: alpha,
                reason: "parameters must be finite".to_string(),
            });
        }

        let log_norm = alpha * beta.ln() - special::lgamma(alpha);
        Ok(Self {
            alpha,
            beta,
            log_norm,
        })
    }

    /// Create an inverse gamma distribution from shape α and scale β.
    pub fn from_shape_scale(shape: f64, scale: f64) -> StatsResult<Self> {
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "must be positive".to_string(),
            });
        }
        Self::new(shape, scale)
    }

    /// Get the shape parameter α.
    pub fn shape(&self) -> f64 {
        self.alpha
    }

    /// Get the scale parameter β.
    pub fn scale(&self) -> f64 {
        self.beta
    }
}

impl Distribution for InverseGamma {
    fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn var(&self) -> f64 {
        if self.alpha > 2.0 {
            (self.beta * self.beta) / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        } else {
            f64::INFINITY
        }
    }

    fn entropy(&self) -> f64 {
        // H = α + ln(β * Γ(α)) - (1+α)*ψ(α)
        self.alpha + self.beta.ln() + special::lgamma(self.alpha)
            - (1.0 + self.alpha) * special::digamma(self.alpha)
    }

    fn median(&self) -> f64 {
        // No closed form, use PPF
        self.ppf(0.5).unwrap_or(self.mean())
    }

    fn mode(&self) -> f64 {
        // Mode = β / (α + 1)
        self.beta / (self.alpha + 1.0)
    }

    fn skewness(&self) -> f64 {
        if self.alpha > 3.0 {
            4.0 * ((self.alpha - 2.0) * (self.alpha - 3.0).sqrt()) / (self.alpha - 3.0)
        } else {
            f64::INFINITY
        }
    }

    fn kurtosis(&self) -> f64 {
        if self.alpha > 4.0 {
            (30.0 * self.alpha - 66.0) / ((self.alpha - 3.0) * (self.alpha - 4.0))
        } else {
            f64::INFINITY
        }
    }
}

impl ContinuousDistribution for InverseGamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // log(PDF) = α*ln(β) - ln(Γ(α)) - (α+1)*ln(x) - β/x
        self.log_norm - (self.alpha + 1.0) * x.ln() - self.beta / x
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            // CDF(x) = gammaincc(α, β/x) = 1 - gammainc(α, β/x)
            special::gammaincc(self.alpha, self.beta / x)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else {
            // SF(x) = gammainc(α, β/x)
            special::gammainc(self.alpha, self.beta / x)
        }
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(f64::INFINITY);
        }
        if p == 1.0 {
            return Ok(0.0);
        }
        // PPF(p) = β / gammaincinv(α, 1-p)
        Ok(self.beta / special::gammaincinv(self.alpha, 1.0 - p))
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // log(PDF) = log_norm - (α+1)*ln(x) - β/x
        // PDF = exp(log_pdf)
        self.log_pdf_tensor(x, client)
            .and_then(|log_pdf| client.exp(&log_pdf))
    }

    fn log_pdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // log(PDF) = log_norm - (α+1)*ln(x) - β/x
        let ln_x = client.log(x)?;
        let term1 = client.mul_scalar(&ln_x, -(self.alpha + 1.0))?;

        // Compute β/x = (1/x) * β
        let ones = Tensor::<R>::ones(x.shape(), x.dtype(), client.device())?;
        let recip_x = client.div(&ones, x)?;
        let term2 = client.mul_scalar(&recip_x, -self.beta)?;

        let result = client.add(&term1, &term2)?;
        client.add_scalar(&result, self.log_norm)
    }

    fn cdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // CDF(x) = gammaincc(α, β/x)
        let alpha_t = Tensor::<R>::full_scalar(x.shape(), x.dtype(), self.alpha, client.device())?;
        let ones = Tensor::<R>::ones(x.shape(), x.dtype(), client.device())?;
        let recip_x = client.div(&ones, x)?;
        let beta_over_x = client.mul_scalar(&recip_x, self.beta)?;
        client.gammaincc(&alpha_t, &beta_over_x)
    }

    fn sf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // SF(x) = gammainc(α, β/x)
        let alpha_t = Tensor::<R>::full_scalar(x.shape(), x.dtype(), self.alpha, client.device())?;
        let ones = Tensor::<R>::ones(x.shape(), x.dtype(), client.device())?;
        let recip_x = client.div(&ones, x)?;
        let beta_over_x = client.mul_scalar(&recip_x, self.beta)?;
        client.gammainc(&alpha_t, &beta_over_x)
    }

    fn log_cdf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        x: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // log(CDF) = log(gammaincc(α, β/x))
        let cdf = self.cdf_tensor(x, client)?;
        client.log(&cdf)
    }

    fn ppf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        p: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PPF(p) = β / gammaincinv(α, 1-p)
        let alpha_t = Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.alpha, client.device())?;
        let neg_p = client.mul_scalar(p, -1.0)?;
        let one_minus_p = client.add_scalar(&neg_p, 1.0)?;
        let gamma_inv = client.gammaincinv(&alpha_t, &one_minus_p)?;

        // Compute β / gamma_inv
        let beta_t = Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.beta, client.device())?;
        client.div(&beta_t, &gamma_inv)
    }

    fn isf_tensor<R: Runtime<DType = DType>, C>(
        &self,
        p: &Tensor<R>,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // ISF(p) = PPF(1 - p) = β / gammaincinv(α, p)
        let alpha_t = Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.alpha, client.device())?;
        let gamma_inv = client.gammaincinv(&alpha_t, p)?;

        // Compute β / gamma_inv
        let beta_t = Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.beta, client.device())?;
        client.div(&beta_t, &gamma_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_gamma_creation() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();
        assert!((ig.shape() - 2.0).abs() < 1e-10);
        assert!((ig.scale() - 1.0).abs() < 1e-10);

        let ig = InverseGamma::from_shape_scale(3.0, 2.0).unwrap();
        assert!((ig.shape() - 3.0).abs() < 1e-10);
        assert!((ig.scale() - 2.0).abs() < 1e-10);

        assert!(InverseGamma::new(0.0, 1.0).is_err());
        assert!(InverseGamma::new(1.0, 0.0).is_err());
        assert!(InverseGamma::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_inverse_gamma_moments() {
        let ig = InverseGamma::new(3.0, 2.0).unwrap();

        // Mean = β / (α - 1) = 2 / 2 = 1
        assert!((ig.mean() - 1.0).abs() < 1e-10);

        // Var = β² / ((α-1)² * (α-2)) = 4 / (4 * 1) = 1
        assert!((ig.var() - 1.0).abs() < 1e-10);

        // Mode = β / (α + 1) = 2 / 4 = 0.5
        assert!((ig.mode() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_gamma_pdf() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();

        // PDF at boundary should be 0
        assert!(ig.pdf(0.0) <= 1e-10);

        // PDF should be positive for x > 0
        assert!(ig.pdf(1.0) > 0.0);
        assert!(ig.pdf(0.5) > 0.0);

        // Log PDF at boundary
        assert!(ig.log_pdf(0.0).is_infinite() && ig.log_pdf(0.0) < 0.0);
    }

    #[test]
    fn test_inverse_gamma_cdf() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();

        // CDF at 0
        assert!((ig.cdf(0.0) - 0.0).abs() < 1e-10);

        // CDF should be increasing
        let c1 = ig.cdf(1.0);
        let c2 = ig.cdf(2.0);
        assert!(c1 < c2);

        // CDF should approach 1 as x → ∞
        assert!(ig.cdf(100.0) > 0.99);
    }

    #[test]
    fn test_inverse_gamma_ppf() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = ig.ppf(p).unwrap();
            assert!((ig.cdf(x) - p).abs() < 1e-5, "Failed for p={}", p);
        }

        assert!(ig.ppf(-0.1).is_err());
        assert!(ig.ppf(1.1).is_err());
    }

    #[test]
    fn test_inverse_gamma_sf() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();

        // SF + CDF = 1
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((ig.sf(x) + ig.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse_gamma_boundary_values() {
        let ig = InverseGamma::new(2.0, 1.0).unwrap();

        // PPF(0) = ∞
        assert!(ig.ppf(0.0).unwrap().is_infinite());

        // PPF(1) = 0
        assert!((ig.ppf(1.0).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_gamma_mean_undefined_for_alpha_le_1() {
        let ig = InverseGamma::new(0.5, 1.0).unwrap();

        // Mean should be infinite for α ≤ 1
        assert!(ig.mean().is_infinite());
    }

    #[test]
    fn test_inverse_gamma_var_undefined_for_alpha_le_2() {
        let ig = InverseGamma::new(1.5, 1.0).unwrap();

        // Variance should be infinite for α ≤ 2
        assert!(ig.var().is_infinite());
    }
}
