//! Truncated Normal distribution.

use super::special::{self, INV_SQRT_2PI, LN_SQRT_2PI};
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Truncated Normal distribution.
///
/// The truncated normal distribution is a normal distribution bounded to the interval [a, b].
/// It has parameters μ (mean of parent normal), σ (std dev of parent normal), a (lower bound), b (upper bound).
///
/// The PDF is:
///
/// f(x) = φ((x-μ)/σ) / (σ * (Φ((b-μ)/σ) - Φ((a-μ)/σ)))  for a ≤ x ≤ b, 0 otherwise
///
/// where φ is the standard normal PDF and Φ is the standard normal CDF.
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{TruncatedNormal, ContinuousDistribution, Distribution};
///
/// // Truncate standard normal to [-1, 1]
/// let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();
/// assert!(tn.pdf(-0.5) > 0.0);
/// assert!(tn.pdf(2.0) == 0.0);  // Outside support
/// assert!(tn.cdf(1.0) == 1.0);  // At upper bound
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TruncatedNormal {
    /// Mean of parent normal distribution (μ)
    mu: f64,
    /// Standard deviation of parent normal (σ)
    sigma: f64,
    /// Lower bound (a)
    a: f64,
    /// Upper bound (b)
    b: f64,
    /// Standardized lower bound: α = (a - μ) / σ
    alpha: f64,
    /// Standardized upper bound: β = (b - μ) / σ
    beta: f64,
    /// CDF of standard normal at upper bound: Φ(β)
    phi_beta: f64,
    /// CDF of standard normal at lower bound: Φ(α)
    phi_alpha: f64,
    /// Normalization constant: Z = Φ(β) - Φ(α)
    z_norm: f64,
}

impl TruncatedNormal {
    /// Create a new truncated normal distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of parent normal distribution
    /// * `sigma` - Standard deviation (must be positive)
    /// * `a` - Lower truncation bound
    /// * `b` - Upper truncation bound (must be > a)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `sigma <= 0`
    /// - `a >= b`
    /// - Any parameter is not finite
    pub fn new(mu: f64, sigma: f64, a: f64, b: f64) -> StatsResult<Self> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "sigma".to_string(),
                value: sigma,
                reason: "must be positive".to_string(),
            });
        }

        if a >= b {
            return Err(StatsError::InvalidParameter {
                name: "a, b".to_string(),
                value: a,
                reason: "lower bound must be < upper bound".to_string(),
            });
        }

        if !mu.is_finite() || !sigma.is_finite() || !a.is_finite() || !b.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "parameters".to_string(),
                value: mu,
                reason: "all parameters must be finite".to_string(),
            });
        }

        let alpha = (a - mu) / sigma;
        let beta = (b - mu) / sigma;
        let phi_alpha = special::norm_cdf(alpha);
        let phi_beta = special::norm_cdf(beta);
        let z_norm = phi_beta - phi_alpha;

        if z_norm <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "bounds".to_string(),
                value: z_norm,
                reason: "normalization constant must be positive (bounds may be too tight)"
                    .to_string(),
            });
        }

        Ok(Self {
            mu,
            sigma,
            a,
            b,
            alpha,
            beta,
            phi_alpha,
            phi_beta,
            z_norm,
        })
    }

    /// Get the mean parameter of parent normal.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation parameter of parent normal.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get the lower bound.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the upper bound.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Standardize a value: z = (x - μ) / σ
    fn standardize(&self, x: f64) -> f64 {
        (x - self.mu) / self.sigma
    }

    /// Standard normal PDF value: φ(z) = (1/√(2π)) * exp(-z²/2)
    fn std_normal_pdf(&self, z: f64) -> f64 {
        INV_SQRT_2PI * (-0.5 * z * z).exp()
    }

    /// Standard normal CDF value: Φ(z)
    fn std_normal_cdf(&self, z: f64) -> f64 {
        special::norm_cdf(z)
    }
}

impl Distribution for TruncatedNormal {
    fn mean(&self) -> f64 {
        let phi_alpha = self.std_normal_pdf(self.alpha);
        let phi_beta = self.std_normal_pdf(self.beta);
        self.mu + self.sigma * (phi_alpha - phi_beta) / self.z_norm
    }

    fn var(&self) -> f64 {
        let phi_alpha = self.std_normal_pdf(self.alpha);
        let phi_beta = self.std_normal_pdf(self.beta);
        let z = self.z_norm;

        let term1 = 1.0 + (self.alpha * phi_alpha - self.beta * phi_beta) / z;
        let term2 = ((phi_alpha - phi_beta) / z).powi(2);

        self.sigma * self.sigma * (term1 - term2)
    }

    fn entropy(&self) -> f64 {
        let phi_alpha = self.std_normal_pdf(self.alpha);
        let phi_beta = self.std_normal_pdf(self.beta);
        let z = self.z_norm;

        // H = ln(σ * Z * √(2πe)) - (α*φ(α) - β*φ(β))/(2*Z)
        let constant_term = (self.sigma * z * (2.0 * PI * std::f64::consts::E).sqrt()).ln();
        let variance_term = (self.alpha * phi_alpha - self.beta * phi_beta) / (2.0 * z);

        constant_term - variance_term
    }

    fn median(&self) -> f64 {
        // Median is at F(x) = 0.5
        let p_median = self.phi_alpha + 0.5 * self.z_norm;
        self.mu + self.sigma * special::norm_ppf(p_median)
    }

    fn mode(&self) -> f64 {
        // Mode of truncated normal is the mode of parent normal if within [a, b], else at nearest bound
        if self.a < self.mu && self.mu < self.b {
            self.mu
        } else if self.mu <= self.a {
            self.a
        } else {
            self.b
        }
    }

    fn skewness(&self) -> f64 {
        let phi_a = self.std_normal_pdf(self.alpha);
        let phi_b = self.std_normal_pdf(self.beta);
        let z = self.z_norm;

        // First three standardized moments about the mean of the truncated normal
        // E'1 = (φ(α) - φ(β)) / Z
        let m1 = (phi_a - phi_b) / z;
        // E'2 = 1 + (α·φ(α) - β·φ(β)) / Z
        let m2 = 1.0 + (self.alpha * phi_a - self.beta * phi_b) / z;
        // Variance of standardized truncated normal
        let var = m2 - m1 * m1;

        if var <= 0.0 {
            return 0.0;
        }

        // E'3 = 2·E'1 + (α²·φ(α) - β²·φ(β)) / Z
        let m3 = 2.0 * m1 + (self.alpha.powi(2) * phi_a - self.beta.powi(2) * phi_b) / z;
        // Central third moment: μ3 = E'3 - 3·E'1·E'2 + 2·E'1³
        let mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.powi(3);
        mu3 / var.powf(1.5)
    }

    fn kurtosis(&self) -> f64 {
        let phi_alpha = self.std_normal_pdf(self.alpha);
        let phi_beta = self.std_normal_pdf(self.beta);
        let z = self.z_norm;

        let xi = (self.alpha * phi_alpha - self.beta * phi_beta) / z;
        let delta_sq = 1.0 + xi * (self.alpha - self.beta) - xi * xi;

        if delta_sq <= 0.0 {
            0.0
        } else {
            let delta = delta_sq.sqrt();
            let numerator = self.alpha.powi(4) * phi_alpha
                - self.beta.powi(4) * phi_beta
                - 4.0 * xi * (self.alpha.powi(3) * phi_alpha - self.beta.powi(3) * phi_beta)
                + 6.0 * xi * xi * delta_sq
                + 3.0 * xi.powi(4);
            (numerator / (z * delta.powi(4))) - 3.0
        }
    }
}

impl ContinuousDistribution for TruncatedNormal {
    fn pdf(&self, x: f64) -> f64 {
        // PDF = 0 outside support
        if x < self.a || x > self.b {
            return 0.0;
        }

        let z = self.standardize(x);
        self.std_normal_pdf(z) / (self.sigma * self.z_norm)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        // log(PDF) = -∞ outside support
        if x < self.a || x > self.b {
            return f64::NEG_INFINITY;
        }

        let z = self.standardize(x);
        // ln(φ(z)) - ln(σ) - ln(Z)
        -LN_SQRT_2PI - 0.5 * z * z - self.sigma.ln() - self.z_norm.ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        // CDF = 0 below lower bound
        if x <= self.a {
            return 0.0;
        }

        // CDF = 1 above upper bound
        if x >= self.b {
            return 1.0;
        }

        let z = self.standardize(x);
        let phi_z = self.std_normal_cdf(z);
        (phi_z - self.phi_alpha) / self.z_norm
    }

    fn sf(&self, x: f64) -> f64 {
        // SF = 1 below lower bound
        if x <= self.a {
            return 1.0;
        }

        // SF = 0 above upper bound
        if x >= self.b {
            return 0.0;
        }

        let z = self.standardize(x);
        let phi_z = self.std_normal_cdf(z);
        (self.phi_beta - phi_z) / self.z_norm
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }

        if p == 0.0 {
            return Ok(self.a);
        }

        if p == 1.0 {
            return Ok(self.b);
        }

        // PPF(p) = μ + σ * Φ⁻¹(Φ(α) + p * Z)
        let p_combined = self.phi_alpha + p * self.z_norm;
        let z = special::norm_ppf(p_combined);
        Ok(self.mu + self.sigma * z)
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // z = (x - μ) / σ
        let centered = client.sub_scalar(x, self.mu)?;
        let z = client.mul_scalar(&centered, 1.0 / self.sigma)?;

        // -0.5 * z²
        let z_sq = client.square(&z)?;
        let neg_half_z_sq = client.mul_scalar(&z_sq, -0.5)?;

        // exp(-0.5 * z²)
        let exp_term = client.exp(&neg_half_z_sq)?;

        // φ(z) / (σ * Z)
        let numerator = client.mul_scalar(&exp_term, INV_SQRT_2PI)?;
        let denom = self.sigma * self.z_norm;
        client.mul_scalar(&numerator, 1.0 / denom)
    }

    fn log_pdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    {
        // z = (x - μ) / σ
        let centered = client.sub_scalar(x, self.mu)?;
        let z = client.mul_scalar(&centered, 1.0 / self.sigma)?;

        // -0.5 * z²
        let z_sq = client.square(&z)?;
        let neg_half_z_sq = client.mul_scalar(&z_sq, -0.5)?;

        // log(PDF) = -ln(√(2π)) - ln(σ) - ln(Z) - 0.5*z²
        let constant = -LN_SQRT_2PI - self.sigma.ln() - self.z_norm.ln();
        client.add_scalar(&neg_half_z_sq, constant)
    }

    fn cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // z = (x - μ) / σ
        let centered = client.sub_scalar(x, self.mu)?;
        let z = client.mul_scalar(&centered, 1.0 / self.sigma)?;

        // Φ(z) = 0.5 * erfc(-z / √2)
        let z_scaled = client.mul_scalar(&z, -std::f64::consts::FRAC_1_SQRT_2)?;
        let erfc_val = client.erfc(&z_scaled)?;
        let phi_z = client.mul_scalar(&erfc_val, 0.5)?;

        // CDF = (Φ(z) - Φ(α)) / Z
        let phi_alpha_tensor =
            Tensor::<R>::full_scalar(x.shape(), x.dtype(), self.phi_alpha, client.device());
        let numerator = client.sub(&phi_z, &phi_alpha_tensor)?;
        let cdf_unbounded = client.mul_scalar(&numerator, 1.0 / self.z_norm)?;

        // Clamp to [0, 1]
        let zero = Tensor::<R>::full_scalar(x.shape(), x.dtype(), 0.0, client.device());
        let one = Tensor::<R>::full_scalar(x.shape(), x.dtype(), 1.0, client.device());
        let clamped = client.maximum(&cdf_unbounded, &zero)?;
        client.minimum(&clamped, &one)
    }

    fn sf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // z = (x - μ) / σ
        let centered = client.sub_scalar(x, self.mu)?;
        let z = client.mul_scalar(&centered, 1.0 / self.sigma)?;

        // Φ(z) = 0.5 * erfc(-z / √2)
        let z_scaled = client.mul_scalar(&z, -std::f64::consts::FRAC_1_SQRT_2)?;
        let erfc_val = client.erfc(&z_scaled)?;
        let phi_z = client.mul_scalar(&erfc_val, 0.5)?;

        // SF = (Φ(β) - Φ(z)) / Z
        let phi_beta_tensor =
            Tensor::<R>::full_scalar(x.shape(), x.dtype(), self.phi_beta, client.device());
        let numerator = client.sub(&phi_beta_tensor, &phi_z)?;
        let sf_unbounded = client.mul_scalar(&numerator, 1.0 / self.z_norm)?;

        // Clamp to [0, 1]
        let zero = Tensor::<R>::full_scalar(x.shape(), x.dtype(), 0.0, client.device());
        let one = Tensor::<R>::full_scalar(x.shape(), x.dtype(), 1.0, client.device());
        let clamped = client.maximum(&sf_unbounded, &zero)?;
        client.minimum(&clamped, &one)
    }

    fn log_cdf_tensor<R: Runtime, C>(&self, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        let cdf = self.cdf_tensor(x, client)?;
        client.log(&cdf)
    }

    fn ppf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // p_combined = Φ(α) + p * Z
        let phi_alpha_tensor =
            Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.phi_alpha, client.device());
        let z_norm_tensor =
            Tensor::<R>::full_scalar(p.shape(), p.dtype(), self.z_norm, client.device());
        let p_scaled = client.mul(p, &z_norm_tensor)?;
        let p_combined = client.add(&phi_alpha_tensor, &p_scaled)?;

        // z = √2 * erfinv(2*p_combined - 1)
        let two_p_minus_1 = client.sub_scalar(&client.mul_scalar(&p_combined, 2.0)?, 1.0)?;
        let erfinv_val = client.erfinv(&two_p_minus_1)?;
        let z = client.mul_scalar(&erfinv_val, std::f64::consts::SQRT_2)?;

        // x = μ + σ * z
        let scaled = client.mul_scalar(&z, self.sigma)?;
        client.add_scalar(&scaled, self.mu)
    }

    fn isf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // ISF(p) = PPF(1 - p)
        let neg_p = client.mul_scalar(p, -1.0)?;
        let one_minus_p = client.add_scalar(&neg_p, 1.0)?;
        self.ppf_tensor(&one_minus_p, client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_normal_creation() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();
        assert!((tn.mu() - 0.0).abs() < 1e-10);
        assert!((tn.sigma() - 1.0).abs() < 1e-10);
        assert!((tn.a() - (-1.0)).abs() < 1e-10);
        assert!((tn.b() - 1.0).abs() < 1e-10);

        // Invalid sigma
        assert!(TruncatedNormal::new(0.0, 0.0, -1.0, 1.0).is_err());
        assert!(TruncatedNormal::new(0.0, -1.0, -1.0, 1.0).is_err());

        // Invalid bounds
        assert!(TruncatedNormal::new(0.0, 1.0, 1.0, 1.0).is_err());
        assert!(TruncatedNormal::new(0.0, 1.0, 1.0, 0.0).is_err());

        // Non-finite parameters
        assert!(TruncatedNormal::new(f64::NAN, 1.0, -1.0, 1.0).is_err());
        assert!(TruncatedNormal::new(0.0, f64::INFINITY, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_truncated_normal_pdf_support() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // PDF should be 0 outside [a, b]
        assert!(tn.pdf(-1.5) == 0.0);
        assert!(tn.pdf(1.5) == 0.0);

        // PDF should be positive inside [a, b]
        assert!(tn.pdf(-0.5) > 0.0);
        assert!(tn.pdf(0.0) > 0.0);
        assert!(tn.pdf(0.5) > 0.0);

        // At boundaries
        assert!(tn.pdf(-1.0) > 0.0);
        assert!(tn.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_truncated_normal_pdf_normalization() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // Simple numerical integration to check normalization
        let mut integral = 0.0;
        let dx = 0.01;
        for i in -100..=100 {
            let x = i as f64 * dx;
            integral += tn.pdf(x) * dx;
        }
        assert!((integral - 1.0).abs() < 0.01, "Integral: {}", integral);
    }

    #[test]
    fn test_truncated_normal_cdf() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // CDF should be 0 at lower bound
        assert!(tn.cdf(-1.0) == 0.0);
        assert!(tn.cdf(-1.1) == 0.0);
        // Just above lower bound should be > 0
        assert!(tn.cdf(-0.99) > 0.0);

        // CDF should be 1 at upper bound
        assert!(tn.cdf(1.0) == 1.0);
        assert!(tn.cdf(1.1) == 1.0);

        // CDF should be ~0.5 at mode (0)
        assert!((tn.cdf(0.0) - 0.5).abs() < 0.01);

        // CDF should be monotonically increasing
        let x_vals = [-0.5, 0.0, 0.5];
        for i in 0..x_vals.len() - 1 {
            assert!(tn.cdf(x_vals[i]) <= tn.cdf(x_vals[i + 1]));
        }
    }

    #[test]
    fn test_truncated_normal_ppf() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // PPF(0) should be lower bound
        assert!((tn.ppf(0.0).unwrap() - (-1.0)).abs() < 1e-10);

        // PPF(1) should be upper bound
        assert!((tn.ppf(1.0).unwrap() - 1.0).abs() < 1e-10);

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = tn.ppf(p).unwrap();
            assert!(
                (tn.cdf(x) - p).abs() < 1e-6,
                "Roundtrip failed for p={}: cdf(ppf(p)) = {}",
                p,
                tn.cdf(x)
            );
        }

        // Invalid probabilities
        assert!(tn.ppf(-0.1).is_err());
        assert!(tn.ppf(1.1).is_err());
    }

    #[test]
    fn test_truncated_normal_sf() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // SF + CDF should equal 1
        for x in [-0.5, 0.0, 0.5] {
            assert!((tn.sf(x) + tn.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_truncated_normal_moments() {
        // Standard normal truncated to [-1, 1] should have mean close to 0 by symmetry
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();
        assert!((tn.mean() - 0.0).abs() < 1e-10); // Symmetric

        // Variance should be less than 1 (parent normal variance)
        assert!(tn.var() < 1.0);

        // Standard deviation
        assert!((tn.std() - tn.var().sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_truncated_normal_mode() {
        // Mode inside bounds should be at mu
        let tn = TruncatedNormal::new(0.5, 1.0, -1.0, 1.0).unwrap();
        assert!((tn.mode() - 0.5).abs() < 1e-10);

        // Mode outside bounds (mu < a) should be at a
        let tn = TruncatedNormal::new(-2.0, 1.0, -1.0, 1.0).unwrap();
        assert!((tn.mode() - (-1.0)).abs() < 1e-10);

        // Mode outside bounds (mu > b) should be at b
        let tn = TruncatedNormal::new(2.0, 1.0, -1.0, 1.0).unwrap();
        assert!((tn.mode() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_truncated_normal_median() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // Median should be close to 0 by symmetry
        assert!((tn.median() - 0.0).abs() < 1e-10);

        // Median should be between a and b
        assert!(tn.a() <= tn.median() && tn.median() <= tn.b());

        // CDF at median should be 0.5
        assert!((tn.cdf(tn.median()) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_truncated_normal_skewness() {
        // Symmetric truncation around mean should have 0 skewness
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();
        assert!(tn.skewness().abs() < 1e-6, "skewness = {}", tn.skewness());

        // Asymmetric truncation should have non-zero skewness
        let tn = TruncatedNormal::new(0.0, 1.0, -0.5, 1.0).unwrap();
        assert!(tn.skewness() > 0.0); // Right skew
    }

    #[test]
    fn test_truncated_normal_asymmetric_bounds() {
        let tn = TruncatedNormal::new(0.5, 1.0, -1.0, 2.0).unwrap();

        // Should still be valid
        assert!(tn.pdf(-0.5) > 0.0);
        assert!(tn.pdf(1.0) > 0.0);

        // CDF should still be monotonic
        assert!(tn.cdf(-0.5) < tn.cdf(0.5));
        assert!(tn.cdf(0.5) < tn.cdf(1.5));

        // PPF roundtrip
        for p in [0.1, 0.5, 0.9] {
            let x = tn.ppf(p).unwrap();
            assert!((tn.cdf(x) - p).abs() < 1e-6);
        }
    }

    #[test]
    fn test_truncated_normal_tight_bounds() {
        // Very tight bounds should still work
        let tn = TruncatedNormal::new(0.0, 1.0, -0.1, 0.1).unwrap();

        // PDF should be much higher than for [-1, 1]
        let tn_wide = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();
        assert!(tn.pdf(0.0) > tn_wide.pdf(0.0));

        // Integration test
        let mut integral = 0.0;
        let dx = 0.001;
        let mut x = tn.a();
        while x <= tn.b() {
            integral += tn.pdf(x) * dx;
            x += dx;
        }
        assert!((integral - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_truncated_normal_log_pdf() {
        let tn = TruncatedNormal::new(0.0, 1.0, -1.0, 1.0).unwrap();

        // log_pdf should match log(pdf) inside support
        for x in [-0.5, 0.0, 0.5] {
            let log_pdf_direct = tn.log_pdf(x);
            let log_pdf_computed = tn.pdf(x).ln();
            assert!((log_pdf_direct - log_pdf_computed).abs() < 1e-10);
        }

        // log_pdf should be -∞ outside support
        assert!(tn.log_pdf(-1.5) == f64::NEG_INFINITY);
        assert!(tn.log_pdf(1.5) == f64::NEG_INFINITY);
    }
}
