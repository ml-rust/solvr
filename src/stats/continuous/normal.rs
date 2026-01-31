//! Normal (Gaussian) distribution.

use super::special::{self, INV_SQRT_2PI, LN_SQRT_2PI};
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};
use std::f64::consts::PI;

/// Normal (Gaussian) distribution.
///
/// The normal distribution with mean μ and standard deviation σ has PDF:
///
/// f(x) = (1 / (σ√(2π))) exp(-(x-μ)² / (2σ²))
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Normal, ContinuousDistribution, Distribution};
///
/// // Standard normal N(0, 1)
/// let n = Normal::standard();
/// assert!((n.pdf(0.0) - 0.3989422804).abs() < 1e-6);
/// assert!((n.cdf(0.0) - 0.5).abs() < 1e-10);
///
/// // Custom normal N(100, 15)
/// let n = Normal::new(100.0, 15.0).unwrap();
/// println!("Mean: {}", n.mean());
/// println!("P(X < 130) = {}", n.cdf(130.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    /// Mean (μ)
    mu: f64,
    /// Standard deviation (σ)
    sigma: f64,
}

impl Normal {
    /// Create a new normal distribution with given mean and standard deviation.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of the distribution
    /// * `sigma` - Standard deviation (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if sigma is not positive.
    pub fn new(mu: f64, sigma: f64) -> StatsResult<Self> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "sigma".to_string(),
                value: sigma,
                reason: "must be positive".to_string(),
            });
        }
        if !mu.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "mu".to_string(),
                value: mu,
                reason: "must be finite".to_string(),
            });
        }
        Ok(Self { mu, sigma })
    }

    /// Create a standard normal distribution N(0, 1).
    pub fn standard() -> Self {
        Self {
            mu: 0.0,
            sigma: 1.0,
        }
    }

    /// Get the mean parameter.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the standard deviation parameter.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Standardize a value: z = (x - μ) / σ
    fn standardize(&self, x: f64) -> f64 {
        (x - self.mu) / self.sigma
    }
}

impl Distribution for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn var(&self) -> f64 {
        self.sigma * self.sigma
    }

    fn std(&self) -> f64 {
        self.sigma
    }

    fn entropy(&self) -> f64 {
        // H = 0.5 * ln(2πeσ²) = 0.5 * (1 + ln(2π) + 2*ln(σ))
        0.5 * (1.0 + (2.0 * PI).ln()) + self.sigma.ln()
    }

    fn median(&self) -> f64 {
        self.mu
    }

    fn mode(&self) -> f64 {
        self.mu
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        0.0 // Excess kurtosis
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        INV_SQRT_2PI * (-0.5 * z * z).exp() / self.sigma
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        -LN_SQRT_2PI - self.sigma.ln() - 0.5 * z * z
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        special::norm_cdf(z)
    }

    fn sf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        special::norm_cdf(-z)
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let z = special::norm_ppf(p);
        Ok(self.mu + self.sigma * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_creation() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.mu() - 0.0).abs() < 1e-10);
        assert!((n.sigma() - 1.0).abs() < 1e-10);

        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_standard_normal() {
        let n = Normal::standard();
        assert!((n.mu() - 0.0).abs() < 1e-10);
        assert!((n.sigma() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_pdf() {
        let n = Normal::standard();

        // PDF at 0 should be 1/sqrt(2π) ≈ 0.3989422804
        assert!((n.pdf(0.0) - 0.3989422804014327).abs() < 1e-10);

        // PDF is symmetric
        assert!((n.pdf(1.0) - n.pdf(-1.0)).abs() < 1e-10);

        // PDF at ±1σ
        let pdf_1sigma = 0.24197072451914337;
        assert!((n.pdf(1.0) - pdf_1sigma).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf() {
        let n = Normal::standard();

        // CDF at 0 should be 0.5
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF at -∞ should approach 0
        assert!(n.cdf(-10.0) < 1e-10);

        // CDF at +∞ should approach 1
        assert!((1.0 - n.cdf(10.0)) < 1e-10);

        // Standard values
        assert!((n.cdf(1.0) - 0.8413447460685429).abs() < 1e-6);
        assert!((n.cdf(-1.0) - 0.15865525393145707).abs() < 1e-6);
        assert!((n.cdf(1.96) - 0.9750021048517796).abs() < 1e-6);
    }

    #[test]
    fn test_normal_ppf() {
        let n = Normal::standard();

        // PPF(0.5) = 0
        assert!((n.ppf(0.5).unwrap() - 0.0).abs() < 1e-10);

        // PPF should be inverse of CDF (roundtrip test)
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let x = n.ppf(p).unwrap();
            assert!(
                (n.cdf(x) - p).abs() < 1e-10,
                "Roundtrip failed for p={}: cdf(ppf(p)) = {}",
                p,
                n.cdf(x)
            );
        }

        // Standard quantiles (tolerance accounts for erf approximation differences)
        // Our implementation is self-consistent: CDF(PPF(p)) = p exactly
        assert!((n.ppf(0.975).unwrap() - 1.96).abs() < 0.001);
        assert!((n.ppf(0.95).unwrap() - 1.645).abs() < 0.001);

        // Invalid probabilities
        assert!(n.ppf(-0.1).is_err());
        assert!(n.ppf(1.1).is_err());
    }

    #[test]
    fn test_normal_moments() {
        let n = Normal::new(5.0, 2.0).unwrap();
        assert!((n.mean() - 5.0).abs() < 1e-10);
        assert!((n.var() - 4.0).abs() < 1e-10);
        assert!((n.std() - 2.0).abs() < 1e-10);
        assert!((n.median() - 5.0).abs() < 1e-10);
        assert!((n.mode() - 5.0).abs() < 1e-10);
        assert!((n.skewness() - 0.0).abs() < 1e-10);
        assert!((n.kurtosis() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_entropy() {
        let n = Normal::standard();
        // H = 0.5 * ln(2πe) ≈ 1.4189385332
        let expected = 0.5 * (2.0 * PI * std::f64::consts::E).ln();
        assert!((n.entropy() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_normal_sf() {
        let n = Normal::standard();
        // SF(x) = 1 - CDF(x)
        assert!((n.sf(0.0) - 0.5).abs() < 1e-10);
        assert!((n.sf(1.96) + n.cdf(1.96) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_interval() {
        let n = Normal::standard();

        // 95% interval should be approximately [-1.96, 1.96]
        let (a, b) = n.interval(0.95).unwrap();
        assert!((a + 1.96).abs() < 0.001, "Lower bound: {}", a);
        assert!((b - 1.96).abs() < 0.001, "Upper bound: {}", b);

        // Verify interval is symmetric and has correct coverage
        assert!((a + b).abs() < 1e-10); // Symmetric around 0
        assert!((n.cdf(b) - n.cdf(a) - 0.95).abs() < 1e-10); // 95% coverage
    }

    #[test]
    fn test_scaled_normal() {
        let n = Normal::new(100.0, 15.0).unwrap();

        // IQ distribution example
        // P(IQ < 130) = P(Z < 2)
        let p = n.cdf(130.0);
        let expected = Normal::standard().cdf(2.0);
        assert!((p - expected).abs() < 1e-10);

        // 95th percentile of IQ
        let q95 = n.ppf(0.95).unwrap();
        let expected = 100.0 + 15.0 * Normal::standard().ppf(0.95).unwrap();
        assert!((q95 - expected).abs() < 1e-10);
    }
}
