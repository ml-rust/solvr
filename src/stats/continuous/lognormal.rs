//! Log-normal distribution.

use super::special::{self, LN_SQRT_2PI};
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Log-normal distribution.
///
/// A log-normal distribution is the distribution of a random variable whose
/// logarithm is normally distributed. If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ).
///
/// f(x) = (1 / (xσ√(2π))) exp(-(ln(x) - μ)² / (2σ²))  for x > 0
///
/// # Parameters
///
/// * `mu` (μ) - Mean of the underlying normal distribution (NOT the mean of X)
/// * `sigma` (σ) - Standard deviation of the underlying normal (must be positive)
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{LogNormal, ContinuousDistribution, Distribution};
///
/// // Standard log-normal (μ=0, σ=1)
/// let ln = LogNormal::standard();
/// println!("Mean: {}", ln.mean()); // exp(0.5) ≈ 1.649
/// println!("Median: {}", ln.median()); // exp(0) = 1
///
/// // Custom parameters
/// let ln = LogNormal::new(0.0, 0.5).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    /// Location parameter (μ) - mean of ln(X)
    mu: f64,
    /// Scale parameter (σ) - std of ln(X)
    sigma: f64,
}

impl LogNormal {
    /// Create a new log-normal distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean of the underlying normal distribution
    /// * `sigma` - Standard deviation of the underlying normal (must be positive)
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

    /// Create a standard log-normal distribution (μ=0, σ=1).
    pub fn standard() -> Self {
        Self {
            mu: 0.0,
            sigma: 1.0,
        }
    }

    /// Create a log-normal distribution from desired mean and variance.
    ///
    /// Given desired mean m and variance v of X, computes μ and σ.
    pub fn from_mean_var(mean: f64, var: f64) -> StatsResult<Self> {
        if mean <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "mean".to_string(),
                value: mean,
                reason: "must be positive".to_string(),
            });
        }
        if var <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "var".to_string(),
                value: var,
                reason: "must be positive".to_string(),
            });
        }

        // σ² = ln(1 + v/m²)
        // μ = ln(m) - σ²/2
        let sigma_sq = (1.0 + var / (mean * mean)).ln();
        let sigma = sigma_sq.sqrt();
        let mu = mean.ln() - sigma_sq / 2.0;

        Self::new(mu, sigma)
    }

    /// Get the location parameter μ.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the scale parameter σ.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Distribution for LogNormal {
    fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    fn var(&self) -> f64 {
        let sigma_sq = self.sigma * self.sigma;
        ((2.0 * self.mu + sigma_sq).exp()) * (sigma_sq.exp() - 1.0)
    }

    fn entropy(&self) -> f64 {
        // H = μ + 0.5 + ln(σ) + 0.5*ln(2π)
        self.mu + 0.5 + self.sigma.ln() + LN_SQRT_2PI
    }

    fn median(&self) -> f64 {
        self.mu.exp()
    }

    fn mode(&self) -> f64 {
        (self.mu - self.sigma * self.sigma).exp()
    }

    fn skewness(&self) -> f64 {
        let sigma_sq = self.sigma * self.sigma;
        (sigma_sq.exp() + 2.0) * (sigma_sq.exp() - 1.0).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let sigma_sq = self.sigma * self.sigma;
        (4.0 * sigma_sq).exp() + 2.0 * (3.0 * sigma_sq).exp() + 3.0 * (2.0 * sigma_sq).exp() - 6.0
    }
}

impl ContinuousDistribution for LogNormal {
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
        let log_x = x.ln();
        let z = (log_x - self.mu) / self.sigma;
        -log_x - LN_SQRT_2PI - self.sigma.ln() - 0.5 * z * z
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        special::norm_cdf(z)
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        special::norm_cdf(-z)
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
        let z = special::norm_ppf(p);
        Ok((self.mu + self.sigma * z).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_lognormal_creation() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert!((ln.mu() - 0.0).abs() < 1e-10);
        assert!((ln.sigma() - 1.0).abs() < 1e-10);

        assert!(LogNormal::new(0.0, 0.0).is_err());
        assert!(LogNormal::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_lognormal_from_mean_var() {
        // Create with desired mean=2, var=1
        let ln = LogNormal::from_mean_var(2.0, 1.0).unwrap();

        // Verify the computed parameters give the correct moments
        assert!((ln.mean() - 2.0).abs() < 1e-6);
        assert!((ln.var() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lognormal_moments() {
        let ln = LogNormal::standard();

        // Mean = exp(μ + σ²/2) = exp(0.5) ≈ 1.6487
        assert!((ln.mean() - 0.5_f64.exp()).abs() < 1e-10);

        // Median = exp(μ) = 1
        assert!((ln.median() - 1.0).abs() < 1e-10);

        // Mode = exp(μ - σ²) = exp(-1) ≈ 0.3679
        assert!((ln.mode() - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_pdf() {
        let ln = LogNormal::standard();

        // PDF at x=1: f(1) = 1/(1*1*√(2π)) * exp(-0/2) = 1/√(2π)
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((ln.pdf(1.0) - expected).abs() < 1e-10);

        // PDF should be 0 for x ≤ 0
        assert!((ln.pdf(0.0) - 0.0).abs() < 1e-10);
        assert!((ln.pdf(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_cdf() {
        let ln = LogNormal::standard();

        // CDF(1) = Φ(0) = 0.5 (since ln(1) = 0)
        assert!((ln.cdf(1.0) - 0.5).abs() < 1e-10);

        // CDF at median should be 0.5
        assert!((ln.cdf(ln.median()) - 0.5).abs() < 1e-10);

        // CDF should be 0 for x ≤ 0
        assert!((ln.cdf(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_ppf() {
        let ln = LogNormal::new(1.0, 0.5).unwrap();

        // PPF(0.5) = median = exp(μ)
        assert!((ln.ppf(0.5).unwrap() - ln.median()).abs() < 1e-10);

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = ln.ppf(p).unwrap();
            assert!((ln.cdf(x) - p).abs() < 1e-8, "Failed for p={}", p);
        }
    }

    #[test]
    fn test_lognormal_sf() {
        let ln = LogNormal::standard();

        // SF + CDF = 1
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((ln.sf(x) + ln.cdf(x) - 1.0).abs() < 1e-10);
        }
    }
}
