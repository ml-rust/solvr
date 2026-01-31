//! Chi-squared distribution.

use super::Gamma;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Chi-squared distribution.
///
/// The chi-squared distribution with k degrees of freedom is a special case
/// of the gamma distribution: χ²(k) = Gamma(k/2, 1/2).
///
/// f(x) = (1 / (2^(k/2) Γ(k/2))) x^(k/2-1) exp(-x/2)  for x > 0
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{ChiSquared, ContinuousDistribution, Distribution};
///
/// let chi2 = ChiSquared::new(5).unwrap();
/// println!("Mean: {}", chi2.mean());
/// println!("95th percentile: {}", chi2.ppf(0.95).unwrap());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    /// Degrees of freedom
    k: f64,
    /// Underlying gamma distribution
    gamma: Gamma,
}

impl ChiSquared {
    /// Create a new chi-squared distribution with k degrees of freedom.
    ///
    /// # Arguments
    ///
    /// * `k` - Degrees of freedom (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if k is not positive.
    pub fn new(k: u64) -> StatsResult<Self> {
        if k == 0 {
            return Err(StatsError::InvalidParameter {
                name: "k".to_string(),
                value: 0.0,
                reason: "degrees of freedom must be positive".to_string(),
            });
        }
        let k_f64 = k as f64;
        // χ²(k) = Gamma(k/2, 1/2)
        let gamma = Gamma::new(k_f64 / 2.0, 0.5)?;
        Ok(Self { k: k_f64, gamma })
    }

    /// Create from a floating-point degrees of freedom (for generalization).
    pub fn new_f64(k: f64) -> StatsResult<Self> {
        if k <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "k".to_string(),
                value: k,
                reason: "degrees of freedom must be positive".to_string(),
            });
        }
        let gamma = Gamma::new(k / 2.0, 0.5)?;
        Ok(Self { k, gamma })
    }

    /// Get the degrees of freedom.
    pub fn df(&self) -> f64 {
        self.k
    }
}

impl Distribution for ChiSquared {
    fn mean(&self) -> f64 {
        self.k
    }

    fn var(&self) -> f64 {
        2.0 * self.k
    }

    fn entropy(&self) -> f64 {
        self.gamma.entropy()
    }

    fn median(&self) -> f64 {
        // Approximation: k * (1 - 2/(9k))³
        self.k * (1.0 - 2.0 / (9.0 * self.k)).powi(3)
    }

    fn mode(&self) -> f64 {
        if self.k >= 2.0 { self.k - 2.0 } else { 0.0 }
    }

    fn skewness(&self) -> f64 {
        (8.0 / self.k).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        12.0 / self.k // Excess kurtosis
    }
}

impl ContinuousDistribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        self.gamma.pdf(x)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        self.gamma.log_pdf(x)
    }

    fn cdf(&self, x: f64) -> f64 {
        self.gamma.cdf(x)
    }

    fn sf(&self, x: f64) -> f64 {
        self.gamma.sf(x)
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        self.gamma.ppf(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_creation() {
        let chi2 = ChiSquared::new(5).unwrap();
        assert!((chi2.df() - 5.0).abs() < 1e-10);

        assert!(ChiSquared::new(0).is_err());
    }

    #[test]
    fn test_chi_squared_moments() {
        let chi2 = ChiSquared::new(10).unwrap();

        assert!((chi2.mean() - 10.0).abs() < 1e-10);
        assert!((chi2.var() - 20.0).abs() < 1e-10);
        assert!((chi2.mode() - 8.0).abs() < 1e-10);
        assert!((chi2.skewness() - (0.8_f64).sqrt()).abs() < 1e-10);
        assert!((chi2.kurtosis() - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_cdf() {
        let chi2 = ChiSquared::new(1).unwrap();

        // For χ²(1), CDF(x) = 2Φ(√x) - 1 where Φ is standard normal CDF
        // At x = 1: CDF ≈ 0.6827
        assert!((chi2.cdf(1.0) - 0.6826894921370859).abs() < 1e-6);

        let chi2 = ChiSquared::new(2).unwrap();
        // χ²(2) = Exponential(1/2), so CDF(x) = 1 - exp(-x/2)
        assert!((chi2.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-6);
    }

    #[test]
    fn test_chi_squared_ppf() {
        let chi2 = ChiSquared::new(5).unwrap();

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let x = chi2.ppf(p).unwrap();
            assert!((chi2.cdf(x) - p).abs() < 1e-6, "Failed for p={}", p);
        }

        // Critical values (from chi-squared table)
        // χ²(5, 0.95) ≈ 11.07
        assert!((chi2.ppf(0.95).unwrap() - 11.0705).abs() < 0.01);
    }
}
