//! Poisson distribution.

use crate::stats::continuous::special;
use crate::stats::distribution::{DiscreteDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Poisson distribution.
///
/// The Poisson distribution models the number of events occurring in a fixed
/// interval when events occur independently at a constant rate λ.
///
/// P(X = k) = λ^k e^(-λ) / k!
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Poisson, DiscreteDistribution, Distribution};
///
/// // Average 5 events per interval
/// let p = Poisson::new(5.0).unwrap();
/// println!("P(X = 5) = {}", p.pmf(5));
/// println!("P(X ≤ 3) = {}", p.cdf(3));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    /// Rate parameter (λ)
    lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution with rate λ.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter (must be positive)
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if lambda <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "lambda".to_string(),
                value: lambda,
                reason: "rate must be positive".to_string(),
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

    /// Get the rate parameter λ.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get the rate parameter (alias for lambda).
    pub fn rate(&self) -> f64 {
        self.lambda
    }
}

impl Distribution for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }

    fn var(&self) -> f64 {
        self.lambda
    }

    fn entropy(&self) -> f64 {
        // Approximate entropy using Stirling
        // H ≈ λ(1 - ln(λ)) + e^(-λ) Σ(λ^k ln(k!) / k!)
        // For large λ: H ≈ 0.5 * ln(2πeλ)
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * self.lambda).ln()
    }

    fn median(&self) -> f64 {
        // Approximate: floor(λ + 1/3 - 0.02/λ)
        (self.lambda + 1.0 / 3.0 - 0.02 / self.lambda).floor()
    }

    fn mode(&self) -> f64 {
        self.lambda.floor()
    }

    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        1.0 / self.lambda // Excess kurtosis
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: u64) -> f64 {
        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: u64) -> f64 {
        let k_f = k as f64;
        k_f * self.lambda.ln() - self.lambda - special::lgamma(k_f + 1.0)
    }

    fn cdf(&self, k: u64) -> f64 {
        // CDF = Q(k+1, λ) = Γ(k+1, λ) / k!
        // Using the regularized upper incomplete gamma
        special::gammaincc((k + 1) as f64, self.lambda)
    }

    fn sf(&self, k: u64) -> f64 {
        // SF = P(k+1, λ) = 1 - Q(k+1, λ)
        special::gammainc((k + 1) as f64, self.lambda)
    }

    fn ppf(&self, prob: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::InvalidProbability { value: prob });
        }
        if prob == 0.0 {
            return Ok(0);
        }
        if prob == 1.0 {
            // Return a large value since Poisson has infinite support
            return Ok(u64::MAX);
        }

        // Use inverse of incomplete gamma for initial estimate
        let initial = special::gammaincinv(self.lambda, prob);
        let mut k = initial.floor() as u64;

        // Refine by searching
        while self.cdf(k) < prob && k < u64::MAX - 1 {
            k += 1;
        }
        while k > 0 && self.cdf(k - 1) >= prob {
            k -= 1;
        }

        Ok(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_creation() {
        let p = Poisson::new(5.0).unwrap();
        assert!((p.lambda() - 5.0).abs() < 1e-10);
        assert!((p.rate() - 5.0).abs() < 1e-10);

        assert!(Poisson::new(0.0).is_err());
        assert!(Poisson::new(-1.0).is_err());
    }

    #[test]
    fn test_poisson_moments() {
        let p = Poisson::new(4.0).unwrap();

        assert!((p.mean() - 4.0).abs() < 1e-10);
        assert!((p.var() - 4.0).abs() < 1e-10);
        assert!((p.std() - 2.0).abs() < 1e-10);
        assert!((p.skewness() - 0.5).abs() < 1e-10);
        assert!((p.kurtosis() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_pmf() {
        let p = Poisson::new(3.0).unwrap();

        // P(X = 0) = e^(-3)
        assert!((p.pmf(0) - (-3.0_f64).exp()).abs() < 1e-10);

        // P(X = 3) = 3^3 * e^(-3) / 3! = 27 * e^(-3) / 6
        let expected = 27.0 * (-3.0_f64).exp() / 6.0;
        assert!((p.pmf(3) - expected).abs() < 1e-10);

        // Sum of PMFs should approach 1 (test with large k)
        let total: f64 = (0..50).map(|k| p.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_cdf() {
        let p = Poisson::new(3.0).unwrap();

        // CDF(0) = P(X ≤ 0) = P(X = 0) = e^(-3)
        assert!((p.cdf(0) - (-3.0_f64).exp()).abs() < 1e-10);

        // CDF should be cumulative
        let cdf_3: f64 = (0..=3).map(|k| p.pmf(k)).sum();
        assert!((p.cdf(3) - cdf_3).abs() < 1e-6);

        // CDF is monotonic
        for k in 0..10 {
            assert!(p.cdf(k) <= p.cdf(k + 1));
        }
    }

    #[test]
    fn test_poisson_ppf() {
        let p = Poisson::new(5.0).unwrap();

        // PPF should give smallest k with CDF(k) >= prob
        for k in 0..15 {
            let prob = p.cdf(k);
            let result = p.ppf(prob).unwrap();
            assert!(p.cdf(result) >= prob);
        }
    }

    #[test]
    fn test_poisson_sf() {
        let p = Poisson::new(3.0).unwrap();

        // SF + CDF = 1
        for k in 0..10 {
            assert!((p.sf(k) + p.cdf(k) - 1.0).abs() < 1e-10);
        }
    }
}
