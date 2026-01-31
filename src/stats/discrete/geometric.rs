//! Geometric distribution.

use crate::stats::distribution::{DiscreteDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Geometric distribution.
///
/// The geometric distribution models the number of failures before the first
/// success in a sequence of independent Bernoulli trials.
///
/// P(X = k) = (1-p)^k * p  for k = 0, 1, 2, ...
///
/// Note: This uses the "number of failures" parameterization where X ∈ {0, 1, 2, ...}.
/// Some texts use "number of trials" where X ∈ {1, 2, 3, ...}.
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Geometric, DiscreteDistribution, Distribution};
///
/// // Success probability 0.3
/// let g = Geometric::new(0.3).unwrap();
/// println!("P(X = 0) = {}", g.pmf(0)); // First trial succeeds
/// println!("Mean failures: {}", g.mean());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    /// Success probability
    p: f64,
    /// Failure probability (1 - p)
    q: f64,
}

impl Geometric {
    /// Create a new geometric distribution.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of success on each trial (must be in (0, 1])
    pub fn new(p: f64) -> StatsResult<Self> {
        if p <= 0.0 || p > 1.0 {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in (0, 1]".to_string(),
            });
        }
        Ok(Self { p, q: 1.0 - p })
    }

    /// Get the success probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for Geometric {
    fn mean(&self) -> f64 {
        self.q / self.p
    }

    fn var(&self) -> f64 {
        self.q / (self.p * self.p)
    }

    fn entropy(&self) -> f64 {
        // H = (-(1-p)ln(1-p) - p*ln(p)) / p
        if self.q == 0.0 {
            return 0.0;
        }
        (-self.q * self.q.ln() - self.p * self.p.ln()) / self.p
    }

    fn median(&self) -> f64 {
        if self.q == 0.0 {
            return 0.0;
        }
        // median = ceil(-1 / log_2(1-p)) - 1
        let val = (-1.0 / (self.q.log2())).ceil() - 1.0;
        val.max(0.0)
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn skewness(&self) -> f64 {
        (2.0 - self.p) / self.q.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        6.0 + (self.p * self.p) / self.q // Excess kurtosis
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: u64) -> f64 {
        if self.q == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        self.q.powi(k as i32) * self.p
    }

    fn log_pmf(&self, k: u64) -> f64 {
        if self.q == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        (k as f64) * self.q.ln() + self.p.ln()
    }

    fn cdf(&self, k: u64) -> f64 {
        if self.q == 0.0 {
            return 1.0;
        }
        // CDF = 1 - (1-p)^(k+1)
        1.0 - self.q.powi((k + 1) as i32)
    }

    fn sf(&self, k: u64) -> f64 {
        if self.q == 0.0 {
            return 0.0;
        }
        // SF = (1-p)^(k+1)
        self.q.powi((k + 1) as i32)
    }

    fn ppf(&self, prob: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::InvalidProbability { value: prob });
        }
        if prob == 0.0 {
            return Ok(0);
        }
        if prob == 1.0 {
            return Ok(u64::MAX);
        }
        if self.q == 0.0 {
            return Ok(0);
        }

        // Solve: 1 - (1-p)^(k+1) >= prob
        // => (1-p)^(k+1) <= 1 - prob
        // => (k+1) * ln(1-p) <= ln(1-prob)
        // => k+1 >= ln(1-prob) / ln(1-p)  (note: ln(1-p) < 0)
        // => k >= ln(1-prob) / ln(1-p) - 1
        let k = ((1.0 - prob).ln() / self.q.ln()).ceil() - 1.0;
        Ok(k.max(0.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_creation() {
        let g = Geometric::new(0.3).unwrap();
        assert!((g.p() - 0.3).abs() < 1e-10);

        assert!(Geometric::new(0.0).is_err());
        assert!(Geometric::new(-0.1).is_err());
        assert!(Geometric::new(1.1).is_err());

        // p = 1 is valid (always succeed first try)
        assert!(Geometric::new(1.0).is_ok());
    }

    #[test]
    fn test_geometric_moments() {
        let g = Geometric::new(0.25).unwrap();

        // Mean = q/p = 0.75/0.25 = 3
        assert!((g.mean() - 3.0).abs() < 1e-10);

        // Var = q/p² = 0.75/0.0625 = 12
        assert!((g.var() - 12.0).abs() < 1e-10);

        // Mode = 0
        assert!((g.mode() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_pmf() {
        let g = Geometric::new(0.5).unwrap();

        // P(X = 0) = p = 0.5
        assert!((g.pmf(0) - 0.5).abs() < 1e-10);

        // P(X = 1) = q*p = 0.25
        assert!((g.pmf(1) - 0.25).abs() < 1e-10);

        // P(X = 2) = q²*p = 0.125
        assert!((g.pmf(2) - 0.125).abs() < 1e-10);

        // Sum should approach 1
        let total: f64 = (0..30).map(|k| g.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_geometric_cdf() {
        let g = Geometric::new(0.5).unwrap();

        // CDF(0) = P(X ≤ 0) = p = 0.5
        assert!((g.cdf(0) - 0.5).abs() < 1e-10);

        // CDF(1) = 1 - q² = 0.75
        assert!((g.cdf(1) - 0.75).abs() < 1e-10);

        // CDF(2) = 1 - q³ = 0.875
        assert!((g.cdf(2) - 0.875).abs() < 1e-10);

        // CDF is monotonic
        for k in 0..10 {
            assert!(g.cdf(k) <= g.cdf(k + 1));
        }
    }

    #[test]
    fn test_geometric_ppf() {
        let g = Geometric::new(0.3).unwrap();

        // PPF should give smallest k with CDF(k) >= prob
        for k in 0..10 {
            let prob = g.cdf(k);
            let result = g.ppf(prob).unwrap();
            assert!(
                g.cdf(result) >= prob,
                "k={}, prob={}, result={}, cdf={}",
                k,
                prob,
                result,
                g.cdf(result)
            );
        }
    }

    #[test]
    fn test_geometric_sf() {
        let g = Geometric::new(0.5).unwrap();

        // SF + CDF = 1
        for k in 0..10 {
            assert!((g.sf(k) + g.cdf(k) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_geometric_p_equals_1() {
        // When p = 1, always succeed on first trial
        let g = Geometric::new(1.0).unwrap();

        assert!((g.pmf(0) - 1.0).abs() < 1e-10);
        assert!((g.pmf(1) - 0.0).abs() < 1e-10);
        assert!((g.cdf(0) - 1.0).abs() < 1e-10);
        assert!((g.mean() - 0.0).abs() < 1e-10);
    }
}
