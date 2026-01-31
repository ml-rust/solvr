//! Binomial distribution.

use super::log_binom;
use crate::stats::continuous::special;
use crate::stats::distribution::{DiscreteDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Binomial distribution.
///
/// The binomial distribution models the number of successes in n independent
/// Bernoulli trials with success probability p.
///
/// P(X = k) = C(n, k) p^k (1-p)^(n-k)
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Binomial, DiscreteDistribution, Distribution};
///
/// // 10 coin flips with fair coin
/// let b = Binomial::new(10, 0.5).unwrap();
/// println!("P(X = 5) = {}", b.pmf(5)); // Most likely outcome
/// println!("P(X ≤ 3) = {}", b.cdf(3)); // At most 3 heads
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    /// Number of trials
    n: u64,
    /// Success probability
    p: f64,
    /// Failure probability (1 - p)
    q: f64,
}

impl Binomial {
    /// Create a new binomial distribution.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials
    /// * `p` - Probability of success on each trial (must be in [0, 1])
    pub fn new(n: u64, p: f64) -> StatsResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }
        Ok(Self { n, p, q: 1.0 - p })
    }

    /// Get the number of trials.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Get the success probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for Binomial {
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn var(&self) -> f64 {
        self.n as f64 * self.p * self.q
    }

    fn entropy(&self) -> f64 {
        // No simple closed form; use sum approximation for large n
        if self.n == 0 {
            return 0.0;
        }
        // For simplicity, use normal approximation entropy for large n
        // H ≈ 0.5 * ln(2πe * npq)
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * self.var()).ln()
    }

    fn median(&self) -> f64 {
        // Approximate: floor(np) or ceil(np)
        (self.n as f64 * self.p).floor()
    }

    fn mode(&self) -> f64 {
        ((self.n + 1) as f64 * self.p).floor()
    }

    fn skewness(&self) -> f64 {
        if self.var() == 0.0 {
            return 0.0;
        }
        (self.q - self.p) / self.var().sqrt()
    }

    fn kurtosis(&self) -> f64 {
        if self.var() == 0.0 {
            return 0.0;
        }
        (1.0 - 6.0 * self.p * self.q) / self.var()
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k == self.n { 1.0 } else { 0.0 };
        }

        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }

        let k_f = k as f64;
        let n_f = self.n as f64;

        log_binom(self.n, k) + k_f * self.p.ln() + (n_f - k_f) * self.q.ln()
    }

    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n {
            return 1.0;
        }
        if self.p == 0.0 {
            return 1.0;
        }
        if self.p == 1.0 {
            return 0.0;
        }

        // CDF = I_{1-p}(n-k, k+1) = 1 - I_p(k+1, n-k)
        1.0 - special::betainc((k + 1) as f64, (self.n - k) as f64, self.p)
    }

    fn sf(&self, k: u64) -> f64 {
        if k >= self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return 0.0;
        }
        if self.p == 1.0 {
            return 1.0;
        }

        // SF = P(X > k) = I_p(k+1, n-k)
        special::betainc((k + 1) as f64, (self.n - k) as f64, self.p)
    }

    fn ppf(&self, prob: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::InvalidProbability { value: prob });
        }
        if prob == 0.0 {
            return Ok(0);
        }
        if prob == 1.0 {
            return Ok(self.n);
        }

        // Binary search for smallest k with CDF(k) >= prob
        let mut lo = 0u64;
        let mut hi = self.n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) < prob {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_creation() {
        let b = Binomial::new(10, 0.5).unwrap();
        assert_eq!(b.n(), 10);
        assert!((b.p() - 0.5).abs() < 1e-10);

        assert!(Binomial::new(10, -0.1).is_err());
        assert!(Binomial::new(10, 1.1).is_err());
    }

    #[test]
    fn test_binomial_moments() {
        let b = Binomial::new(10, 0.3).unwrap();

        // Mean = np = 3
        assert!((b.mean() - 3.0).abs() < 1e-10);

        // Var = npq = 2.1
        assert!((b.var() - 2.1).abs() < 1e-10);

        // Skewness = (q-p)/sqrt(npq)
        let expected_skew = 0.4 / 2.1_f64.sqrt();
        assert!((b.skewness() - expected_skew).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_pmf() {
        let b = Binomial::new(10, 0.5).unwrap();

        // P(X = 5) for fair coin is C(10,5) * 0.5^10 = 252/1024
        let expected = 252.0 / 1024.0;
        assert!((b.pmf(5) - expected).abs() < 1e-10);

        // Sum of all PMFs should be 1
        let total: f64 = (0..=10).map(|k| b.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-10);

        // PMF(k) = 0 for k > n
        assert!((b.pmf(11) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_cdf() {
        let b = Binomial::new(10, 0.5).unwrap();

        // CDF should be cumulative
        let cdf_5: f64 = (0..=5).map(|k| b.pmf(k)).sum();
        assert!((b.cdf(5) - cdf_5).abs() < 1e-6);

        // CDF(n) = 1
        assert!((b.cdf(10) - 1.0).abs() < 1e-10);

        // CDF is monotonic
        for k in 0..10 {
            assert!(b.cdf(k) <= b.cdf(k + 1));
        }
    }

    #[test]
    fn test_binomial_ppf() {
        let b = Binomial::new(10, 0.5).unwrap();

        // PPF should give smallest k with CDF(k) >= p
        for k in 0..=10 {
            let p = b.cdf(k);
            let result = b.ppf(p).unwrap();
            assert!(b.cdf(result) >= p);
            if result > 0 {
                assert!(b.cdf(result - 1) < p);
            }
        }
    }

    #[test]
    fn test_binomial_edge_cases() {
        // p = 0: always 0 successes
        let b = Binomial::new(10, 0.0).unwrap();
        assert!((b.pmf(0) - 1.0).abs() < 1e-10);
        assert!((b.pmf(1) - 0.0).abs() < 1e-10);

        // p = 1: always n successes
        let b = Binomial::new(10, 1.0).unwrap();
        assert!((b.pmf(10) - 1.0).abs() < 1e-10);
        assert!((b.pmf(9) - 0.0).abs() < 1e-10);

        // n = 0: always 0
        let b = Binomial::new(0, 0.5).unwrap();
        assert!((b.pmf(0) - 1.0).abs() < 1e-10);
    }
}
