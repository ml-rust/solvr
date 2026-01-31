//! Hypergeometric distribution.

use crate::stats::discrete::log_binom;
use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{DiscreteDistribution, Distribution};

/// Hypergeometric distribution.
///
/// The hypergeometric distribution models the number of successes in n draws
/// without replacement from a population containing N items, of which K are
/// successes. It has PMF:
///
/// P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
///
/// where:
/// - N is the population size
/// - K is the number of success states in the population
/// - n is the number of draws
/// - k is the number of observed successes
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Hypergeometric, DiscreteDistribution};
///
/// // Urn with 20 balls, 7 red, draw 12. What's P(exactly 4 red)?
/// let h = Hypergeometric::new(20, 7, 12).unwrap();
/// println!("P(X = 4): {}", h.pmf(4));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Hypergeometric {
    /// Population size (N)
    pop_size: u64,
    /// Number of success states in population (K)
    num_success: u64,
    /// Number of draws (n)
    num_draws: u64,
}

impl Hypergeometric {
    /// Create a new hypergeometric distribution.
    ///
    /// # Arguments
    ///
    /// * `pop_size` - Total population size (N)
    /// * `num_success` - Number of success states (K ≤ N)
    /// * `num_draws` - Number of draws (n ≤ N)
    pub fn new(pop_size: u64, num_success: u64, num_draws: u64) -> StatsResult<Self> {
        if pop_size == 0 {
            return Err(StatsError::InvalidParameter {
                name: "pop_size".to_string(),
                value: pop_size as f64,
                reason: "population size must be positive".to_string(),
            });
        }
        if num_success > pop_size {
            return Err(StatsError::InvalidParameter {
                name: "num_success".to_string(),
                value: num_success as f64,
                reason: "number of successes cannot exceed population size".to_string(),
            });
        }
        if num_draws > pop_size {
            return Err(StatsError::InvalidParameter {
                name: "num_draws".to_string(),
                value: num_draws as f64,
                reason: "number of draws cannot exceed population size".to_string(),
            });
        }
        Ok(Self {
            pop_size,
            num_success,
            num_draws,
        })
    }

    /// Get the population size.
    pub fn pop_size(&self) -> u64 {
        self.pop_size
    }

    /// Get the number of success states.
    pub fn num_success(&self) -> u64 {
        self.num_success
    }

    /// Get the number of draws.
    pub fn num_draws(&self) -> u64 {
        self.num_draws
    }

    /// Get the minimum possible value.
    pub fn min_val(&self) -> u64 {
        // min(k) = max(0, n - (N - K)) = max(0, n + K - N)
        let n = self.num_draws;
        let n_minus_k = self.pop_size - self.num_success;
        n.saturating_sub(n_minus_k)
    }

    /// Get the maximum possible value.
    pub fn max_val(&self) -> u64 {
        // max(k) = min(n, K)
        self.num_draws.min(self.num_success)
    }
}

impl Distribution for Hypergeometric {
    fn mean(&self) -> f64 {
        let n = self.num_draws as f64;
        let k = self.num_success as f64;
        let big_n = self.pop_size as f64;
        n * k / big_n
    }

    fn var(&self) -> f64 {
        let n = self.num_draws as f64;
        let k = self.num_success as f64;
        let big_n = self.pop_size as f64;

        // Variance = n * K * (N-K) * (N-n) / (N² * (N-1))
        if big_n <= 1.0 {
            return 0.0;
        }
        n * k * (big_n - k) * (big_n - n) / (big_n * big_n * (big_n - 1.0))
    }

    fn entropy(&self) -> f64 {
        let min_k = self.min_val();
        let max_k = self.max_val();

        let mut h = 0.0;
        for k in min_k..=max_k {
            let p = self.pmf(k);
            if p > 1e-300 {
                h -= p * p.ln();
            }
        }
        h
    }

    fn median(&self) -> f64 {
        // No closed form; use ppf
        self.ppf(0.5).unwrap_or(self.mean().round() as u64) as f64
    }

    fn mode(&self) -> f64 {
        let n = self.num_draws as f64;
        let k = self.num_success as f64;
        let big_n = self.pop_size as f64;

        // Mode = floor((n+1)(K+1)/(N+2))
        (((n + 1.0) * (k + 1.0)) / (big_n + 2.0)).floor()
    }

    fn skewness(&self) -> f64 {
        let n = self.num_draws as f64;
        let k = self.num_success as f64;
        let big_n = self.pop_size as f64;

        if big_n <= 2.0 {
            return f64::NAN;
        }

        let _p = k / big_n;

        let numerator = (big_n - 2.0 * k) * (big_n - 1.0).sqrt() * (big_n - 2.0 * n);
        let denominator = (n * k * (big_n - k) * (big_n - n)).sqrt() * (big_n - 2.0);

        numerator / denominator
    }

    fn kurtosis(&self) -> f64 {
        let n = self.num_draws as f64;
        let k = self.num_success as f64;
        let big_n = self.pop_size as f64;

        if big_n <= 3.0 {
            return f64::NAN;
        }

        // Complex formula for excess kurtosis
        let a = (big_n - 1.0)
            * big_n
            * big_n
            * (big_n * (big_n + 1.0) - 6.0 * k * (big_n - k) - 6.0 * n * (big_n - n))
            + 6.0 * n * k * (big_n - k) * (big_n - n) * (5.0 * big_n - 6.0);

        let b = n * k * (big_n - k) * (big_n - n) * (big_n - 2.0) * (big_n - 3.0);

        if b == 0.0 {
            return f64::NAN;
        }

        a / b
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn pmf(&self, k: u64) -> f64 {
        let log_p = self.log_pmf(k);
        if log_p.is_finite() { log_p.exp() } else { 0.0 }
    }

    fn log_pmf(&self, k: u64) -> f64 {
        let min_k = self.min_val();
        let max_k = self.max_val();

        if k < min_k || k > max_k {
            return f64::NEG_INFINITY;
        }

        let big_n = self.pop_size;
        let big_k = self.num_success;
        let n = self.num_draws;

        // log(C(K,k)) + log(C(N-K, n-k)) - log(C(N, n))
        log_binom(big_k, k) + log_binom(big_n - big_k, n - k) - log_binom(big_n, n)
    }

    fn cdf(&self, k: u64) -> f64 {
        let max_k = self.max_val();
        if k >= max_k {
            return 1.0;
        }

        let min_k = self.min_val();
        if k < min_k {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in min_k..=k {
            sum += self.pmf(i);
        }
        sum.min(1.0)
    }

    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }

    fn ppf(&self, prob: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: prob,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }

        let min_k = self.min_val();
        let max_k = self.max_val();

        if prob == 0.0 {
            return Ok(min_k);
        }
        if prob >= 1.0 {
            return Ok(max_k);
        }

        // Linear search (range is bounded)
        let mut cumulative = 0.0;
        for k in min_k..=max_k {
            cumulative += self.pmf(k);
            if cumulative >= prob {
                return Ok(k);
            }
        }

        Ok(max_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergeometric_creation() {
        assert!(Hypergeometric::new(20, 7, 12).is_ok());
        assert!(Hypergeometric::new(0, 0, 0).is_err());
        assert!(Hypergeometric::new(10, 15, 5).is_err()); // K > N
        assert!(Hypergeometric::new(10, 5, 15).is_err()); // n > N
    }

    #[test]
    fn test_hypergeometric_bounds() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // min = max(0, 12 + 7 - 20) = max(0, -1) = 0
        // But actually: min = max(0, n - (N-K)) = max(0, 12 - 13) = 0
        assert_eq!(h.min_val(), 0);

        // max = min(n, K) = min(12, 7) = 7
        assert_eq!(h.max_val(), 7);
    }

    #[test]
    fn test_hypergeometric_pmf() {
        // Classic urn problem: 20 balls, 7 red, draw 12
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // PMF should sum to 1 over valid range
        let sum: f64 = (h.min_val()..=h.max_val()).map(|k| h.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // PMF outside range is 0
        assert!((h.pmf(8) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_cdf() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // CDF at max should be 1
        assert!((h.cdf(h.max_val()) - 1.0).abs() < 1e-10);

        // CDF should be monotonically increasing
        let mut prev = 0.0;
        for k in h.min_val()..=h.max_val() {
            let curr = h.cdf(k);
            assert!(curr >= prev);
            prev = curr;
        }
    }

    #[test]
    fn test_hypergeometric_mean() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // Mean = n * K / N = 12 * 7 / 20 = 4.2
        assert!((h.mean() - 4.2).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_variance() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // Var = n * K * (N-K) * (N-n) / (N² * (N-1))
        // = 12 * 7 * 13 * 8 / (400 * 19)
        let expected = 12.0 * 7.0 * 13.0 * 8.0 / (400.0 * 19.0);
        assert!((h.var() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_ppf() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // PPF(0) = min value
        assert_eq!(h.ppf(0.0).unwrap(), h.min_val());

        // PPF(1) = max value
        assert_eq!(h.ppf(1.0).unwrap(), h.max_val());

        // Round-trip
        for k in h.min_val()..=h.max_val() {
            let p = h.cdf(k);
            let recovered = h.ppf(p).unwrap();
            assert!(recovered == k || recovered == k + 1);
        }
    }

    #[test]
    fn test_hypergeometric_mode() {
        let h = Hypergeometric::new(20, 7, 12).unwrap();

        // Mode should be where PMF is maximum
        let mode = h.mode() as u64;
        let pmf_mode = h.pmf(mode);

        for k in h.min_val()..=h.max_val() {
            assert!(h.pmf(k) <= pmf_mode + 1e-10);
        }
    }

    #[test]
    fn test_hypergeometric_extreme() {
        // All draws are successes
        let h = Hypergeometric::new(10, 10, 5).unwrap();
        assert!((h.pmf(5) - 1.0).abs() < 1e-10);

        // No successes possible
        let h = Hypergeometric::new(10, 0, 5).unwrap();
        assert!((h.pmf(0) - 1.0).abs() < 1e-10);
    }
}
