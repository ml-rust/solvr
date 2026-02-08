//! Multinomial distribution.

use crate::stats::continuous::special;
use crate::stats::distribution::Distribution;
use crate::stats::error::{StatsError, StatsResult};

/// Multinomial distribution.
///
/// The multinomial distribution models the outcomes of n independent trials
/// where each trial has k possible outcomes with fixed probabilities.
///
/// P(X = x) = n! / (x₁! * x₂! * ... * xₖ!) * p₁^x₁ * p₂^x₂ * ... * pₖ^xₖ
///
/// # Parameters
///
/// * `n` - Number of trials
/// * `p` - Vector of k probabilities (must sum to 1, all ≥ 0)
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Multinomial, Distribution};
///
/// // 3-sided die rolled 10 times with probabilities [0.3, 0.3, 0.4]
/// let m = Multinomial::new(10, vec![0.3, 0.3, 0.4]).unwrap();
/// println!("P(X = [3, 3, 4]) = {}", m.pmf(&[3, 3, 4]));
/// println!("Mean vector: {:?}", m.mean_vec());
/// ```
#[derive(Debug, Clone)]
pub struct Multinomial {
    /// Number of trials
    n: u64,
    /// Probability vector (length k)
    p: Vec<f64>,
    /// Number of categories
    k: usize,
}

impl Multinomial {
    /// Create a new multinomial distribution.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials
    /// * `p` - Vector of probabilities (must sum to 1, all ≥ 0)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any probability is negative
    /// - Probabilities don't sum to approximately 1.0 (within 1e-9)
    /// - p is empty
    pub fn new(n: u64, p: Vec<f64>) -> StatsResult<Self> {
        if p.is_empty() {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: 0.0,
                reason: "probability vector cannot be empty".to_string(),
            });
        }

        // Check all probabilities are non-negative
        for (i, &prob) in p.iter().enumerate() {
            if prob < 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: format!("p[{}]", i),
                    value: prob,
                    reason: "probability must be non-negative".to_string(),
                });
            }
        }

        // Check probabilities sum to 1 (with tolerance for floating point)
        let sum: f64 = p.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: sum,
                reason: "probabilities must sum to 1.0".to_string(),
            });
        }

        let k = p.len();
        Ok(Self { n, p, k })
    }

    /// Get the number of trials.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Get the probability vector.
    pub fn p(&self) -> &[f64] {
        &self.p
    }

    /// Get the number of categories.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Compute the probability mass function.
    ///
    /// PMF(x) = n! / (x₁! * ... * xₖ!) * p₁^x₁ * ... * pₖ^xₖ
    ///
    /// # Arguments
    ///
    /// * `x` - Vector of outcomes (must have length k and sum to n)
    ///
    /// # Panics
    ///
    /// Panics if x.len() != k
    pub fn pmf(&self, x: &[u64]) -> f64 {
        assert_eq!(x.len(), self.k, "x must have length k");

        // Check that x sums to n
        let sum: u64 = x.iter().sum();
        if sum != self.n {
            return 0.0;
        }

        self.log_pmf(x).exp()
    }

    /// Compute the log probability mass function.
    ///
    /// log(PMF(x)) = log(n!) - Σlog(xᵢ!) + Σ xᵢ * log(pᵢ)
    ///
    /// # Arguments
    ///
    /// * `x` - Vector of outcomes (must have length k and sum to n)
    ///
    /// # Panics
    ///
    /// Panics if x.len() != k
    pub fn log_pmf(&self, x: &[u64]) -> f64 {
        assert_eq!(x.len(), self.k, "x must have length k");

        // Check that x sums to n
        let sum: u64 = x.iter().sum();
        if sum != self.n {
            return f64::NEG_INFINITY;
        }

        // log(n!) - Σlog(xᵢ!)
        let mut log_result = special::lgamma((self.n + 1) as f64);
        for &xi in x.iter() {
            log_result -= special::lgamma((xi + 1) as f64);
        }

        // + Σ xᵢ * log(pᵢ)
        for (xi, pi) in x.iter().zip(self.p.iter()) {
            if *xi > 0 {
                log_result += (*xi as f64) * pi.ln();
            }
        }

        log_result
    }

    /// Compute the mean vector.
    ///
    /// E[Xᵢ] = n * pᵢ
    pub fn mean_vec(&self) -> Vec<f64> {
        let n_f = self.n as f64;
        self.p.iter().map(|&pi| n_f * pi).collect()
    }

    /// Compute the covariance matrix.
    ///
    /// Cov(Xᵢ, Xⱼ) = n * pᵢ * pⱼ * (-1) for i ≠ j
    /// Cov(Xᵢ, Xᵢ) = n * pᵢ * (1 - pᵢ)
    pub fn cov_matrix(&self) -> Vec<Vec<f64>> {
        let n_f = self.n as f64;
        let mut cov = vec![vec![0.0; self.k]; self.k];

        for (i, row) in cov.iter_mut().enumerate().take(self.k) {
            for (j, cell) in row.iter_mut().enumerate().take(self.k) {
                if i == j {
                    *cell = n_f * self.p[i] * (1.0 - self.p[i]);
                } else {
                    *cell = -n_f * self.p[i] * self.p[j];
                }
            }
        }

        cov
    }

    /// Compute the variance of the first category (used for Distribution trait).
    fn var_first(&self) -> f64 {
        self.n as f64 * self.p[0] * (1.0 - self.p[0])
    }
}

impl Distribution for Multinomial {
    fn mean(&self) -> f64 {
        // For scalar methods, use first category
        self.n as f64 * self.p[0]
    }

    fn var(&self) -> f64 {
        // Variance of first category
        self.var_first()
    }

    fn entropy(&self) -> f64 {
        // Multinomial entropy: H = log(n!) - Σ log(xᵢ!) + ...
        // For the distribution, the expected entropy is:
        // H = log(n!) - Σ pᵢ * ψ(n*pᵢ + 1) + (n+1)*ψ(n+1)
        // where ψ is the digamma function
        // Simplified approximation for constant distributions:
        // H ≈ log(n) + Σ (-pᵢ * log(pᵢ))  [entropy of the probability vector]
        let n_f = self.n as f64;

        // Multinomial entropy approximation
        // Start with log(n!)
        let mut h = special::lgamma(n_f + 1.0);

        // Subtract expected value of Σ log(xᵢ!)
        // For large n, E[log(xᵢ!)] ≈ pᵢ * (log(n*pᵢ) - 1 + digamma(n*pᵢ))
        for &pi in self.p.iter() {
            if pi > 0.0 {
                let n_pi = n_f * pi;
                // Approximation: E[log(Xᵢ!)] ≈ n*pᵢ * log(n*pᵢ) - n*pᵢ
                h -= n_pi * (n_pi.ln() - 1.0);
                // More accurate: add digamma term
                h -= n_pi * special::digamma(n_pi + 1.0);
            }
        }

        // Add entropy of the categorical distribution
        // H += -Σ pᵢ * log(pᵢ)
        for &pi in self.p.iter() {
            if pi > 0.0 {
                h += -pi * pi.ln();
            }
        }

        h
    }

    fn median(&self) -> f64 {
        // Median of first category ≈ floor(n * p[0])
        (self.n as f64 * self.p[0]).floor()
    }

    fn mode(&self) -> f64 {
        // Mode of first category ≈ floor((n+1) * p[0])
        ((self.n as f64 + 1.0) * self.p[0]).floor()
    }

    fn skewness(&self) -> f64 {
        let var = self.var_first();
        if var == 0.0 {
            return 0.0;
        }

        // Skewness of first category: (1 - 2*p[0]) / sqrt(n*p[0]*(1-p[0]))
        let numerator = 1.0 - 2.0 * self.p[0];
        numerator / var.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let var = self.var_first();
        if var == 0.0 {
            return 0.0;
        }

        // Excess kurtosis of first category: (1 - 6*p[0]*(1-p[0])) / (n*p[0]*(1-p[0]))
        let numerator = 1.0 - 6.0 * self.p[0] * (1.0 - self.p[0]);
        numerator / var
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multinomial_creation() {
        let m = Multinomial::new(10, vec![0.3, 0.3, 0.4]).unwrap();
        assert_eq!(m.n(), 10);
        assert_eq!(m.k(), 3);
        assert_eq!(m.p(), &[0.3, 0.3, 0.4]);

        // Invalid: probabilities don't sum to 1
        assert!(Multinomial::new(10, vec![0.3, 0.3, 0.3]).is_err());

        // Invalid: negative probability
        assert!(Multinomial::new(10, vec![-0.1, 0.6, 0.5]).is_err());

        // Invalid: empty probability vector
        assert!(Multinomial::new(10, vec![]).is_err());
    }

    #[test]
    fn test_multinomial_mean_vec() {
        let m = Multinomial::new(10, vec![0.2, 0.3, 0.5]).unwrap();
        let mean = m.mean_vec();

        assert!((mean[0] - 2.0).abs() < 1e-10); // 10 * 0.2
        assert!((mean[1] - 3.0).abs() < 1e-10); // 10 * 0.3
        assert!((mean[2] - 5.0).abs() < 1e-10); // 10 * 0.5
    }

    #[test]
    fn test_multinomial_covariance() {
        let m = Multinomial::new(10, vec![0.5, 0.5]).unwrap();
        let cov = m.cov_matrix();

        // Var(X₀) = 10 * 0.5 * 0.5 = 2.5
        assert!((cov[0][0] - 2.5).abs() < 1e-10);

        // Var(X₁) = 10 * 0.5 * 0.5 = 2.5
        assert!((cov[1][1] - 2.5).abs() < 1e-10);

        // Cov(X₀, X₁) = -10 * 0.5 * 0.5 = -2.5
        assert!((cov[0][1] - (-2.5)).abs() < 1e-10);
        assert!((cov[1][0] - (-2.5)).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_pmf() {
        // Fair 3-sided die, 3 trials
        let m = Multinomial::new(3, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]).unwrap();

        // P(X = [1, 1, 1]) = 3! / (1! * 1! * 1!) * (1/3)^3 = 6 * 1/27 = 2/9
        let pmf = m.pmf(&[1, 1, 1]);
        let expected = 6.0 / 27.0;
        assert!((pmf - expected).abs() < 1e-10);

        // P(X = [3, 0, 0]) = 3! / (3! * 0! * 0!) * (1/3)^3 = 1/27
        let pmf = m.pmf(&[3, 0, 0]);
        let expected = 1.0 / 27.0;
        assert!((pmf - expected).abs() < 1e-10);

        // P(X = [2, 1, 0]) = 3! / (2! * 1! * 0!) * (1/3)^3 = 3 * 1/27 = 1/9
        let pmf = m.pmf(&[2, 1, 0]);
        let expected = 3.0 / 27.0;
        assert!((pmf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_pmf_invalid() {
        let m = Multinomial::new(3, vec![0.5, 0.5]).unwrap();

        // x doesn't sum to n
        assert!((m.pmf(&[1, 1]) - 0.0).abs() < 1e-10);

        // x sums to n but wrong length
        // This will panic due to assertion, so we skip this test
    }

    #[test]
    fn test_multinomial_pmf_sums_to_one() {
        let m = Multinomial::new(2, vec![0.4, 0.6]).unwrap();

        // All possible outcomes for n=2
        let outcomes = [vec![0, 2], vec![1, 1], vec![2, 0]];

        let total: f64 = outcomes.iter().map(|x| m.pmf(x)).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_log_pmf() {
        let m = Multinomial::new(3, vec![0.5, 0.5]).unwrap();

        let log_pmf = m.log_pmf(&[1, 2]);
        let pmf = m.pmf(&[1, 2]);

        assert!((log_pmf.exp() - pmf).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_moments() {
        let m = Multinomial::new(100, vec![0.3, 0.7]).unwrap();

        // Mean of first category = 100 * 0.3 = 30
        assert!((m.mean() - 30.0).abs() < 1e-10);

        // Variance of first category = 100 * 0.3 * 0.7 = 21
        assert!((m.var() - 21.0).abs() < 1e-10);

        // Skewness = (1 - 2*0.3) / sqrt(21) = 0.4 / sqrt(21)
        let expected_skew = 0.4 / 21.0_f64.sqrt();
        assert!((m.skewness() - expected_skew).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_edge_case_zero_probability() {
        let m = Multinomial::new(5, vec![1.0, 0.0]).unwrap();

        // All trials go to first category
        assert!((m.pmf(&[5, 0]) - 1.0).abs() < 1e-10);
        assert!((m.pmf(&[4, 1]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_uniform() {
        // Uniform distribution over k categories
        let m = Multinomial::new(4, vec![0.25, 0.25, 0.25, 0.25]).unwrap();

        // P(X = [1, 1, 1, 1]) = 4! / (1!)^4 * (0.25)^4 = 24 * (1/256) = 3/32
        let pmf = m.pmf(&[1, 1, 1, 1]);
        let expected = 24.0 / 256.0;
        assert!((pmf - expected).abs() < 1e-10);
    }
}
