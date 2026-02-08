//! Dirichlet distribution.

use super::special;
use crate::stats::distribution::Distribution;
use crate::stats::error::{StatsError, StatsResult};

/// Dirichlet distribution.
///
/// The Dirichlet distribution is a multivariate generalization of the Beta distribution.
/// It is parameterized by a vector of concentration parameters α = (α₁, ..., αₖ).
///
/// PDF: f(x) = (1/B(α)) ∏ xᵢ^(αᵢ - 1)
///
/// where B(α) = ∏ Γ(αᵢ) / Γ(Σ αᵢ) is the multivariate beta function.
///
/// The support is the (k-1)-simplex: xᵢ > 0, Σ xᵢ = 1.
#[derive(Debug, Clone)]
pub struct Dirichlet {
    /// Concentration parameters
    alpha: Vec<f64>,
    /// Number of categories
    k: usize,
    /// Sum of alpha parameters
    alpha_sum: f64,
    /// Log normalizing constant: Σ lgamma(αᵢ) - lgamma(Σ αᵢ)
    log_beta: f64,
}

impl Dirichlet {
    /// Create a new Dirichlet distribution.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters (all must be positive, length >= 2)
    pub fn new(alpha: Vec<f64>) -> StatsResult<Self> {
        if alpha.len() < 2 {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha.len() as f64,
                reason: "Dirichlet requires at least 2 categories".to_string(),
            });
        }

        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 || !a.is_finite() {
                return Err(StatsError::InvalidParameter {
                    name: format!("alpha[{}]", i),
                    value: a,
                    reason: "concentration parameter must be positive and finite".to_string(),
                });
            }
        }

        let k = alpha.len();
        let alpha_sum: f64 = alpha.iter().sum();
        let log_beta: f64 =
            alpha.iter().map(|&a| special::lgamma(a)).sum::<f64>() - special::lgamma(alpha_sum);

        Ok(Self {
            alpha,
            k,
            alpha_sum,
            log_beta,
        })
    }

    /// Get the concentration parameters.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Get the number of categories.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the sum of concentration parameters.
    pub fn alpha_sum(&self) -> f64 {
        self.alpha_sum
    }

    /// Compute the log-PDF at point x on the simplex.
    ///
    /// x must have length k and all elements must be positive and sum to ~1.
    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.k, "x must have length k");

        let sum: f64 = x.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return f64::NEG_INFINITY;
        }

        for &xi in x {
            if xi <= 0.0 {
                return f64::NEG_INFINITY;
            }
        }

        let mut log_p = -self.log_beta;
        for (xi, ai) in x.iter().zip(self.alpha.iter()) {
            log_p += (ai - 1.0) * xi.ln();
        }
        log_p
    }

    /// Compute the PDF at point x on the simplex.
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Mean vector: E[Xᵢ] = αᵢ / α₀ where α₀ = Σ αᵢ
    pub fn mean_vec(&self) -> Vec<f64> {
        self.alpha.iter().map(|&a| a / self.alpha_sum).collect()
    }

    /// Variance vector: Var(Xᵢ) = αᵢ(α₀ - αᵢ) / (α₀²(α₀ + 1))
    pub fn var_vec(&self) -> Vec<f64> {
        let a0 = self.alpha_sum;
        let denom = a0 * a0 * (a0 + 1.0);
        self.alpha.iter().map(|&a| a * (a0 - a) / denom).collect()
    }

    /// Covariance matrix.
    ///
    /// Cov(Xᵢ, Xⱼ) = -αᵢαⱼ / (α₀²(α₀+1))  for i ≠ j
    /// Var(Xᵢ) = αᵢ(α₀-αᵢ) / (α₀²(α₀+1))
    pub fn cov_matrix(&self) -> Vec<Vec<f64>> {
        let a0 = self.alpha_sum;
        let denom = a0 * a0 * (a0 + 1.0);
        let mut cov = vec![vec![0.0; self.k]; self.k];

        for (i, row) in cov.iter_mut().enumerate().take(self.k) {
            for (j, cell) in row.iter_mut().enumerate().take(self.k) {
                if i == j {
                    *cell = self.alpha[i] * (a0 - self.alpha[i]) / denom;
                } else {
                    *cell = -self.alpha[i] * self.alpha[j] / denom;
                }
            }
        }
        cov
    }

    /// Mode vector: mode_i = (αᵢ - 1) / (α₀ - k) for all αᵢ > 1.
    ///
    /// Returns None if any αᵢ <= 1.
    pub fn mode_vec(&self) -> Option<Vec<f64>> {
        if self.alpha.iter().any(|&a| a <= 1.0) {
            return None;
        }
        let denom = self.alpha_sum - self.k as f64;
        Some(self.alpha.iter().map(|&a| (a - 1.0) / denom).collect())
    }
}

impl Distribution for Dirichlet {
    fn mean(&self) -> f64 {
        self.alpha[0] / self.alpha_sum
    }

    fn var(&self) -> f64 {
        let a0 = self.alpha_sum;
        self.alpha[0] * (a0 - self.alpha[0]) / (a0 * a0 * (a0 + 1.0))
    }

    fn entropy(&self) -> f64 {
        let a0 = self.alpha_sum;
        let mut h = self.log_beta + (a0 - self.k as f64) * special::digamma(a0);
        for &ai in &self.alpha {
            h -= (ai - 1.0) * special::digamma(ai);
        }
        h
    }

    fn median(&self) -> f64 {
        // No closed-form; approximate as mean
        self.mean()
    }

    fn mode(&self) -> f64 {
        if self.alpha[0] > 1.0 {
            (self.alpha[0] - 1.0) / (self.alpha_sum - self.k as f64)
        } else {
            0.0
        }
    }

    fn skewness(&self) -> f64 {
        let a0 = self.alpha_sum;
        let ai = self.alpha[0];
        // Skewness of Xᵢ ~ Beta(αᵢ, α₀ - αᵢ)
        let b = a0 - ai;
        2.0 * (b - ai) * (a0 + 1.0).sqrt() / ((a0 + 2.0) * (ai * b).sqrt())
    }

    fn kurtosis(&self) -> f64 {
        let a0 = self.alpha_sum;
        let ai = self.alpha[0];
        let b = a0 - ai;
        // Excess kurtosis of Beta(αᵢ, α₀ - αᵢ)
        let num = 6.0
            * (ai.powi(3) - ai.powi(2) * (2.0 * b - 1.0) + b.powi(2) * (b + 1.0)
                - 2.0 * ai * b * (b + 2.0));
        let den = ai * b * (a0 + 2.0) * (a0 + 3.0);
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_creation() {
        let d = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(d.k(), 3);
        assert!((d.alpha_sum() - 6.0).abs() < 1e-10);

        // Too few categories
        assert!(Dirichlet::new(vec![1.0]).is_err());
        // Negative alpha
        assert!(Dirichlet::new(vec![1.0, -1.0]).is_err());
        // Zero alpha
        assert!(Dirichlet::new(vec![0.0, 1.0]).is_err());
    }

    #[test]
    fn test_dirichlet_mean() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        let mean = d.mean_vec();
        assert!((mean[0] - 0.2).abs() < 1e-10);
        assert!((mean[1] - 0.3).abs() < 1e-10);
        assert!((mean[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_pdf_at_mean() {
        let d = Dirichlet::new(vec![2.0, 2.0, 2.0]).unwrap();
        let mean = d.mean_vec();
        let pdf_val = d.pdf(&mean);
        assert!(pdf_val > 0.0);
        assert!(pdf_val.is_finite());
    }

    #[test]
    fn test_dirichlet_pdf_outside_simplex() {
        let d = Dirichlet::new(vec![2.0, 3.0]).unwrap();
        // Doesn't sum to 1
        assert_eq!(d.pdf(&[0.3, 0.3]), 0.0);
        // Negative component
        assert_eq!(d.pdf(&[-0.1, 1.1]), 0.0);
    }

    #[test]
    fn test_dirichlet_uniform() {
        // Dirichlet(1, 1, 1) is uniform on the simplex
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        let p1 = d.pdf(&[0.2, 0.3, 0.5]);
        let p2 = d.pdf(&[0.1, 0.1, 0.8]);
        assert!((p1 - p2).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_covariance() {
        let d = Dirichlet::new(vec![1.0, 1.0]).unwrap();
        let cov = d.cov_matrix();
        // Beta(1,1) = Uniform(0,1), var = 1/12
        assert!((cov[0][0] - 1.0 / 12.0).abs() < 1e-10);
        assert!((cov[0][1] - (-1.0 / 12.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_mode() {
        let d = Dirichlet::new(vec![3.0, 5.0, 2.0]).unwrap();
        let mode = d.mode_vec().unwrap();
        // mode_i = (α_i - 1) / (α₀ - k) = (α_i - 1) / 7
        assert!((mode[0] - 2.0 / 7.0).abs() < 1e-10);
        assert!((mode[1] - 4.0 / 7.0).abs() < 1e-10);
        assert!((mode[2] - 1.0 / 7.0).abs() < 1e-10);

        // No mode when alpha_i <= 1
        let d2 = Dirichlet::new(vec![0.5, 2.0]).unwrap();
        assert!(d2.mode_vec().is_none());
    }

    #[test]
    fn test_dirichlet_entropy() {
        // Dirichlet(1,1) = Beta(1,1) = Uniform, entropy = 0
        let d = Dirichlet::new(vec![1.0, 1.0]).unwrap();
        assert!(d.entropy().abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_symmetric_skewness() {
        // Symmetric Dirichlet should have 0 skewness for first component
        // when all alpha are equal
        let d = Dirichlet::new(vec![5.0, 5.0]).unwrap();
        assert!(d.skewness().abs() < 1e-10);
    }
}
