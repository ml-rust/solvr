//! Beta distribution.

use super::special;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Beta distribution on [0, 1].
///
/// The beta distribution with shape parameters α and β has PDF:
///
/// f(x) = x^(α-1) (1-x)^(β-1) / B(α, β)  for 0 < x < 1
///
/// where B(α, β) is the beta function.
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Beta, ContinuousDistribution, Distribution};
///
/// // Uniform distribution as Beta(1, 1)
/// let b = Beta::new(1.0, 1.0).unwrap();
/// assert!((b.pdf(0.5) - 1.0).abs() < 1e-10);
///
/// // Arcsine distribution as Beta(0.5, 0.5)
/// let b = Beta::new(0.5, 0.5).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    /// Shape parameter α
    alpha: f64,
    /// Shape parameter β
    beta: f64,
    /// Log of normalizing constant: -ln(B(α, β))
    log_norm: f64,
}

impl Beta {
    /// Create a new beta distribution with shape parameters α and β.
    ///
    /// # Arguments
    ///
    /// * `alpha` - First shape parameter (must be positive)
    /// * `beta` - Second shape parameter (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are not positive.
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha,
                reason: "must be positive".to_string(),
            });
        }
        if beta <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "beta".to_string(),
                value: beta,
                reason: "must be positive".to_string(),
            });
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "alpha/beta".to_string(),
                value: alpha,
                reason: "parameters must be finite".to_string(),
            });
        }

        let log_norm = -special::lbeta(alpha, beta);
        Ok(Self {
            alpha,
            beta,
            log_norm,
        })
    }

    /// Get the first shape parameter α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the second shape parameter β.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Distribution for Beta {
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn var(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    fn entropy(&self) -> f64 {
        let sum = self.alpha + self.beta;
        special::lbeta(self.alpha, self.beta)
            - (self.alpha - 1.0) * special::digamma(self.alpha)
            - (self.beta - 1.0) * special::digamma(self.beta)
            + (sum - 2.0) * special::digamma(sum)
    }

    fn median(&self) -> f64 {
        // No closed form in general; use PPF
        // Approximation for α, β > 1
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0 / 3.0) / (self.alpha + self.beta - 2.0 / 3.0)
        } else {
            self.ppf(0.5).unwrap_or(self.mean())
        }
    }

    fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else if self.alpha <= 1.0 && self.beta > 1.0 {
            0.0
        } else if self.alpha > 1.0 && self.beta <= 1.0 {
            1.0
        } else {
            // α ≤ 1 and β ≤ 1: bimodal at 0 and 1, return NaN
            f64::NAN
        }
    }

    fn skewness(&self) -> f64 {
        let sum = self.alpha + self.beta;
        2.0 * (self.beta - self.alpha) * (sum + 1.0).sqrt()
            / ((sum + 2.0) * (self.alpha * self.beta).sqrt())
    }

    fn kurtosis(&self) -> f64 {
        let sum = self.alpha + self.beta;
        let num = 6.0
            * ((self.alpha - self.beta).powi(2) * (sum + 1.0)
                - self.alpha * self.beta * (sum + 2.0));
        let denom = self.alpha * self.beta * (sum + 2.0) * (sum + 3.0);
        num / denom
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            // Handle boundary cases based on parameters
            if x == 0.0 && self.alpha < 1.0 {
                return f64::INFINITY;
            }
            if x == 1.0 && self.beta < 1.0 {
                return f64::INFINITY;
            }
            if x == 0.0 && self.alpha == 1.0 {
                return self.log_norm.exp();
            }
            if x == 1.0 && self.beta == 1.0 {
                return self.log_norm.exp();
            }
            return 0.0;
        }
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        self.log_norm + (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            special::betainc(self.alpha, self.beta, x)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else if x >= 1.0 {
            0.0
        } else {
            // SF(x; α, β) = I_{1-x}(β, α)
            special::betainc(self.beta, self.alpha, 1.0 - x)
        }
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        Ok(special::betaincinv(self.alpha, self.beta, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_creation() {
        let b = Beta::new(2.0, 3.0).unwrap();
        assert!((b.alpha() - 2.0).abs() < 1e-10);
        assert!((b.beta() - 3.0).abs() < 1e-10);

        assert!(Beta::new(0.0, 1.0).is_err());
        assert!(Beta::new(1.0, 0.0).is_err());
        assert!(Beta::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_beta_uniform() {
        // Beta(1, 1) = Uniform(0, 1)
        let b = Beta::new(1.0, 1.0).unwrap();

        assert!((b.mean() - 0.5).abs() < 1e-10);
        assert!((b.var() - 1.0 / 12.0).abs() < 1e-10);
        assert!((b.pdf(0.5) - 1.0).abs() < 1e-10);
        assert!((b.cdf(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_moments() {
        let b = Beta::new(2.0, 5.0).unwrap();

        // Mean = α/(α+β) = 2/7
        assert!((b.mean() - 2.0 / 7.0).abs() < 1e-10);

        // Var = αβ/((α+β)²(α+β+1)) = 10/(49*8) = 10/392
        assert!((b.var() - 10.0 / 392.0).abs() < 1e-10);

        // Mode = (α-1)/(α+β-2) = 1/5
        assert!((b.mode() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_beta_pdf() {
        let b = Beta::new(2.0, 2.0).unwrap();

        // Symmetric around 0.5
        assert!((b.pdf(0.3) - b.pdf(0.7)).abs() < 1e-10);

        // Mode at 0.5
        let mode_pdf = b.pdf(0.5);
        assert!(b.pdf(0.3) < mode_pdf);
        assert!(b.pdf(0.7) < mode_pdf);

        // PDF = 0 outside [0, 1]
        assert!((b.pdf(-0.1) - 0.0).abs() < 1e-10);
        assert!((b.pdf(1.1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_cdf() {
        let b = Beta::new(2.0, 2.0).unwrap();

        assert!((b.cdf(0.0) - 0.0).abs() < 1e-10);
        assert!((b.cdf(0.5) - 0.5).abs() < 1e-6); // Symmetric
        assert!((b.cdf(1.0) - 1.0).abs() < 1e-10);

        // CDF is monotonic
        assert!(b.cdf(0.3) < b.cdf(0.5));
        assert!(b.cdf(0.5) < b.cdf(0.7));
    }

    #[test]
    fn test_beta_ppf() {
        let b = Beta::new(2.0, 5.0).unwrap();

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = b.ppf(p).unwrap();
            assert!((b.cdf(x) - p).abs() < 1e-6, "Failed for p={}", p);
        }

        assert!(b.ppf(-0.1).is_err());
        assert!(b.ppf(1.1).is_err());
    }

    #[test]
    fn test_beta_sf() {
        let b = Beta::new(2.0, 3.0).unwrap();

        // SF + CDF = 1
        for x in [0.2, 0.4, 0.6, 0.8] {
            assert!((b.sf(x) + b.cdf(x) - 1.0).abs() < 1e-10);
        }
    }
}
