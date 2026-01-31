//! Gamma distribution.

use super::special;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Gamma distribution.
///
/// The gamma distribution with shape α and rate β has PDF:
///
/// f(x) = (β^α / Γ(α)) x^(α-1) exp(-βx)  for x > 0
///
/// Alternatively parameterized by shape α and scale θ = 1/β:
///
/// f(x) = (1 / (Γ(α) θ^α)) x^(α-1) exp(-x/θ)
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Gamma, ContinuousDistribution, Distribution};
///
/// // Shape = 2, rate = 1
/// let g = Gamma::new(2.0, 1.0).unwrap();
/// assert!((g.mean() - 2.0).abs() < 1e-10);
///
/// // From shape and scale
/// let g = Gamma::from_shape_scale(2.0, 0.5).unwrap();
/// assert!((g.mean() - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    /// Shape parameter (α)
    alpha: f64,
    /// Rate parameter (β)
    beta: f64,
    /// Log of normalizing constant: α*ln(β) - ln(Γ(α))
    log_norm: f64,
}

impl Gamma {
    /// Create a new gamma distribution with shape α and rate β.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter (must be positive)
    /// * `beta` - Rate parameter (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are not positive.
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha,
                reason: "shape must be positive".to_string(),
            });
        }
        if beta <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "beta".to_string(),
                value: beta,
                reason: "rate must be positive".to_string(),
            });
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "alpha/beta".to_string(),
                value: alpha,
                reason: "parameters must be finite".to_string(),
            });
        }

        let log_norm = alpha * beta.ln() - special::lgamma(alpha);
        Ok(Self {
            alpha,
            beta,
            log_norm,
        })
    }

    /// Create a gamma distribution from shape α and scale θ = 1/β.
    pub fn from_shape_scale(shape: f64, scale: f64) -> StatsResult<Self> {
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "must be positive".to_string(),
            });
        }
        Self::new(shape, 1.0 / scale)
    }

    /// Get the shape parameter α.
    pub fn shape(&self) -> f64 {
        self.alpha
    }

    /// Get the rate parameter β.
    pub fn rate(&self) -> f64 {
        self.beta
    }

    /// Get the scale parameter θ = 1/β.
    pub fn scale(&self) -> f64 {
        1.0 / self.beta
    }
}

impl Distribution for Gamma {
    fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    fn var(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }

    fn entropy(&self) -> f64 {
        // H = α - ln(β) + ln(Γ(α)) + (1-α)ψ(α)
        self.alpha - self.beta.ln()
            + special::lgamma(self.alpha)
            + (1.0 - self.alpha) * special::digamma(self.alpha)
    }

    fn median(&self) -> f64 {
        // No closed form, use PPF
        self.ppf(0.5).unwrap_or(self.mean())
    }

    fn mode(&self) -> f64 {
        if self.alpha >= 1.0 {
            (self.alpha - 1.0) / self.beta
        } else {
            0.0
        }
    }

    fn skewness(&self) -> f64 {
        2.0 / self.alpha.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        6.0 / self.alpha // Excess kurtosis
    }
}

impl ContinuousDistribution for Gamma {
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
        self.log_norm + (self.alpha - 1.0) * x.ln() - self.beta * x
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            special::gammainc(self.alpha, self.beta * x)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else {
            special::gammaincc(self.alpha, self.beta * x)
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
            return Ok(f64::INFINITY);
        }
        Ok(special::gammaincinv(self.alpha, p) / self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_creation() {
        let g = Gamma::new(2.0, 1.0).unwrap();
        assert!((g.shape() - 2.0).abs() < 1e-10);
        assert!((g.rate() - 1.0).abs() < 1e-10);
        assert!((g.scale() - 1.0).abs() < 1e-10);

        let g = Gamma::from_shape_scale(2.0, 0.5).unwrap();
        assert!((g.rate() - 2.0).abs() < 1e-10);

        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::new(1.0, 0.0).is_err());
        assert!(Gamma::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_gamma_moments() {
        let g = Gamma::new(3.0, 2.0).unwrap();

        // Mean = α/β = 3/2
        assert!((g.mean() - 1.5).abs() < 1e-10);

        // Var = α/β² = 3/4
        assert!((g.var() - 0.75).abs() < 1e-10);

        // Mode = (α-1)/β = 2/2 = 1
        assert!((g.mode() - 1.0).abs() < 1e-10);

        // Skewness = 2/√α = 2/√3
        assert!((g.skewness() - 2.0 / 3.0_f64.sqrt()).abs() < 1e-10);

        // Excess kurtosis = 6/α = 2
        assert!((g.kurtosis() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_pdf() {
        // Gamma(1, 1) = Exponential(1)
        let g = Gamma::new(1.0, 1.0).unwrap();
        assert!((g.pdf(0.0) - 0.0).abs() < 1e-10); // PDF at 0 for α=1
        assert!((g.pdf(1.0) - (-1.0_f64).exp()).abs() < 1e-10);

        // Gamma(2, 1) has mode at x = 1
        let g = Gamma::new(2.0, 1.0).unwrap();
        let mode_pdf = g.pdf(1.0);
        assert!(g.pdf(0.5) < mode_pdf);
        assert!(g.pdf(2.0) < mode_pdf);
    }

    #[test]
    fn test_gamma_cdf() {
        // Gamma(1, λ) = Exponential(λ)
        let g = Gamma::new(1.0, 2.0).unwrap();

        assert!((g.cdf(0.0) - 0.0).abs() < 1e-10);

        // For exponential: CDF(x) = 1 - exp(-λx)
        let x: f64 = 1.0;
        let expected = 1.0 - (-2.0 * x).exp();
        assert!((g.cdf(x) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_ppf() {
        let g = Gamma::new(2.0, 1.0).unwrap();

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = g.ppf(p).unwrap();
            assert!((g.cdf(x) - p).abs() < 1e-6, "Failed for p={}", p);
        }

        assert!(g.ppf(-0.1).is_err());
        assert!(g.ppf(1.1).is_err());
    }

    #[test]
    fn test_gamma_special_cases() {
        // Gamma(1, β) = Exponential(β)
        let g = Gamma::new(1.0, 2.0).unwrap();
        assert!((g.mean() - 0.5).abs() < 1e-10);
        assert!((g.var() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_sf() {
        let g = Gamma::new(2.0, 1.0).unwrap();

        // SF + CDF = 1
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((g.sf(x) + g.cdf(x) - 1.0).abs() < 1e-10);
        }
    }
}
