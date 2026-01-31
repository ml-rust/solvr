//! Weibull distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};

use super::special::lgamma;

/// Weibull distribution.
///
/// The Weibull distribution is a continuous probability distribution with PDF:
///
/// f(x; k, λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)  for x ≥ 0
///
/// where:
/// - k > 0 is the shape parameter
/// - λ > 0 is the scale parameter
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Weibull, ContinuousDistribution};
///
/// let w = Weibull::new(2.0, 1.0).unwrap();  // shape=2, scale=1
/// println!("PDF at 1.0: {}", w.pdf(1.0));
/// println!("CDF at 1.0: {}", w.cdf(1.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Weibull {
    /// Shape parameter (k > 0)
    shape: f64,
    /// Scale parameter (λ > 0)
    scale: f64,
}

impl Weibull {
    /// Create a new Weibull distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter k (must be > 0)
    /// * `scale` - Scale parameter λ (must be > 0)
    pub fn new(shape: f64, scale: f64) -> StatsResult<Self> {
        if shape <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "shape".to_string(),
                value: shape,
                reason: "shape parameter must be positive".to_string(),
            });
        }
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "scale parameter must be positive".to_string(),
            });
        }
        Ok(Self { shape, scale })
    }

    /// Create an exponential distribution (Weibull with shape=1).
    pub fn exponential(scale: f64) -> StatsResult<Self> {
        Self::new(1.0, scale)
    }

    /// Create a Rayleigh distribution (Weibull with shape=2).
    pub fn rayleigh(scale: f64) -> StatsResult<Self> {
        Self::new(2.0, scale)
    }

    /// Get the shape parameter.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution for Weibull {
    fn mean(&self) -> f64 {
        self.scale
            * (1.0 + 1.0 / self.shape)
                .exp()
                .powf(lgamma(1.0 + 1.0 / self.shape).exp().ln())
    }

    fn var(&self) -> f64 {
        let k = self.shape;
        let lambda = self.scale;
        let g1 = lgamma(1.0 + 1.0 / k).exp();
        let g2 = lgamma(1.0 + 2.0 / k).exp();
        lambda * lambda * (g2 - g1 * g1)
    }

    fn entropy(&self) -> f64 {
        let k = self.shape;
        let lambda = self.scale;
        let euler_mascheroni = 0.5772156649015329;
        euler_mascheroni * (1.0 - 1.0 / k) + (lambda / k).ln() + 1.0
    }

    fn median(&self) -> f64 {
        self.scale * std::f64::consts::LN_2.powf(1.0 / self.shape)
    }

    fn mode(&self) -> f64 {
        if self.shape <= 1.0 {
            0.0
        } else {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        }
    }

    fn skewness(&self) -> f64 {
        let k = self.shape;
        let g1 = lgamma(1.0 + 1.0 / k).exp();
        let g2 = lgamma(1.0 + 2.0 / k).exp();
        let g3 = lgamma(1.0 + 3.0 / k).exp();
        let mu = g1;
        let sigma2 = g2 - g1 * g1;
        let sigma = sigma2.sqrt();
        (g3 - 3.0 * mu * sigma2 - mu.powi(3)) / sigma.powi(3)
    }

    fn kurtosis(&self) -> f64 {
        let k = self.shape;
        let g1 = lgamma(1.0 + 1.0 / k).exp();
        let g2 = lgamma(1.0 + 2.0 / k).exp();
        let g3 = lgamma(1.0 + 3.0 / k).exp();
        let g4 = lgamma(1.0 + 4.0 / k).exp();
        let mu = g1;
        let sigma2 = g2 - g1 * g1;
        let mu3 = g3 - 3.0 * mu * sigma2 - mu.powi(3);
        (g4 - 4.0 * mu3 * mu - 6.0 * sigma2 * mu * mu - mu.powi(4)) / sigma2.powi(2) - 3.0
    }
}

impl ContinuousDistribution for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            if self.shape < 1.0 {
                return f64::INFINITY;
            } else if self.shape == 1.0 {
                return 1.0 / self.scale;
            } else {
                return 0.0;
            }
        }

        let k = self.shape;
        let lambda = self.scale;
        let z = x / lambda;
        (k / lambda) * z.powf(k - 1.0) * (-z.powf(k)).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            if self.shape < 1.0 {
                return f64::INFINITY;
            } else if self.shape == 1.0 {
                return -self.scale.ln();
            } else {
                return f64::NEG_INFINITY;
            }
        }

        let k = self.shape;
        let lambda = self.scale;
        k.ln() - lambda.ln() + (k - 1.0) * (x / lambda).ln() - (x / lambda).powf(k)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        1.0 - (-(x / self.scale).powf(self.shape)).exp()
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-(x / self.scale).powf(self.shape)).exp()
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.scale * (-((1.0 - p).ln())).powf(1.0 / self.shape))
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(f64::INFINITY);
        }
        if p == 1.0 {
            return Ok(0.0);
        }
        Ok(self.scale * (-(p.ln())).powf(1.0 / self.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_creation() {
        assert!(Weibull::new(1.0, 1.0).is_ok());
        assert!(Weibull::new(0.0, 1.0).is_err());
        assert!(Weibull::new(1.0, 0.0).is_err());
        assert!(Weibull::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_weibull_pdf() {
        let w = Weibull::new(2.0, 1.0).unwrap();

        // PDF(0) = 0 for k > 1
        assert!((w.pdf(0.0) - 0.0).abs() < 1e-10);

        // PDF(1) for k=2, λ=1 is 2 * exp(-1) ≈ 0.7358
        assert!((w.pdf(1.0) - 2.0 * (-1.0_f64).exp()).abs() < 1e-10);

        // PDF negative is 0
        assert!((w.pdf(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_cdf() {
        let w = Weibull::new(2.0, 1.0).unwrap();

        assert!((w.cdf(0.0) - 0.0).abs() < 1e-10);
        assert!((w.cdf(-1.0) - 0.0).abs() < 1e-10);

        // CDF(1) for k=2, λ=1 is 1 - exp(-1) ≈ 0.6321
        assert!((w.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_ppf() {
        let w = Weibull::new(2.0, 1.0).unwrap();

        assert!((w.ppf(0.0).unwrap() - 0.0).abs() < 1e-10);
        assert!(w.ppf(1.0).unwrap().is_infinite());

        // Round-trip
        let x = 0.5;
        let p = w.cdf(x);
        assert!((w.ppf(p).unwrap() - x).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_median() {
        let w = Weibull::new(2.0, 1.0).unwrap();

        // Median should satisfy CDF(median) = 0.5
        let med = w.median();
        assert!((w.cdf(med) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_special_case() {
        // Weibull with k=1 is exponential
        let w = Weibull::exponential(2.0).unwrap();
        assert!((w.shape() - 1.0).abs() < 1e-10);

        // PDF of exponential at x=0 is 1/λ
        assert!((w.pdf(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_sf() {
        let w = Weibull::new(2.0, 1.0).unwrap();

        // SF + CDF = 1
        let x = 1.5;
        assert!((w.sf(x) + w.cdf(x) - 1.0).abs() < 1e-10);
    }
}
