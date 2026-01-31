//! Cauchy (Lorentz) distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};

use std::f64::consts::PI;

/// Cauchy (Lorentz) distribution.
///
/// The Cauchy distribution is a continuous probability distribution with PDF:
///
/// f(x; x₀, γ) = 1 / (πγ * [1 + ((x - x₀)/γ)²])
///
/// where:
/// - x₀ is the location parameter (median)
/// - γ > 0 is the scale parameter (half-width at half-maximum)
///
/// Notable properties:
/// - The mean, variance, and higher moments are undefined
/// - Heavy tails (polynomial decay vs exponential for normal)
/// - The ratio of two independent standard normals follows a Cauchy distribution
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Cauchy, ContinuousDistribution};
///
/// let c = Cauchy::new(0.0, 1.0).unwrap();  // standard Cauchy
/// println!("PDF at 0: {}", c.pdf(0.0));    // 1/π ≈ 0.318
/// println!("CDF at 0: {}", c.cdf(0.0));    // 0.5
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Cauchy {
    /// Location parameter (median)
    loc: f64,
    /// Scale parameter (half-width at half-maximum)
    scale: f64,
}

impl Cauchy {
    /// Create a new Cauchy distribution.
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (median)
    /// * `scale` - Scale parameter (must be > 0)
    pub fn new(loc: f64, scale: f64) -> StatsResult<Self> {
        if scale <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
                reason: "scale parameter must be positive".to_string(),
            });
        }
        Ok(Self { loc, scale })
    }

    /// Create the standard Cauchy distribution (loc=0, scale=1).
    pub fn standard() -> Self {
        Self {
            loc: 0.0,
            scale: 1.0,
        }
    }

    /// Get the location parameter.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution for Cauchy {
    fn mean(&self) -> f64 {
        // Mean is undefined for Cauchy
        f64::NAN
    }

    fn var(&self) -> f64 {
        // Variance is undefined for Cauchy
        f64::NAN
    }

    fn std(&self) -> f64 {
        f64::NAN
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(4πγ)
        (4.0 * PI * self.scale).ln()
    }

    fn median(&self) -> f64 {
        self.loc
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn skewness(&self) -> f64 {
        // Skewness is undefined
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        // Kurtosis is undefined
        f64::NAN
    }
}

impl ContinuousDistribution for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -(PI * self.scale).ln() - (1.0 + z * z).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 + z.atan() / PI
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        0.5 - z.atan() / PI
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
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.loc + self.scale * (PI * (p - 0.5)).tan())
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }
        self.ppf(1.0 - p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cauchy_creation() {
        assert!(Cauchy::new(0.0, 1.0).is_ok());
        assert!(Cauchy::new(0.0, 0.0).is_err());
        assert!(Cauchy::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_cauchy_pdf() {
        let c = Cauchy::standard();

        // PDF at mode is 1/π
        assert!((c.pdf(0.0) - 1.0 / PI).abs() < 1e-10);

        // Symmetric
        assert!((c.pdf(1.0) - c.pdf(-1.0)).abs() < 1e-10);

        // PDF at ±1 is 1/(2π)
        assert!((c.pdf(1.0) - 1.0 / (2.0 * PI)).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_cdf() {
        let c = Cauchy::standard();

        // CDF at median is 0.5
        assert!((c.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF(-∞) → 0, CDF(+∞) → 1
        assert!(c.cdf(-1e10) < 0.001);
        assert!(c.cdf(1e10) > 0.999);

        // CDF(1) = 0.75 for standard Cauchy
        assert!((c.cdf(1.0) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_ppf() {
        let c = Cauchy::standard();

        // PPF(0.5) = 0 (median)
        assert!((c.ppf(0.5).unwrap() - 0.0).abs() < 1e-10);

        // PPF(0.75) = 1
        assert!((c.ppf(0.75).unwrap() - 1.0).abs() < 1e-10);

        // PPF(0.25) = -1
        assert!((c.ppf(0.25).unwrap() + 1.0).abs() < 1e-10);

        // Round-trip
        let x = 2.5;
        let p = c.cdf(x);
        assert!((c.ppf(p).unwrap() - x).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_undefined_moments() {
        let c = Cauchy::standard();

        assert!(c.mean().is_nan());
        assert!(c.var().is_nan());
        assert!(c.std().is_nan());
        assert!(c.skewness().is_nan());
        assert!(c.kurtosis().is_nan());
    }

    #[test]
    fn test_cauchy_location_scale() {
        let c = Cauchy::new(5.0, 2.0).unwrap();

        // Median is at location
        assert!((c.median() - 5.0).abs() < 1e-10);

        // CDF at location is 0.5
        assert!((c.cdf(5.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_sf() {
        let c = Cauchy::standard();

        // SF + CDF = 1
        let x = 1.5;
        assert!((c.sf(x) + c.cdf(x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_entropy() {
        let c = Cauchy::standard();

        // Entropy of standard Cauchy is ln(4π)
        assert!((c.entropy() - (4.0 * PI).ln()).abs() < 1e-10);
    }
}
