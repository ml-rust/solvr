//! Pareto distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};

/// Pareto distribution (Type I).
///
/// The Pareto distribution is a power-law probability distribution with PDF:
///
/// f(x; α, xₘ) = α * xₘ^α / x^(α+1)  for x ≥ xₘ
///
/// where:
/// - α > 0 is the shape parameter (tail index)
/// - xₘ > 0 is the scale parameter (minimum value)
///
/// The Pareto distribution is used to model many real-world phenomena:
/// - Wealth distribution (Pareto's 80-20 rule)
/// - City population sizes
/// - File sizes, web traffic
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Pareto, ContinuousDistribution};
///
/// let p = Pareto::new(2.0, 1.0).unwrap();  // shape=2, scale=1
/// println!("PDF at 1: {}", p.pdf(1.0));
/// println!("Mean: {}", p.mean());  // 2.0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Pareto {
    /// Shape parameter (tail index, α > 0)
    shape: f64,
    /// Scale parameter (minimum value, xₘ > 0)
    scale: f64,
}

impl Pareto {
    /// Create a new Pareto distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter α (must be > 0)
    /// * `scale` - Scale parameter xₘ (must be > 0)
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

    /// Create a standard Pareto distribution (scale=1).
    pub fn standard(shape: f64) -> StatsResult<Self> {
        Self::new(shape, 1.0)
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

impl Distribution for Pareto {
    fn mean(&self) -> f64 {
        if self.shape <= 1.0 {
            f64::INFINITY
        } else {
            self.shape * self.scale / (self.shape - 1.0)
        }
    }

    fn var(&self) -> f64 {
        if self.shape <= 2.0 {
            f64::INFINITY
        } else {
            let alpha = self.shape;
            let xm = self.scale;
            (xm * xm * alpha) / ((alpha - 1.0).powi(2) * (alpha - 2.0))
        }
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(xₘ/α) + 1 + 1/α
        (self.scale / self.shape).ln() + 1.0 + 1.0 / self.shape
    }

    fn median(&self) -> f64 {
        self.scale * 2.0_f64.powf(1.0 / self.shape)
    }

    fn mode(&self) -> f64 {
        self.scale
    }

    fn skewness(&self) -> f64 {
        if self.shape <= 3.0 {
            f64::NAN
        } else {
            let alpha = self.shape;
            2.0 * (1.0 + alpha) / (alpha - 3.0) * ((alpha - 2.0) / alpha).sqrt()
        }
    }

    fn kurtosis(&self) -> f64 {
        if self.shape <= 4.0 {
            f64::NAN
        } else {
            let alpha = self.shape;
            6.0 * (alpha.powi(3) + alpha.powi(2) - 6.0 * alpha - 2.0)
                / (alpha * (alpha - 3.0) * (alpha - 4.0))
        }
    }
}

impl ContinuousDistribution for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            return 0.0;
        }
        self.shape * self.scale.powf(self.shape) / x.powf(self.shape + 1.0)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            return f64::NEG_INFINITY;
        }
        self.shape.ln() + self.shape * self.scale.ln() - (self.shape + 1.0) * x.ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.scale {
            return 0.0;
        }
        1.0 - (self.scale / x).powf(self.shape)
    }

    fn sf(&self, x: f64) -> f64 {
        if x < self.scale {
            return 1.0;
        }
        (self.scale / x).powf(self.shape)
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
            return Ok(self.scale);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.scale / (1.0 - p).powf(1.0 / self.shape))
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
            return Ok(self.scale);
        }
        Ok(self.scale / p.powf(1.0 / self.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_creation() {
        assert!(Pareto::new(2.0, 1.0).is_ok());
        assert!(Pareto::new(0.0, 1.0).is_err());
        assert!(Pareto::new(2.0, 0.0).is_err());
        assert!(Pareto::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_pareto_pdf() {
        let p = Pareto::new(2.0, 1.0).unwrap();

        // PDF at scale is α/xₘ = 2
        assert!((p.pdf(1.0) - 2.0).abs() < 1e-10);

        // PDF below scale is 0
        assert!((p.pdf(0.5) - 0.0).abs() < 1e-10);

        // PDF at 2: 2 * 1^2 / 2^3 = 0.25
        assert!((p.pdf(2.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_cdf() {
        let p = Pareto::new(2.0, 1.0).unwrap();

        // CDF below scale is 0
        assert!((p.cdf(0.5) - 0.0).abs() < 1e-10);

        // CDF at scale is 0
        assert!((p.cdf(1.0) - 0.0).abs() < 1e-10);

        // CDF at 2: 1 - (1/2)^2 = 0.75
        assert!((p.cdf(2.0) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_ppf() {
        let p = Pareto::new(2.0, 1.0).unwrap();

        // PPF(0) = scale
        assert!((p.ppf(0.0).unwrap() - 1.0).abs() < 1e-10);

        // PPF(1) = infinity
        assert!(p.ppf(1.0).unwrap().is_infinite());

        // Round-trip
        for &x in &[1.0, 1.5, 2.0, 3.0, 10.0] {
            let prob = p.cdf(x);
            assert!((p.ppf(prob).unwrap() - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pareto_mean() {
        // Mean is α*xₘ/(α-1) for α > 1
        let p = Pareto::new(2.0, 1.0).unwrap();
        assert!((p.mean() - 2.0).abs() < 1e-10);

        // Mean is infinite for α ≤ 1
        let p = Pareto::new(1.0, 1.0).unwrap();
        assert!(p.mean().is_infinite());

        let p = Pareto::new(0.5, 1.0).unwrap();
        assert!(p.mean().is_infinite());
    }

    #[test]
    fn test_pareto_variance() {
        // Variance is xₘ²α/((α-1)²(α-2)) for α > 2
        let p = Pareto::new(3.0, 1.0).unwrap();
        let expected = 3.0 / (4.0 * 1.0); // 3 / ((3-1)^2 * (3-2)) = 3/4
        assert!((p.var() - expected).abs() < 1e-10);

        // Variance is infinite for α ≤ 2
        let p = Pareto::new(2.0, 1.0).unwrap();
        assert!(p.var().is_infinite());
    }

    #[test]
    fn test_pareto_median() {
        let p = Pareto::new(2.0, 1.0).unwrap();

        // Median = xₘ * 2^(1/α) = 1 * 2^0.5 ≈ 1.414
        let med = p.median();
        assert!((med - 2.0_f64.sqrt()).abs() < 1e-10);

        // Verify CDF(median) ≈ 0.5
        assert!((p.cdf(med) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_mode() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        assert!((p.mode() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_sf() {
        let p = Pareto::new(2.0, 1.0).unwrap();

        // SF + CDF = 1
        for &x in &[1.0, 1.5, 2.0, 5.0] {
            assert!((p.sf(x) + p.cdf(x) - 1.0).abs() < 1e-10);
        }

        // SF below scale is 1
        assert!((p.sf(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_skewness_kurtosis() {
        // Skewness undefined for α ≤ 3
        let p = Pareto::new(3.0, 1.0).unwrap();
        assert!(p.skewness().is_nan());

        // Kurtosis undefined for α ≤ 4
        let p = Pareto::new(4.0, 1.0).unwrap();
        assert!(p.kurtosis().is_nan());

        // Both defined for α > 4
        let p = Pareto::new(5.0, 1.0).unwrap();
        assert!(!p.skewness().is_nan());
        assert!(!p.kurtosis().is_nan());
    }
}
