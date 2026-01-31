//! Gumbel distribution (Extreme Value Type I).

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};

use std::f64::consts::PI;

/// Gumbel distribution (Extreme Value Type I).
///
/// The Gumbel distribution is used to model the maximum (or minimum) of a number
/// of samples of various distributions. It has PDF:
///
/// f(x; μ, β) = (1/β) * exp(-(z + exp(-z)))
///
/// where z = (x - μ)/β and:
/// - μ is the location parameter (mode)
/// - β > 0 is the scale parameter
///
/// There are two types:
/// - Right-skewed (maximum): models the maximum of samples
/// - Left-skewed (minimum): models the minimum of samples
///
/// This implementation is for the right-skewed (maximum) version.
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Gumbel, ContinuousDistribution};
///
/// let g = Gumbel::new(0.0, 1.0).unwrap();  // standard Gumbel
/// println!("PDF at 0: {}", g.pdf(0.0));
/// println!("Mode: {}", g.mode());  // 0.0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gumbel {
    /// Location parameter (mode)
    loc: f64,
    /// Scale parameter
    scale: f64,
}

/// Euler-Mascheroni constant
const EULER_MASCHERONI: f64 = 0.5772156649015329;

impl Gumbel {
    /// Create a new Gumbel distribution (right-skewed / maximum).
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (mode)
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

    /// Create the standard Gumbel distribution (loc=0, scale=1).
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

impl Distribution for Gumbel {
    fn mean(&self) -> f64 {
        // Mean = μ + β*γ where γ is Euler-Mascheroni constant
        self.loc + self.scale * EULER_MASCHERONI
    }

    fn var(&self) -> f64 {
        // Variance = (π²/6) * β²
        (PI * PI / 6.0) * self.scale * self.scale
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(β) + γ + 1
        self.scale.ln() + EULER_MASCHERONI + 1.0
    }

    fn median(&self) -> f64 {
        // Median = μ - β * ln(ln(2))
        self.loc - self.scale * 2.0_f64.ln().ln()
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn skewness(&self) -> f64 {
        // Skewness ≈ 1.14 (constant for Gumbel)
        // Exact value: 12*sqrt(6)*ζ(3)/π³ where ζ(3) ≈ 1.202
        1.1395470994046486
    }

    fn kurtosis(&self) -> f64 {
        // Excess kurtosis = 12/5 = 2.4
        2.4
    }
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let exp_neg_z = (-z).exp();
        (1.0 / self.scale) * exp_neg_z * (-exp_neg_z).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -self.scale.ln() - z - (-z).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (-(-z).exp()).exp()
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
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
        Ok(self.loc - self.scale * (-p.ln()).ln())
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }
}

/// Left-skewed Gumbel distribution (for modeling minimums).
///
/// The left-skewed Gumbel (also called Gumbel minimum) has PDF:
///
/// f(x; μ, β) = (1/β) * exp((z - exp(z)))
///
/// where z = (x - μ)/β
#[derive(Debug, Clone, Copy)]
pub struct GumbelMin {
    /// Location parameter (mode)
    loc: f64,
    /// Scale parameter
    scale: f64,
}

impl GumbelMin {
    /// Create a new left-skewed Gumbel distribution.
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

    /// Create the standard left-skewed Gumbel distribution.
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

impl Distribution for GumbelMin {
    fn mean(&self) -> f64 {
        self.loc - self.scale * EULER_MASCHERONI
    }

    fn var(&self) -> f64 {
        (PI * PI / 6.0) * self.scale * self.scale
    }

    fn entropy(&self) -> f64 {
        self.scale.ln() + EULER_MASCHERONI + 1.0
    }

    fn median(&self) -> f64 {
        self.loc + self.scale * 2.0_f64.ln().ln()
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn skewness(&self) -> f64 {
        // Negative skewness (left-skewed)
        -1.1395470994046486
    }

    fn kurtosis(&self) -> f64 {
        2.4
    }
}

impl ContinuousDistribution for GumbelMin {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let exp_z = z.exp();
        (1.0 / self.scale) * exp_z * (-exp_z).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -self.scale.ln() + z - z.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 - (-z.exp()).exp()
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (-z.exp()).exp()
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
        Ok(self.loc + self.scale * (-(1.0 - p).ln()).ln())
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gumbel_creation() {
        assert!(Gumbel::new(0.0, 1.0).is_ok());
        assert!(Gumbel::new(0.0, 0.0).is_err());
        assert!(Gumbel::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_gumbel_pdf() {
        let g = Gumbel::standard();

        // PDF at mode (x=0) for standard Gumbel
        // f(0) = (1/β) * exp(-z) * exp(-exp(-z)) where z=0
        // f(0) = 1 * exp(0) * exp(-exp(0)) = 1 * 1 * exp(-1) ≈ 0.3679
        let pdf_at_mode = (-1.0_f64).exp();
        assert!((g.pdf(0.0) - pdf_at_mode).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_cdf() {
        let g = Gumbel::standard();

        // CDF(0) = exp(-1) ≈ 0.3679
        assert!((g.cdf(0.0) - (-1.0_f64).exp()).abs() < 1e-10);

        // CDF should be monotonically increasing
        assert!(g.cdf(-1.0) < g.cdf(0.0));
        assert!(g.cdf(0.0) < g.cdf(1.0));
    }

    #[test]
    fn test_gumbel_ppf() {
        let g = Gumbel::standard();

        // Round-trip
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
            let p = g.cdf(x);
            assert!((g.ppf(p).unwrap() - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gumbel_mean() {
        let g = Gumbel::standard();

        // Mean = γ (Euler-Mascheroni constant)
        assert!((g.mean() - EULER_MASCHERONI).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_variance() {
        let g = Gumbel::standard();

        // Variance = π²/6
        assert!((g.var() - PI * PI / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_median() {
        let g = Gumbel::standard();

        // Verify CDF(median) = 0.5
        let med = g.median();
        assert!((g.cdf(med) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_mode() {
        let g = Gumbel::new(5.0, 2.0).unwrap();
        assert!((g.mode() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_sf() {
        let g = Gumbel::standard();

        // SF + CDF = 1
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!((g.sf(x) + g.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gumbel_min_creation() {
        assert!(GumbelMin::new(0.0, 1.0).is_ok());
        assert!(GumbelMin::new(0.0, 0.0).is_err());
    }

    #[test]
    fn test_gumbel_min_pdf() {
        let g = GumbelMin::standard();

        // PDF at mode
        let pdf_at_mode = (-1.0_f64).exp();
        assert!((g.pdf(0.0) - pdf_at_mode).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_min_skewness() {
        let g = Gumbel::standard();
        let gm = GumbelMin::standard();

        // Opposite skewness
        assert!((g.skewness() + gm.skewness()).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_min_ppf() {
        let g = GumbelMin::standard();

        // Round-trip
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = g.cdf(x);
            assert!((g.ppf(p).unwrap() - x).abs() < 1e-10);
        }
    }
}
