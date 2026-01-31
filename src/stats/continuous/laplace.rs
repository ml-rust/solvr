//! Laplace (double exponential) distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{ContinuousDistribution, Distribution};

/// Laplace (double exponential) distribution.
///
/// The Laplace distribution is a continuous probability distribution with PDF:
///
/// f(x; μ, b) = 1/(2b) * exp(-|x - μ|/b)
///
/// where:
/// - μ is the location parameter (mean, median, mode)
/// - b > 0 is the scale parameter
///
/// It can be thought of as two exponential distributions spliced at the mean,
/// or as the distribution of the difference of two i.i.d. exponential random variables.
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Laplace, ContinuousDistribution};
///
/// let l = Laplace::new(0.0, 1.0).unwrap();  // standard Laplace
/// println!("PDF at 0: {}", l.pdf(0.0));     // 0.5
/// println!("CDF at 0: {}", l.cdf(0.0));     // 0.5
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Laplace {
    /// Location parameter (mean)
    loc: f64,
    /// Scale parameter
    scale: f64,
}

impl Laplace {
    /// Create a new Laplace distribution.
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter (mean, median, mode)
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

    /// Create the standard Laplace distribution (loc=0, scale=1).
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

impl Distribution for Laplace {
    fn mean(&self) -> f64 {
        self.loc
    }

    fn var(&self) -> f64 {
        2.0 * self.scale * self.scale
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(2be) = 1 + ln(2b)
        1.0 + (2.0 * self.scale).ln()
    }

    fn median(&self) -> f64 {
        self.loc
    }

    fn mode(&self) -> f64 {
        self.loc
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        // Excess kurtosis is 3
        3.0
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        (-z).exp() / (2.0 * self.scale)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc).abs() / self.scale;
        -z - (2.0 * self.scale).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }

    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        if z < 0.0 {
            1.0 - 0.5 * z.exp()
        } else {
            0.5 * (-z).exp()
        }
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

        let x = if p <= 0.5 {
            self.loc + self.scale * (2.0 * p).ln()
        } else {
            self.loc - self.scale * (2.0 * (1.0 - p)).ln()
        };
        Ok(x)
    }

    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_creation() {
        assert!(Laplace::new(0.0, 1.0).is_ok());
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Laplace::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_laplace_pdf() {
        let l = Laplace::standard();

        // PDF at 0 is 0.5
        assert!((l.pdf(0.0) - 0.5).abs() < 1e-10);

        // Symmetric
        assert!((l.pdf(1.0) - l.pdf(-1.0)).abs() < 1e-10);

        // PDF at 1 is 0.5 * exp(-1) ≈ 0.184
        assert!((l.pdf(1.0) - 0.5 * (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_cdf() {
        let l = Laplace::standard();

        // CDF at median is 0.5
        assert!((l.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF at 0 from both sides
        assert!((l.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF(-1) = 0.5 * exp(-1) ≈ 0.184
        assert!((l.cdf(-1.0) - 0.5 * (-1.0_f64).exp()).abs() < 1e-10);

        // CDF(1) = 1 - 0.5 * exp(-1) ≈ 0.816
        assert!((l.cdf(1.0) - (1.0 - 0.5 * (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_ppf() {
        let l = Laplace::standard();

        // PPF(0.5) = 0 (median)
        assert!((l.ppf(0.5).unwrap() - 0.0).abs() < 1e-10);

        // Round-trip
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = l.cdf(x);
            assert!((l.ppf(p).unwrap() - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplace_moments() {
        let l = Laplace::new(5.0, 2.0).unwrap();

        assert!((l.mean() - 5.0).abs() < 1e-10);
        assert!((l.var() - 8.0).abs() < 1e-10); // 2 * 2^2 = 8
        assert!((l.skewness() - 0.0).abs() < 1e-10);
        assert!((l.kurtosis() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_sf() {
        let l = Laplace::standard();

        // SF + CDF = 1
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!((l.sf(x) + l.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplace_entropy() {
        let l = Laplace::standard();

        // Entropy of standard Laplace is 1 + ln(2)
        assert!((l.entropy() - (1.0 + 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_median_mode() {
        let l = Laplace::new(3.0, 2.0).unwrap();

        assert!((l.median() - 3.0).abs() < 1e-10);
        assert!((l.mode() - 3.0).abs() < 1e-10);
    }
}
