//! F distribution (Fisher-Snedecor).

use super::special;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// F distribution (Fisher-Snedecor distribution).
///
/// The F distribution with d1 and d2 degrees of freedom has PDF:
///
/// f(x) = √((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2))
///
/// The F distribution arises as the ratio of two chi-squared random variables
/// divided by their degrees of freedom.
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{FDistribution, ContinuousDistribution};
///
/// // ANOVA F-test with 3 and 20 degrees of freedom
/// let f = FDistribution::new(3.0, 20.0).unwrap();
/// let f_stat = 3.5;
/// let p_value = f.sf(f_stat); // Right-tail probability
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FDistribution {
    /// Numerator degrees of freedom (d1)
    d1: f64,
    /// Denominator degrees of freedom (d2)
    d2: f64,
    /// Log of normalizing constant
    log_norm: f64,
}

impl FDistribution {
    /// Create a new F distribution.
    ///
    /// # Arguments
    ///
    /// * `d1` - Numerator degrees of freedom (must be positive)
    /// * `d2` - Denominator degrees of freedom (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if either parameter is not positive.
    pub fn new(d1: f64, d2: f64) -> StatsResult<Self> {
        if d1 <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "d1".to_string(),
                value: d1,
                reason: "numerator df must be positive".to_string(),
            });
        }
        if d2 <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "d2".to_string(),
                value: d2,
                reason: "denominator df must be positive".to_string(),
            });
        }
        if !d1.is_finite() || !d2.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "d1/d2".to_string(),
                value: d1,
                reason: "parameters must be finite".to_string(),
            });
        }

        // log_norm = (d1/2)*ln(d1) + (d2/2)*ln(d2) - ln(B(d1/2, d2/2))
        let log_norm =
            (d1 / 2.0) * d1.ln() + (d2 / 2.0) * d2.ln() - special::lbeta(d1 / 2.0, d2 / 2.0);

        Ok(Self { d1, d2, log_norm })
    }

    /// Get the numerator degrees of freedom.
    pub fn dfn(&self) -> f64 {
        self.d1
    }

    /// Get the denominator degrees of freedom.
    pub fn dfd(&self) -> f64 {
        self.d2
    }
}

impl Distribution for FDistribution {
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::NAN
        }
    }

    fn var(&self) -> f64 {
        if self.d2 > 4.0 {
            let num = 2.0 * self.d2 * self.d2 * (self.d1 + self.d2 - 2.0);
            let denom = self.d1 * (self.d2 - 2.0).powi(2) * (self.d2 - 4.0);
            num / denom
        } else {
            f64::NAN
        }
    }

    fn entropy(&self) -> f64 {
        let half_d1 = self.d1 / 2.0;
        let half_d2 = self.d2 / 2.0;
        let half_sum = (self.d1 + self.d2) / 2.0;

        (self.d1 / self.d2).ln()
            + special::lbeta(half_d1, half_d2)
            + (1.0 - half_d1) * special::digamma(half_d1)
            - (1.0 + half_d2) * special::digamma(half_d2)
            + half_sum * special::digamma(half_sum)
    }

    fn median(&self) -> f64 {
        self.ppf(0.5).unwrap_or(self.mean())
    }

    fn mode(&self) -> f64 {
        if self.d1 > 2.0 {
            ((self.d1 - 2.0) / self.d1) * (self.d2 / (self.d2 + 2.0))
        } else {
            0.0
        }
    }

    fn skewness(&self) -> f64 {
        if self.d2 > 6.0 {
            let num = (2.0 * self.d1 + self.d2 - 2.0) * (8.0 * (self.d2 - 4.0)).sqrt();
            let denom = (self.d2 - 6.0) * (self.d1 * (self.d1 + self.d2 - 2.0)).sqrt();
            num / denom
        } else {
            f64::NAN
        }
    }

    fn kurtosis(&self) -> f64 {
        if self.d2 > 8.0 {
            let a = self.d1;
            let b = self.d2;
            let num = 12.0 * a * (5.0 * b - 22.0) * (a + b - 2.0) + (b - 4.0) * (b - 2.0).powi(2);
            let denom = a * (b - 6.0) * (b - 8.0) * (a + b - 2.0);
            12.0 * num / denom
        } else {
            f64::NAN
        }
    }
}

impl ContinuousDistribution for FDistribution {
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
        let half_d1 = self.d1 / 2.0;
        let half_d2 = self.d2 / 2.0;

        self.log_norm + (half_d1 - 1.0) * x.ln()
            - (half_d1 + half_d2) * (self.d1 * x + self.d2).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // CDF = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
        let t = self.d1 * x / (self.d1 * x + self.d2);
        special::betainc(self.d1 / 2.0, self.d2 / 2.0, t)
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        // SF = I_{d2/(d1*x+d2)}(d2/2, d1/2)
        let t = self.d2 / (self.d1 * x + self.d2);
        special::betainc(self.d2 / 2.0, self.d1 / 2.0, t)
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

        // Solve: I_t(d1/2, d2/2) = p for t, then x = d2*t / (d1*(1-t))
        let t = special::betaincinv(self.d1 / 2.0, self.d2 / 2.0, p);
        let x = self.d2 * t / (self.d1 * (1.0 - t));

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_creation() {
        let f = FDistribution::new(5.0, 10.0).unwrap();
        assert!((f.dfn() - 5.0).abs() < 1e-10);
        assert!((f.dfd() - 10.0).abs() < 1e-10);

        assert!(FDistribution::new(0.0, 10.0).is_err());
        assert!(FDistribution::new(5.0, 0.0).is_err());
        assert!(FDistribution::new(-1.0, 10.0).is_err());
    }

    #[test]
    fn test_f_moments() {
        let f = FDistribution::new(5.0, 10.0).unwrap();

        // Mean = d2 / (d2 - 2) = 10/8 = 1.25
        assert!((f.mean() - 1.25).abs() < 1e-10);

        // Mode = ((d1-2)/d1) * (d2/(d2+2)) = (3/5) * (10/12) = 0.5
        assert!((f.mode() - 0.5).abs() < 1e-10);

        // Test low df cases
        let f = FDistribution::new(2.0, 3.0).unwrap();
        // d2 = 3 > 2, so mean exists: d2/(d2-2) = 3/1 = 3
        assert!((f.mean() - 3.0).abs() < 1e-10);

        // Mean is undefined when d2 <= 2
        let f = FDistribution::new(5.0, 2.0).unwrap();
        assert!(f.mean().is_nan());
    }

    #[test]
    fn test_f_cdf() {
        let f = FDistribution::new(5.0, 10.0).unwrap();

        assert!((f.cdf(0.0) - 0.0).abs() < 1e-10);

        // F(5,10) critical value at p=0.95 is approximately 3.33
        assert!((f.cdf(3.33) - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_f_ppf() {
        let f = FDistribution::new(5.0, 10.0).unwrap();

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let x = f.ppf(p).unwrap();
            assert!((f.cdf(x) - p).abs() < 1e-5, "Failed for p={}", p);
        }

        // Critical value F(5,10, 0.95) ≈ 3.33
        assert!((f.ppf(0.95).unwrap() - 3.33).abs() < 0.1);
    }

    #[test]
    fn test_f_sf() {
        let f = FDistribution::new(5.0, 10.0).unwrap();

        // SF + CDF = 1
        for x in [0.5, 1.0, 2.0, 3.0] {
            assert!((f.sf(x) + f.cdf(x) - 1.0).abs() < 1e-10);
        }
    }
}
