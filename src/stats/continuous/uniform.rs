//! Uniform distribution.

use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};

/// Continuous uniform distribution on [a, b].
///
/// The uniform distribution has constant PDF on the interval [a, b]:
///
/// f(x) = 1 / (b - a)  for a ≤ x ≤ b
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{Uniform, ContinuousDistribution, Distribution};
///
/// // Standard uniform U(0, 1)
/// let u = Uniform::standard();
/// assert!((u.pdf(0.5) - 1.0).abs() < 1e-10);
/// assert!((u.cdf(0.5) - 0.5).abs() < 1e-10);
///
/// // Custom uniform U(2, 8)
/// let u = Uniform::new(2.0, 8.0).unwrap();
/// assert!((u.mean() - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    /// Lower bound
    a: f64,
    /// Upper bound
    b: f64,
    /// Range (b - a)
    range: f64,
}

impl Uniform {
    /// Create a new uniform distribution on [a, b].
    ///
    /// # Errors
    ///
    /// Returns an error if a >= b or if bounds are not finite.
    pub fn new(a: f64, b: f64) -> StatsResult<Self> {
        if !a.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "a".to_string(),
                value: a,
                reason: "must be finite".to_string(),
            });
        }
        if !b.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "b".to_string(),
                value: b,
                reason: "must be finite".to_string(),
            });
        }
        if a >= b {
            return Err(StatsError::InvalidParameter {
                name: "a, b".to_string(),
                value: a,
                reason: format!("a must be less than b (got a={}, b={})", a, b),
            });
        }
        Ok(Self { a, b, range: b - a })
    }

    /// Create a standard uniform distribution U(0, 1).
    pub fn standard() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            range: 1.0,
        }
    }

    /// Get the lower bound.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get the upper bound.
    pub fn b(&self) -> f64 {
        self.b
    }
}

impl Distribution for Uniform {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn var(&self) -> f64 {
        self.range * self.range / 12.0
    }

    fn entropy(&self) -> f64 {
        self.range.ln()
    }

    fn median(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn mode(&self) -> f64 {
        // Any value in [a, b] is a mode; return midpoint
        (self.a + self.b) / 2.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        -6.0 / 5.0 // Excess kurtosis
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / self.range
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            f64::NEG_INFINITY
        } else {
            -self.range.ln()
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.a {
            0.0
        } else if x > self.b {
            1.0
        } else {
            (x - self.a) / self.range
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x < self.a {
            1.0
        } else if x > self.b {
            0.0
        } else {
            (self.b - x) / self.range
        }
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        Ok(self.a + p * self.range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_creation() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert!((u.a() - 0.0).abs() < 1e-10);
        assert!((u.b() - 1.0).abs() < 1e-10);

        assert!(Uniform::new(1.0, 0.0).is_err());
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_standard_uniform() {
        let u = Uniform::standard();
        assert!((u.a() - 0.0).abs() < 1e-10);
        assert!((u.b() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_pdf() {
        let u = Uniform::new(2.0, 5.0).unwrap();

        // PDF = 1/3 on [2, 5]
        assert!((u.pdf(3.0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((u.pdf(2.0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((u.pdf(5.0) - 1.0 / 3.0).abs() < 1e-10);

        // PDF = 0 outside [2, 5]
        assert!((u.pdf(1.9) - 0.0).abs() < 1e-10);
        assert!((u.pdf(5.1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_cdf() {
        let u = Uniform::new(2.0, 5.0).unwrap();

        assert!((u.cdf(2.0) - 0.0).abs() < 1e-10);
        assert!((u.cdf(3.5) - 0.5).abs() < 1e-10);
        assert!((u.cdf(5.0) - 1.0).abs() < 1e-10);

        // Outside support
        assert!((u.cdf(1.0) - 0.0).abs() < 1e-10);
        assert!((u.cdf(6.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_ppf() {
        let u = Uniform::new(2.0, 5.0).unwrap();

        assert!((u.ppf(0.0).unwrap() - 2.0).abs() < 1e-10);
        assert!((u.ppf(0.5).unwrap() - 3.5).abs() < 1e-10);
        assert!((u.ppf(1.0).unwrap() - 5.0).abs() < 1e-10);

        // PPF should be inverse of CDF
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = u.ppf(p).unwrap();
            assert!((u.cdf(x) - p).abs() < 1e-10);
        }
    }

    #[test]
    fn test_uniform_moments() {
        let u = Uniform::new(2.0, 8.0).unwrap();

        assert!((u.mean() - 5.0).abs() < 1e-10);
        assert!((u.var() - 3.0).abs() < 1e-10); // (8-2)²/12 = 36/12 = 3
        assert!((u.median() - 5.0).abs() < 1e-10);
        assert!((u.skewness() - 0.0).abs() < 1e-10);
        assert!((u.kurtosis() - (-1.2)).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_entropy() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        assert!((u.entropy() - 10.0_f64.ln()).abs() < 1e-10);
    }
}
