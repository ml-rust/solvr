//! Student's t distribution.

use super::special;
use crate::stats::distribution::{ContinuousDistribution, Distribution};
use crate::stats::error::{StatsError, StatsResult};
use std::f64::consts::PI;

/// Student's t distribution.
///
/// The Student's t distribution with ν degrees of freedom has PDF:
///
/// f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) (1 + x²/ν)^(-(ν+1)/2)
///
/// As ν → ∞, the t distribution approaches the standard normal.
///
/// # Examples
///
/// ```ignore
/// use solvr::stats::{StudentT, ContinuousDistribution, Distribution};
///
/// let t = StudentT::new(10.0).unwrap();
/// println!("95th percentile: {}", t.ppf(0.95).unwrap());
///
/// // Two-tailed critical value for α = 0.05
/// let alpha = 0.05;
/// let t_crit = t.ppf(1.0 - alpha/2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    /// Degrees of freedom (ν)
    nu: f64,
    /// Log of normalizing constant
    log_norm: f64,
}

impl StudentT {
    /// Create a new Student's t distribution with ν degrees of freedom.
    ///
    /// # Arguments
    ///
    /// * `nu` - Degrees of freedom (must be positive)
    ///
    /// # Errors
    ///
    /// Returns an error if nu is not positive.
    pub fn new(nu: f64) -> StatsResult<Self> {
        if nu <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "nu".to_string(),
                value: nu,
                reason: "degrees of freedom must be positive".to_string(),
            });
        }
        if !nu.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "nu".to_string(),
                value: nu,
                reason: "must be finite".to_string(),
            });
        }

        // log_norm = ln(Γ((ν+1)/2)) - ln(√(νπ)) - ln(Γ(ν/2))
        let log_norm =
            special::lgamma((nu + 1.0) / 2.0) - 0.5 * (nu * PI).ln() - special::lgamma(nu / 2.0);

        Ok(Self { nu, log_norm })
    }

    /// Get the degrees of freedom.
    pub fn df(&self) -> f64 {
        self.nu
    }
}

impl Distribution for StudentT {
    fn mean(&self) -> f64 {
        if self.nu > 1.0 { 0.0 } else { f64::NAN }
    }

    fn var(&self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else if self.nu > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }

    fn entropy(&self) -> f64 {
        let half_nu = self.nu / 2.0;
        let half_nu_plus_1 = (self.nu + 1.0) / 2.0;
        half_nu_plus_1 * (special::digamma(half_nu_plus_1) - special::digamma(half_nu))
            + 0.5 * (self.nu * PI).ln()
            + special::lbeta(half_nu, 0.5)
    }

    fn median(&self) -> f64 {
        0.0
    }

    fn mode(&self) -> f64 {
        0.0
    }

    fn skewness(&self) -> f64 {
        if self.nu > 3.0 { 0.0 } else { f64::NAN }
    }

    fn kurtosis(&self) -> f64 {
        if self.nu > 4.0 {
            6.0 / (self.nu - 4.0)
        } else if self.nu > 2.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

impl ContinuousDistribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: f64) -> f64 {
        self.log_norm - ((self.nu + 1.0) / 2.0) * (1.0 + x * x / self.nu).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == 0.0 {
            return 0.5;
        }

        // Use incomplete beta function
        // CDF(x) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2) for x > 0
        // CDF(x) = 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2) for x < 0
        let x2 = x * x;
        let t = self.nu / (self.nu + x2);
        let beta_inc = special::betainc(self.nu / 2.0, 0.5, t);

        if x > 0.0 {
            1.0 - 0.5 * beta_inc
        } else {
            0.5 * beta_inc
        }
    }

    fn sf(&self, x: f64) -> f64 {
        // Use symmetry: SF(x) = CDF(-x)
        self.cdf(-x)
    }

    fn ppf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        if p == 0.5 {
            return Ok(0.0);
        }

        // Use inverse incomplete beta function
        // For p > 0.5: x = √(ν * (1/I⁻¹ - 1))
        // For p < 0.5: x = -√(ν * (1/I⁻¹ - 1))
        let (q, sign) = if p > 0.5 {
            (2.0 * (1.0 - p), 1.0)
        } else {
            (2.0 * p, -1.0)
        };

        let t = special::betaincinv(self.nu / 2.0, 0.5, q);
        let x = sign * (self.nu * (1.0 / t - 1.0)).sqrt();

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_t_creation() {
        let t = StudentT::new(10.0).unwrap();
        assert!((t.df() - 10.0).abs() < 1e-10);

        assert!(StudentT::new(0.0).is_err());
        assert!(StudentT::new(-1.0).is_err());
    }

    #[test]
    fn test_student_t_moments() {
        let t = StudentT::new(10.0).unwrap();

        assert!((t.mean() - 0.0).abs() < 1e-10);
        assert!((t.var() - 10.0 / 8.0).abs() < 1e-10);
        assert!((t.median() - 0.0).abs() < 1e-10);
        assert!((t.mode() - 0.0).abs() < 1e-10);
        assert!((t.skewness() - 0.0).abs() < 1e-10);
        assert!((t.kurtosis() - 1.0).abs() < 1e-10); // 6/(10-4) = 1
    }

    #[test]
    fn test_student_t_pdf_symmetry() {
        let t = StudentT::new(5.0).unwrap();

        // PDF should be symmetric around 0
        for x in [0.5, 1.0, 2.0, 3.0] {
            assert!((t.pdf(x) - t.pdf(-x)).abs() < 1e-10);
        }

        // Maximum at x = 0
        assert!(t.pdf(0.0) > t.pdf(1.0));
    }

    #[test]
    fn test_student_t_cdf() {
        let t = StudentT::new(10.0).unwrap();

        // CDF(0) = 0.5 by symmetry
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-10);

        // CDF(-x) + CDF(x) = 1
        for x in [0.5, 1.0, 2.0] {
            assert!((t.cdf(-x) + t.cdf(x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_student_t_ppf() {
        let t = StudentT::new(10.0).unwrap();

        // PPF(0.5) = 0
        assert!((t.ppf(0.5).unwrap() - 0.0).abs() < 1e-10);

        // PPF should be inverse of CDF (roundtrip test)
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let x = t.ppf(p).unwrap();
            assert!(
                (t.cdf(x) - p).abs() < 1e-4,
                "Roundtrip failed for p={}: cdf(ppf(p)) = {}",
                p,
                t.cdf(x)
            );
        }

        // Critical value t(10, 0.975) ≈ 2.228
        assert!((t.ppf(0.975).unwrap() - 2.228).abs() < 0.01);
    }

    #[test]
    fn test_student_t_convergence_to_normal() {
        // As ν → ∞, t distribution approaches standard normal
        let t = StudentT::new(1000.0).unwrap();
        let normal_cdf_1 = 0.8413447460685429; // Φ(1)

        assert!((t.cdf(1.0) - normal_cdf_1).abs() < 0.01);
    }

    #[test]
    fn test_student_t_low_df() {
        // t(1) = Cauchy distribution
        let t = StudentT::new(1.0).unwrap();
        assert!(t.mean().is_nan());
        assert!(t.var().is_nan());

        // t(2) has infinite variance
        let t = StudentT::new(2.0).unwrap();
        assert!((t.mean() - 0.0).abs() < 1e-10);
        assert!(t.var().is_infinite());
    }
}
