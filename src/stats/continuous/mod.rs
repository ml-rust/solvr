//! Continuous probability distributions.

mod beta;
mod cauchy;
mod chi_squared;
mod exponential;
mod f_distribution;
mod gamma;
mod gumbel;
mod laplace;
mod lognormal;
mod normal;
mod pareto;
mod student_t;
mod uniform;
mod weibull;

pub use beta::Beta;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use f_distribution::FDistribution;
pub use gamma::Gamma;
pub use gumbel::{Gumbel, GumbelMin};
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use normal::Normal;
pub use pareto::Pareto;
pub use student_t::StudentT;
pub use uniform::Uniform;
pub use weibull::Weibull;

/// Helper module for special functions used in distributions.
pub(crate) mod special {
    use numr::algorithm::special::scalar as numr_special;

    /// Standard normal PDF constant: 1/sqrt(2π)
    pub const INV_SQRT_2PI: f64 = 0.3989422804014327;

    /// ln(sqrt(2π))
    pub const LN_SQRT_2PI: f64 = 0.9189385332046727;

    /// Error function using Horner's method approximation.
    /// Accurate to ~1.5e-7.
    #[allow(dead_code)]
    pub fn erf(x: f64) -> f64 {
        numr_special::erf_scalar(x)
    }

    /// Complementary error function: erfc(x) = 1 - erf(x)
    pub fn erfc(x: f64) -> f64 {
        numr_special::erfc_scalar(x)
    }

    /// Inverse error function.
    pub fn erfinv(x: f64) -> f64 {
        numr_special::erfinv_scalar(x)
    }

    /// Standard normal CDF: Φ(x)
    pub fn norm_cdf(x: f64) -> f64 {
        0.5 * erfc(-x / std::f64::consts::SQRT_2)
    }

    /// Standard normal quantile function: Φ⁻¹(p)
    pub fn norm_ppf(p: f64) -> f64 {
        std::f64::consts::SQRT_2 * erfinv(2.0 * p - 1.0)
    }

    /// Gamma function.
    #[allow(dead_code)]
    pub fn gamma(x: f64) -> f64 {
        numr_special::gamma_scalar(x)
    }

    /// Log-gamma function.
    pub fn lgamma(x: f64) -> f64 {
        numr_special::lgamma_scalar(x)
    }

    /// Beta function: B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    #[allow(dead_code)]
    pub fn beta(a: f64, b: f64) -> f64 {
        numr_special::beta_scalar(a, b)
    }

    /// Log-beta function.
    pub fn lbeta(a: f64, b: f64) -> f64 {
        lgamma(a) + lgamma(b) - lgamma(a + b)
    }

    /// Regularized incomplete beta function: I_x(a, b)
    pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
        numr_special::betainc_scalar(a, b, x)
    }

    /// Inverse regularized incomplete beta function.
    pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
        numr_special::betaincinv_scalar(a, b, p)
    }

    /// Regularized lower incomplete gamma function: P(a, x) = γ(a,x)/Γ(a)
    pub fn gammainc(a: f64, x: f64) -> f64 {
        numr_special::gammainc_scalar(a, x)
    }

    /// Regularized upper incomplete gamma function: Q(a, x) = Γ(a,x)/Γ(a) = 1 - P(a,x)
    pub fn gammaincc(a: f64, x: f64) -> f64 {
        numr_special::gammaincc_scalar(a, x)
    }

    /// Inverse of the regularized lower incomplete gamma function.
    pub fn gammaincinv(a: f64, p: f64) -> f64 {
        numr_special::gammaincinv_scalar(a, p)
    }

    /// Digamma (psi) function: ψ(x) = d/dx ln(Γ(x))
    pub fn digamma(x: f64) -> f64 {
        numr_special::digamma_scalar(x)
    }
}
