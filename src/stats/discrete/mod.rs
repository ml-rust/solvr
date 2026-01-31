//! Discrete probability distributions.

mod binomial;
mod discrete_uniform;
mod geometric;
mod hypergeometric;
mod negative_binomial;
mod poisson;

pub use binomial::Binomial;
pub use discrete_uniform::DiscreteUniform;
pub use geometric::Geometric;
pub use hypergeometric::Hypergeometric;
pub use negative_binomial::NegativeBinomial;
pub use poisson::Poisson;

/// Helper for computing log-binomial coefficients.
pub(crate) fn log_binom(n: u64, k: u64) -> f64 {
    use super::continuous::special::lgamma;

    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    let n_f = n as f64;
    let k_f = k as f64;

    lgamma(n_f + 1.0) - lgamma(k_f + 1.0) - lgamma(n_f - k_f + 1.0)
}

/// Compute binomial coefficient C(n, k).
#[allow(dead_code)]
pub(crate) fn binom(n: u64, k: u64) -> f64 {
    log_binom(n, k).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_binom() {
        // C(5, 2) = 10
        assert!((log_binom(5, 2).exp() - 10.0).abs() < 1e-10);

        // C(10, 5) = 252
        assert!((log_binom(10, 5).exp() - 252.0).abs() < 1e-6);

        // Edge cases
        assert!((log_binom(5, 0) - 0.0).abs() < 1e-10); // C(n,0) = 1
        assert!((log_binom(5, 5) - 0.0).abs() < 1e-10); // C(n,n) = 1
        assert!(log_binom(3, 5).is_infinite()); // k > n
    }
}
