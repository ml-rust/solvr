//! Window function generation algorithms.
//!
//! Pure Rust implementations of window functions as f64 vectors.

use std::f64::consts::PI;

/// Generate Hann window values as f64.
pub fn generate_hann_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

/// Generate Hamming window values as f64.
pub fn generate_hamming_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.54 - 0.46 * x.cos()
        })
        .collect()
}

/// Generate Blackman window values as f64.
pub fn generate_blackman_f64(size: usize) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = 2.0 * PI * (i as f64) / n;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect()
}

/// Generate Kaiser window values as f64.
pub fn generate_kaiser_f64(size: usize, beta: f64) -> Vec<f64> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }

    let n = size as f64;
    let half_n = (n - 1.0) / 2.0;
    let i0_beta = bessel_i0(beta);

    (0..size)
        .map(|i| {
            let x = (i as f64 - half_n) / half_n;
            let arg = beta * (1.0 - x * x).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0.
///
/// Uses the polynomial approximation from Abramowitz & Stegun.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        let exp_ax = ax.exp();
        (exp_ax / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = generate_hann_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0].abs() < 1e-10);
        assert!((window[1] - window[7]).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_window() {
        let window = generate_hamming_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0] > 0.05);
    }

    #[test]
    fn test_blackman_window() {
        let window = generate_blackman_f64(8);
        assert_eq!(window.len(), 8);
        assert!(window[0].abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_window() {
        let window = generate_kaiser_f64(8, 5.0);
        assert_eq!(window.len(), 8);
        for &w in &window {
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_window_size_edge_cases() {
        assert!(generate_hann_f64(0).is_empty());
        assert_eq!(generate_hann_f64(1), vec![1.0]);
    }
}
