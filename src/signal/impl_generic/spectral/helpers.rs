//! Helper functions for spectral analysis algorithms.
//!
//! Window generation for spectral analysis. Uses Tensor operations where possible.
use crate::DType;

use crate::signal::traits::spectral::SpectralWindow;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Generate window coefficients as a tensor.
///
/// Window generation is a setup operation (not hot path) and happens once per
/// spectral computation. The coefficients are computed analytically and placed
/// directly on the target device.
pub fn generate_window<R: Runtime<DType = DType>>(
    window: &SpectralWindow<R>,
    length: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    match window {
        SpectralWindow::Rectangular => Tensor::from_slice(&vec![1.0f64; length], &[length], device),
        SpectralWindow::Hann => {
            let w: Vec<f64> = (0..length)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (length - 1) as f64).cos()))
                .collect();
            Tensor::from_slice(&w, &[length], device)
        }
        SpectralWindow::Hamming => {
            let w: Vec<f64> = (0..length)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (length - 1) as f64).cos())
                .collect();
            Tensor::from_slice(&w, &[length], device)
        }
        SpectralWindow::Blackman => {
            let w: Vec<f64> = (0..length)
                .map(|i| {
                    let n = i as f64 / (length - 1) as f64;
                    0.42 - 0.5 * (2.0 * PI * n).cos() + 0.08 * (4.0 * PI * n).cos()
                })
                .collect();
            Tensor::from_slice(&w, &[length], device)
        }
        SpectralWindow::Kaiser(beta) => {
            let i0_beta = bessel_i0(*beta);
            let w: Vec<f64> = (0..length)
                .map(|i| {
                    let x = 2.0 * i as f64 / (length - 1) as f64 - 1.0;
                    let arg = beta * (1.0 - x * x).sqrt();
                    bessel_i0(arg) / i0_beta
                })
                .collect();
            Tensor::from_slice(&w, &[length], device)
        }
        SpectralWindow::Custom(w) => Ok(w.clone()),
    }
}

/// Modified Bessel function of the first kind, order 0.
///
/// Used for Kaiser window computation.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let ans = 0.39894228
            + y * (0.01328592
                + y * (0.00225319
                    + y * (-0.00157565
                        + y * (0.00916281
                            + y * (-0.02057706
                                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        ans * ax.exp() / ax.sqrt()
    }
}
