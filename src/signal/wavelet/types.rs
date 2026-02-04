//! Wavelet types and families.

use std::f64::consts::PI;

/// Wavelet family enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveletFamily {
    /// Haar wavelet (simplest orthogonal wavelet).
    #[default]
    Haar,
    /// Daubechies wavelets (dbN where N is the order).
    Daubechies(usize),
    /// Symlet wavelets (symN, near-symmetric Daubechies).
    Symlet(usize),
    /// Coiflet wavelets (coifN).
    Coiflet(usize),
    /// Morlet wavelet (for CWT).
    Morlet,
    /// Mexican Hat wavelet (Ricker, for CWT).
    MexicanHat,
}

/// Wavelet with filter coefficients.
#[derive(Debug, Clone)]
pub struct Wavelet {
    /// Wavelet family.
    pub family: WaveletFamily,
    /// Low-pass decomposition filter.
    pub dec_lo: Vec<f64>,
    /// High-pass decomposition filter.
    pub dec_hi: Vec<f64>,
    /// Low-pass reconstruction filter.
    pub rec_lo: Vec<f64>,
    /// High-pass reconstruction filter.
    pub rec_hi: Vec<f64>,
}

impl Wavelet {
    /// Create a wavelet from a family specification.
    pub fn new(family: WaveletFamily) -> Self {
        let (dec_lo, dec_hi, rec_lo, rec_hi) = match family {
            WaveletFamily::Haar => haar_coefficients(),
            WaveletFamily::Daubechies(n) => daubechies_coefficients(n),
            WaveletFamily::Symlet(n) => symlet_coefficients(n),
            WaveletFamily::Coiflet(n) => coiflet_coefficients(n),
            WaveletFamily::Morlet | WaveletFamily::MexicanHat => {
                // CWT wavelets don't have discrete filter coefficients
                // Use empty vectors; CWT uses continuous wavelet function
                (vec![], vec![], vec![], vec![])
            }
        };

        Self {
            family,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }

    /// Get filter length.
    pub fn filter_length(&self) -> usize {
        self.dec_lo.len()
    }

    /// Check if this is a CWT wavelet.
    pub fn is_cwt_wavelet(&self) -> bool {
        matches!(
            self.family,
            WaveletFamily::Morlet | WaveletFamily::MexicanHat
        )
    }

    /// Evaluate CWT wavelet at given points.
    /// For discrete wavelets, returns None.
    pub fn evaluate(&self, t: &[f64], scale: f64) -> Option<Vec<f64>> {
        match self.family {
            WaveletFamily::Morlet => {
                // Morlet wavelet: exp(-t^2/2) * cos(5t)
                let omega = 5.0;
                Some(
                    t.iter()
                        .map(|&ti| {
                            let x = ti / scale;
                            let envelope = (-x * x / 2.0).exp();
                            envelope * (omega * x).cos() / scale.sqrt()
                        })
                        .collect(),
                )
            }
            WaveletFamily::MexicanHat => {
                // Mexican Hat (Ricker): (1 - t^2) * exp(-t^2/2)
                // Normalized: 2/(sqrt(3)*pi^(1/4)) * (1 - t^2) * exp(-t^2/2)
                let norm = 2.0 / (3.0_f64.sqrt() * PI.powf(0.25));
                Some(
                    t.iter()
                        .map(|&ti| {
                            let x = ti / scale;
                            let x2 = x * x;
                            norm * (1.0 - x2) * (-x2 / 2.0).exp() / scale.sqrt()
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }
}

/// Generate Haar wavelet coefficients.
fn haar_coefficients() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let dec_lo = vec![sqrt2_inv, sqrt2_inv];
    let dec_hi = vec![sqrt2_inv, -sqrt2_inv];
    let rec_lo = vec![sqrt2_inv, sqrt2_inv];
    let rec_hi = vec![-sqrt2_inv, sqrt2_inv];
    (dec_lo, dec_hi, rec_lo, rec_hi)
}

/// Generate Daubechies wavelet coefficients.
fn daubechies_coefficients(order: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Daubechies coefficients for common orders
    let coeffs = match order {
        1 => vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()], // Same as Haar
        2 => vec![
            0.48296291314469025,
            0.836516303737469,
            0.22414386804185735,
            -0.12940952255092145,
        ],
        3 => vec![
            0.3326705529509569,
            0.8068915093133388,
            0.4598775021193313,
            -0.13501102001039084,
            -0.08544127388224149,
            0.035226291882100656,
        ],
        4 => vec![
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            -0.02798376941698385,
            -0.18703481171888114,
            0.030841381835986965,
            0.032883011666982945,
            -0.010597401784997278,
        ],
        _ => {
            // For unsupported orders, fall back to db2
            vec![
                0.48296291314469025,
                0.836516303737469,
                0.22414386804185735,
                -0.12940952255092145,
            ]
        }
    };

    orthogonal_filter_bank(&coeffs)
}

/// Generate Symlet wavelet coefficients.
fn symlet_coefficients(order: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Symlet coefficients (near-symmetric Daubechies)
    let coeffs = match order {
        2 => vec![
            -0.12940952255092145,
            0.22414386804185735,
            0.836516303737469,
            0.48296291314469025,
        ],
        3 => vec![
            0.035226291882100656,
            -0.08544127388224149,
            -0.13501102001039084,
            0.4598775021193313,
            0.8068915093133388,
            0.3326705529509569,
        ],
        4 => vec![
            -0.07576571478927333,
            -0.02963552764599851,
            0.49761866763201545,
            0.8037387518059161,
            0.29785779560527736,
            -0.09921954357684722,
            -0.012603967262037833,
            0.0322231006040427,
        ],
        _ => {
            // Fall back to sym2
            vec![
                -0.12940952255092145,
                0.22414386804185735,
                0.836516303737469,
                0.48296291314469025,
            ]
        }
    };

    orthogonal_filter_bank(&coeffs)
}

/// Generate Coiflet wavelet coefficients.
fn coiflet_coefficients(order: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Coiflet coefficients
    let coeffs = match order {
        1 => vec![
            -0.01565572813546454,
            -0.0727326195128539,
            0.38486484686420286,
            0.8525720202122554,
            0.337_897_662_457_809_2,
            -0.0727326195128539,
        ],
        2 => vec![
            -0.0007205494453645122,
            -0.0018232088707029932,
            0.0056114348193944995,
            0.023680171946334084,
            -0.0594344186464569,
            -0.0764885990783064,
            0.41700518442169254,
            0.8127236354455423,
            0.3861100668211622,
            -0.06737255472196302,
            -0.04146493678175915,
            0.016387336463522112,
        ],
        _ => {
            // Fall back to coif1
            vec![
                -0.01565572813546454,
                -0.0727326195128539,
                0.38486484686420286,
                0.8525720202122554,
                0.337_897_662_457_809_2,
                -0.0727326195128539,
            ]
        }
    };

    orthogonal_filter_bank(&coeffs)
}

/// Create orthogonal filter bank from low-pass coefficients.
fn orthogonal_filter_bank(lo: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let _n = lo.len();
    let dec_lo = lo.to_vec();

    // High-pass decomposition: alternating flip
    let dec_hi: Vec<f64> = lo
        .iter()
        .enumerate()
        .map(|(i, &c)| if i % 2 == 0 { -c } else { c })
        .rev()
        .collect();

    // Reconstruction filters are time-reversed
    let rec_lo: Vec<f64> = lo.iter().rev().cloned().collect();
    let rec_hi: Vec<f64> = dec_hi.iter().rev().cloned().collect();

    (dec_lo, dec_hi, rec_lo, rec_hi)
}
