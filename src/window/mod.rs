//! Window function algorithms for signal processing
//!
//! This module defines the [`WindowFunctions`] trait that provides window functions
//! commonly used in spectral analysis and signal processing. All window functions
//! use the periodic formulation (suitable for FFT-based analysis).
//!
//! # Supported Windows
//!
//! - **Hann (Hanning)**: General-purpose window with good frequency resolution
//! - **Hamming**: Similar to Hann but with better sidelobe suppression
//! - **Blackman**: Excellent sidelobe suppression at the cost of main lobe width
//! - **Kaiser**: Parametric window with adjustable frequency resolution vs sidelobe tradeoff
//!
//! # Window Function Comparison
//!
//! | Window     | First Sidelobe | Sidelobe Rolloff | Main Lobe Width | Best For |
//! |------------|---------------|------------------|-----------------|----------|
//! | Rectangular| -13 dB        | -6 dB/octave     | Narrowest       | Transient analysis |
//! | Hann       | -31.5 dB      | -18 dB/octave    | Moderate        | General purpose |
//! | Hamming    | -42.7 dB      | -6 dB/octave     | Moderate        | Audio processing |
//! | Blackman   | -58 dB        | -18 dB/octave    | Wide            | High dynamic range |
//! | Kaiser     | Adjustable    | Adjustable       | Adjustable      | Custom requirements |
//!
//! # Choosing a Window
//!
//! - **Hann**: Start here. Best all-around choice for most applications.
//! - **Hamming**: Use when you need consistent sidelobe attenuation (-42 dB floor).
//! - **Blackman**: Use when spectral leakage must be minimized (e.g., detecting weak signals).
//! - **Kaiser**: Use when you need precise control over the resolution/leakage tradeoff.
//!
//! ## Kaiser Beta Guidelines
//!
//! The Kaiser window's `beta` parameter controls the tradeoff:
//!
//! | Beta | Approximate Sidelobe | Equivalent Window |
//! |------|---------------------|-------------------|
//! | 0    | -13 dB              | Rectangular       |
//! | 5    | -50 dB              | Hamming           |
//! | 6    | -60 dB              | Hann              |
//! | 8.6  | -90 dB              | Blackman          |
//! | 14   | -120 dB             | (very narrow)     |
//!
//! # Implementation Notes
//!
//! Window functions are implemented on CPU regardless of the target device, as they are
//! typically small arrays where GPU acceleration provides no benefit. The generated
//! window is transferred to the target device when needed.
//!
//! # Mathematical Definitions (Periodic Formulation)
//!
//! For a window of size N with n = 0, 1, ..., N-1:
//!
//! ```text
//! Hann:     w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
//! Hamming:  w[n] = 0.54 - 0.46 * cos(2*pi*n / N)
//! Blackman: w[n] = 0.42 - 0.5 * cos(2*pi*n / N) + 0.08 * cos(4*pi*n / N)
//! Kaiser:   w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
//! ```
//!
//! Where I0 is the modified Bessel function of the first kind, order 0.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod generators_reexport;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
mod wgpu;

// Re-export traits and validators
pub use traits::{WindowFunctions, validate_window_dtype, validate_window_size};

// Re-export commonly used generator functions for backward compatibility
pub use generators_reexport::{
    bessel_i0, generate_blackman_f64, generate_hamming_f64, generate_hann_f64, generate_kaiser_f64,
};
