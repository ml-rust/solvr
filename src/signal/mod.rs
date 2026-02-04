//! Signal processing algorithms.
//!
//! This module provides signal processing operations including:
//! - 1D and 2D convolution (FFT-based)
//! - 1D and 2D cross-correlation
//! - STFT (Short-Time Fourier Transform)
//! - ISTFT (Inverse STFT)
//! - Spectrogram
//! - Digital filter design (IIR: butter, cheby1, cheby2, ellip, bessel; FIR: firwin)
//! - Filter representation conversions (tf, zpk, sos)
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! signal/
//! ├── mod.rs                # Exports + validation helpers
//! ├── traits/               # Algorithm trait definitions
//! │   ├── convolution.rs
//! │   ├── stft.rs
//! │   └── spectrogram.rs
//! ├── impl_generic/         # Generic implementations (written once)
//! │   ├── convolution.rs
//! │   ├── stft.rs
//! │   ├── helpers.rs
//! │   ├── padding.rs
//! │   └── slice.rs
//! ├── cpu/                  # CPU trait impl (pure delegation)
//! │   ├── convolution.rs
//! │   ├── stft.rs
//! │   └── spectrogram.rs
//! ├── cuda/                 # CUDA trait impl (pure delegation)
//! │   ├── convolution.rs
//! │   ├── stft.rs
//! │   └── spectrogram.rs
//! └── wgpu/                 # WebGPU trait impl (pure delegation)
//!     ├── convolution.rs
//!     ├── stft.rs
//!     └── spectrogram.rs
//! ```
//!
//! # Backend Support
//!
//! - CPU (F32, F64)
//! - CUDA (F32, F64) - requires `cuda` feature
//! - WebGPU (F32 only) - requires `wgpu` feature
//!
//! # Algorithm: FFT-based Convolution
//!
//! ```text
//! convolve(signal, kernel, mode):
//!
//! 1. Compute output length based on mode:
//!    - full: len(signal) + len(kernel) - 1
//!    - same: max(len(signal), len(kernel))
//!    - valid: |len(signal) - len(kernel)| + 1
//!
//! 2. Pad both to next power-of-2 >= (len(signal) + len(kernel) - 1)
//!
//! 3. FFT convolution:
//!    X = rfft(pad(signal, padded_len))
//!    H = rfft(pad(kernel, padded_len))
//!    Y = X * H  (element-wise complex multiply)
//!    result = irfft(Y, n=padded_len)
//!
//! 4. Slice output based on mode
//! ```

mod cpu;
pub mod filter;
pub mod impl_generic;
pub mod traits;
pub mod wavelet;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::dtype::DType;
use numr::error::{Error, Result};

pub use traits::analysis::{
    DecimateFilterImpl, DecimateParams, HilbertResult, PeakParams, PeakResult,
    SignalAnalysisAlgorithms,
};
pub use traits::convolution::ConvMode;
pub use traits::filter_apply::{
    FilterApplicationAlgorithms, LfilterResult, PadType, SosfiltResult,
};
pub use traits::frequency_response::{FrequencyResponseAlgorithms, FreqzResult, FreqzSpec};
pub use traits::spectral::{
    CoherenceResult, CsdResult, Detrend, PeriodogramParams, PeriodogramResult, PsdScaling,
    SpectralAnalysisAlgorithms, SpectralWindow, WelchParams, WelchResult,
};
pub use traits::{ConvolutionAlgorithms, SpectrogramAlgorithms, StftAlgorithms};
pub use wavelet::{
    CwtAlgorithms, CwtResult, DwtAlgorithms, DwtResult, WavedecResult, Wavelet, WaveletFamily,
};

// Re-export filter types and traits
pub use filter::{
    FilterConversions, FilterOutput, FilterType, FirDesignAlgorithms, FirWindow,
    IirDesignAlgorithms, IirDesignResult, SosFilter, SosPairing, TransferFunction, ZpkFilter,
};

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate signal dtype for convolution (must be F32 or F64).
pub fn validate_signal_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate that kernel is 1D.
pub fn validate_kernel_1d(kernel: &[usize], op: &'static str) -> Result<()> {
    if kernel.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "kernel",
            reason: format!("{op} requires 1D kernel, got {}-D", kernel.len()),
        });
    }
    Ok(())
}

/// Validate that kernel is 2D.
pub fn validate_kernel_2d(kernel: &[usize], op: &'static str) -> Result<()> {
    if kernel.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "kernel",
            reason: format!("{op} requires 2D kernel, got {}-D", kernel.len()),
        });
    }
    Ok(())
}

/// Validate STFT parameters.
pub fn validate_stft_params(n_fft: usize, hop_length: usize, op: &'static str) -> Result<()> {
    if n_fft == 0 || !n_fft.is_power_of_two() {
        return Err(Error::InvalidArgument {
            arg: "n_fft",
            reason: format!("{op} requires n_fft to be a positive power of 2, got {n_fft}"),
        });
    }
    if hop_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "hop_length",
            reason: format!("{op} requires hop_length > 0, got {hop_length}"),
        });
    }
    Ok(())
}

/// Calculate next power of 2 >= n.
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// Calculate number of STFT frames.
pub fn stft_num_frames(signal_len: usize, n_fft: usize, hop_length: usize, center: bool) -> usize {
    let padded_len = if center {
        signal_len + n_fft
    } else {
        signal_len
    };

    if padded_len < n_fft {
        0
    } else {
        (padded_len - n_fft) / hop_length + 1
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_num_frames() {
        let frames = stft_num_frames(1000, 256, 64, true);
        let expected = (1000 + 256 - 256) / 64 + 1;
        assert_eq!(frames, expected);
    }
}
