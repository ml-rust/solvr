//! Generic signal processing implementations.
//!
//! This module provides Runtime-generic implementations of signal processing
//! algorithms. All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! All signal processing operations are fully tensor-based - data stays on device
//! with no GPU->CPU->GPU roundtrips in algorithm loops. Operations use numr's
//! tensor ops: `narrow`, `pad`, `mul`, `add`, `cat`, `rfft`, `irfft`, etc.
//!
//! The key benefit: **zero code duplication** across backends. CPU, CUDA, and
//! WebGPU all use these same implementations.

mod analysis;
mod convolution;
mod frequency_response;
mod helpers;
mod padding;
mod slice;
mod spectral;
mod stft;

// Re-export only what backends need
// GPU-accelerable algorithms:
pub use analysis::{hilbert_impl, resample_impl};
// CPU helper functions used by cpu/ implementations:
pub use analysis::{
    apply_butter_lowpass, apply_fir_lowpass, compute_prominences, compute_savgol_coeffs,
    filter_by_distance,
};
// CPU-only algorithms (decimate, find_peaks, savgol, extrema, medfilt, wiener) live in cpu/
pub use convolution::{convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl};
pub use frequency_response::{freqz_impl, group_delay_impl, sosfreqz_impl};
pub use spectral::{coherence_impl, csd_impl, lombscargle_impl, periodogram_impl, welch_impl};
pub use stft::{istft_impl, spectrogram_impl, stft_impl};

// Note: filter_apply (lfilter, filtfilt, sosfilt, sosfiltfilt) is CPU-only
// because IIR filtering is inherently sequential. See cpu/filter_apply.rs.
