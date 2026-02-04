//! CPU implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for CPU
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! Some algorithms (extrema, medfilt, wiener, decimate, find_peaks, savgol)
//! are CPU-only due to their sequential access patterns.

mod analysis;
mod convolution;
mod extrema;
mod filter_apply;
mod frequency_response;
mod medfilt;
mod spectral;
mod spectrogram;
mod stft;
mod wiener;

// Trait implementations are available when importing the traits.
// No explicit re-exports needed since impls are automatically in scope.
