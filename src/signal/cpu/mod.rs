//! CPU implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for CPU
//! by delegating to the generic implementations in `impl_generic/`.

mod analysis;
mod convolution;
mod filter_apply;
mod frequency_response;
mod spectral;
mod spectrogram;
mod stft;
