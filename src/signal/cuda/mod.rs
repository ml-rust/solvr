//! CUDA implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for CUDA
//! by delegating to the generic implementations in `impl_generic/`.

mod convolution;
mod spectrogram;
mod stft;

pub use convolution::*;
pub use spectrogram::*;
pub use stft::*;
