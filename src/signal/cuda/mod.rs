//! CUDA implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for CUDA
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! Note: Some algorithms (extrema, medfilt, wiener) are CPU-only and return
//! errors when called on CUDA tensors.

mod convolution;
mod extrema;
mod medfilt;
mod spectrogram;
mod stft;
mod wiener;

pub use convolution::*;
pub use spectrogram::*;
pub use stft::*;
