//! WebGPU implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for WebGPU
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! # Limitations
//!
//! - Only F32 is supported (WGSL doesn't support F64)
//! - Some algorithms (extrema, medfilt, wiener) are CPU-only and return errors

mod convolution;
mod extrema;
mod medfilt;
mod spectrogram;
mod stft;
mod wiener;

pub use convolution::*;
pub use spectrogram::*;
pub use stft::*;
