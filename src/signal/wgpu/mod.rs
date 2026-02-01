//! WebGPU implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for WebGPU
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! # Limitations
//!
//! - Only F32 is supported (WGSL doesn't support F64)

mod convolution;
mod spectrogram;
mod stft;

pub use convolution::*;
pub use spectrogram::*;
pub use stft::*;
