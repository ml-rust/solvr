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

mod convolution;
mod helpers;
mod padding;
mod slice;
mod stft;

// Re-export only what backends need - internal helpers are used within impl_generic
pub use convolution::{convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl};
pub use stft::{istft_impl, spectrogram_impl, stft_impl};
