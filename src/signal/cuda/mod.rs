//! CUDA implementation of signal processing algorithms.
//!
//! This module implements the signal processing algorithm traits for CUDA
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! Note: Some algorithms (extrema, medfilt, wiener) are CPU-only and return
//! errors when called on CUDA tensors.

mod convolution;
mod edge;
mod extrema;
mod medfilt;
mod nd_filters;
mod spectrogram;
mod stft;
mod wiener;
