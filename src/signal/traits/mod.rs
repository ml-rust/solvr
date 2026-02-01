//! Signal processing algorithm traits.
//!
//! This module defines the algorithmic contracts for signal processing operations.
//! Each trait represents a logical group of related algorithms.

pub mod convolution;
pub mod spectrogram;
pub mod stft;

pub use convolution::{ConvMode, ConvolutionAlgorithms};
pub use spectrogram::SpectrogramAlgorithms;
pub use stft::StftAlgorithms;
