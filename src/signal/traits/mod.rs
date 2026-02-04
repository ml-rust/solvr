//! Signal processing algorithm traits.
//!
//! This module defines the algorithmic contracts for signal processing operations.
//! Each trait represents a logical group of related algorithms.

pub mod analysis;
pub mod convolution;
pub mod filter_apply;
pub mod frequency_response;
pub mod spectral;
pub mod spectrogram;
pub mod stft;

pub use analysis::{
    DecimateParams, HilbertResult, PeakParams, PeakResult, SignalAnalysisAlgorithms,
};
pub use convolution::{ConvMode, ConvolutionAlgorithms};
pub use filter_apply::{FilterApplicationAlgorithms, LfilterResult, PadType, SosfiltResult};
pub use frequency_response::{FrequencyResponseAlgorithms, FreqzResult, FreqzSpec};
pub use spectral::{
    CoherenceResult, CsdResult, Detrend, PeriodogramParams, PeriodogramResult, PsdScaling,
    SpectralAnalysisAlgorithms, SpectralWindow, WelchParams, WelchResult,
};
pub use spectrogram::SpectrogramAlgorithms;
pub use stft::StftAlgorithms;
