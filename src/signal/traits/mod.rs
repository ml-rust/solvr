//! Signal processing algorithm traits.
//!
//! This module defines the algorithmic contracts for signal processing operations.
//! Each trait represents a logical group of related algorithms.

pub mod analysis;
pub mod convolution;
pub mod extrema;
pub mod filter_apply;
pub mod frequency_response;
pub mod medfilt;
pub mod spectral;
pub mod spectrogram;
pub mod stft;
pub mod wiener;

pub use analysis::{
    DecimateParams, HilbertResult, PeakParams, PeakResult, SignalAnalysisAlgorithms,
};
pub use convolution::{ConvMode, ConvolutionAlgorithms};
pub use extrema::{ExtremaAlgorithms, ExtremaResult, ExtremumMode};
pub use filter_apply::{FilterApplicationAlgorithms, LfilterResult, PadType, SosfiltResult};
pub use frequency_response::{FrequencyResponseAlgorithms, FreqzResult, FreqzSpec};
pub use medfilt::MedianFilterAlgorithms;
pub use spectral::{
    CoherenceResult, CsdResult, Detrend, PeriodogramParams, PeriodogramResult, PsdScaling,
    SpectralAnalysisAlgorithms, SpectralWindow, WelchParams, WelchResult,
};
pub use spectrogram::SpectrogramAlgorithms;
pub use stft::StftAlgorithms;
pub use wiener::WienerAlgorithms;
