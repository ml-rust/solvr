//! Generic implementations of signal analysis algorithms.
//!
//! GPU-accelerable algorithms (hilbert, resample) live here.
//! CPU-only algorithms (decimate, find_peaks, savgol, extrema, medfilt, wiener)
//! live in cpu/ and use helpers from this module.

pub mod helpers;
mod hilbert;
mod resample;

pub use helpers::{
    apply_butter_lowpass, apply_fir_lowpass, compute_prominences, compute_savgol_coeffs,
    filter_by_distance,
};
pub use hilbert::hilbert_impl;
pub use resample::resample_impl;
