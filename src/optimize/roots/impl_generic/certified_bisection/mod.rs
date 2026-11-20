//! Certified increasing roots with bounded residual callbacks.

pub(crate) mod admission;
pub(crate) mod classification;
mod controller;
pub(crate) mod convergence;
mod extended;
pub(crate) mod iteration;
pub(crate) mod probe;
use super::certified::{evaluate, native};
mod scalar_tolerance;
pub(crate) mod stopping;

#[cfg(test)]
mod correction_fixtures;
#[cfg(test)]
mod extended_tests;
#[cfg(test)]
mod scalar_tolerance_tests;

pub use controller::bisect_certified_impl;
