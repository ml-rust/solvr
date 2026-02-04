//! Trait definitions and types for integration algorithms.

mod algorithms;
mod types;

pub use algorithms::IntegrationAlgorithms;
pub use types::{
    BVPResult, MonteCarloMethod, MonteCarloOptions, MonteCarloResult, NQuadOptions, QMCOptions,
    QMCSequence, QuadOptions, QuadResult, RombergOptions, SymplecticResult, TanhSinhOptions,
};
