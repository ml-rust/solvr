//! Trait definitions and types for integration algorithms.

mod algorithms;
mod certified_simpson;
mod certified_simpson_types;
pub mod events;
mod types;

pub use algorithms::IntegrationAlgorithms;
pub use events::EventFunction;
pub use types::{
    BVPResult, MonteCarloMethod, MonteCarloOptions, MonteCarloResult, NQuadOptions, QMCOptions,
    QMCSequence, QuadOptions, QuadResult, RombergOptions, SymplecticResult, TanhSinhOptions,
};

pub use certified_simpson::CertifiedSimpsonAlgorithms;
pub use certified_simpson_types::{
    CertifiedSimpsonOptions, CertifiedSimpsonPanel, CertifiedSimpsonQuery, CertifiedSimpsonResult,
    CertifiedSimpsonValues,
};
